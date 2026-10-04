"""No-reward calibration of clearance, odor transmission and motor sensitivity.

Two fixed inhibitory gains are declared before native loading. They are
not searched using game scores. A clearing but unresponsive circuit does not
pass; neither does a responsive circuit with self-sustaining post-cue activity.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.balanced_native import BalancedNativeBrain, BalanceConfig, VERSION
from flysnake.research.paired import writer_lock, digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as parent_sources, inputs, immutable_arrays


def sources():
    result = parent_sources()
    for name in ('scripts/assay_native_balance.py', 'flysnake/research/input_isolation.py',
                 'flysnake/research/physical_native.py', 'flysnake/research/balanced_native.py'):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output / 'status.json'
        if path.exists(): raise ValueError('Preserve previous physical evidence')
        source = sources()
        report = {'protocol': VERSION, 'status': 'loading', 'sources': source, 'process_id': os.getpid(),
                  'started_at': time.time(), 'rows': [], 'learning': False, 'game_played': False,
                  'settings': {'inhibitory_gains': [2., 4.], 'excitatory_gain': 1., 'pre_ms': 300, 'cue_ms': 300,
                               'wash_ms': 3000, 'motor_current_mV': 12.,
                               'selection': 'No-reward clearance/transmission gates; no game return.'}}
        for name in source:
            target = args.output / 'sources' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        atomic_json(path, report)
        try:
            brain = BalancedNativeBrain(); b = brain.b
            unique_pre = np.unique(brain.pre); report['original_brain'] = brain.identity.copy()
            samples = []; phase_totals = []
            for scale in (2., 4.):
                brain.set_balance(BalanceConfig(inhibitory_gain=scale))
                condition = {'configuration': asdict(brain.balance_config), 'brain': brain.identity.copy(),
                             'cues': [], 'perturbed': []}
                for cue, rgb, concentration in inputs(brain)[1:]:
                    brain.reset_trial(); rows = []
                    for phase, bins, physical in (('pre', 3, np.zeros((2, 2))),
                                                 ('cue', 3, concentration),
                                                 ('wash', 30, np.zeros((2, 2)))):
                        total = np.zeros(b.n, np.int32)
                        for index in range(bins):
                            receipt = brain.integrate(rgb, 100, physical, learning=False)
                            emitted = brain.last_counts.copy(); total += emitted; samples.append(emitted[unique_pre])
                            if phase != 'cue':
                                expected = np.zeros(b.n, np.float32)
                                expected[np.r_[brain.groups['left'], brain.groups['right']]] = 12.
                                if not np.array_equal(b.drive, expected):
                                    raise AssertionError('Undeclared external drive during clearance calibration')
                            rows.append({'phase': phase, 'bin': index + 1, 'receipt': receipt,
                                         'gamma1_KC_active': int(np.count_nonzero(emitted[unique_pre])),
                                         'gamma1_KC_spikes': int(emitted[unique_pre].sum())})
                            report.update(status='running', phase=f'{scale}/{cue}/{phase}/{index+1}',
                                          active_rows=rows, updated_at=time.time()); atomic_json(path, report)
                        phase_totals.append(total)
                    tail = [r for r in rows if r['phase'] == 'wash'][-5:]
                    cued = [r for r in rows if r['phase'] == 'cue']
                    condition['cues'].append({'cue': cue, 'rows': rows,
                        'last_500ms_gamma1_KC_spikes': sum(r['gamma1_KC_spikes'] for r in tail),
                        'last_500ms_PPL101_spikes': np.sum([r['receipt']['readouts']['PPL101'] for r in tail], axis=0).tolist(),
                        'cue_gamma1_KC_spikes': sum(r['gamma1_KC_spikes'] for r in cued),
                        'cue_MBON11_spikes': np.sum([r['receipt']['readouts']['gamma1_MBON11'] for r in cued], axis=0).tolist(),
                        'cue_motor_counts': {k: np.sum([r['receipt']['counts'][k] for r in cued], axis=0).tolist()
                                             for k in brain.groups}})
                if digest(b.weight) != brain.initial_all_weights_sha256:
                    raise AssertionError('Frozen physics calibration changed weights')
                # Existing gamma1 intervention checks whether the retained route
                # can still influence downstream cells at this coupling level.
                baseline = []
                for cue, rgb, concentration in inputs(brain)[1:]:
                    brain.reset_trial(); baseline.append(brain.integrate(rgb, 300, concentration))
                brain.set_fraction(np.full(len(brain.edges), .8))
                altered = []
                for cue, rgb, concentration in inputs(brain)[1:]:
                    brain.reset_trial(); altered.append(brain.integrate(rgb, 300, concentration))
                brain.reset_memory()
                condition['perturbed'] = [{'cue': cue, 'before': a, 'after': c}
                                          for cue, a, c in zip(('DM1', 'DM4'), baseline, altered)]
                condition['gates'] = {
                    'odor_reaches_gamma1_KC': all(r['cue_gamma1_KC_spikes'] > 0 for r in condition['cues']),
                    'odor_reaches_gamma1_MBON': all(sum(r['cue_MBON11_spikes']) > 0 for r in condition['cues']),
                    'postcue_gamma1_clears': all(r['last_500ms_gamma1_KC_spikes'] == 0 and
                                                not any(r['last_500ms_PPL101_spikes']) for r in condition['cues']),
                    'gamma1_intervention_changes_MBON': any(a['readouts']['gamma1_MBON11'] != c['readouts']['gamma1_MBON11']
                                                            for a, c in zip(baseline, altered)),
                    'gamma1_intervention_changes_motor_counts': any(a['counts'] != c['counts'] for a, c in zip(baseline, altered)),
                    'gamma1_intervention_changes_DNa02_counts': any(any(a['counts'][side] != c['counts'][side] for side in ('left', 'right')) for a, c in zip(baseline, altered))}
                condition['all_gates_pass'] = all(condition['gates'].values())
                report['rows'].append(condition); report.pop('active_rows', None); atomic_json(path, report)
                print(json.dumps({'inhibitory_gain': scale, 'gates': condition['gates'],
                                  'cues': [{k: v for k, v in r.items() if k != 'rows'} for r in condition['cues']]}), flush=True)
            report['native_spikes'] = {'file': 'native-spikes.npz', 'sha256': immutable_arrays(
                args.output / 'native-spikes.npz', counts=np.array(phase_totals), gamma1_counts=np.array(samples),
                cell_ids=b.ids, gamma1_KC_indices=unique_pre)}
            report.update(status='complete', completed_at=time.time(),
                          interpretation='Whole-graph excitation/inhibition balance sensitivity only. No biologically fitted parameter or learned game score.')
            atomic_json(path, report); print(json.dumps({'status': 'complete'}), flush=True)
        except Exception:
            report.update(status='error', error=traceback.format_exc()); atomic_json(path, report); raise


if __name__ == '__main__': main()
