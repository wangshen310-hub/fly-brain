"""Full-native stimulus clearance and readout-current causal control.

Weights remain frozen. No reward, optimizer, game or action teacher. The test
checks whether the retained model returns to the pre-cue population state when
all declared sensory currents are removed, with and without motor tonic drive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.input_isolation import IsolatedInputBrain, InputConfig
from flysnake.research.paired import writer_lock, digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as parent_sources, inputs, immutable_arrays

VERSION = 'native-frozen-stimulus-clearance-motor-current-control-v1'


def sources():
    result = parent_sources()
    for name in ('scripts/assay_native_clearance.py', 'flysnake/research/input_isolation.py'):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output / 'status.json'
        if path.exists(): raise ValueError('Preserve previous clearance evidence')
        source = sources()
        report = {'protocol': VERSION, 'status': 'loading', 'sources': source, 'process_id': os.getpid(),
                  'started_at': time.time(), 'rows': [], 'learning': False, 'game_played': False,
                  'settings': {'pre_ms': 300, 'cue_ms': 300, 'wash_ms': 3000, 'bin_ms': 100,
                               'motor_tonic_mV': [12., 0.], 'central_extra_mV': 0.,
                               'vision_input': False, 'lamina_bias_mV': 0., 'inactive_odor_current': 0.}}
        for name in source:
            target = args.output / 'sources' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        atomic_json(path, report)
        try:
            brain = IsolatedInputBrain(inputs=InputConfig(vision_input=False, lamina_bias_mV=0.,
                                                         zero_odor_is_silent=True))
            b = brain.b; report['brain'] = brain.identity
            unique_pre = np.unique(brain.pre)
            motors = np.r_[brain.groups['left'], brain.groups['right']]
            apl = np.array([np.flatnonzero(b.ids == cell)[0] for cell in (10540, 10977)])
            selected = np.unique(np.r_[unique_pre, apl, *brain.memory_readouts.values(),
                                      *[ix for groups in brain.orn.values() for ix in groups.values()]])
            samples = []; phase_totals = []; blank = np.full((180, 320, 3), 85, np.uint8)
            for motor_current in (12., 0.):
                b.tonic[motors] = motor_current
                for cue, rgb, concentration in inputs(brain)[1:]:
                    brain.reset_trial(); rows = []
                    for phase, bins, physical in (('pre', 3, np.zeros((2, 2))),
                                                 ('cue', 3, concentration),
                                                 ('wash', 30, np.zeros((2, 2)))):
                        total = np.zeros(b.n, np.int32)
                        for index in range(bins):
                            receipt = brain.integrate(blank, 100, physical, learning=False)
                            emitted = brain.last_counts.copy(); samples.append(emitted[selected]); total += emitted
                            if phase != 'cue':
                                expected = np.zeros(b.n, np.float32); expected[motors] = motor_current
                                if not np.array_equal(b.drive, expected):
                                    raise AssertionError('Undeclared external drive during stimulus removal')
                            row = {'phase': phase, 'bin': index + 1, 'receipt': receipt,
                                   'gamma1_KC_active': int(np.count_nonzero(emitted[unique_pre])),
                                   'gamma1_KC_spikes': int(emitted[unique_pre].sum()),
                                   'APL_spikes': emitted[apl].tolist(),
                                   'nonzero_external_drive_ids': b.ids[np.flatnonzero(b.drive)].tolist(),
                                   'nonzero_external_drive_mV': b.drive[np.flatnonzero(b.drive)].tolist()}
                            rows.append(row)
                            report.update(status='running', phase=f'{motor_current}/{cue}/{phase}/{index+1}',
                                          updated_at=time.time(), active_rows=rows)
                            atomic_json(path, report)
                            if phase == 'wash' and (index + 1) % 5 == 0:
                                print(json.dumps({'motor_tonic': motor_current, 'cue': cue,
                                    'wash_ms': 100 * (index + 1), 'gamma1_KC_spikes': row['gamma1_KC_spikes'],
                                    'PPL101': receipt['readouts']['PPL101']}), flush=True)
                        phase_totals.append(total)
                    tail = [r for r in rows if r['phase'] == 'wash'][-5:]
                    result = {'motor_tonic_mV': motor_current, 'cue': cue, 'rows': rows,
                              'final_500ms_gamma1_KC_spikes': sum(r['gamma1_KC_spikes'] for r in tail),
                              'final_500ms_PPL101_spikes': np.sum([r['receipt']['readouts']['PPL101'] for r in tail], axis=0).tolist()}
                    result['returns_to_silent_gamma1_baseline'] = (
                        result['final_500ms_gamma1_KC_spikes'] == 0 and not any(result['final_500ms_PPL101_spikes']))
                    report['rows'].append(result); report.pop('active_rows', None); atomic_json(path, report)
            if digest(b.weight) != brain.initial_all_weights_sha256:
                raise AssertionError('Frozen clearance test changed graph weights')
            report['native_spikes'] = {'file': 'native-spikes.npz', 'sha256': immutable_arrays(
                args.output / 'native-spikes.npz', counts=np.array(phase_totals), selected_counts=np.array(samples),
                selected_indices=selected, cell_ids=b.ids, gamma1_KC_indices=unique_pre)}
            report.update(status='complete', completed_at=time.time(), all_weights_unchanged=True,
                          interpretation='Cue removal and motor-current causal control only; no learning or gameplay.')
            atomic_json(path, report); print(json.dumps({'status': 'complete'}), flush=True)
        except Exception:
            report.update(status='error', error=traceback.format_exc()); atomic_json(path, report); raise


if __name__ == '__main__': main()
