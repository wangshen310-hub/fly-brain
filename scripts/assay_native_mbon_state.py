"""Frozen membrane and afferent-budget measurement, no additional tuning.

Uses three already-tested physical conditions. v/g/drive are actual native
states. Weighted presynaptic spike budgets are explicitly potential deliveries:
the 1.8ms delay and refractory rejection are not reconstructed from count bins.
No budget is used to choose an action or update a neural parameter.
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
from flysnake.research.tonic_apl_native import TonicAPLBrain, TonicConfig, VERSION
from flysnake.research.apl_native_v2 import APLConfig
from flysnake.research.paired import writer_lock, digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as parent_sources, inputs, immutable_arrays


def sources():
    result = parent_sources()
    for name in ('scripts/assay_native_mbon_state.py', 'flysnake/research/input_isolation.py',
                 'flysnake/research/physical_native.py', 'flysnake/research/apl_native_v2.py',
                 'flysnake/research/tonic_apl_native.py'):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output / 'status.json'
        if path.exists(): raise ValueError('Preserve previous membrane evidence')
        source = sources()
        report = {'protocol': 'native-gamma1-membrane-and-potential-budget-v1', 'status': 'loading',
                  'sources': source, 'process_id': os.getpid(), 'started_at': time.time(),
                  'learning': False, 'game_played': False, 'rows': [],
                  'settings': {'pre_ms': 300, 'cue_ms': 300, 'wash_ms': 1000,
                               'bin_ms': 10, 'conditions': [[1., 0.], [2., 0.], [2., 5.]],
                               'budget_scope': 'Actual spike counts times existing weights; potential deliveries only.'}}
        for name in source:
            target = args.output / 'sources' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        atomic_json(path, report)
        try:
            brain = TonicAPLBrain(); b = brain.b
            from stonkfly.neural.common import annotations
            a = annotations(b.ids); types = a.type.fillna('unknown').astype(str).to_numpy()
            target = brain.memory_readouts['gamma1_MBON11']
            incoming = np.flatnonzero(np.isin(b.post, target)).astype(np.int64)
            pre = (np.searchsorted(b.ptr, incoming, side='right')-1).astype(np.int32)
            incoming = incoming[~b.modulation_mask[pre].astype(bool)]
            pre = (np.searchsorted(b.ptr, incoming, side='right')-1).astype(np.int32)
            unique_pre = np.unique(brain.pre); trace = []; totals = []
            report['brain'] = brain.identity
            for apl_gain, intrinsic in ((1., 0.), (2., 0.), (2., 5.)):
                brain.set_apl(APLConfig(inhibitory_gain=apl_gain))
                brain.set_mbon_current(TonicConfig(mbon_current_mV=intrinsic))
                # KC and APL masks use actual identities; remaining sources are
                # split by retained edge sign, including target-specific signs.
                groups = np.where(b.weight[incoming] >= 0, 2, 3)
                groups[np.isin(pre, brain.apl)] = 1; groups[b.circuit['kc_mask'][pre].astype(bool)] = 0
                group_names = ['KC', 'APL', 'other_positive', 'other_negative']
                for cue, rgb, concentration in inputs(brain)[1:]:
                    brain.reset_trial(); rows = []; phase_counts = []; phase_budgets = []
                    for phase, bins, physical in (('pre', 30, np.zeros((2, 2))), ('cue', 30, concentration),
                                                 ('wash', 100, np.zeros((2, 2)))):
                        aggregate = np.zeros(b.n, np.int32); by_edge = np.zeros(len(incoming), float)
                        for index in range(bins):
                            receipt = brain.integrate(rgb, 10, physical, learning=False)
                            c = brain.last_counts.copy(); aggregate += c
                            budget = b.weight[incoming].astype(float) * c[pre]; by_edge += budget
                            rows.append({'phase': phase, 'bin': index+1, 'full_counts_sha256': digest(c),
                                'gamma1_KC_spikes': int(c[unique_pre].sum()), 'MBON_spikes': c[target].tolist(),
                                'PPL101_spikes': receipt['readouts']['PPL101'],
                                'v_mV': b.v[target].tolist(), 'g_mV': b.g[target].tolist(),
                                'drive_mV': b.drive[target].tolist(), 'rest_mV': b.rest[target].tolist(),
                                'potential_signed_budget': {name: float(budget[groups == j].sum())
                                                           for j, name in enumerate(group_names)}})
                            trace.append(np.c_[b.v[target], b.g[target], b.drive[target]])
                        phase_counts.append(aggregate); totals.append(aggregate)
                        categories = {}
                        for kind in np.unique(types[pre]):
                            categories[kind] = float(by_edge[types[pre] == kind].sum())
                        positive = sorted(((k, v) for k, v in categories.items() if v > 0), key=lambda x: -x[1])[:8]
                        negative = sorted(((k, v) for k, v in categories.items() if v < 0), key=lambda x: x[1])[:8]
                        phase_budgets.append({'phase': phase,
                            'group_totals': {name: float(by_edge[groups == j].sum()) for j, name in enumerate(group_names)},
                            'largest_positive_source_types': positive, 'largest_negative_source_types': negative})
                    result = {'apl_gain': apl_gain, 'MBON_intrinsic_mV': intrinsic, 'cue': cue,
                              'rows': rows, 'potential_budgets': phase_budgets,
                              'phase_MBON_counts': [c[target].tolist() for c in phase_counts]}
                    report['rows'].append(result); report.update(status='running', phase=f'{apl_gain}/{intrinsic}/{cue}',
                                                                 updated_at=time.time()); atomic_json(path, report)
                    print(json.dumps({'apl_gain': apl_gain, 'intrinsic': intrinsic, 'cue': cue,
                                      'MBON_counts': result['phase_MBON_counts'],
                                      'cue_budget': phase_budgets[1]['group_totals']}), flush=True)
                if digest(b.weight) != brain.initial_all_weights_sha256:
                    raise AssertionError('Frozen membrane diagnostic changed native weights')
            report['native_states'] = {'file': 'native-states.npz', 'sha256': immutable_arrays(
                args.output / 'native-states.npz', phase_counts=np.array(totals), mbon_state=np.array(trace),
                cell_ids=b.ids, mbon_ids=b.ids[target], incoming_edges=incoming, incoming_pre=pre)}
            report.update(status='complete', completed_at=time.time(),
                          interpretation='Actual native membrane/drive/g and potential afferent budgets; no learning or fit.')
            atomic_json(path, report); print(json.dumps({'status': 'complete'}), flush=True)
        except Exception:
            report.update(status='error', error=traceback.format_exc()); atomic_json(path, report); raise


if __name__ == '__main__': main()
