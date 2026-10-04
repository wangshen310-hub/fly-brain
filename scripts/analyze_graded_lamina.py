"""Offline frozen candidate analysis: task information, controls, stability."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.apl_local import file_hash
from flysnake.lab.experiment import atomic_json


def comparison(run, a, b, groups):
    n = max(int(ix.max()) for ix in groups.values()) + 1
    changed_counts = np.zeros(n, bool)
    changed_v = np.zeros(n, bool)
    max_delta = np.zeros(n)
    for ar, br in zip(a['bins'], b['bins']):
        with np.load(run / ar['arrays']['file'], allow_pickle=False) as az, \
             np.load(run / br['arrays']['file'], allow_pickle=False) as bz:
            delta = np.abs(az['voltage_mV'].astype(np.float64) - bz['voltage_mV'])
            changed_counts |= az['full_counts'] != bz['full_counts']
            changed_v |= delta > .001
            np.maximum(max_delta, delta, out=max_delta)
    return {name: {'cells_with_different_10ms_counts': int(changed_counts[ix].sum()),
        'cells_with_boundary_voltage_difference_over_1e_3_mV': int(changed_v[ix].sum()),
        'maximum_boundary_voltage_difference_mV': float(max_delta[ix].max())}
        for name, ix in groups.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous candidate analysis')
    status = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise ValueError('Completed frozen candidate required')
    for name, sha in status['sources'].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError('Locked source differs')
    if file_hash(args.run / 'anatomy.npz') != status['anatomy_sha256']:
        raise ValueError('Anatomy differs')
    with np.load(args.run / 'anatomy.npz', allow_pickle=False) as z:
        groups = {name: z[name + '_indices'] for name in status['groups']}
        allowed = z['external_allowed']
    checked = 0
    conditions = {}
    all_finite = True
    outside_drive = 0
    for row in status['rows']:
        voltage_min = float('inf')
        voltage_max = -float('inf')
        rates_min = float('inf')
        rates_max = -float('inf')
        below_minus_100 = np.zeros(len(allowed), bool)
        kc_counts = np.zeros(len(groups['all_KC']), np.int64)
        for item in row['bins']:
            path = args.run / item['arrays']['file']
            if file_hash(path) != item['arrays']['sha256']:
                raise ValueError('Saved candidate array differs')
            with np.load(path, allow_pickle=False) as z:
                all_finite &= all(np.isfinite(z[name]).all() for name in z.files)
                outside_drive = max(outside_drive, int(np.count_nonzero(z['external_drive'][~allowed])))
                voltage_min = min(voltage_min, float(z['voltage_mV'].min()))
                voltage_max = max(voltage_max, float(z['voltage_mV'].max()))
                rates_min = min(rates_min, float(z['graded_release_hz'].min()))
                rates_max = max(rates_max, float(z['graded_release_hz'].max()))
                below_minus_100 |= z['voltage_mV'] < -100
                kc_counts += z['full_counts'][groups['all_KC']]
            checked += 1
        conditions[row['label']] = {
            'groups': {name: {'spikes': sum(r['summary'][name]['spikes'] for r in row['bins']),
                'minimum_boundary_voltage_mV': min(r['summary'][name]['boundary_V_min_mV'] for r in row['bins']),
                'maximum_boundary_voltage_mV': max(r['summary'][name]['boundary_V_max_mV'] for r in row['bins'])}
                for name in groups},
            'motor_counts': row['cue_motor_counts'], 'full_V_min': voltage_min, 'full_V_max': voltage_max,
            'graded_rate_min_hz': rates_min, 'graded_rate_max_hz': rates_max,
            'cells_ever_below_minus_100_mV_at_saved_boundaries': int(below_minus_100.sum()),
            'active_KC_in_300ms': int((kc_counts > 0).sum()),
            'active_KC_fraction_in_300ms': float((kc_counts > 0).mean())}
    rows = {r['label']: r for r in status['rows']}
    comparisons = {name: comparison(args.run, rows[a], rows[b], groups)
        for name, a, b in (
            ('food_position', 'food-A', 'food-B'), ('head_orientation', 'food-A', 'head-rotated'),
            ('head_vs_body', 'food-A', 'head-as-body'),
            ('off_food_position', 'food-A-retinal-off', 'food-B-retinal-off'),
            ('repeat', 'food-A', 'food-A-repeat'))}
    report = {'status': 'complete', 'completed_at': time.time(), 'source_sha256': file_hash(__file__),
        'run_status_sha256': file_hash(args.run / 'status.json'), 'immutable_arrays_checked': checked,
        'conditions': conditions, 'comparisons': comparisons,
        'gates': {'all_saved_arrays_finite': bool(all_finite),
            'nonvisual_external_current_zero': outside_drive == 0,
            'food_changes_KC_spike_counts': comparisons['food_position']['all_KC']['cells_with_different_10ms_counts'] > 0,
            'food_changes_motor_spike_counts': comparisons['food_position']['motor']['cells_with_different_10ms_counts'] > 0,
            'physiological_fit': False, 'association_learning_tested': False,
            'behavioral_learning_proven': False},
        'scope': 'One predeclared graded-release candidate. V range uses 10ms boundaries, '
            'not all internal evaluations. Changed neural counts are not correct decisions or learning.'}
    atomic_json(args.output, report)
    print(json.dumps({'status': 'complete', 'gates': report['gates'],
        'food_position': {name: comparisons['food_position'][name] for name in ('lamina', 'all_KC', 'MBON11', 'motor')}}))


if __name__ == '__main__':
    main()
