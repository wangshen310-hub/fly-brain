"""Offline diagnosis of immutable native relay observations; no action policy."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.apl_local import file_hash
from flysnake.research.visual_relay_trace import EVENT_COLUMNS
from flysnake.lab.experiment import atomic_json


def compare(run, left, right, groups, types):
    n = len(types)
    changes = {name: np.zeros(n, bool) for name in
        ('counts', 'voltage_exact', 'voltage_over_1e_6_mV', 'voltage_over_1e_3_mV',
         'native_voltage_range_exact', 'synaptic_g_exact', 'delivery_budget_exact')}
    maximum_voltage = np.zeros(n)
    maximum_events = np.zeros(n)
    max_g = np.zeros(n)
    for la, rb in zip(left['bins'], right['bins']):
        with np.load(run / la['arrays']['file'], allow_pickle=False) as a, \
             np.load(run / rb['arrays']['file'], allow_pickle=False) as b:
            delta = np.abs(a['voltage_mV'].astype(np.float64) - b['voltage_mV'])
            event_delta = np.max(np.abs(a['fast_event_budgets'] - b['fast_event_budgets']), axis=0)
            g_delta = np.abs(a['synaptic_g'].astype(np.float64) - b['synaptic_g'])
            changes['counts'] |= a['full_counts'] != b['full_counts']
            changes['voltage_exact'] |= delta != 0
            changes['voltage_over_1e_6_mV'] |= delta > 1e-6
            changes['voltage_over_1e_3_mV'] |= delta > 1e-3
            changes['native_voltage_range_exact'] |= (
                (a['native_evaluation_min_mV'] != b['native_evaluation_min_mV']) |
                (a['native_evaluation_max_mV'] != b['native_evaluation_max_mV']))
            changes['synaptic_g_exact'] |= g_delta != 0
            changes['delivery_budget_exact'] |= event_delta != 0
            np.maximum(maximum_voltage, delta, out=maximum_voltage)
            np.maximum(maximum_events, event_delta, out=maximum_events)
            np.maximum(max_g, g_delta, out=max_g)
    result = {}
    for name, ix in groups.items():
        result[name] = dict({key: int(mask[ix].sum()) for key, mask in changes.items()},
            largest_boundary_voltage_difference_mV=float(maximum_voltage[ix].max()),
            largest_synaptic_g_difference=float(max_g[ix].max()),
            largest_delivery_budget_difference=float(maximum_events[ix].max()))
    nonretinal = groups['nonreceptors']
    changed = nonretinal[changes['voltage_over_1e_3_mV'][nonretinal]]
    unique, counts = np.unique(types[changed], return_counts=True)
    ranking = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0]))
    result['nonreceptor_types_with_voltage_differences_over_1e_3_mV'] = ranking
    return result


def aggregate(row, group):
    summaries = [b['summary'][group] for b in row['bins']]
    return {'spikes': sum(s['spikes'] for s in summaries),
        'minimum_native_evaluated_voltage_mV': min(s['V_native_evaluation_min_mV'] for s in summaries),
        'maximum_native_evaluated_voltage_mV': max(s['V_native_evaluation_max_mV'] for s in summaries),
        'threshold_minus_maximum_mV': -45 - max(s['V_native_evaluation_max_mV'] for s in summaries),
        'maximum_external_drive_nonzero_cells': max(s['external_drive_nonzero'] for s in summaries),
        'actual_delayed_fast_event_budgets': {name: sum(s['fast_edge_delivery_budgets'][name] for s in summaries)
                                           for name in EVENT_COLUMNS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing analysis')
    status = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise ValueError('Completed relay audit required')
    for name, sha in status['sources'].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError('Locked source changed: ' + name)
    if file_hash(args.run / 'anatomy.npz') != status['anatomy_sha256']:
        raise ValueError('Anatomy hash differs')
    with np.load(args.run / 'anatomy.npz', allow_pickle=False) as a:
        types = a['cell_types']
        groups = {name[:-8]: a[name] for name in a.files if name.endswith('_indices')}
    rows = {row['label']: row for row in status['rows']}
    arrays_checked = 0
    for row in rows.values():
        for b in row['bins']:
            if file_hash(args.run / b['arrays']['file']) != b['arrays']['sha256']:
                raise ValueError('Native array hash differs')
            arrays_checked += 1
    comparisons = {}
    for name, a, b in (
        ('food_position', 'food-A', 'food-B'),
        ('head_orientation', 'food-A', 'head-rotated'),
        ('head_vs_body', 'food-A', 'head-as-body'),
        ('visual_on_vs_off', 'food-A', 'food-A-retinal-off'),
        ('off_food_position', 'food-A-retinal-off', 'food-B-retinal-off'),
        ('repeat', 'food-A', 'food-A-repeat')):
        comparisons[name] = compare(args.run, rows[a], rows[b], groups, types)
    report = {'status': 'complete', 'analysis_source_sha256': file_hash(__file__),
        'run_status_sha256': file_hash(args.run / 'status.json'), 'completed_at': time.time(),
        'immutable_arrays_checked': arrays_checked,
        'comparisons': comparisons,
        'conditions': {label: {g: aggregate(row, g) for g in groups} for label, row in rows.items()},
        'anatomical_edge_audit': status['anatomical_edge_audit'],
        'scope': 'Boundary v/g differences and actual delayed fast synaptic events. '
            'Event budgets are g increments, not physical currents/charge; min/max are evaluated native states, '
            'not exhaustive continuous membrane traces. Labels are used only in this offline report.',
        'behavioral_learning_proven': False}
    atomic_json(args.output, report)
    print(json.dumps({'status': 'complete', 'arrays_checked': arrays_checked,
        'food_position_groups': {g: comparisons['food_position'][g] for g in
                                ('lamina', 'all_KC', 'MBON11', 'motor')}}))


if __name__ == '__main__':
    main()
