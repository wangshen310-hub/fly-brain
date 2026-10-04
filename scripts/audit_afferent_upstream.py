"""Read-only full-graph tracing of a fixed LEFT/FORWARD visual counterfactual.

All labels and annotations belong to this audit, never to the neural process's
RGB observation boundary. Frozen existing parameters and decoder are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_afferent import PlasticAfferentBrain
from flysnake.research.teacher_spiking import Dynamics, STATE_ATOL, decode_spikes, file_hash


def grouped(values, release_change, voltage_change, count_change, distance):
    rows = []
    for value in np.unique(values):
        selected = values == value
        release = selected & release_change
        voltage = selected & voltage_change
        if not release.any() and not voltage.any():
            continue
        reached = distance[release]
        reached = reached[reached >= 0]
        rows.append({'name': str(value), 'neurons': int(selected.sum()),
                     'changed_arrival_time_series': int(release.sum()),
                     'changed_voltage_over_atol': int(voltage.sum()),
                     'changed_window_spike_counts': int((selected & count_change).sum()),
                     'nearest_fast_hops_to_turn': int(reached.min()) if reached.size else None})
    return sorted(rows, key=lambda row: (-row['changed_arrival_time_series'],
                                         -row['changed_voltage_over_atol']))


def main():
    p = argparse.ArgumentParser()
    for name in ('graph', 'dataset', 'run', 'catalog', 'paths', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    point = json.loads((Path(args.run) / 'committed.json').read_text(encoding='utf-8'))['checkpoint']
    payload = torch.load(point['path'], map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    brain = PlasticAfferentBrain.load_graph(args.graph,
                                          config=Dynamics(**payload['identity']['dynamics']))
    brain.restore_checkpoint(point['path'], point['sha256'])
    parameter_hash = brain.parameter_hash()
    meta = json.loads((Path(args.dataset) / 'manifest.json').read_text(encoding='utf-8'))
    if file_hash(Path(args.dataset) / 'dataset.npz') != meta['dataset_sha256']:
        raise ValueError('Dataset changed')
    rows = meta['rows']['train']
    pose = rows[0]['pose_key']
    with np.load(Path(args.dataset) / 'dataset.npz', allow_pickle=False) as data:
        indices = [i for i, row in enumerate(rows) if row['pose_key'] == pose and row['adjacent_eat']]
        indices.sort(key=lambda i: int(data['train_labels'][i]))
        if data['train_labels'][indices].tolist() != [0, 1, 2]:
            raise ValueError('Physical LEFT/FORWARD/RIGHT training counterfactual required')
        frames = data['train_rgb'][indices[:2]].copy()
    with np.load(Path(args.catalog) / 'catalog.npz', allow_pickle=False) as catalog:
        ids = catalog['ids'].copy()
        types = catalog['cell_type'].copy()
        classes = catalog['superclass'].copy()
    catalog_meta = json.loads((Path(args.catalog) / 'manifest.json').read_text(encoding='utf-8'))
    if file_hash(Path(args.catalog) / 'catalog.npz') != catalog_meta['catalog_sha256']:
        raise ValueError('Offline catalog changed')
    with np.load(Path(args.graph) / 'graph.npz', allow_pickle=False) as graph:
        if not np.array_equal(ids, graph['ids']):
            raise ValueError('Offline annotations are not aligned')
        exc_crow, exc_col, inh_crow, inh_col = [graph[k] for k in
                                             ('exc_crow', 'exc_col', 'inh_crow', 'inh_col')]
    motor = brain.motor.cpu().numpy()
    paths_meta = json.loads((Path(args.paths) / 'summary.json').read_text(encoding='utf-8'))
    with np.load(Path(args.paths) / 'distances.npz', allow_pickle=False) as paths:
        if paths_meta['graph_sha256'] != brain.graph_identity or not np.array_equal(ids, paths['ids']):
            raise ValueError('Offline anatomical paths are incompatible')
        distance = paths['hops'].copy()

    def fast_inputs(cell):
        return np.r_[exc_col[exc_crow[cell]:exc_crow[cell + 1]],
                     inh_col[inh_crow[cell]:inh_crow[cell + 1]]]
    light = brain.encode(frames)
    # Both columns are independent cold-state trajectories. Pairing does not
    # create any neural connection or input other than each frame's radiance.
    state = brain.zero_state(2)
    arrivals, voltage_max = [], np.zeros(brain.n, dtype=np.float32)
    conductance_max = np.zeros((2, brain.n), dtype=np.float32)
    counts = torch.zeros((brain.n, 2), device=light.device)
    with torch.no_grad():
        for _ in range(brain.config.window_ms):
            arrivals.append(state['release_queue'][state['cursor']].cpu().numpy())
            c, _, state = brain.integrate_encoded(light, state, steps=1)
            counts += c
            voltage_max = np.maximum(voltage_max, torch.abs(state['v'][:, 0] - state['v'][:, 1]).cpu().numpy())
            for slot, channel in enumerate(('ge', 'gi')):
                conductance_max[slot] = np.maximum(conductance_max[slot],
                    torch.abs(state[channel][:, 0] - state[channel][:, 1]).cpu().numpy())
    arrivals = np.stack(arrivals)
    arrival_difference = np.abs(arrivals[:, :, 0] - arrivals[:, :, 1])
    release_max = arrival_difference.max(axis=0)
    release_change = release_max > STATE_ATOL
    voltage_change = voltage_max > STATE_ATOL
    window_counts = counts.cpu().numpy().astype(np.int64)
    count_change = window_counts[:, 0] != window_counts[:, 1]
    receipts = []
    for column in range(2):
        raw = window_counts[motor, column].tolist()
        receipts.append(decode_spikes({'left': [raw[0]], 'right': [raw[1]], 'forward': raw[2:]}))
    fast_motor_sources = np.unique(np.concatenate([fast_inputs(cell) for cell in motor]))
    near_changes = np.flatnonzero(release_change & (distance >= 0))
    nearest = int(distance[near_changes].min()) if len(near_changes) else None
    frontier_ids = near_changes[distance[near_changes] == nearest] if nearest is not None else np.empty(0, int)
    cells = [{'cell_id': int(ids[k]), 'cell_type': str(types[k]), 'superclass': str(classes[k]),
              'fast_hops_to_turn': int(distance[k]), 'max_actual_release_difference': float(release_max[k]),
              'max_voltage_difference': float(voltage_max[k]),
              'window_spike_counts': window_counts[k].tolist()} for k in frontier_ids]
    # Offline localization only: which immediate turn-input relays receive a
    # differing actual release yet fail to pass a differing release onward?
    relay_rows = []
    for relay in np.unique(brain.aff_pre.cpu().numpy()):
        sources = np.unique(fast_inputs(relay))
        changing = sources[release_change[sources]]
        if not len(changing):
            continue
        relay_rows.append({'cell_id': int(ids[relay]), 'cell_type': str(types[relay]),
                           'superclass': str(classes[relay]),
                           'changed_incoming_source_count': len(changing),
                           'changed_incoming_source_ids': ids[changing].tolist(),
                           'own_actual_release_changes': bool(release_change[relay]),
                           'max_voltage_difference': float(voltage_max[relay]),
                           'max_exc_conductance_difference': float(conductance_max[0, relay]),
                           'max_inh_conductance_difference': float(conductance_max[1, relay]),
                           'window_spike_counts': window_counts[relay].tolist()})
    relay_rows.sort(key=lambda row: -max(row['max_exc_conductance_difference'],
                                        row['max_inh_conductance_difference']))
    depth_rows = [{'hops_to_turn': hop, 'neurons': int(np.sum(distance == hop)),
                   'changed_arrival_time_series': int(np.sum(release_change & (distance == hop))),
                   'changed_voltage_over_atol': int(np.sum(voltage_change & (distance == hop)))}
                  for hop in range(9)]
    radiance_diff = torch.abs(light[:, 0] - light[:, 1]).cpu().numpy()
    result = {'status': 'complete', 'checkpoint': point, 'parameter_sha256': parameter_hash,
              'frames': [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
              'train_indices': indices[:2], 'physical_scene_order': ['left_food', 'forward_food'],
              'actual_receipts': receipts, 'all_neurons': brain.n,
              'anatomical_edges': brain.manifest['anatomical_edges'],
              'photoreceptors_with_different_radiance': int(np.sum(radiance_diff > STATE_ATOL)),
              'maximum_receptor_radiance_difference': float(radiance_diff.max()),
              'actual_delayed_release_changed_cells': int(release_change.sum()),
              'non_graded_release_changed_cells': int(np.sum(release_change & ~brain.graded_mask.cpu().numpy())),
              'changed_window_spike_count_cells': int(count_change.sum()),
              'motor_input_neurons': len(fast_motor_sources),
              'motor_input_arrivals_byte_equal': bool(np.array_equal(arrivals[:, fast_motor_sources, 0],
                                                                     arrivals[:, fast_motor_sources, 1])),
              'max_motor_input_release_difference': float(release_max[fast_motor_sources].max()),
              'nearest_changed_release_hops_to_turn': nearest,
              'nearest_changed_release_cells': cells, 'depth_profile': depth_rows,
              'turn_input_relays_receiving_changed_releases': relay_rows,
              'by_superclass': grouped(classes, release_change, voltage_change, count_change, distance),
              'by_cell_type': grouped(types, release_change, voltage_change, count_change, distance),
              'release_change_atol': STATE_ATOL,
              'scope': 'One frozen trained state, one fixed pose, two independent 100ms cold trajectories. '
                       'No inference of global incapacity or a unique missing physiological mechanism.',
              'source_sha256': file_hash(__file__), 'catalog_sha256': catalog_meta['catalog_sha256']}
    np.savez_compressed(out / 'signal_arrays.npz', ids=ids, arrivals=arrivals,
                        voltage_max=voltage_max, conductance_max=conductance_max,
                        release_max=release_max, window_counts=window_counts,
                        fast_hops_to_turn=distance, receptor_radiance_difference=radiance_diff)
    result['arrays_sha256'] = file_hash(out / 'signal_arrays.npz')
    if brain.parameter_hash() != parameter_hash:
        raise ValueError('Read-only tracing changed memory')
    if any(name.endswith('teacher_data') for name in sys.modules):
        raise ValueError('Expert entered neural inference')
    with (out / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('photoreceptors_with_different_radiance',
        'actual_delayed_release_changed_cells', 'non_graded_release_changed_cells',
        'changed_window_spike_count_cells', 'motor_input_arrivals_byte_equal',
        'nearest_changed_release_hops_to_turn', 'nearest_changed_release_cells', 'by_superclass')}), flush=True)


if __name__ == '__main__':
    main()
