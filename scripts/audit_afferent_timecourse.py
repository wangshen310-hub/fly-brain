"""Frozen persistent-state/latency counterfactuals, with no teacher in inference."""
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
from flysnake.research.teacher_spiking import Dynamics, decode_spikes, file_hash


def main():
    p = argparse.ArgumentParser()
    for name in ('graph', 'dataset', 'run', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--windows', type=int, default=10)
    args = p.parse_args()
    if args.windows < 2:
        raise ValueError('A persistent-state continuation is required')
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
            raise ValueError('Physical counterfactual order changed')
        frames = data['train_rgb'][indices].copy()
    with np.load(Path(args.graph) / 'graph.npz', allow_pickle=False) as graph:
        ids = graph['ids'].copy()
    tracked_ids = np.array([21744, 61354, 19331, 10067, 10260, 13920], dtype=np.int64)
    positions = np.searchsorted(ids, tracked_ids)
    if not np.array_equal(ids[positions], tracked_ids):
        raise ValueError('Tracked neuronal identities changed')
    tracked = torch.as_tensor(positions, device=brain.base_rest.device)
    motor = brain.motor
    conditions = {}
    for name, frame, enabled in [('left-food', frames[0], True),
                                  ('forward-food', frames[1], True),
                                  ('right-food', frames[2], True),
                                  ('vision-off', frames[0], False)]:
        light, state, records = brain.encode(frame[None]), None, []
        with torch.no_grad():
            for window in range(args.windows):
                counts, m, state = brain.integrate_encoded(light, state, visual_enabled=enabled)
                raw = m[0].cpu().to(torch.int64).tolist()
                receipt = decode_spikes({'left': [raw[0]], 'right': [raw[1]], 'forward': raw[2:]})
                records.append({'start_tick': state['ticks'] - brain.config.window_ms,
                                'end_tick': state['ticks'], **receipt,
                                'tracked_neuron_window_counts': counts[tracked, 0].cpu().to(torch.int64).tolist(),
                                'motor_final_voltage': state['v'][motor, 0].cpu().tolist(),
                                'actual_total_spikes': int(counts.sum()),
                                'full_counts_sha256': hashlib.sha256(counts.cpu().numpy().tobytes()).hexdigest()})
        conditions[name] = {'input_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                            'visual_enabled': enabled, 'windows': records}
        print(json.dumps({'condition': name, 'actions': [r['action'] for r in records],
                          'left_right_counts': [[r['counts']['left'][0], r['counts']['right'][0]]
                                                for r in records]}), flush=True)
    left = conditions['left-food']['windows']
    forward = conditions['forward-food']['windows']
    result = {'status': 'complete', 'checkpoint': point, 'parameter_sha256': parameter_hash,
              'conditions': conditions, 'tracked_neuron_ids': tracked_ids.tolist(),
              'left_forward_motor_count_differences_at_windows':
                  [i for i, (a, b) in enumerate(zip(left, forward)) if a['counts'] != b['counts']],
              'left_forward_action_differences_at_windows':
                  [i for i, (a, b) in enumerate(zip(left, forward)) if a['action'] != b['action']],
              'source_sha256': file_hash(__file__),
              'scope': 'Same fixed training pose; constant images held for ten continuous neural windows. '
                       'Read-only latency/state probe, not moving-game performance or learning.'}
    if brain.parameter_hash() != parameter_hash:
        raise ValueError('Frozen time course changed memory')
    if any(name.endswith('teacher_data') for name in sys.modules):
        raise ValueError('Expert entered inference')
    with (out / 'summary.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
