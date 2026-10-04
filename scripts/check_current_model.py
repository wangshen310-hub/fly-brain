"""Verify portable artifact hashes and actual full-graph CUDA inference/replay.

No expert module, optimizer step or policy controller is imported here.
Source bytes and Torch/CUDA identity must match the saved checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from flysnake.research.teacher_spiking import Dynamics, file_hash, STATE_ATOL
from flysnake.research.teacher_afferent import PlasticAfferentBrain


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare_receipt(actual, expected):
    floats = {'voltage_min', 'voltage_max'}
    exact = set(actual) == set(expected) and all(actual[k] == v for k, v in expected.items() if k not in floats)
    float_ok = all(abs(actual[k] - expected[k]) <= STATE_ATOL for k in floats)
    return {'integer_spikes_actions_pixels_and_timing_exact': exact,
            'receipt_floats_within_atol': float_ok}


def checked(path, record):
    if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
        raise ValueError('Artifact integrity mismatch: ' + str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/current')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.model_dir.resolve()
    manifest = read(bundle / 'model.json')
    for name, record in manifest['files'].items():
        checked(bundle / name, record)
    for name, record in manifest['related_artifacts'].items():
        checked(ROOT / name, record)
    for name, sha256 in manifest['source_sha256'].items():
        if file_hash(ROOT / name) != sha256:
            raise ValueError('Source bytes changed: ' + name)
    graph = manifest['complete_graph']
    checked(args.graph / graph['asset'], graph)
    if file_hash(args.graph / graph['manifest_asset']) != graph['manifest_sha256']:
        raise ValueError('Complete graph manifest changed')
    if not torch.cuda.is_available():
        raise RuntimeError('This saved CUDA checkpoint requires an NVIDIA CUDA device; CPU tests are separate.')
    payload = torch.load(bundle / manifest['checkpoint'], map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    brain = PlasticAfferentBrain.load_graph(args.graph, config=Dynamics(**payload['identity']['dynamics']))
    _, extra = brain.restore_checkpoint(bundle / manifest['training_memory'], manifest['files'][manifest['training_memory']]['sha256'])
    if extra.get('step') != 60 or not extra.get('optimizer', {}).get('state'):
        raise ValueError('The committed training memory must preserve step and optimizer state')
    brain.restore_checkpoint(bundle / manifest['checkpoint'], manifest['files'][manifest['checkpoint']]['sha256'])
    if brain.parameter_hash() != manifest['parameter_sha256']:
        raise ValueError('Learned parameter identity mismatch')
    smoke = []
    for sample in manifest['smoke']:
        with Image.open(bundle / sample['path']) as image:
            frame = np.asarray(image.convert('RGB')).copy()
        if hashlib.sha256(frame.tobytes()).hexdigest() != sample['rgb_sha256']:
            raise ValueError('Decoded PNG pixels changed')
        receipt, _ = brain.act(frame)
        smoke.append({'path': sample['path'], 'checks': compare_receipt(receipt, sample['expected_receipt']),
                      'actual_receipt': receipt})

    record = read(bundle / 'replay.json')
    if record['float_state_atol'] != STATE_ATOL:
        raise ValueError('Saved continuation tolerance mismatch')
    for name, field in [('replay-frames.npz', 'frames_sha256'), ('replay-states.npz', 'expected_states_sha256')]:
        if file_hash(bundle / name) != record[field]:
            raise ValueError('Continuation fixture changed')
    state, _ = brain.restore_checkpoint(bundle / manifest['replay_checkpoint'], record['checkpoint']['sha256'])
    start_tick = state['ticks']
    checks = []
    with np.load(bundle / 'replay-frames.npz', allow_pickle=False) as frames, np.load(bundle / 'replay-states.npz', allow_pickle=False) as expected_states:
        for i, frame in enumerate(frames['frames']):
            receipt, state = brain.act(frame, state)
            reference = record['expected'][i]
            numeric = {k: float(np.max(np.abs(v.cpu().numpy() - expected_states[f'{i}_{k}'])))
                       for k, v in state.items() if isinstance(v, torch.Tensor)}
            check = {**compare_receipt(receipt, reference['receipt']),
                     'refractory_exact': numeric['refractory'] == 0,
                     'timing_exact': all(state[k] == reference['state'][k] for k in ('ticks', 'cursor')),
                     'float_state_within_atol': all(value <= STATE_ATOL for k, value in numeric.items() if k != 'refractory')}
            checks.append({'checks': check, 'max_abs_error': numeric, 'actual_receipt': receipt})
    expert_imported = any(name.endswith('teacher_data') for name in sys.modules)
    matched = (not expert_imported and all(all(row['checks'].values()) for row in smoke + checks))
    result = {'status': 'complete', 'matched': matched, 'fresh_process': True,
              'complete_graph_neurons': brain.n, 'complete_graph_anatomical_edges': brain.manifest['anatomical_edges'],
              'graph_sha256': graph['sha256'], 'parameter_sha256': brain.parameter_hash(),
              'training_memory_step': extra['step'], 'optimizer_state_saved': True,
              'restored_nonzero_state_start_tick': start_tick, 'smoke': smoke, 'continuation': checks,
              'source_bytes_and_artifact_hashes_verified': True, 'expert_imported': expert_imported,
              'float_state_atol': STATE_ATOL, 'GPU_bitwise_state_replay_claimed': False,
              'torch': str(torch.__version__), 'cuda': torch.version.cuda,
              'ordinary_game_learning_validated': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'matched': matched, 'smoke_conditions': len(smoke), 'continuation_conditions': len(checks),
                      'graph_neurons': brain.n, 'graph_edges': brain.manifest['anatomical_edges'], 'output': str(args.output)}))
    if not matched:
        raise ValueError('Actual saved-model inference or continuation mismatch')


if __name__ == '__main__':
    main()
