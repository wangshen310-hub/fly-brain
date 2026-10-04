"""Read-only, matched held-out game evaluation of committed native memories.

Training directories are never written. Both memories play the same worlds
with exploration disabled and synapses frozen. Parameter change alone is not
evidence of learning; this runner records actual food and collision outcomes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.research.paired import (VERSION, NativeSearchBrain, PairedSearch,
                                     restore_checkpoint, writer_lock)
from train_paired_native_snake import episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=ROOT/'runtime/live-paired-snake-v1')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=12)
    p.add_argument('--limit', type=int, default=16)
    p.add_argument('--seed-start', type=int, default=970001)
    args = p.parse_args()
    if args.n < 1 or args.limit < 1 or args.seed_start < 200000:
        raise ValueError('Positive evaluation size and disjoint world seeds required')
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output/'status.json'
        if path.exists():
            raise ValueError('Preserve existing evaluation: use a fresh output')
        raw = (args.run/'status.json').read_bytes()
        parent = json.loads(raw)
        provenance = {'protocol': parent['protocol'], 'settings': parent['settings'],
                      'source': parent['source'], 'brain_protocol': VERSION}
        for name, expected in parent['source'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Current native source differs from checkpoint: '+name)
        report = {'protocol': 'heldout-committed-native-memory-evaluation-v1',
                  'status': 'loading', 'n': args.n, 'limit': args.limit,
                  'seed_start': args.seed_start, 'stage': 'uniform_food',
                  'size': parent['settings']['size'], 'exploration': False,
                  'training_status_sha256': hashlib.sha256(raw).hexdigest(),
                  'source_checkpoints': [parent['checkpoints'][0], parent['checkpoints'][-1]],
                  'assessments': {'initial': [], 'trained': []},
                  'interpretation': 'Development assessment; independent training seeds and matched shuffled-feedback controls remain required.'}
        (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
        (args.output/'training-status-snapshot.json').write_bytes(raw)
        atomic_json(path, report)
        try:
            brain = NativeSearchBrain()
            for label, checkpoint in zip(('initial', 'trained'), report['source_checkpoints']):
                search = PairedSearch(parent['settings']['seed'])
                restore_checkpoint(args.run/checkpoint['file'], checkpoint['sha256'],
                                   brain, search, provenance)
                expected_weight = brain.weight_hash()
                report.update(status='running', phase=label)
                for seed in range(args.seed_start, args.seed_start+args.n):
                    row = episode(brain, seed, 'uniform_food', report['size'], args.limit)
                    if brain.weight_hash() != expected_weight:
                        raise RuntimeError('Frozen evaluation changed synapses')
                    report['assessments'][label].append(row)
                    atomic_json(path, report)
                    print(json.dumps({'phase': label, 'through': len(report['assessments'][label]),
                                      'food': row['food'], 'steps': row['steps'],
                                      'reason': row['reason']}), flush=True)
            before = np.array([r['food'] for r in report['assessments']['initial']])
            after = np.array([r['food'] for r in report['assessments']['trained']])
            difference = after-before
            rng = np.random.default_rng(930003)
            bootstrap = difference[rng.integers(0, args.n, (10000, args.n))].mean(axis=1)
            report['summary'] = {'initial_food': int(before.sum()), 'trained_food': int(after.sum()),
                                 'paired_food_per_episode_change': float(difference.mean()),
                                 'paired_bootstrap_95pct': np.quantile(bootstrap, [.025, .975]).tolist(),
                                 'initial_successful_episodes': int((before>0).sum()),
                                 'trained_successful_episodes': int((after>0).sum())}
            report.update(status='complete', completed_at=time.time())
            atomic_json(path, report)
            print(json.dumps(report['summary']), flush=True)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__ == '__main__':
    main()
