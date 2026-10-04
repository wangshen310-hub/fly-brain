"""Independent native reload replay of an immutable curriculum checkpoint.

This checks reproducibility, not learning. The training writer is untouched.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.research.covered_learning import CoveredNodeBrain, CoveredLearningConfig
from flysnake.research.paired import digest
from train_covered_controls import ARMS, foods, pose_split, source_hashes, trial


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing verification')
    status = json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    committed = status['checkpoints'][-1]
    if committed['blocks_completed']<1:
        raise ValueError('A trained committed batch is required')
    checkpoint = args.run/committed['file']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=committed['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    if status['sources']!=source_hashes():
        raise ValueError('Current source differs from committed experiment')
    with np.load(checkpoint, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
        memory = {arm: z[arm].copy() for arm in ARMS}
    for arm, theta in memory.items():
        if digest(theta)!=committed['parameters_sha256'][arm]:
            raise ValueError('Saved parameter hash mismatch')
    training, _ = pose_split()
    replays = []
    for repeat in range(2):
        brain = CoveredNodeBrain(CoveredLearningConfig(**metadata['settings']['config']))
        if json.loads(json.dumps(brain.identity))!=metadata['identity']:
            raise ValueError('Restored graph, base weights or calibration differs')
        rng = np.random.default_rng()
        rng.bit_generator.state = metadata['rng']
        scenarios = []
        for _ in range(2):
            pose = training[int(rng.integers(len(training)))]
            available = foods(pose)
            scenarios.append((pose, available[int(rng.integers(len(available)))],
                              rng.normal(0, brain.config.noise_sigma, 2)))
        replay = []
        for arm in ARMS:
            brain.set_parameters(memory[arm])
            for pose, food, noise in scenarios:
                row, eligibility = trial(brain, pose, food, noise)
                replay.append({'arm': arm, 'image_sha256': row['image_sha256'],
                               'noise': noise.tolist(), 'counts': row['receipt']['counts'],
                               'action': row['receipt']['action'], 'return': row['return'],
                               'feedback_spikes': row['feedback']['delivered_spikes'],
                               'weight_sha256': brain.weight_hash(),
                               'eligibility_sha256': digest(eligibility)})
        replays.append(replay)
        del brain
        print(json.dumps({'independent_reload': repeat+1, 'native_trials': len(replay)}), flush=True)
    if replays[0]!=replays[1]:
        raise RuntimeError('Independent native replay did not reproduce RNG, spikes, actions or eligibility')
    report = {'status': 'complete', 'checkpoint': committed, 'replays_equal': True,
              'native_trials_per_reload': len(replays[0]), 'replay': replays[0],
              'completed_at': time.time(),
              'interpretation': 'Native checkpoint reproducibility only; not behavioral learning evidence.'}
    atomic_json(args.output, report)


if __name__=='__main__':
    main()
