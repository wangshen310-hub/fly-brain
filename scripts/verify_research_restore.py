"""Replay a committed research checkpoint in an isolated real neural model."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot_node_perturbation import PerturbedBrain
from validate_node_learning import restore_training
from flysnake.lab.environment import cue_scene


def replay(brain):
    result = []
    for i, (side, kind) in enumerate((('left', 'reward'), ('right', 'aversive'), ('left', 'aversive'))):
        observation = brain.train_observe(cue_scene(51901+i, side))
        feedback = brain.learn(kind, eta=.3)
        result.append({'action': observation['action'], 'counts': observation['counts'],
                       'noise': brain.noise.tolist(), 'memory_before': feedback['memory_before'],
                       'memory_after': feedback['memory_after'], 'delivered_spikes': feedback['delivered_spikes']})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    state = json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    checkpoint = state['checkpoints'][-1].copy()
    path = args.run/checkpoint['file']
    brain = PerturbedBrain(state['seed'], 12.)
    restore_training(brain, path, checkpoint['sha256'])
    assert brain.weight_hash() == checkpoint['memory_sha256']
    first = replay(brain)
    first_stats = brain.mean_pre.copy()
    first_baseline = brain.reward_baseline
    restore_training(brain, path, checkpoint['sha256'])
    second = replay(brain)
    assert first == second
    np.testing.assert_array_equal(first_stats, brain.mean_pre)
    assert first_baseline == brain.reward_baseline
    report = {'checkpoint': str(path), 'file_sha256': checkpoint['sha256'],
              'memory_sha256': checkpoint['memory_sha256'], 'real_native_replay_exact': True,
              'next_noise_counts_actions_feedback_and_weights_equal': True, 'rows': first,
              'interpretation': 'Clone-only reproducibility test. Fixed stimulus outcomes are fixtures, not learning evidence. Original training files unmodified.'}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
