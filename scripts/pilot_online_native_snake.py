"""Local reward-modulated neural learning in a bounded Snake curriculum.

The environment places nearby food to increase reward opportunities. It does
not correct actions, expose coordinates to the brain, or prevent collisions.
This is a development experiment, not an automatic bypass of validation gates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot_node_perturbation import PerturbedBrain
from pilot_native_snake_search import FixedLargeGame
from validate_node_learning import save_training, restore_training
from flysnake.lab.experiment import atomic_json
from flysnake.game import DIRECTIONS


class NearbyFoodGame(FixedLargeGame):
    def _sample_food(self):
        hx, hy = self.snake[0]
        available = [(hx+dx, hy+dy) for dx, dy in DIRECTIONS
                     if self._inside((hx+dx, hy+dy)) and (hx+dx, hy+dy) not in self.snake]
        return self._rng.choice(available) if available else super()._sample_food()


class SnakeBrain(PerturbedBrain):
    def __init__(self, seed):
        super().__init__(seed, 12.)
        self.credit = np.zeros_like(self.theta)

    def observe(self, rgb):
        result = self.brain.observe(rgb, learning=False, task='snake', reset=False)
        result['memory_sha256'] = self.weight_hash()
        return result

    def train_observe(self, rgb):
        result = super().train_observe(rgb)
        # Delayed feedback can affect neural activity from preceding decisions.
        self.credit = np.exp(-.3/1.5)*self.credit + np.clip(self.eligibility, -10, 10)
        self.eligibility = self.credit.copy()
        return result

    def learn(self, kind, learning=True, eta=.03):
        result = super().learn(kind, learning=learning, eta=eta)
        self.credit.fill(0)
        return result


def episode(brain, seed, training=False, nearby=True, limit=24):
    cls = NearbyFoodGame if nearby else FixedLargeGame
    game = cls(seed=seed, size=4, calibration=brain.brain.calibration)
    brain.brain.reset_trial()
    brain.credit.fill(0)
    rows, silent = [], 0
    for _ in range(limit):
        rgb = game.render_observation()
        receipt = brain.train_observe(rgb) if training else brain.observe(rgb)
        receipt.update(episode_id=game.episode_id, step_id=game.step_id)
        if receipt['action'] is None:
            silent += 1
            feedback = brain.learn('aversive', learning=training)
            rows.append({'receipt': receipt, 'no_action': True, 'feedback': feedback})
            continue
        event = game.apply_receipt(receipt)
        if not event['accepted']:
            raise RuntimeError(f'Invalid neural receipt: {event}')
        feedback = None
        if event['kind'] != 'none':
            feedback = brain.learn(event['kind'], learning=training)
            if feedback['delivered_spikes'] <= 0:
                raise RuntimeError('Feedback not delivered')
            game.complete_feedback(event['event_id'])
        else:
            brain.brain.b.tonic[np.r_[brain.brain.native.groups['left'], brain.brain.native.groups['right']]] = brain.tonic
        rows.append({'receipt': receipt, 'event': event, 'feedback': feedback})
        if game.terminal:
            break
    return {'seed': seed, 'training': training, 'nearby_food': nearby, 'food': game.score,
            'steps': game.step_id, 'decisions': len(rows), 'silent': silent,
            'reason': game.death_reason or ('win' if game.outcome == 'win' else 'time_limit'), 'rows': rows}


def assess(brain):
    before = brain.weight_hash()
    result = {}
    for name, nearby in (('nearby_food', True), ('uniform_food', False)):
        rows = [episode(brain, s, nearby=nearby) for s in range(2801, 2813)]
        result[name] = {'food': sum(r['food'] for r in rows), 'steps': sum(r['steps'] for r in rows),
                        'episodes': rows, 'n': len(rows)}
    if brain.weight_hash() != before:
        raise RuntimeError('Frozen game assessment changed weights')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=92501)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing = (args.output/'status.json').exists()
    if existing and not args.resume:
        raise ValueError('Use a new output directory or explicitly resume')
    if not existing:
        (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    brain = SnakeBrain(args.seed)
    report = {'protocol': 'online-native-snake-development-v1', 'status': 'running',
              'seed': args.seed, 'brain': brain.report, 'eta': .03,
              'planned_episodes': args.episodes,
              'credit_seconds': 1.5, 'view': 'fixed full board magnified 2x',
              'calibration': 'locked before training; neutral six-cell-board scenes',
              'episodes': [], 'checkpoints': []}
    if existing:
        saved = json.loads((args.output/'status.json').read_text(encoding='utf-8'))
        for field in ('protocol', 'seed', 'eta', 'planned_episodes'):
            if report[field] != saved[field]:
                raise ValueError(f'Resume configuration mismatch: {field}')
        if (args.output/'source.py').read_bytes() != Path(__file__).read_bytes():
            raise ValueError('Resume source version mismatch')
        report = saved
        if report['status'] == 'complete':
            print(json.dumps({'status': 'already_complete'}), flush=True)
            return
        if report['checkpoints']:
            c = report['checkpoints'][-1]
            restore_training(brain, args.output/c['file'], c['sha256'])
    else:
        report['before'] = assess(brain)
        atomic_json(args.output/'status.json', report)
    print(json.dumps({'before': {k: v['food'] for k, v in report['before'].items()}}), flush=True)
    start = time.time()
    for i in range(len(report['episodes']), args.episodes):
        row = episode(brain, 1801+i, training=True, nearby=True)
        report['episodes'].append(row)
        path = args.output/f'episode-{i+1}-{time.time_ns()}.npz'
        save_training(brain, path)
        report['checkpoints'].append({'episode': i+1, 'file': path.name,
                                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'memory_sha256': brain.weight_hash()})
        atomic_json(args.output/'status.json', report)
        if (i+1) % 10 == 0:
            recent = report['episodes'][-10:]
            print(json.dumps({'through': i+1, 'food_recent10': sum(r['food'] for r in recent),
                              'steps_recent10': sum(r['steps'] for r in recent), 'elapsed': time.time()-start}), flush=True)
    report['after'] = assess(brain)
    report.update(status='complete', interpretation='Exploratory curriculum only. No controls or independent training seeds yet; do not claim robust game learning.')
    atomic_json(args.output/'status.json', report)
    print(json.dumps({'status': 'complete', 'before': {k: v['food'] for k, v in report['before'].items()},
                      'after': {k: v['food'] for k, v in report['after'].items()}}), flush=True)


if __name__ == '__main__':
    main()
