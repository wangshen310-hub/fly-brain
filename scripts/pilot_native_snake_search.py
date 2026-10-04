"""Direct Snake development test; not a promotion past the formal learning gate.

Full native dynamics, existing signed synapses, fixed view and fixed decoder.
Only scalar game outcomes guide parameter selection. This tests a different
question from reversal: can actual game reward improve heldout food capture?
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probe_individual_synapses import IndividualBrain
from flysnake.lab.brain import LabBrain
from flysnake.lab.environment import LabGame
from flysnake.lab.experiment import atomic_json


class FixedLargeGame(LabGame):
    def render_observation(self):
        if self._feedback_observation is not None:
            return self._feedback_observation.copy()
        original = super().render_observation()
        # Fixed 2x magnification about the BOARD center, never about the head.
        return np.asarray(Image.fromarray(original[45:135, 80:240]).resize((320, 180), Image.Resampling.NEAREST)).copy()


class GameBrain(IndividualBrain):
    def __init__(self):
        self.brain = LabBrain()
        neutral = []
        for seed in range(901, 909):
            game = FixedLargeGame(seed=seed, size=4)
            game.food = None
            neutral.append(game.render_observation())
        self.brain.calibrate(neutral)
        b = self.brain.b
        targets = np.r_[self.brain.native.groups['left'], self.brain.native.groups['right']]
        edges = np.flatnonzero(np.isin(b.post, targets) & (b.weight != 0))
        pre = np.searchsorted(b.ptr, edges, side='right') - 1
        fast = b.modulation_mask[pre] == 0
        self.edges, self.pre = edges[fast], pre[fast]
        self.base = b.weight[self.edges].copy()
        side = np.isin(b.post[self.edges], self.brain.native.groups['right']).astype(int)
        self.group = side * 2 + (self.base < 0).astype(int)
        self.theta = np.zeros(4)
        self.report = {'mutable_edges': len(self.edges), 'group_sizes': np.bincount(self.group, minlength=4).tolist(),
                       'calibration': self.brain.calibration, 'selected_cells': self.brain.native.ids,
                       'view': 'fixed board; constant 2x magnification; RGB only',
                       'dynamics': 'reset only at episode boundaries, continuous between game decisions',
                       'learning': 'scalar reward search of native synaptic gains; engineering optimizer'}

    def observe_game(self, rgb):
        result = self.brain.observe(rgb, task='snake', learning=False, reset=False)
        result['memory_sha256'] = self.weight_hash()
        return result


def episodes(brain, seeds, maximum_steps=24):
    outcomes = []
    for seed in seeds:
        game = FixedLargeGame(seed=seed, size=4, calibration=brain.brain.calibration)
        brain.brain.reset_trial()
        records = []
        silent = 0
        for _ in range(maximum_steps):
            receipt = brain.observe_game(game.render_observation())
            receipt.update(episode_id=game.episode_id, step_id=game.step_id)
            if receipt['action'] is None:
                silent += 1
                feedback = brain.feedback('aversive')
                records.append({'receipt': receipt, 'no_action': True, 'feedback': feedback})
                continue
            event = game.apply_receipt(receipt)
            if not event['accepted']:
                raise RuntimeError(f'Invalid neural receipt: {event}')
            feedback = None
            if event['kind'] != 'none':
                feedback = brain.feedback(event['kind'])
                if feedback['delivered_spikes'] <= 0:
                    raise RuntimeError('Undelivered reward/punishment')
                game.complete_feedback(event['event_id'])
            records.append({'receipt': receipt, 'event': event, 'feedback': feedback})
            if game.terminal:
                break
        # Explicit finite-experiment shaping: timeout costs the same as death,
        # so surviving in an empty loop cannot beat a food-capturing trajectory.
        outcome = {'seed': seed, 'food': game.score, 'steps': game.step_id, 'silent': silent,
                   'reason': game.death_reason or ('win' if game.outcome == 'win' else 'time_limit'),
                   'fitness': 10 * game.score - 5 * (game.outcome != 'win') - silent,
                   'rows': records}
        outcomes.append(outcome)
    return float(np.mean([r['fitness'] for r in outcomes])), outcomes


def compact(rows):
    return [{k: r[k] for k in ('seed', 'food', 'steps', 'silent', 'reason', 'fitness')} for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidates', type=int, default=24)
    p.add_argument('--seed', type=int, default=92351)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'status.json').exists():
        raise ValueError('Use a fresh output directory')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    brain = GameBrain()
    rng = np.random.default_rng(args.seed)
    report = {'protocol': 'native-snake-gain-development-v1', 'status': 'running', **brain.report,
              'train_seeds': [731, 932, 1133], 'heldout_seeds': list(range(2031, 2043)),
              'seed': args.seed, 'maximum_steps': 24, 'iterations': []}
    print(json.dumps(brain.report), flush=True)
    baseline_fitness, baseline = episodes(brain, report['heldout_seeds'])
    report['baseline'] = {'fitness': baseline_fitness, 'episodes': baseline}
    print(json.dumps({'baseline': compact(baseline)}), flush=True)
    elite, best = [], -float('inf')
    start = time.time()
    for i in range(args.candidates):
        if i == 0:
            theta = np.zeros(4)
        elif i % 3 == 1:
            theta = rng.uniform(-4, 4, 4)
        else:
            theta = np.clip(elite[int(rng.integers(len(elite)))] + rng.normal(0, 1, 4), -4, 4)
        brain.set_parameters(theta)
        fitness, rows = episodes(brain, report['train_seeds'])
        if fitness > best:
            best, elite = fitness, []
        if fitness == best:
            elite.append(theta.copy())
            elite = elite[-8:]
        report['iterations'].append({'i': i, 'theta': theta.tolist(), 'fitness': fitness, 'episodes': rows})
        report.update(best_fitness=best, best_theta=elite[-1].tolist())
        brain.save(args.output / f'candidate-{i}.npz')
        atomic_json(args.output / 'status.json', report)
        print(json.dumps({'i': i, 'fitness': fitness, 'best': best, 'episodes': compact(rows), 'elapsed': time.time()-start}), flush=True)
    brain.set_parameters(elite[-1])
    fitness, rows = episodes(brain, report['heldout_seeds'])
    brain.save(args.output / 'candidate.npz')
    report.update(status='complete', test_fitness=fitness, test=rows,
                  interpretation='Exploratory only. Final heldout results never guide selection; controls and independent replication remain required.')
    atomic_json(args.output / 'status.json', report)
    print(json.dumps({'status': 'complete', 'baseline': compact(baseline), 'test': compact(rows)}), flush=True)


if __name__ == '__main__':
    main()
