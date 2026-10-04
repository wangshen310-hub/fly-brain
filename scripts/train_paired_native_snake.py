"""Finite, resumable reward-only Snake learning on a full native connectome.

Training and heldout assessments use actual game trajectories. No state-derived
action labels or distance shaping enter learning. This is a development runner;
completed generations do not by themselves constitute a learning success.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.paired import (VERSION, SearchConfig, PairedSearch, NativeSearchBrain,
                                     writer_lock, save_checkpoint, restore_checkpoint)
from flysnake.research.vision import ReadableGame
from flysnake.lab.experiment import atomic_json
from flysnake.game import DIRECTIONS


class CurriculumGame(ReadableGame):
    def __init__(self, *, stage, **kwargs):
        if stage not in ('fixed_pose', 'nearby_food', 'uniform_food'):
            raise ValueError('Invalid curriculum stage')
        self.stage = stage
        super().__init__(**kwargs)

    def new_episode(self):
        super().new_episode()
        if self.stage == 'fixed_pose':
            self.direction = 0
            self.snake = [(self.size//2, y) for y in (self.size-3, self.size-2, self.size-1)]
            self.food = self._sample_food()
        return self.snapshot()

    def _sample_food(self):
        if self.stage != 'uniform_food':
            hx, hy = self.snake[0]
            free = [(hx+dx, hy+dy) for dx, dy in DIRECTIONS
                    if self._inside((hx+dx, hy+dy)) and (hx+dx, hy+dy) not in self.snake]
            if free:
                return self._rng.choice(free)
        return super()._sample_food()


def episode(brain, seed, stage, size, limit, on_step=None):
    game = CurriculumGame(seed=seed, stage=stage, size=size, calibration=brain.brain.calibration)
    brain.brain.reset_trial()
    rows = []; total_return = 0.
    for _ in range(limit):
        rgb = game.render_observation()
        if on_step is not None:
            on_step('input', game, rgb, None, None, None)
        receipt = brain.observe(rgb)
        receipt.update(episode_id=game.episode_id, step_id=game.step_id)
        event = None; feedback = None
        if receipt['action'] is None:
            # No invented fallback move and no waiting loop: punish this window,
            # then immediately integrate the next; the episode has a finite cap.
            feedback = brain.feedback('aversive')
            total_return -= .1
        else:
            event = game.apply_receipt(receipt)
            if not event['accepted']:
                raise RuntimeError(f'Invalid native neural receipt: {event}')
            if event['kind'] != 'none':
                feedback = brain.feedback(event['kind'])
                # The task return reflects the outcome, not small variations
                # in DAN response to the same outcome. Delivery is checked.
                total_return += 1. if event['kind'] == 'reward' else -1.
                game.complete_feedback(event['event_id'])
        if feedback is not None and feedback['delivered_spikes'] <= 0:
            raise RuntimeError('Outcome stimulus failed to reach the native circuit')
        rows.append({'receipt': receipt, 'event': event, 'feedback': feedback})
        if on_step is not None:
            on_step('result', game, rgb, receipt, event, feedback)
        if game.terminal:
            break
    if not game.terminal:
        # Explicit experimental timeout penalty. It is not a collision and is
        # reported separately. Prevents endless loops from earning survival pay.
        if brain.brain.last is None:
            brain.observe(game.render_observation())
        timeout = brain.feedback('aversive')
        if timeout['delivered_spikes'] <= 0:
            raise RuntimeError('Timeout stimulus not delivered')
        total_return -= 1.
    else:
        timeout = None
    return {'seed': seed, 'stage': stage, 'size': size, 'food': game.score,
            'steps': game.step_id, 'decisions': len(rows),
            'silent': sum(r['receipt']['action'] is None for r in rows),
            'reason': game.death_reason or ('win' if game.outcome == 'win' else 'time_limit'),
            'return': float(total_return), 'rows': rows, 'timeout_feedback': timeout}


def batch(brain, seeds, stage, size, limit):
    weight = brain.weight_hash()
    rows = [episode(brain, int(s), stage, size, limit) for s in seeds]
    if brain.weight_hash() != weight:
        raise RuntimeError('Weights changed during a frozen candidate rollout')
    return float(np.mean([r['return'] for r in rows])), rows


def assess(brain, n, size, limit):
    result = {}
    for stage in ('fixed_pose', 'nearby_food', 'uniform_food'):
        value, rows = batch(brain, range(96001, 96001+n), stage, size, limit)
        result[stage] = {'food': sum(r['food'] for r in rows), 'n': len(rows),
                         'mean_return': value, 'episodes': rows}
    return result


def source_manifest():
    paths = [Path(__file__)]
    for directory in ('flysnake', 'vendor/stonkfly/neural'):
        paths += [p for p in (ROOT/directory).rglob('*')
                  if p.suffix in ('.py', '.cpp', '.dll', '.json')]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def train(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output):
        path = output/'status.json'
        existing = path.exists()
        if existing and not args.resume:
            raise ValueError('Existing run preserved: use --resume or a fresh directory')
        if args.resume and not existing:
            raise ValueError('No committed experiment exists to resume')
        config = SearchConfig()
        settings = {k: getattr(args, k) for k in ('seed', 'arm', 'generations', 'pairs',
                                                 'batch_size', 'size', 'limit', 'evaluation_n')}
        sources = source_manifest()
        report = json.loads(path.read_text(encoding='utf-8')) if existing else None
        if existing:
            if report['settings'] != settings or report['sources'] != sources:
                raise ValueError('Source/configuration differs from committed run')
            if report['status'] == 'complete':
                print(json.dumps({'status': 'already_complete'}), flush=True)
                return
        else:
            report = {'protocol': VERSION, 'status': 'initializing', 'arm': args.arm,
                      'settings': settings, 'sources': sources, 'iterations': [], 'checkpoints': [],
                      'interpretation': 'Development evidence only. Frozen controls, independent seeds and larger boards remain required.',
                      'reward': {'food': 1., 'death': -1., 'no_output': -.1, 'timeout': -1.,
                                 'DAN_role': 'Native stimulation is delivered and verified; amplitude does not redefine the game score.'},
                      'progress': {'phase': 'loading_native_graph'}}
            atomic_json(path, report)
        search = PairedSearch(args.seed, config)
        brain = NativeSearchBrain(config.dimensions)
        provenance = {'settings': settings, 'sources': sources}
        report['brain'] = brain.identity
        if report['checkpoints']:
            c = report['checkpoints'][-1]
            restore_checkpoint(output/c['file'], c['sha256'], brain, search, provenance)
            if search.updates != len(report['iterations']):
                raise ValueError('Checkpoint/committed generation mismatch')
        start = time.time()

        def commit():
            name = f'memory-{search.updates:04d}-{time.time_ns()}.npz'
            sha = save_checkpoint(output/name, brain, search, provenance)
            report['checkpoints'].append({'generation': search.updates, 'file': name,
                                          'sha256': sha, 'memory_sha256': brain.weight_hash()})
            atomic_json(path, report)

        try:
            if 'before' not in report:
                report['progress'] = {'phase': 'baseline_evaluation'}
                atomic_json(path, report)
                report['before'] = assess(brain, args.evaluation_n, args.size, args.limit)
                report['status'] = 'running'
                commit()
                print(json.dumps({'before': {k: v['food'] for k, v in report['before'].items()}}), flush=True)
            for gen in range(search.updates, args.generations):
                if (output/'STOP_AFTER_GENERATION').exists():
                    raise KeyboardInterrupt('Stop requested at a committed generation boundary')
                stage = ('fixed_pose', 'nearby_food', 'uniform_food')[min(2, gen*3//args.generations)]
                # Independent world-seed stream; never expose these to the brain.
                seeds = np.random.default_rng(args.seed+10000+gen).integers(100000, 200000, args.batch_size)
                directions = search.directions(args.pairs)
                scores = []; records = []
                for j, direction in enumerate(directions):
                    paired = []; outcomes = []
                    for sign in (1, -1):
                        report['status'] = 'running'
                        report['progress'] = {'phase': 'training', 'generation': gen+1,
                                              'stage': stage, 'pair': j+1, 'sign': sign}
                        atomic_json(path, report)
                        brain.set_parameters(search.candidate(direction, sign))
                        value, rows = batch(brain, seeds, stage, args.size, args.limit)
                        paired.append(value)
                        candidate_file = f'rollout-{gen+1}-{j+1}-{sign}-{time.time_ns()}.json'
                        atomic_json(output/candidate_file, {'episodes': rows, 'weights': brain.weight_hash()})
                        outcomes.append({'sign': sign, 'return': value,
                                         'food': sum(r['food'] for r in rows), 'n': len(rows),
                                         'file': candidate_file})
                    scores.append(paired); records.append(outcomes)
                update = search.update(directions, scores, args.arm)
                brain.set_parameters(search.theta)
                report['iterations'].append({'generation': gen+1, 'stage': stage,
                                              'candidates': records, 'update': update,
                                              'elapsed_seconds': time.time()-start})
                commit()
                print(json.dumps({'generation': gen+1, 'stage': stage, 'returns': scores, **update}), flush=True)
            report['progress'] = {'phase': 'final_frozen_evaluation'}
            atomic_json(path, report)
            report['after'] = assess(brain, args.evaluation_n, args.size, args.limit)
            report['status'] = 'complete'
            report['progress'] = {'phase': 'finished'}
            atomic_json(path, report)
            print(json.dumps({'status': 'complete', 'after': {k: v['food'] for k, v in report['after'].items()}}), flush=True)
        except BaseException as exc:
            # Do not promote half-finished generation state. Resume reloads the
            # last immutable boundary checkpoint and replays that generation.
            committed = json.loads(path.read_text(encoding='utf-8'))
            committed['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            committed['error'] = f'{type(exc).__name__}: {exc}'
            atomic_json(path, committed)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('learning', 'frozen', 'shuffled'), default='learning')
    p.add_argument('--seed', type=int, default=92601)
    p.add_argument('--generations', type=int, default=9)
    p.add_argument('--pairs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=3)
    p.add_argument('--size', type=int, choices=(4, 6, 8, 12), default=4)
    p.add_argument('--limit', type=int, default=8)
    p.add_argument('--evaluation-n', type=int, default=6)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.generations, a.batch_size, a.limit, a.evaluation_n) < 1 or a.pairs < 2:
        p.error('Positive counts and at least two perturbation pairs are required')
    train(a)
