"""Centered native learning, frozen and shuffled-feedback curriculum pilot.

This first-decision food curriculum is a necessary development test, NOT an
acceptance test for ordinary multi-step Snake. It uses actual game transitions
and native stimulation; no target action is given to the neural learner.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.game import DIRECTIONS
from flysnake.lab.experiment import atomic_json
from flysnake.research.paired import writer_lock, digest
from flysnake.research.projection import CoveredGame
from flysnake.research.centered_learning import (VERSION, CenteredCreditBrain as CoveredNodeBrain,
                                                CenteredLearningConfig as CoveredLearningConfig, reward_update)

ARMS = ('learning', 'frozen', 'shuffled')


def pose_split():
    training, heldout = [], []
    for heading, (dx, dy) in enumerate(DIRECTIONS):
        poses = []
        for x in range(4):
            for y in range(4):
                body = [(x-i*dx, y-i*dy) for i in range(3)]
                if all(0<=a<4 and 0<=b<4 for a, b in body):
                    poses.append({'heading': heading, 'body': body})
        indices = np.random.default_rng(930061+heading).permutation(len(poses))
        heldout.extend(poses[i] for i in indices[:2])
        training.extend(poses[i] for i in indices[2:])
    return training, heldout


def foods(pose):
    x, y = pose['body'][0]
    return [(x+dx, y+dy) for dx, dy in DIRECTIONS
            if 0<=x+dx<4 and 0<=y+dy<4 and (x+dx, y+dy) not in pose['body']]


def trial(brain, pose, food, noise=None):
    game = CoveredGame(seed=0, size=4, geometry=brain.geometry,
                       calibration=brain.brain.calibration)
    game.snake = pose['body'].copy(); game.direction = pose['heading']; game.food = food
    brain.reset_trial()
    rgb = game.render_observation()
    receipt, eligibility = brain.observe(rgb, noise)
    if receipt['selected_cell_ids'] != brain.identity['selected_cells']:
        raise RuntimeError('Motor group provenance changed')
    receipt.update(episode_id=game.episode_id, step_id=game.step_id)
    if receipt['action'] is None:
        feedback = brain.feedback('aversive')
        event = None; value = -1.1; reason = 'window_expired_no_output'
    else:
        event = game.apply_receipt(receipt)
        if not event['accepted']:
            raise RuntimeError('Rejected actual neural receipt: '+str(event))
        if event['kind']=='reward':
            value = 1.; reason = 'food'; feedback = brain.feedback('reward')
            game.complete_feedback(event['event_id'])
        elif event['kind']=='aversive':
            value = -1.; reason = game.death_reason; feedback = brain.feedback('aversive')
            game.complete_feedback(event['event_id'])
        else:
            # Explicit one-decision curriculum deadline, separate from death.
            # Eating a food closes this short curriculum window successfully.
            value = -1.; reason = 'window_expired'; feedback = brain.feedback('aversive')
    row = {'image_sha256': receipt['observation_sha256'], 'food': game.score,
           'return': value, 'reason': reason, 'receipt': receipt, 'event': event,
           'feedback': feedback, 'noise': None if noise is None else np.asarray(noise).tolist()}
    return row, eligibility


def assessment(brain, theta, heldout):
    brain.set_parameters(theta); weight = brain.weight_hash()
    rows = []
    for pose in heldout:
        for food in foods(pose):
            row, _ = trial(brain, pose, food)
            rows.append(row)
    if brain.weight_hash()!=weight:
        raise RuntimeError('Frozen assessment changed synapses')
    hashes = [r['image_sha256'] for r in rows]
    if len(set(hashes))!=len(hashes):
        raise RuntimeError('Held-out scenes are not distinct')
    return {'food': sum(r['food'] for r in rows), 'n': len(rows),
            'unique_images': len(set(hashes)), 'memory_sha256': weight, 'rows': rows}


def source_hashes():
    files = [Path(__file__), ROOT/'flysnake/research/covered_learning.py',
             ROOT/'flysnake/research/centered_learning.py',
             ROOT/'flysnake/research/projection.py', ROOT/'flysnake/research/paired.py',
             ROOT/'flysnake/research/vision.py', ROOT/'flysnake/research/operating.py',
             ROOT/'flysnake/lab/brain.py', ROOT/'flysnake/lab/environment.py',
             ROOT/'flysnake/lab/protocol.py', ROOT/'flysnake/game.py',
             ROOT/'flysnake/neural.py', ROOT/'vendor/stonkfly/neural/kernel.cpp']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def checkpoint(output, report, memory, rng, brain):
    name = f'memory-{report["blocks_completed"]:04d}-{time.time_ns()}.npz'
    metadata = {'protocol': VERSION, 'sources': report['sources'],
                'settings': report['settings'], 'identity': brain.identity,
                'rng': rng.bit_generator.state, 'blocks_completed': report['blocks_completed']}
    with (output/name).open('xb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True), **memory)
        stream.flush(); os.fsync(stream.fileno())
    report['checkpoints'].append({'file': name, 'sha256': hashlib.sha256((output/name).read_bytes()).hexdigest(),
                                  'blocks_completed': report['blocks_completed'],
                                  'parameters_sha256': {arm: digest(theta) for arm, theta in memory.items()}})
    atomic_json(output/'status.json', report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=930101)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=.3)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.blocks<1 or args.batch_size<2:
        raise ValueError('Positive block count and at least two actual returns required')
    config = CoveredLearningConfig(learning_rate=args.learning_rate)
    settings = {'seed': args.seed, 'blocks': args.blocks, 'batch_size': args.batch_size,
                'config': asdict(config), 'curriculum': 'one_decision_adjacent_food', 'size': 4}
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output/'status.json'; sources = source_hashes()
        exists = path.is_file()
        if exists != args.resume:
            raise ValueError('Resume an existing run explicitly; preserve prior directories')
        training, heldout = pose_split()
        report = (json.loads(path.read_text(encoding='utf-8')) if exists else
                  {'protocol': VERSION, 'status': 'loading', 'settings': settings,
                   'sources': sources, 'blocks_completed': 0, 'blocks': [], 'checkpoints': [],
                   'training_poses': training, 'heldout_poses': heldout,
                   'before': None, 'after': {}, 'training_image_hashes': [],
                   'interpretation': 'Development curriculum only. Ordinary multi-step Snake, independent training seeds, reversal and larger-board tests are still required.'})
        if report['settings']!=settings or report['sources']!=sources:
            raise ValueError('Committed configuration or source mismatch')
        if report['status']=='complete':
            print('already_complete', flush=True); return
        if not exists:
            for name in sources:
                target = args.output/'source-snapshot'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
            atomic_json(path, report)
        try:
            brain = CoveredNodeBrain(config)
            if report.get('activity_baseline'):
                baseline = report['activity_baseline']
                brain.restore_activity_baseline(args.output/baseline['file'], baseline['sha256'])
            else:
                # A label-free neural normalization pool uses training poses
                # only. Correct actions and reward are never computed here.
                baseline_rng = np.random.default_rng(930075)
                pool = [(pose, food) for pose in training for food in foods(pose)]
                selected = baseline_rng.choice(len(pool), min(32, len(pool)), replace=False)
                frames = []
                for index in selected:
                    pose, food = pool[int(index)]
                    game = CoveredGame(seed=0, size=4, geometry=brain.geometry)
                    game.snake = pose['body'].copy(); game.direction = pose['heading']; game.food = food
                    frames.append(game.render_observation())
                report.update(status='running', phase='label_free_activity_calibration')
                atomic_json(path, report)
                measured = brain.fit_activity_baseline(frames)
                baseline_path = args.output/f'activity-baseline-{time.time_ns()}.npz'
                sha = brain.save_activity_baseline(baseline_path)
                report['activity_baseline'] = {'file': baseline_path.name, 'sha256': sha, **measured}
                atomic_json(path, report)
            rng = np.random.default_rng(args.seed)
            memory = {arm: np.zeros_like(brain.theta) for arm in ARMS}
            if report['checkpoints']:
                latest = report['checkpoints'][-1]; file = args.output/latest['file']
                if hashlib.sha256(file.read_bytes()).hexdigest()!=latest['sha256']:
                    raise ValueError('Memory file checksum mismatch')
                with np.load(file, allow_pickle=False) as z:
                    meta = json.loads(str(z['metadata']))
                    if (meta['identity']!=json.loads(json.dumps(brain.identity))
                            or meta['settings']!=settings or meta['sources']!=sources
                            or meta['blocks_completed']!=report['blocks_completed']):
                        raise ValueError('Memory provenance mismatch')
                    memory = {arm: z[arm].copy() for arm in ARMS}
                    rng.bit_generator.state = meta['rng']
                for arm, theta in memory.items():
                    brain.set_parameters(theta)
                    if digest(theta)!=latest['parameters_sha256'][arm]:
                        raise ValueError('Memory parameter digest mismatch')
            if report['before'] is None:
                report.update(status='running', phase='initial_frozen_assessment', brain=brain.identity)
                atomic_json(path, report)
                report['before'] = assessment(brain, memory['frozen'], heldout)
                checkpoint(args.output, report, memory, rng, brain)
                print(json.dumps({'phase': 'before', 'food': report['before']['food'],
                                  'n': report['before']['n']}), flush=True)
            heldout_hashes = {r['image_sha256'] for r in report['before']['rows']}
            if heldout_hashes & set(report['activity_baseline']['image_hashes']):
                raise RuntimeError('Label-free calibration overlaps held-out pixels')
            for block in range(report['blocks_completed'], args.blocks):
                scenarios = []
                for _ in range(args.batch_size):
                    pose = training[int(rng.integers(len(training)))]
                    available = foods(pose)
                    scenarios.append((pose, available[int(rng.integers(len(available)))],
                                      rng.normal(0, config.noise_sigma, 2)))
                block_record = {'block': block+1, 'arms': {}}
                for arm in ARMS:
                    brain.set_parameters(memory[arm])
                    rows = []; traces = []
                    for i, (pose, food, noise) in enumerate(scenarios):
                        report.update(status='running', phase='training', arm=arm,
                                      active_block=block+1, active_trial=i+1)
                        atomic_json(path, report)
                        row, trace = trial(brain, pose, food, noise)
                        if row['image_sha256'] in heldout_hashes:
                            raise RuntimeError('Training/held-out pixel overlap')
                        rows.append(row); traces.append(trace)
                    memory[arm], update = reward_update(memory[arm], traces,
                        [r['return'] for r in rows], arm=arm, config=config,
                        shuffle_seed=args.seed+10000+block)
                    block_record['arms'][arm] = {'food': sum(r['food'] for r in rows),
                                                  'rows': rows, 'update': update}
                    print(json.dumps({'block': block+1, 'arm': arm,
                                      'food': block_record['arms'][arm]['food'],
                                      'update': update}), flush=True)
                report['blocks'].append(block_record)
                report['blocks_completed'] = block+1
                report['training_image_hashes'] = sorted(set(report['training_image_hashes']) |
                    {r['image_sha256'] for value in block_record['arms'].values() for r in value['rows']})
                checkpoint(args.output, report, memory, rng, brain)
                if (args.output/'STOP_AFTER_BLOCK').exists():
                    report.update(status='stopped', phase='committed_boundary')
                    atomic_json(path, report); return
            for arm in ARMS:
                report.update(phase='final_frozen_assessment', arm=arm)
                atomic_json(path, report)
                report['after'][arm] = assessment(brain, memory[arm], heldout)
                atomic_json(path, report)
                print(json.dumps({'phase': 'after', 'arm': arm,
                                  'food': report['after'][arm]['food'],
                                  'n': report['after'][arm]['n']}), flush=True)
            if report['after']['frozen']['rows']!=report['before']['rows']:
                # Neural absolute timestamps differ, so compare behavior and
                # count provenance directly rather than process-local clocks.
                keys = ('image_sha256', 'food', 'return', 'reason')
                for a, b in zip(report['before']['rows'], report['after']['frozen']['rows']):
                    if (any(a[k]!=b[k] for k in keys)
                            or a['receipt']['counts']!=b['receipt']['counts']):
                        raise RuntimeError('Frozen control did not reproduce initial behavior')
            report.update(status='complete', phase='complete', completed_at=time.time())
            atomic_json(path, report)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__=='__main__':
    main()
