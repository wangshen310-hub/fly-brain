"""Matched controls, frozen readouts, novelty and reversal for local neural learning.

Runs one bounded arm per directory. Seeds/arms must all be reported together;
this file never promotes a model to the user-facing game by itself.
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
from flysnake.lab.environment import cue_scene
from flysnake.lab.experiment import atomic_json, wilson


def novel_scene(seed, side):
    # Disjoint rectangle coordinates, not just a new seed for the same pixels.
    rng = np.random.default_rng(seed)
    rgb = np.full((180, 320, 3), 85, dtype=np.uint8)
    y = int(rng.integers(18, 26))
    x = (26 if side == 'left' else 186) + int(rng.integers(0, 8))
    rgb[y:y+148, x:x+124] = (180, 200, 255)
    return rgb


def assessment(brain, mapping):
    before = brain.weight_hash()
    output = {}
    for name, scene in (('familiar_distribution', cue_scene), ('novel_pixels', novel_scene)):
        rows, seen = [], set()
        for seed in range(40700, 41700):
            pair = [scene(seed, side) for side in ('left', 'right')]
            h = hashlib.sha256(pair[0].tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            for side, rgb in zip(('left', 'right'), pair):
                r = brain.observe(rgb)
                target = side if mapping == 'same' else ('right' if side == 'left' else 'left')
                rows.append({'seed': seed, 'cue': side, 'correct': r['action'] == target,
                             'action': r['action'], 'counts': r['counts'],
                             'observation_sha256': r['observation_sha256'], 'memory_sha256': r['memory_sha256']})
            if len(rows) == 32:
                break
        correct = sum(r['correct'] for r in rows)
        output[name] = {'correct': correct, 'n': len(rows), 'accuracy': correct / len(rows),
                        'wilson95_descriptive': wilson(correct, len(rows)), 'rows': rows}
    if brain.weight_hash() != before:
        raise RuntimeError('Frozen assessment changed memory')
    return output


def save_training(brain, path):
    brain.save(path)
    # Sufficient slow/optimizer state to resume at a completed trial boundary.
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k].copy() for k in z.files}
    values.update(mean_pre=brain.mean_pre, reward_baseline=brain.reward_baseline,
                  rng_state=json.dumps(brain.rng.bit_generator.state), tonic=brain.tonic,
                  memory_sha256=brain.weight_hash())
    with path.open('wb') as stream:
        np.savez_compressed(stream, **values)


def restore_training(brain, path, expected_sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Checkpoint file hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        for name in ('edges', 'base', 'group'):
            if not np.array_equal(z[name], getattr(brain, name)):
                raise ValueError(f'Incompatible checkpoint {name}')
        if json.loads(str(z['calibration'])) != brain.brain.calibration or float(z['tonic']) != brain.tonic:
            raise ValueError('Incompatible neural calibration or tonic drive')
        mean_pre = z['mean_pre']
        if mean_pre.shape != brain.mean_pre.shape or not np.isfinite(mean_pre).all():
            raise ValueError('Invalid eligibility statistics')
        brain.set_parameters(z['theta'])
        if brain.weight_hash() != str(z['memory_sha256']):
            raise ValueError('Restored neural weight hash mismatch')
        brain.mean_pre[:] = mean_pre
        brain.reward_baseline = float(z['reward_baseline'])
        brain.rng.bit_generator.state = json.loads(str(z['rng_state']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('learning', 'frozen', 'shuffled'), required=True)
    p.add_argument('--seed', type=int, default=92401)
    p.add_argument('--trials', type=int, default=320)
    p.add_argument('--initial-mapping', choices=('same', 'opposite'), default='opposite')
    p.add_argument('--no-reversal', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.trials % 2:
        raise ValueError('Balanced trial count required')
    args.output.mkdir(parents=True, exist_ok=True)
    existing = (args.output / 'status.json').exists()
    if existing and not args.resume:
        raise ValueError('Use a new output directory or explicitly resume')
    if not existing:
        (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    brain = PerturbedBrain(args.seed, 12.)
    report = {'protocol': 'native-node-perturbation-validation-v1', 'status': 'running', 'arm': args.arm,
              'seed': args.seed, 'trials_per_phase': args.trials, 'eta': .3, 'brain': brain.report,
              'initial_mapping': args.initial_mapping, 'no_reversal': args.no_reversal,
              'rows': [], 'evaluations': {}, 'checkpoints': []}
    mapping = args.initial_mapping
    if existing:
        saved = json.loads((args.output / 'status.json').read_text(encoding='utf-8'))
        for field in ('protocol', 'arm', 'seed', 'trials_per_phase', 'eta', 'initial_mapping', 'no_reversal'):
            if report[field] != saved[field]:
                raise ValueError(f'Resume configuration mismatch: {field}')
        if (args.output / 'source.py').read_bytes() != Path(__file__).read_bytes():
            raise ValueError('Resume source version mismatch')
        report = saved
        if report['status'] == 'complete':
            print(json.dumps({'status': 'already_complete'}), flush=True)
            return
        if report['checkpoints']:
            checkpoint = report['checkpoints'][-1]
            restore_training(brain, args.output / checkpoint['file'], checkpoint['sha256'])
    else:
        report['evaluations']['before'] = assessment(brain, mapping)
        atomic_json(args.output / 'status.json', report)
    phases = ['initial'] if args.no_reversal else ['initial', 'reversal']
    start = time.time()
    for phase_index, phase in enumerate(phases):
        if phase_index:
            mapping = 'same' if mapping == 'opposite' else 'opposite'
        fake = np.array(['reward', 'aversive'] * (args.trials // 2))
        np.random.default_rng(args.seed + 9183 + phase_index).shuffle(fake)
        if phase in report['evaluations']:
            continue
        completed = sum(r['phase'] == phase for r in report['rows'])
        for i in range(completed, args.trials):
            side = 'left' if i % 2 == 0 else 'right'
            rgb = cue_scene(6200 + i // 2, side)
            result = brain.train_observe(rgb)
            target = side if mapping == 'same' else ('right' if side == 'left' else 'left')
            correct = result['action'] == target
            kind = str(fake[i]) if args.arm == 'shuffled' else ('reward' if correct else 'aversive')
            feedback = brain.learn(kind, learning=args.arm != 'frozen', eta=.3)
            if feedback['delivered_spikes'] <= 0:
                raise RuntimeError('Undelivered feedback')
            report['rows'].append({'phase': phase, 'i': i, 'mapping': mapping, 'cue': side,
                                   'action': result['action'], 'correct': correct,
                                   'counts': result['counts'], 'noise': brain.noise.tolist(), 'feedback': feedback})
            path = args.output / f'{phase}-{i+1}-{time.time_ns()}.npz'
            save_training(brain, path)
            report['checkpoints'].append({'phase': phase, 'through': i+1, 'file': path.name,
                                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                          'memory_sha256': brain.weight_hash()})
            atomic_json(args.output / 'status.json', report)
            if (i + 1) % 40 == 0:
                print(json.dumps({'arm': args.arm, 'phase': phase, 'through': i+1,
                                  'recent_correct': sum(r['correct'] for r in report['rows'][-40:]),
                                  'elapsed': time.time()-start}), flush=True)
        report['evaluations'][phase] = assessment(brain, mapping)
        atomic_json(args.output / 'status.json', report)
        print(json.dumps({'arm': args.arm, 'phase': phase, 'evaluation':
                         {k: v['accuracy'] for k, v in report['evaluations'][phase].items()}}), flush=True)
    report.update(status='complete', interpretation='Report all arms and seeds together. No standalone arm proves robust learning or Snake competence.')
    atomic_json(args.output / 'status.json', report)


if __name__ == '__main__':
    main()
