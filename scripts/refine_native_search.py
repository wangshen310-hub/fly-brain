"""Multi-layout reward-only refinement, with pixel-disjoint untouched test cues.

Continues a declared development checkpoint; not an independent replication.
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
from probe_individual_synapses import IndividualBrain
from probe_motor_controllability import evaluate
from flysnake.lab.environment import cue_scene
from flysnake.lab.experiment import atomic_json


def key(seed):
    return hashlib.sha256(cue_scene(seed, 'left').tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidates', type=int, default=48)
    p.add_argument('--seed', type=int, default=92381)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'status.json').exists():
        raise ValueError('Use a new output directory')
    (args.output / 'source.py').write_bytes(Path(__file__).read_bytes())
    brain = IndividualBrain('dna02', 'motor', 'activity', 6, 12.)
    with np.load(args.source, allow_pickle=False) as z:
        for name in ('edges', 'base', 'group'):
            if not np.array_equal(z[name], getattr(brain, name)):
                raise ValueError(f'Incompatible candidate {name}')
        initial = z['theta'].copy()
        if json.loads(str(z['calibration'])) != brain.brain.calibration:
            raise ValueError('Incompatible calibration')
    # No test image may duplicate earlier development evaluation or calibration.
    seen = {key(s) for s in [6121, *range(7310, 7318), *range(4081, 4087), 444, 555]}
    train = [6121]
    for s in range(19700, 20000):
        if key(s) not in seen:
            seen.add(key(s))
            train.append(s)
            if len(train) == 4:
                break
    test = []
    for s in range(29700, 30000):
        if key(s) not in seen:
            seen.add(key(s)); test.append(s)
            if len(test) == 8:
                break
    if len(test) != 8:
        raise RuntimeError('Insufficient unique heldout images')
    report = {'protocol': 'native-multi-layout-refinement-dev-v1', 'status': 'running',
              'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'train_seeds': train, 'test_seeds': test, 'pixel_disjoint': True, 'seed': args.seed,
              'brain': brain.report, 'iterations': []}
    atomic_json(args.output / 'status.json', report)
    rng = np.random.default_rng(args.seed)
    elite, best = [], -float('inf')
    start = time.time()
    for i in range(args.candidates):
        if i == 0:
            theta = initial.copy()
        elif i % 8 == 7:
            theta = rng.uniform(-4, 4, len(initial))
        else:
            theta = np.clip(elite[int(rng.integers(len(elite)))] + rng.normal(0, .4 if i % 3 else 1., len(initial)), -4, 4)
        brain.set_parameters(theta)
        fitness, rows = evaluate(brain, train, 'opposite', feedback=True)
        if fitness > best:
            best, elite = fitness, []
        if fitness == best:
            elite.append(theta.copy()); elite = elite[-12:]
        report['iterations'].append({'i': i, 'theta': theta.tolist(), 'fitness': fitness, 'rows': rows})
        report.update(best_fitness=best, best_theta=elite[-1].tolist())
        brain.save(args.output / f'candidate-{i}.npz')
        atomic_json(args.output / 'status.json', report)
        print(json.dumps({'i': i, 'fitness': fitness, 'best': best, 'elapsed': time.time()-start}), flush=True)
    brain.set_parameters(elite[-1])
    fit, rows = evaluate(brain, test, 'opposite')
    brain.save(args.output / 'candidate.npz')
    brain.set_parameters(np.zeros_like(initial))
    baseline_fit, baseline_rows = evaluate(brain, test, 'opposite')
    report.update(status='complete', test_fitness=fit, test=rows, baseline_fitness=baseline_fit, baseline=baseline_rows,
                  interpretation='Development only; final test labels never used to select weights. Independent training seeds and reversal controls required.')
    atomic_json(args.output / 'status.json', report)
    print(json.dumps({'status': 'complete', 'test_fitness': fit, 'baseline_fitness': baseline_fit}), flush=True)


if __name__ == '__main__':
    main()
