"""Summarize frozen native food outcomes and detect constant-turn bias.

Environmental target classes are diagnostic only. They never enter the learner
or action channel. The short curriculum cannot establish ordinary Snake skill.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.game import DIRECTIONS
from flysnake.research.projection import CoveredGame


def paired(a, b):
    difference = np.asarray(a)-np.asarray(b)
    rng = np.random.default_rng(930081)
    bootstrap = difference[rng.integers(0, len(a), (20000, len(a)))].mean(1)
    gains = int((difference>0).sum()); losses = int((difference<0).sum())
    discordant = gains+losses
    p = (min(1., 2*sum(math.comb(discordant, i) for i in range(min(gains, losses)+1))/2**discordant)
         if discordant else 1.)
    return {'gains': gains, 'losses': losses, 'mean_food_difference': float(difference.mean()),
            'paired_bootstrap_95pct': np.quantile(bootstrap, [.025, .975]).tolist(),
            'exact_mcnemar_two_sided_p': p}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing evidence')
    raw = (args.run/'status.json').read_bytes(); run = json.loads(raw)
    if run['status']!='complete' or set(run['after'])!={'learning', 'frozen', 'shuffled'}:
        raise ValueError('All actual frozen assessments must complete')
    for name, sha in run['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Source differs from assessment: '+name)
    expected = []; hashes = []
    for pose in run['heldout_poses']:
        body = [tuple(cell) for cell in pose['body']]
        x, y = body[0]
        for dx, dy in DIRECTIONS:
            food = (x+dx, y+dy)
            if not (0<=food[0]<4 and 0<=food[1]<4) or food in body:
                continue
            relative = (DIRECTIONS.index((dx, dy))-pose['heading'])%4
            expected.append({3: 'left', 0: 'forward', 1: 'right'}[relative])
            game = CoveredGame(seed=0, size=4, geometry=run['brain']['geometry'])
            game.snake = body; game.direction = pose['heading']; game.food = food
            hashes.append(game.observation_hash())
    assessments = {'before': run['before'], **run['after']}
    for label, a in assessments.items():
        if [r['image_sha256'] for r in a['rows']]!=hashes:
            raise ValueError('Environmental diagnostic does not match actual neural pixels')
        if any((r['receipt']['action']==target)!=(r['food']==1) for r, target in zip(a['rows'], expected)):
            raise ValueError('Recorded native food outcome disagrees with transition diagnostic')
    scores = {label: np.array([r['food'] for r in a['rows']]) for label, a in assessments.items()}
    result = {'status': 'complete', 'training_status_sha256': hashlib.sha256(raw).hexdigest(),
              'training_seed': run['settings']['seed'], 'assessment_n': len(expected),
              'target_distribution': dict(Counter(expected)),
              'analytical_constant_action_ceiling': max(Counter(expected).values()),
              'assessments': {label: {'food': int(scores[label].sum()),
                  'actions': dict(Counter(r['receipt']['action'] for r in a['rows']))}
                  for label, a in assessments.items()},
              'comparisons': {reference: paired(scores['learning'], scores[reference])
                              for reference in ('before', 'frozen', 'shuffled')},
              'ordinary_snake_learning_proven': False,
              'interpretation': 'Single-seed short-window development experiment. Constant-action counts are analysis only; no such scripted controller exists in the game.'}
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__=='__main__':
    main()
