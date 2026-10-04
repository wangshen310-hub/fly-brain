"""Frozen representation audit, never a game controller or learning result.

Compare the retained viewport and anatomically covered fixed viewport on the
same balanced adjacent-food scenes. Diagnostic predictors are discarded; only
their held-out predictions are saved. Whole body poses stay within one fold.
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
from flysnake.game import DIRECTIONS
from flysnake.lab.experiment import atomic_json
from flysnake.research.paired import NativeSearchBrain, writer_lock
from flysnake.research.projection import CoveredGame, coverage_geometry
from flysnake.research.vision import ReadableGame


def scenes(n):
    by_label = [[] for _ in range(3)]
    for heading, (dx, dy) in enumerate(DIRECTIONS):
        for y in range(4):
            for x in range(4):
                body = [(x-i*dx, y-i*dy) for i in range(3)]
                if any(not (0<=a<4 and 0<=b<4) for a, b in body):
                    continue
                for label, turn in enumerate((-1, 0, 1)):
                    fx, fy = DIRECTIONS[(heading+turn)%4]
                    food = (x+fx, y+fy)
                    if not (0<=food[0]<4 and 0<=food[1]<4) or food in body:
                        continue
                    by_label[label].append({'body': body, 'heading': heading,
                                            'food': food, 'label': label,
                                            'pose': str((body, heading))})
    rng = np.random.default_rng(930031)
    for rows in by_label:
        rng.shuffle(rows)
    n = min(n//3, min(map(len, by_label)))
    return [by_label[label][i] for i in range(n) for label in range(3)]


def folds(rows):
    mapping = {}
    for heading in range(4):
        poses = sorted({r['pose'] for r in rows if r['heading']==heading})
        np.random.default_rng(930040+heading).shuffle(poses)
        mapping.update({pose: i%4 for i, pose in enumerate(poses)})
    return np.array([mapping[r['pose']] for r in rows])


def diagnose(x, y, partition, kind):
    x = np.asarray(x, dtype=float)
    x = x[:, np.ptp(x, axis=0)>0]
    classes = int(y.max())+1
    guesses = np.full(len(y), -1)
    majority = np.full(len(y), -1)
    for fold in range(4):
        test = partition==fold; train = ~test
        if not test.any():
            continue
        scale = np.maximum(x[train].std(0), 1.)
        mean = x[train].mean(0)
        a = (x[train]-mean)/scale; b = (x[test]-mean)/scale
        majority[test] = np.argmax(np.bincount(y[train], minlength=classes))
        if kind=='linear':
            a = np.c_[a, np.ones(len(a))]; b = np.c_[b, np.ones(len(b))]
            gram = a@a.T; cross = b@a.T; ridge = 1.
        else:
            p = max(x.shape[1], 1)
            gram = np.exp(-np.maximum((a*a).sum(1)[:,None]+(a*a).sum(1)[None,:]-2*a@a.T, 0)/p)
            cross = np.exp(-np.maximum((b*b).sum(1)[:,None]+(a*a).sum(1)[None,:]-2*b@a.T, 0)/p)
            ridge = .1
        guesses[test] = np.argmax(cross@np.linalg.solve(gram+ridge*np.eye(len(a)),
                                                       np.eye(classes)[y[train]]), axis=1)
    return {'correct': int((guesses==y).sum()), 'n': len(y),
            'accuracy': float((guesses==y).mean()),
            'majority_accuracy': float((majority==y).mean()),
            'variable_features': x.shape[1], 'predictions': guesses.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenes', type=int, default=48)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output/'status.json'
        if path.exists():
            raise ValueError('Use a new output; preserve prior evidence')
        rows = scenes(args.scenes); partition = folds(rows)
        report = {'protocol': 'covered-fixed-board-representation-audit-v1',
                  'status': 'loading', 'scenes': rows, 'folds': partition.tolist(),
                  'completed': {}, 'results': {},
                  'inference_use': 'None. Diagnostic predictor coefficients are neither saved nor used for game control.',
                  'interpretation': 'Engineering perception diagnostic, not learned Snake behavior.'}
        (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
        atomic_json(path, report)
        try:
            native = NativeSearchBrain(); brain = native.brain; b = brain.b
            geometry = coverage_geometry(b.uv)
            report['geometry'] = geometry
            motors = np.r_[brain.native.groups['left'], brain.native.groups['right']]
            direct = native.edges[np.isin(b.post[native.edges], motors)]
            afferents = np.unique(np.searchsorted(b.ptr, direct, side='right')-1)
            upstream = np.unique(np.searchsorted(b.ptr, native.edges, side='right')-1)
            for mode in ('original', 'covered'):
                samples = []; hashes = []
                report.update(status='running', phase=mode)
                start = time.time()
                for i, row in enumerate(rows):
                    game = (ReadableGame(seed=0, size=4) if mode=='original' else
                            CoveredGame(seed=0, size=4, geometry=geometry))
                    game.snake = row['body']; game.direction = row['heading']; game.food = row['food']
                    rgb = game.render_observation()
                    brain.observe(rgb, learning=False, reset=True)
                    samples.append(brain.last['counts'].copy())
                    hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                    report['completed'][mode] = i+1
                    atomic_json(path, report)
                    if (i+1)%6==0:
                        print(json.dumps({'mode': mode, 'through': i+1,
                                          'elapsed': time.time()-start}), flush=True)
                counts = np.array(samples)
                np.savez_compressed(args.output/f'{mode}-native-spikes.npz',
                                    counts=counts, cell_ids=b.ids, image_hashes=hashes)
                result = {}
                for name, indices in (('retina', b.retina), ('motor_afferents', afferents),
                                      ('two_hop_presynaptic', upstream)):
                    features = counts[:, indices]
                    result[name] = {}
                    for target, y in (('relative_food', np.array([r['label'] for r in rows])),
                                      ('heading', np.array([r['heading'] for r in rows]))):
                        result[name][target] = {kind: diagnose(features, y, partition, kind)
                                               for kind in ('linear', 'rbf')}
                report['results'][mode] = result
                atomic_json(path, report)
                print(json.dumps({'mode': mode, 'results': result}), flush=True)
            report.update(status='complete', completed_at=time.time())
            atomic_json(path, report)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__=='__main__':
    main()
