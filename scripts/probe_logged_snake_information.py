"""Read-only information diagnostic; fitted predictors NEVER control the game.

Uses already-recorded native afferent spike counts, reconstructs the matching
initial frames, and groups identical frames across folds. No model coefficients
are saved. These are the legacy visual/motor afferents in receipts, not all
DNa02 inputs or the full brain; failure cannot prove the whole brain lacks info.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot_online_native_snake import NearbyFoodGame
from flysnake.game import DIRECTIONS


def predictions(x, y, groups, kind, seed=93001):
    classes=int(np.max(y))+1
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    result = np.full(len(y), -1)
    majority = np.full(len(y), -1)
    for fold in np.array_split(unique, min(5, len(unique))):
        test = np.isin(groups, fold)
        train = ~test
        if not train.any():
            raise ValueError('Insufficient distinct scenes')
        mean = x[train].mean(axis=0)
        scale = np.maximum(x[train].std(axis=0), 1.)
        a = (x[train] - mean) / scale
        b = (x[test] - mean) / scale
        majority[test] = np.argmax(np.bincount(y[train], minlength=classes))
        target = np.eye(classes)[y[train]]
        if kind == 'linear':
            a = np.c_[a, np.ones(len(a))]
            b = np.c_[b, np.ones(len(b))]
            gram = a @ a.T
            cross = b @ a.T
            ridge = 1.
        else:
            p = max(x.shape[1], 1)
            gram = np.exp(-np.maximum(np.sum(a*a, 1)[:, None] + np.sum(a*a, 1)[None, :] - 2*a@a.T, 0)/p)
            cross = np.exp(-np.maximum(np.sum(b*b, 1)[:, None] + np.sum(a*a, 1)[None, :] - 2*b@a.T, 0)/p)
            ridge = .1
        score = cross @ np.linalg.solve(gram + ridge*np.eye(len(a)), target)
        result[test] = np.argmax(score, axis=1)
    return result, majority


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raw = (args.run/'status.json').read_bytes()
    d = json.loads(raw)
    rows, features, motor = [], [], []
    for episode in d['episodes']:
        first = episode['rows'][0]['receipt']
        game = NearbyFoodGame(seed=episode['seed'], size=4, calibration=d['brain']['calibration'])
        rgb = game.render_observation()
        h = hashlib.sha256(rgb.tobytes()).hexdigest()
        if h != first['observation_sha256']:
            raise ValueError('Reconstructed frame does not match recorded brain input')
        hx, hy = game.snake[0]
        delta = (game.food[0]-hx, game.food[1]-hy)
        if delta not in DIRECTIONS:
            raise ValueError('Initial food is not adjacent')
        relative = (DIRECTIONS.index(delta)-game.direction) % 4
        label = {3: 0, 0: 1, 1: 2}[relative]
        rows.append({'seed': episode['seed'], 'image_sha256': h, 'label': label,
                     'action': first['action']})
        features.append(first['afferent_counts'])
        motor.append([float(np.mean(first['counts'][a])) for a in ('left', 'forward', 'right')])
    groups = np.array([r['image_sha256'] for r in rows])
    labels = np.array([r['label'] for r in rows])
    x = np.array(features, dtype=float)
    unique_cols = np.unique(x.T, axis=0).T
    unique_cols = unique_cols[:, np.ptp(unique_cols, axis=0) > 0]
    report = {'status_snapshot_sha256': hashlib.sha256(raw).hexdigest(), 'episodes': len(rows),
              'unique_images': len(np.unique(groups)), 'source_afferents': x.shape[1],
              'distinct_nonconstant_features': unique_cols.shape[1], 'class_order': ['left', 'forward', 'right'],
              'class_counts': np.bincount(labels, minlength=3).tolist(), 'rows': rows, 'results': {},
              'limits': 'Exploratory diagnostic on changing weights and exploratory currents. Only recorded legacy afferents; no whole-brain impossibility claim. No predictor is used or saved for game control.'}
    for representation, activity in (('recorded_afferents', unique_cols), ('motor_counts', np.array(motor))):
        for kind in ('linear', 'rbf'):
            pred, majority = predictions(activity, labels, groups, kind)
            accuracy = float(np.mean(pred == labels))
            null = []
            ug = np.unique(groups)
            group_labels = np.array([labels[np.flatnonzero(groups == g)[0]] for g in ug])
            for seed in range(93020, 93030):
                shuffled = group_labels.copy()
                np.random.default_rng(seed).shuffle(shuffled)
                assigned = np.array([shuffled[np.flatnonzero(ug == g)[0]] for g in groups])
                guess, _ = predictions(activity, assigned, groups, kind)
                null.append(float(np.mean(guess == assigned)))
            key = representation+'_'+kind
            report['results'][key] = {'accuracy': accuracy, 'correct': int(np.sum(pred == labels)),
                                     'majority_accuracy': float(np.mean(majority == labels)),
                                     'shuffled_label_mean': float(np.mean(null)), 'shuffled_label_range': [min(null), max(null)],
                                     'confusion': [[int(np.sum((labels == a) & (pred == b))) for b in range(3)] for a in range(3)]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows',)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
