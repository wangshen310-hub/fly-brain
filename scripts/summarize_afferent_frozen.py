"""Offline paired/clustered interpretation of fixed frozen audit results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ACTIONS = ('left', 'forward', 'right')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    evaluation, dataset = Path(args.evaluation), Path(args.dataset)
    result = json.loads((evaluation / 'summary.json').read_text(encoding='utf-8'))
    if result['status'] != 'complete':
        raise ValueError('Evaluate only completed fixed runs')
    meta = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    if sha(dataset / 'dataset.npz') != meta['dataset_sha256']:
        raise ValueError('Dataset changed')
    with np.load(dataset / 'dataset.npz', allow_pickle=False) as data:
        labels = data['test_labels'].copy()
    poses = np.array([row['pose_key'] for row in meta['rows']['test']])
    groups = np.unique(poses)
    weights = 1. / (3 * np.bincount(labels, minlength=3)[labels])
    predictions = {}
    episodes = {}
    for arm in result['arms']:
        scenes = json.loads((evaluation / (arm + '-test-scenes.json')).read_text(encoding='utf-8'))
        predictions[arm] = np.array([ACTIONS.index(r['action']) if r['action'] is not None else -1
                                     for r in scenes['receipts']])
        if len(predictions[arm]) != len(labels):
            raise ValueError('Incomplete paired scenes')
        if 'ordinary_games' in result['arms'][arm]:
            episodes[arm] = json.loads((evaluation / (arm + '-episodes.json')).read_text(encoding='utf-8'))
            if [row['seed'] for row in episodes[arm]] != result['world_seeds']:
                raise ValueError('Ordinary worlds are not paired')
    rng = np.random.default_rng(81003)
    draws = rng.integers(0, len(groups), size=(20000, len(groups)))
    comparisons = {}
    correct = predictions['trained'] == labels
    for arm in ('mismatch', 'reset-afferents', 'vision-off'):
        baseline = predictions[arm] == labels
        difference = (correct.astype(float) - baseline.astype(float)) * weights
        contributions = np.array([difference[poses == group].sum() for group in groups])
        distribution = contributions[draws].sum(axis=1)
        row = {'extra_correct_of_72': int(correct.sum() - baseline.sum()),
               'balanced_accuracy_difference': float(difference.sum()),
               'pose_cluster_bootstrap_95_ci': np.quantile(distribution, [.025, .975]).tolist(),
               'improved_scene_indices': np.flatnonzero(correct & ~baseline).tolist(),
               'worsened_scene_indices': np.flatnonzero(~correct & baseline).tolist()}
        if arm in episodes:
            score_delta = np.array([a['score'] - b['score'] for a, b in
                                    zip(episodes['trained'], episodes[arm])], dtype=float)
            game_draws = rng.integers(0, len(score_delta), size=(20000, len(score_delta)))
            row.update(ordinary_mean_food_difference=float(score_delta.mean()),
                       paired_world_bootstrap_95_ci=np.quantile(score_delta[game_draws].mean(axis=1),
                                                               [.025, .975]).tolist(),
                       episodes_with_more_food=int(np.sum(score_delta > 0)),
                       episodes_with_less_food=int(np.sum(score_delta < 0)))
        comparisons[arm] = row
    summary = {'status': 'complete', 'evaluation_sha256': sha(evaluation / 'summary.json'),
               'test_pose_clusters': len(groups), 'test_frames': len(labels),
               'bootstrap_draws': 20000, 'bootstrap_seed': 81003,
               'comparisons': comparisons, 'source_sha256': sha(__file__),
               'bootstrap_method': 'Resample whole test poses; retain original complete-test '
                                   'class weights in paired balanced-accuracy contributions. '
                                   'Resample paired world seeds for mean ordinary food scores.',
               'interpretation': 'Exploratory single-training-seed precision, not a confirmatory '
                                 'multi-seed learning claim; game vision-off control not performed.'}
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
