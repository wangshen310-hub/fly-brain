"""Independent frozen generalization and ordinary-game audit of afferent training.

The fixed 60-step checkpoints are chosen before this evaluation. Test labels
are used only after neuronal inference, and never supplied to the brain. No
expert module, optimizer or reward-driven update is available in this process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import ACTIONS, Dynamics, file_hash
from flysnake.research.teacher_afferent import PlasticAfferentBrain
from flysnake.research.teacher_game import SpikingSnake, attach_game_receipt


def atomic(path, value):
    temp = path.with_suffix('.partial')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05 * (attempt + 1))


def committed(run):
    run = Path(run)
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    point = json.loads((run / 'committed.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete' or status['step'] != 60 or point['step'] != 60:
        raise ValueError('This bounded audit requires the completed fixed 60-step protocol')
    return point['checkpoint'], status


def scene_metrics(receipts, labels):
    pred = np.array([ACTIONS.index(r['action']) if r['action'] is not None else -1
                     for r in receipts])
    recall = [float(np.mean(pred[labels == k] == k)) for k in range(3)]
    return {'cases': len(labels), 'accuracy': float(np.mean(pred == labels)),
            'balanced_accuracy': float(np.mean(recall)), 'recall': recall,
            'class_counts': np.bincount(labels, minlength=3).tolist(),
            'predicted_actions': [int(np.sum(pred == k)) for k in range(3)],
            'silent': int(np.sum(pred < 0))}


def ordinary_games(brain, seeds, max_decisions, visual=True):
    parameter_hash = brain.parameter_hash()
    episodes, events = [], []
    for episode, seed in enumerate(seeds):
        # The same preselected episode seeds are used by every frozen arm.
        game = SpikingSnake(seed=int(seed), size=12, parameter_sha256=parameter_hash,
                            window_ms=brain.config.window_ms)
        state, actions, total_reward = None, [], 0.
        initial = game.snapshot()
        for decision in range(max_decisions):
            receipt, state = brain.act(game.render_observation(), state, visual_enabled=visual)
            event = game.apply_receipt(attach_game_receipt(receipt, game, parameter_hash))
            if not event['accepted']:
                raise ValueError('Actual neural receipt rejected: ' + event['reason'])
            actions.append(receipt['action'])
            total_reward += event['scalar_reward']
            events.append({'episode': episode, 'decision': decision, 'receipt': receipt,
                           'event': event})
            if game.pending_feedback:
                game.complete_feedback(event['event_id'])
            if game.terminal:
                break
        episodes.append({'episode': episode, 'seed': int(seed), 'score': game.score,
                         'decisions': len(actions), 'silent': actions.count(None),
                         'reward': total_reward, 'terminal': game.terminal,
                         'death_reason': game.death_reason, 'truncated': not game.terminal,
                         'initial': initial, 'final': game.snapshot(), 'actions': actions})
    all_actions = [action for row in episodes for action in row['actions']]
    summary = {'episodes': len(episodes), 'board_size': 12,
               'total_food': sum(row['score'] for row in episodes),
               'mean_food_score': float(np.mean([row['score'] for row in episodes])),
               'deaths': sum(row['terminal'] and row['death_reason'] is not None for row in episodes),
               'truncated': sum(row['truncated'] for row in episodes),
               'death_reasons': {reason: sum(row['death_reason'] == reason for row in episodes)
                                 for reason in sorted({row['death_reason'] for row in episodes
                                                       if row['death_reason'] is not None})},
               'actual_decisions': len(all_actions),
               'action_counts': {str(action): all_actions.count(action) for action in (*ACTIONS, None)},
               'reward_learning_enabled': False,
               'neural_state': 'continuous within each episode; reset between independent episodes'}
    if brain.parameter_hash() != parameter_hash:
        raise ValueError('Frozen game changed parameters')
    return summary, episodes, events


def main():
    p = argparse.ArgumentParser()
    for name in ('graph', 'dataset', 'normal-run', 'mismatch-run', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--episodes', type=int, default=24)
    p.add_argument('--max-decisions', type=int, default=100)
    p.add_argument('--world-seed', type=int, default=81003)
    args = p.parse_args()
    if args.episodes < 1 or args.max_decisions < 1:
        raise ValueError('Positive bounded evaluation required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    normal, normal_status = committed(args.normal_run)
    mismatch, mismatch_status = committed(args.mismatch_run)
    if normal_status['warm_checkpoint'] != mismatch_status['warm_checkpoint']:
        raise ValueError('Matched afferent arms must share the same upstream checkpoint')
    meta = json.loads((Path(args.dataset) / 'manifest.json').read_text(encoding='utf-8'))
    if file_hash(Path(args.dataset) / 'dataset.npz') != meta['dataset_sha256']:
        raise ValueError('Frozen dataset bytes changed')
    with np.load(Path(args.dataset) / 'dataset.npz', allow_pickle=False) as data:
        frames = data['test_rgb'].copy()
        labels = data['test_labels'].copy()
    train_poses = {r['pose_key'] for r in meta['rows']['train']}
    test_poses = {r['pose_key'] for r in meta['rows']['test']}
    if train_poses & test_poses:
        raise ValueError('Test poses overlap training')
    seeds = np.random.default_rng(args.world_seed).integers(1, 2**31, args.episodes).tolist()
    payload = torch.load(normal['path'], map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    brain = PlasticAfferentBrain.load_graph(args.graph,
                                          config=Dynamics(**payload['identity']['dynamics']))
    sources = out / 'sources'
    sources.mkdir()
    paths = [Path(__file__), ROOT / 'flysnake/research/teacher_afferent.py',
             ROOT / 'flysnake/research/teacher_spiking.py',
             ROOT / 'flysnake/research/teacher_game.py',
             ROOT / 'flysnake/research/optical_sampling.py']
    for path in paths:
        shutil.copyfile(path, sources / path.name)
    result = {'status': 'running', 'pid': os.getpid(), 'config': vars(args),
              'normal_checkpoint': normal, 'mismatch_checkpoint': mismatch,
              'common_warm_checkpoint': normal_status['warm_checkpoint'],
              'dataset_sha256': meta['dataset_sha256'], 'source_sha256': file_hash(__file__),
              'brain_identity': brain.identity, 'world_seeds': seeds,
              'test_training_pose_overlap': 0, 'training_seed_count': 1,
              'control_scope': 'New afferent stage only; both share the normal upstream 200-step fit.',
              'inference_has_expert': False, 'reward_learning_enabled': False,
              'ordinary_game_learning_validated': False, 'arms': {}}
    atomic(out / 'status.json', result)
    start = time.perf_counter()
    for arm in ('trained', 'mismatch', 'reset-afferents', 'vision-off'):
        point = mismatch if arm == 'mismatch' else normal
        brain.restore_checkpoint(point['path'], point['sha256'])
        if arm == 'reset-afferents':
            with torch.no_grad():
                brain.afferent_log_gain.zero_()
        parameter_hash = brain.parameter_hash()
        receipts = []
        # Isolated cold-state scenes are deliberately distinct from continuous
        # ordinary games below. Test labels cannot influence these receipts.
        for frame in frames:
            receipt, _ = brain.act(frame, visual_enabled=arm != 'vision-off')
            receipts.append(receipt)
        metrics = scene_metrics(receipts, labels)
        row = {'parameter_sha256': parameter_hash, 'test_scenes': metrics}
        atomic(out / (arm + '-test-scenes.json'), {'summary': metrics, 'receipts': receipts})
        if arm != 'vision-off':
            game_summary, episodes, events = ordinary_games(brain, seeds, args.max_decisions)
            row['ordinary_games'] = game_summary
            atomic(out / (arm + '-episodes.json'), episodes)
            atomic(out / (arm + '-events.json'), events)
        if brain.parameter_hash() != parameter_hash:
            raise ValueError('Frozen inference changed parameters')
        result['arms'][arm] = row
        atomic(out / 'status.json', result)
        print(json.dumps({'arm': arm, **row}), flush=True)
    if any(name.endswith('teacher_data') for name in sys.modules):
        raise ValueError('Expert module entered frozen inference')
    result.update(status='complete', wall_seconds=time.perf_counter() - start,
                  interpretation='Fixed training checkpoints, new heldout poses and ordinary games. '
                  'Any three-image fit is not proof of whole-game learning or natural fly plasticity.')
    atomic(out / 'summary.json', result)
    atomic(out / 'status.json', result)
    print(json.dumps({'status': 'complete', 'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
