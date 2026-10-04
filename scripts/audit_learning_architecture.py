"""Read-only audit of committed learning evidence; never loads or trains a brain.

Offline target labels are diagnostic only. This program cannot emit actions or
modify experiment directories. Its only write is the requested audit JSON.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    'native-sensory-basis-controls-20260930-pilot-v1',
    'native-actor-credit-controls-20260930-pilot-v1',
    'native-pulse-balanced-controls-20260930-pilot-v1',
    'native-chemical-causal-controls-20260930-pilot-v1',
)
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def target(episode):
    """Used only in this offline audit, never available to any native actor."""
    x, y = episode['pose']['body'][0]
    fx, fy = episode['food_start']
    relative = (DIRECTIONS.index((fx-x, fy-y))-episode['pose']['heading']) % 4
    return {0: 'forward', 1: 'right', 3: 'left'}[relative]


def eval_summary(ev):
    if not ev:
        return None
    episodes = ev['episodes']
    return {
        'food': ev['food'], 'n': ev['n'],
        'actions': dict(Counter(r['receipt']['action'] for e in episodes for r in e['rows'])),
        'parameters_frozen': ev.get('parameters_frozen'),
        'exploration_off': ev.get('exploration_off'),
    }


def process_snapshot():
    # Read current OS processes, not stale experiment status or a lock filename.
    command = r"""$p = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -match 'fly-brain-snake|serve_.*snake|train_chemical|durable_worker'
    } | Select-Object ProcessId,ParentProcessId,Name,CommandLine)
    ConvertTo-Json -InputObject $p -Compress"""
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, encoding='utf-8', check=True)
    processes = json.loads(result.stdout or '[]')
    return [p for p in processes if 'audit_learning_architecture.py' not in p['CommandLine']]


def chemical_details(s):
    feedback = []
    mismatches = []
    repetitions = defaultdict(set)
    counts_by_kind = defaultdict(list)
    cycles = []
    for cycle in s['cycles']:
        pose_actions = defaultdict(set)
        action_targets = Counter()
        for scenario in cycle['batch']:
            action_targets[target({'pose': scenario['pose'], 'food_start': scenario['food']})] += 1
        for arm, arm_data in cycle['arms'].items():
            for group in arm_data['groups']:
                for episode in group['episodes']:
                    for row in episode['rows']:
                        receipt = row['receipt']
                        left, right = receipt['counts']['left'][0], receipt['counts']['right'][0]
                        stronger = 'left' if left > right else 'right' if right > left else None
                        if receipt['action'] in ('left', 'right') and stronger != receipt['action']:
                            mismatches.append({'cycle': cycle['cycle'], 'arm': arm,
                                'candidate': group['candidate'], 'left': left, 'right': right,
                                'action': receipt['action'], 'rgb': receipt['observation_sha256']})
                        key = (group['weights_sha256'], receipt['observation_sha256'],
                               tuple(row['sense']['concentration']))
                        repetitions[key].add(json.dumps(receipt['counts'], sort_keys=True))
                        if row['feedback']:
                            feedback.append(row['feedback'])
                            counts_by_kind[row['feedback']['kind']].append(row['feedback'])
                        if arm == 'learning' and group['candidate'] == 0:
                            pose_actions[(episode['pose']['heading'],
                                          tuple(map(tuple, episode['pose']['body'])))].add(receipt['action'])
        cycles.append({'cycle': cycle['cycle'], 'worlds': len(cycle['batch']),
            'offline_target_distribution': dict(action_targets),
            'current_normal_poses': len(pose_actions),
            'current_normal_poses_with_food_dependent_action': sum(len(v)>1 for v in pose_actions.values()),
            'normal_food_current_plus_minus': [g['food'] for g in cycle['arms']['learning']['groups']],
            'accepted': cycle['arms']['learning']['update']['accepted']})
    return {
        'feedback': {
            'events': len(feedback),
            'learning_enabled_events': sum(f['learning'] for f in feedback),
            'local_changed_edges_total': sum(f['changed_edges'] for f in feedback),
            'local_memory_hash_changed_events': sum(f['memory_before'] != f['memory_after'] for f in feedback),
            'delivered_dan_spikes_range': [min(f['delivered_spikes'] for f in feedback),
                                          max(f['delivered_spikes'] for f in feedback)],
            'kinds': {k: {'n': len(v), 'modulation_min': min(f['modulation'] for f in v),
                          'modulation_max': max(f['modulation'] for f in v)} for k,v in counts_by_kind.items()},
            'hash_scope': 'feedback hashes describe inherited LabBrain fraction, not all native weights; '
                          'NativeSensoryBasisBrain.feedback separately asserts mutable-weight hash unchanged',
        },
        'calibrated_turn_opposes_stronger_raw_side': {'n': len(mismatches), 'examples': mismatches[:5],
            'interpretation': 'Documented decoder transform, not random fallback; counts are recorded native output.'},
        'identical_weights_pixels_odor_with_different_turn_counts': sum(len(v)>1 for v in repetitions.values()),
        'unique_weights_pixels_odor_groups': len(repetitions),
        'cycles': cycles,
        'acceptance_scope': 'Only current randomly sampled small batch, no cumulative replay objective',
        'native_rollout_learning': False,
        'weight_learning_method': 'External scalar-game-return candidate search on existing native edges',
        'final_assessment_available': set(s['after']) == {'learning', 'frozen', 'shuffled'},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    processes = process_snapshot()
    results = {}
    for name in RUNS:
        path = ROOT/'runtime'/name/'status.json'
        s = read(path)
        results[name] = {
            'protocol': s['protocol'], 'recorded_status': s['status'],
            'completed_cycles': s['cycles_completed'], 'active_cycle': s.get('active_cycle'),
            'mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            'status_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'before': eval_summary(s['before']),
            'after': {k: eval_summary(v) for k,v in s['after'].items()},
            'heldout_is_development_set': True,
            'source_hash_mismatches': [relative for relative, sha in s['sources'].items()
                if not (ROOT/relative).exists() or
                hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != sha],
        }
        if s.get('checkpoints'):
            cp = s['checkpoints'][-1]
            if Path(cp['file']).name != cp['file']:
                raise ValueError('Unsafe checkpoint filename')
            cp_path = path.parent/cp['file']
            results[name]['latest_checkpoint'] = {
                'file': cp['file'], 'cycles_completed': cp['cycles_completed'],
                'file_sha256_verified': hashlib.sha256(cp_path.read_bytes()).hexdigest() == cp['sha256'],
            }
        if s['settings']['stage'] == 'adjacent_one_step' and s['before']:
            labels = Counter(target(e) for e in s['before']['episodes'])
            results[name]['offline_constant_action_baseline'] = {
                'target_distribution': dict(labels), 'best_food': max(labels.values()),
                'labels_available_to_learner': False,
            }
        if 'chemical-causal' in name:
            results[name]['mechanism'] = chemical_details(s)
    snapshots = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
        'vendor/stonkfly/neural/rule.py', 'vendor/stonkfly/neural/brain.py',
        'vendor/stonkfly/neural/circuit.py', 'flysnake/lab/brain.py',
        'flysnake/lab/protocol.py', 'flysnake/research/paired.py',
        'flysnake/research/sensory_basis.py', 'flysnake/research/actor_credit.py',
        'flysnake/research/pulse_balance.py', 'flysnake/research/chemical_basis.py',
        'scripts/train_actor_credit.py', 'scripts/train_chemical_controls.py',
        'scripts/run_ordinary_chemical_snake.py')}
    report = {
        'audit': 'architecture-learning-causal-chain-2026-10-02-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Committed logs and source; no new neural rollout, training, or checkpoint mutation',
        'project_processes': processes,
        'stale_running_record_without_project_process': not processes and
            any(r['recorded_status'] == 'running' for r in results.values()),
        'source_sha256': snapshots, 'experiments': results,
        'ordinary_snake_learning_established': False,
        'neural_conditioning_gate_passed': False,
        'interpretation': 'Failure of these protocols is not proof that a connectome-constrained model cannot learn.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'audit': report['audit'], 'processes': len(processes),
        'chemical_feedback': results[RUNS[-1]]['mechanism']['feedback'],
        'decoder_opposed_turns': results[RUNS[-1]]['mechanism']['calibrated_turn_opposes_stronger_raw_side']['n'],
        'nonreproducible_groups': results[RUNS[-1]]['mechanism']['identical_weights_pixels_odor_with_different_turn_counts'],
        'output': str(args.output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
