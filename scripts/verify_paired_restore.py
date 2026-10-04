"""Independent full-native replay of an immutable research checkpoint."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.paired import NativeSearchBrain, PairedSearch, SearchConfig, restore_checkpoint
from train_paired_native_snake import episode
from flysnake.lab.experiment import atomic_json


def comparable(row):
    return {k: row[k] for k in ('food', 'steps', 'decisions', 'silent', 'reason', 'return')} | {
        'neural': [{'action': r['receipt']['action'], 'counts': r['receipt']['counts'],
                    'pixels': r['receipt']['observation_sha256'],
                    'feedback': None if r['feedback'] is None else {
                        k: r['feedback'][k] for k in ('kind', 'delivered_spikes', 'modulation')}}
                   for r in row['rows']]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    c = report['checkpoints'][-1]
    provenance = {'settings': report['settings'], 'sources': report['sources']}
    brain = NativeSearchBrain(); search = PairedSearch(0, SearchConfig())
    restore_checkpoint(args.run/c['file'], c['sha256'], brain, search, provenance)
    expected_noise = search.directions(3)
    expected_hash = brain.weight_hash()
    first = comparable(episode(brain, 97901, 'uniform_food', 4, 4))
    brain.set_parameters(np.full_like(search.theta, .15))
    search.theta[:] = .15; search.m[:] = 1.; search.v[:] = 2.
    restore_checkpoint(args.run/c['file'], c['sha256'], brain, search, provenance)
    actual_noise = search.directions(3)
    second = comparable(episode(brain, 97901, 'uniform_food', 4, 4))
    assert brain.weight_hash() == expected_hash
    np.testing.assert_array_equal(expected_noise, actual_noise)
    assert first == second
    result = {'protocol': 'native-paired-restore-verification-v1', 'status': 'passed',
              'checkpoint': c, 'native_neurons': brain.brain.b.n,
              'native_edges': len(brain.brain.b.post), 'replay': first,
              'next_parameter_perturbations_identical': True,
              'interpretation': 'Persistence and deterministic neural replay only; not evidence of learned Snake behavior.'}
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('replay', 'checkpoint')}), flush=True)


if __name__ == '__main__':
    main()
