"""Fresh full-model restore and replay of the finite gamma1 diagnostic.

This verifies committed nonzero weights and actual neural receipts. It is not
a gameplay test or evidence that the failed conditioning mechanism learned.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.input_isolation import IsolatedInputBrain, InputConfig
from flysnake.research.paired import digest
from flysnake.lab.experiment import atomic_json
from pilot_gamma1_conditioning import assessment, trial, sources


def biological(receipt):
    return {key: value for key, value in receipt.items() if key != 'neural_time_end'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier restore evidence')
    recorded = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    if recorded['status'] != 'complete' or recorded['settings']['rounds'] != 1:
        raise ValueError('This verifier replays the completed one-round pilot')
    if recorded['sources'] != sources():
        raise ValueError('Pilot sources changed')
    result = {'started_at': time.time(), 'status': 'loading', 'run': str(args.run),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'arms': {}, 'learning_proven': False}
    atomic_json(args.output, result)
    brain = IsolatedInputBrain(inputs=InputConfig(vision_input=False,
                              lamina_bias_mV=0., zero_odor_is_silent=True))
    reference_outside = None
    for arm, record in recorded['arms'].items():
        first, final = record['checkpoints']
        brain.restore(args.run / first['file'], first['sha256'])
        replay = trial(brain, arm, 'DM4')
        expected = record['trials'][1]
        for key in ('weight_before_sha256', 'weight_after_sha256', 'changed_edges',
                    'mean_fraction', 'minimum_fraction'):
            if replay[key] != expected[key]:
                raise AssertionError(f'{arm} replay differs: {key}')
        for original, repeated in zip(expected['rows'], replay['rows']):
            if biological(original['receipt']) != biological(repeated['receipt']):
                raise AssertionError(f'{arm} actual native replay differs: {original["phase"]}')
        if brain.weight_hash() != final['memory_sha256']:
            raise AssertionError('Replayed final weights differ from committed checkpoint')
        metadata = brain.restore(args.run / final['file'], final['sha256'])
        restored = assessment(brain)
        for original, repeated in zip(recorded['retained'][arm], restored):
            if original['cue'] != repeated['cue'] or biological(original['receipt']) != biological(repeated['receipt']):
                raise AssertionError(f'{arm} frozen restored cue counts differ')
        mask = np.ones(len(brain.b.weight), dtype=bool); mask[brain.edges] = False
        outside = digest(brain.b.weight[mask])
        if reference_outside is None:
            reference_outside = outside
        elif outside != reference_outside:
            raise AssertionError('Nonplastic native graph weights differ across arms')
        result['arms'][arm] = {'final_memory_sha256': brain.weight_hash(),
                              'state': metadata['state'], 'native_trial_replay_equal': True,
                              'restored_frozen_probes_equal': 3,
                              'nonplastic_weights_sha256': outside}
        atomic_json(args.output, result)
        print(json.dumps({'arm': arm, 'replay_equal': True, 'frozen_probes_equal': 3}), flush=True)
    result.update(status='complete', completed_at=time.time(),
                  scope='Fresh process, one full native model; four second trials and twelve frozen probes. '
                        'Trial-boundary memory only, no mid-window or ordinary Snake recovery claim.')
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()
