"""Sequential native interface, gamma1 reachability and causality assay.

No game is played or scored. All conditions are declared before integration.
The source snapshot is committed before the full model is constructed. Evidence
contains actual full-neuron count arrays and physical inputs, not synthetic
fixture spikes. Failed mechanism gates remain failed; there is no policy fallback.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.compartment_native import VERSION, NativeCompartmentBrain
from flysnake.research.projection import CoveredGame
from flysnake.research.paired import writer_lock, digest
from flysnake.lab.experiment import atomic_json


SOURCE_FILES = (
    'scripts/assay_native_compartment.py', 'flysnake/research/compartment_native.py',
    'flysnake/research/raw_turn.py', 'flysnake/research/olfaction.py',
    'flysnake/research/operating.py', 'flysnake/research/vision.py',
    'flysnake/research/projection.py', 'flysnake/lab/protocol.py',
    'flysnake/neural.py', 'vendor/stonkfly/neural/kernel.cpp',
    'vendor/stonkfly/neural/brain.py', 'vendor/stonkfly/neural/circuit.py',
    'vendor/stonkfly/neural/visual.py', 'vendor/stonkfly/neural/rule.py',
)


def sources():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def inputs(brain):
    rgb = np.full((180, 320, 3), 85, np.uint8)
    blank = np.zeros((2, 2))
    dm1 = np.array([[.8, .8], [0., 0.]])
    dm4 = np.array([[0., 0.], [.8, .8]])
    return [('neutral', rgb, blank), ('DM1', rgb, dm1), ('DM4', rgb, dm4)]


def immutable_arrays(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output/'status.json'
        if path.exists():
            raise ValueError('Use a fresh assay output; earlier evidence is immutable')
        source = sources()
        report = {'protocol': VERSION, 'status': 'loading', 'sources': source,
                  'process_id': os.getpid(), 'started_at': time.time(),
                  'rows': [], 'gates': {}, 'game_played': False, 'behavioral_learning_proven': False}
        for relative in source:
            target = args.output/'sources'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/relative).read_bytes())
        atomic_json(path, report)
        try:
            brain = NativeCompartmentBrain()
            report['brain'] = brain.identity
            samples = []
            def measure(label, rgb, concentrations, *, stimulation=None, expected=None):
                brain.reset_trial()
                result = brain.integrate(rgb, 300, concentrations, stimulation=stimulation)
                samples.append(brain.last_counts.copy())
                row = {'label': label, 'receipt': result, 'expected_for_motor_intervention': expected}
                report['rows'].append(row)
                report.update(status='running', phase=label, updated_at=time.time())
                atomic_json(path, report)
                print(json.dumps({'label': label, 'KC_active': result['KC_active'],
                    'KC_spikes': result['KC_spikes'], 'plastic_pre_active': result['plastic_pre_active'],
                    'readouts': result['readouts'], 'motor_counts': result['counts'],
                    'action': result['action']}), flush=True)
                return result
            # Both model working points are declared a priori; no reward,
            # correct action or candidate score is used to choose a condition.
            for headroom in (0., 4.):
                brain.set_working_point(headroom)
                for label, rgb, concentration in inputs(brain):
                    measure(f'headroom-{headroom:g}/{label}', rgb, concentration)
            brain.set_working_point(4.)
            rgb = inputs(brain)[0][1]
            all_motors = np.concatenate(list(brain.groups.values()))
            for label, active, opposite in (('left', 'left', 'right'), ('right', 'right', 'left')):
                measure('motor-intervention/'+label, rgb, np.zeros((2,2)),
                        stimulation=[(brain.groups[active], 30.), (brain.groups[opposite], -100.)], expected=label)
            measure('motor-intervention/silent', rgb, np.zeros((2,2)),
                    stimulation=[(all_motors, -100.)], expected=None)
            # Same pose, mirrored physical odor gradient. Pixels remain fixed.
            game = CoveredGame(seed=100201, size=4, geometry=brain.geometry)
            game.snake = [(1,1),(1,2),(1,3)]
            game.direction = 0
            game.food = None
            scene = game.render_observation()
            measure('bilateral-gradient/L', scene, [[.8,.2],[0.,0.]])
            measure('bilateral-gradient/R', scene, [[.2,.8],[0.,0.]])
            # Untaught synaptic intervention: can gamma1 memory affect motor
            # outputs at all? Preserve and restore baseline; no learning score.
            baseline = {}
            for label, rgb, concentration in inputs(brain)[1:]:
                baseline[label] = measure('memory-causality/baseline/'+label, rgb, concentration)
            brain.set_fraction(np.full(len(brain.edges), .8))
            altered = {}
            for label, rgb, concentration in inputs(brain)[1:]:
                altered[label] = measure('memory-causality/0.8/'+label, rgb, concentration)
            brain.reset_memory()
            alpha_mask = np.isin(brain.b.post[brain.b.circuit['edges']],
                                 brain.memory_readouts['alpha1_MBON07'])
            report['all_selected_edges_gamma1_only'] = not bool(alpha_mask.any())
            sensory = {row['label'].split('/')[-1]: row['receipt'] for row in report['rows']
                       if row['label'].startswith('headroom-4/')}
            report['gates'] = {
                'motor_interface': all(row['receipt']['action'] == row['expected_for_motor_intervention']
                    for row in report['rows'] if row['label'].startswith('motor-intervention/')),
                'odor_reaches_gamma1_presynapses': all(sensory[label]['plastic_pre_active']>0 for label in ('DM1','DM4')),
                'gamma1_presynaptic_patterns_differ': not np.array_equal(samples[4][brain.pre], samples[5][brain.pre]),
                'untaught_gamma1_intervention_changes_motor_counts': any(
                    baseline[label]['counts'] != altered[label]['counts'] for label in baseline),
                'untaught_gamma1_intervention_changes_mbon_counts': any(
                    baseline[label]['readouts']['gamma1_MBON11'] != altered[label]['readouts']['gamma1_MBON11']
                    for label in baseline),
            }
            report['native_spikes'] = {'file': 'native-spikes.npz', 'sha256': immutable_arrays(
                args.output/'native-spikes.npz', counts=np.array(samples), cell_ids=brain.b.ids,
                plastic_edges=brain.edges, plastic_pre=brain.pre, base_weights=brain.base)}
            report.update(status='complete', phase='completed', completed_at=time.time(),
                all_weights_restored=digest(brain.b.weight)==brain.initial_all_weights_sha256,
                interpretation='Sensory, raw motor BCI and existing gamma1-path causality only; no learned game behavior.')
            atomic_json(path, report)
            print(json.dumps({'status': report['status'], 'gates': report['gates'],
                              'all_weights_restored': report['all_weights_restored']}), flush=True)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__ == '__main__':
    main()
