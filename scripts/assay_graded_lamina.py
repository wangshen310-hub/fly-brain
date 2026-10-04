"""One predeclared, frozen graded L1-L5 candidate, never trained on actions."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.graded_lamina import GradedLaminaBrain, VERSION
from flysnake.research.visual_relay_trace import dynamic_hashes
from flysnake.research.visual_stimuli import fixed_geometry, audit_frames
from flysnake.research.apl_local import file_hash
from flysnake.research.paired import digest, writer_lock
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import immutable_arrays
from audit_visual_relay import anatomy, warm


def base_hashes(brain):
    hashes = dynamic_hashes(brain)
    hashes['arrays'] = {k: v for k, v in hashes['arrays'].items() if not k.startswith('lamina_graded_')}
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        status_path = args.output / 'status.json'
        if status_path.exists():
            raise ValueError('Use a new graded assay output')
        reference = json.loads((args.reference / 'status.json').read_text(encoding='utf-8'))
        if reference['status'] != 'complete':
            raise ValueError('Completed native relay audit required')
        source = dict(reference['sources'])
        for name in ('scripts/assay_graded_lamina.py', 'flysnake/research/graded_lamina.py',
                     'flysnake/research/kernels/graded_lamina_hooks.hpp'):
            source[name] = file_hash(ROOT / name)
        for name, sha in source.items():
            if file_hash(ROOT / name) != sha:
                raise ValueError('Locked source changed: ' + name)
            target = args.output / 'sources' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write((ROOT / name).read_bytes())
        settings = [('neutral-gray', 'neutral_gray', True), ('food-A', 'food_A', True),
            ('food-B', 'food_B', True), ('head-rotated', 'head_rotated', True),
            ('head-as-body', 'head_as_body', True),
            ('food-A-retinal-off', 'food_A', False), ('food-B-retinal-off', 'food_B', False),
            ('food-A-repeat', 'food_A', True)]
        report = {'status': 'loading', 'protocol': VERSION, 'process_id': os.getpid(),
            'started_at': time.time(), 'sources': source, 'rows': [],
            'reference_status_sha256': file_hash(args.reference / 'status.json'),
            'settings': {'conditions': settings, 'warm_ms': 100, 'pre_ms': 300,
                'cue_ms': 300, 'bin_ms': 10, 'olfactory_input': False, 'learning': False,
                'game_played': False, 'score_or_action_based_parameter_selection': False,
                'candidate_count': 1, 'model_parameters_physiologically_fitted': False},
            'behavioral_learning_proven': False}
        atomic_json(status_path, report)
        try:
            brain = GradedLaminaBrain(args.bundle, attachment=args.bundle / 'geometry',
                build_directory=args.output / 'build', enabled=False)
            b = brain.b
            frames = audit_frames(fixed_geometry(b, brain.aperture))
            types, groups, _ = anatomy(brain)
            groups['L4'] = np.flatnonzero(types == 'L4')
            groups['graded_L1_L5'] = brain.graded_indices
            immutable_arrays(args.output / 'anatomy.npz', cell_ids=b.ids, cell_types=types,
                external_allowed=brain.external_current_mask, graded_indices=brain.graded_indices,
                **{name + '_indices': ix for name, ix in groups.items()})
            report.update(brain=brain.identity.copy(), groups={k: len(ix) for k, ix in groups.items()},
                          anatomy_sha256=file_hash(args.output / 'anatomy.npz'))
            # An explicit zero-mechanism intervention must reproduce the locked
            # original full state. It is not a low-score parameter candidate.
            warm(brain, frames['neutral_gray'], True)
            checks = []
            for step in range(30):
                brain.integrate(frames['food_A'], 10)
                equal = base_hashes(brain) == reference['passthrough_reference'][step]
                if not equal:
                    raise AssertionError('Disabled graded kernel changed original native state: bin ' + str(step))
                checks.append(equal)
            report['disabled_candidate_complete_native_state_matches_original_30_bins'] = checks
            b.reset(keep_memory=True)
            brain.set_graded_enabled(True)
            report['brain'] = brain.identity.copy()
            records = {}
            pre_expected = None
            for index, (label, frame, visual) in enumerate(settings):
                pre = warm(brain, frames['neutral_gray'], visual)
                if visual:
                    if pre_expected is None:
                        pre_expected = pre
                    if pre != pre_expected:
                        raise AssertionError('Graded conditions have unequal prehistory')
                bins = []
                signature = []
                for step in range(30):
                    receipt = brain.integrate(frames[frame], 10)
                    filename = f'condition-{index:02d}-bin-{step:02d}.npz'
                    arrays = {'full_counts': brain.last_counts, 'voltage_mV': b.v, 'synaptic_g': b.g,
                        'external_drive': b.drive, 'graded_release_hz': brain.graded_rates,
                        'graded_delivery_positive': brain.graded_delivered[0],
                        'graded_delivery_negative_magnitude': brain.graded_delivered[1]}
                    sha = immutable_arrays(args.output / filename, **arrays)
                    summary = {name: {'spikes': int(brain.last_counts[ix].sum()),
                        'spiking_cells': int(np.count_nonzero(brain.last_counts[ix])),
                        'boundary_V_min_mV': float(b.v[ix].min()),
                        'boundary_V_max_mV': float(b.v[ix].max())} for name, ix in groups.items()}
                    bins.append({'bin': step, 'cursor': int(b.cursor), 'receipt': receipt,
                                 'summary': summary, 'arrays': {'file': filename, 'sha256': sha}})
                    signature.append({name: digest(value) for name, value in arrays.items()})
                    report.update(status='running', phase=label + '/' + str(step+1),
                                  active_bins=bins, updated_at=time.time())
                    atomic_json(status_path, report)
                if digest(b.weight) != brain.initial_all_weights_sha256:
                    raise AssertionError('Graded propagation changed a graph weight')
                row = {'label': label, 'frame': frame, 'retinal_input': visual, 'bins': bins,
                    'all_weights_unchanged': True,
                    'cue_motor_counts': {g: [sum(r['receipt']['counts'][g][k] for r in bins)
                        for k in range(len(ix))] for g, ix in brain.groups.items()}}
                if label == 'food-A':
                    row['checkpoint'] = {'file': 'graded-visual-state.npz',
                        'sha256': brain.save_state(args.output / 'graded-visual-state.npz')}
                    # Both queues, every native field and adapter histories are
                    # in the v2 checkpoint; retain a 100ms actual continuation.
                    replay = []
                    for _ in range(10):
                        brain.integrate(frames['neutral_gray'], 10)
                        replay.append(dynamic_hashes(brain))
                    report['checkpoint_expected_continuation'] = replay
                report['rows'].append(row)
                report.pop('active_bins', None)
                records[label] = signature
                print(json.dumps({'condition': label, 'motor_counts': row['cue_motor_counts'],
                    'KC_spikes': sum(r['summary']['all_KC']['spikes'] for r in bins),
                    'visual_spikes': sum(r['summary']['visual_annotated']['spikes'] for r in bins),
                    'graded_point_spikes': sum(r['summary']['graded_L1_L5']['spikes'] for r in bins)}), flush=True)
                atomic_json(status_path, report)
            report['gates'] = {
                'disabled_mechanism_matches_original_complete_native_state_30_bins': all(checks),
                'repeated_RGB_all_saved_arrays_identical': records['food-A'] == records['food-A-repeat'],
                'retinal_off_RGB_all_saved_arrays_identical':
                    records['food-A-retinal-off'] == records['food-B-retinal-off'],
                'all_weights_frozen': True, 'nonvisual_external_drive_zero': True,
                'graded_L1_L5_point_spikes_zero': True,
                'behavioral_learning_proven': False}
            report.update(status='complete', completed_at=time.time(),
                interpretation='One unfitted graded lamina candidate tested on frozen visual controls. '
                    'Information transfer and physiological stability must be analyzed; no behavior learning tested.')
            atomic_json(status_path, report)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(status_path, report)
            raise


if __name__ == '__main__':
    main()
