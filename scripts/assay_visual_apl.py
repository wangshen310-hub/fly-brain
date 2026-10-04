"""Frozen native RGB propagation audit, without any olfactory stimulation.

One full graph is loaded. Separate immutable arrays retain actual neural
counts. No game is scored and no conditioning or action targets are supplied.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import pyarrow.feather as feather
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.visual_apl import VisualAPLBrain, VERSION
from flysnake.research.visual_stimuli import fixed_geometry, audit_frames
from flysnake.research.optical_sampling import sample_aperture_linear
from flysnake.research.apl_local import file_hash
from flysnake.research.paired import digest, writer_lock
from flysnake.research.recurrence_protocol import vector_comparison, cue_windows
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import immutable_arrays


def new_sources(parent):
    result = dict(parent['sources'])
    for relative in ('scripts/assay_visual_apl.py', 'flysnake/research/visual_apl.py',
        'flysnake/research/visual_stimuli.py', 'flysnake/research/apl_local_state_v2.py',
        'flysnake/research/optical_sampling.py', 'flysnake/research/viewport_candidates.py',
        'flysnake/research/recurrence_protocol.py', 'flysnake/lab/environment.py',
        'flysnake/game.py', 'vendor/stonkfly/neural/transmitters.py'):
        result[relative] = file_hash(ROOT / relative)
    return result


def summaries(counts, groups):
    return {name: {'spikes': int(counts[ix].sum()),
                   'active_cells': int(np.count_nonzero(counts[ix]))}
            for name, ix in groups.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output / 'status.json'
        if path.exists():
            raise ValueError('Preserve previous visual evidence; use a new output')
        parent = json.loads((args.bundle / 'status.json').read_text(encoding='utf-8'))
        if parent['status'] != 'complete':
            raise ValueError('A completed immutable local APL bundle is required')
        for relative, sha in parent['sources'].items():
            if file_hash(ROOT / relative) != sha:
                raise ValueError('Locked source changed: ' + relative)
        source = new_sources(parent)
        settings = [
            ('neutral-gray', 'neutral_gray', True, True),
            ('empty-board', 'board_empty', True, True),
            ('board-no-food', 'board_no_food', True, True),
            ('food-A', 'food_A', True, True),
            ('food-B', 'food_B', True, True),
            ('head-rotated', 'head_rotated', True, True),
            ('head-as-body', 'head_as_body', True, True),
            ('pose-moved-food-fixed', 'pose_moved_food_fixed', True, True),
            ('food-A-retinal-off', 'food_A', False, False),
            ('food-B-retinal-off', 'food_B', False, False),
            ('food-A-remove-visual', 'food_A', True, False),
            ('food-A-repeat', 'food_A', True, True),
        ]
        report = {'protocol': VERSION, 'status': 'loading', 'sources': source,
            'process_id': os.getpid(), 'started_at': time.time(), 'rows': [],
            'parent_status_sha256': file_hash(args.bundle / 'status.json'),
            'settings': {'conditions': settings, 'pre_ms': 300, 'cue_ms': 300,
                'cue_bin_ms': 10, 'wash_ms': 1000, 'wash_bin_ms': 100,
                'pre_and_wash_image': 'uniform RGB85', 'learning': False,
                'olfactory_drive': False, 'game_played': False,
                'external_action_policy': False},
            'behavioral_learning_proven': False}
        for relative in source:
            target = args.output / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        atomic_json(path, report)
        try:
            brain = VisualAPLBrain(args.bundle, attachment=args.bundle / 'geometry')
            b = brain.b
            geometry = fixed_geometry(b, brain.aperture)
            frames = audit_frames(geometry)
            annotation = feather.read_table(ROOT / 'data/malecns_v1/annotations.feather',
                columns=['bodyId', 'type', 'class']).to_pandas().set_index('bodyId').loc[b.ids]
            types = annotation.type.fillna('').to_numpy(str)
            classes = annotation['class'].fillna('').to_numpy(str)
            groups = {'R1_R6': b.retina, 'R8': b.r8,
                'lamina': b.lamina,
                'visual_annotated': np.flatnonzero(classes == 'visual'),
                'ORN': np.flatnonzero(np.char.startswith(types, 'ORN_')),
                'ALPN': np.flatnonzero(classes == 'ALPN'),
                'all_KC': brain.memory_readouts['KC'], 'gamma1_KC': np.unique(brain.pre),
                'MBON11': brain.memory_readouts['gamma1_MBON11'],
                'PPL101': brain.memory_readouts['PPL101'],
                'motor': np.r_[brain.groups['left'], brain.groups['forward'], brain.groups['right']]}
            del annotation, types, classes
            if any(not len(ix) for ix in groups.values()):
                raise ValueError('A declared anatomical diagnostic group is empty')
            report['groups'] = {name: {'cells': len(ix), 'indices_sha256': digest(ix)}
                                for name, ix in groups.items()}
            report.update(brain=brain.identity.copy(), geometry=geometry)
            allowed = np.flatnonzero(brain.external_current_mask)
            immutable_arrays(args.output / 'anatomy.npz', cell_ids=b.ids,
                external_drive_allowed_indices=allowed, original_retinal_uv=b.uv,
                original_r8_uv=b.r8_uv, **{name + '_indices': ix for name, ix in groups.items()})
            report['anatomy_sha256'] = file_hash(args.output / 'anatomy.npz')
            optical = {}
            for name, rgb in frames.items():
                Image.fromarray(rgb).save(args.output / (name + '.png'))
                optical[name] = np.r_[sample_aperture_linear(rgb, b.uv, brain.aperture).ravel(),
                    sample_aperture_linear(rgb, b.r8_uv, brain.aperture).ravel()]
            comparisons = [('food_A', 'food_B'), ('food_A', 'board_no_food'),
                           ('food_A', 'head_rotated'), ('food_A', 'head_as_body')]
            report['static_optical_comparisons'] = {a + '_vs_' + c: {
                'changed_scalar_samples': int(np.count_nonzero(optical[a] != optical[c])),
                'maximum_radiance_difference': float(np.abs(optical[a] - optical[c]).max())}
                for a, c in comparisons}
            # Retain only anatomical readout columns between conditions, not
            # seven additional full-brain matrices on this memory-limited host.
            selected = np.unique(np.concatenate(list(groups.values())))
            compact_groups = {name: np.searchsorted(selected, ix) for name, ix in groups.items()}
            cue_records = {}; cue_hashes = {}; pre_history = None
            for index, (label, frame, visual, wash_visual) in enumerate(settings):
                brain.set_visual_input(visual)
                brain.reset_trial(frames['neutral_gray'])
                actual = []; drives = []; rows = []
                checkpoint = None
                for phase, steps, duration, rgb in (
                    ('pre', 3, 100, frames['neutral_gray']),
                    ('cue', 30, 10, frames[frame]),
                    ('wash', 10, 100, frames['neutral_gray'])):
                    if phase == 'wash':
                        brain.set_visual_input(wash_visual)
                        if label == 'food-A':
                            checkpoint = {'file': 'visual-state.npz',
                                'sha256': brain.save_state(args.output / 'visual-state.npz')}
                    for step in range(steps):
                        receipt = brain.integrate(rgb, duration)
                        if np.any(b.drive[groups['ORN']]):
                            raise AssertionError('An ORN received external current in a visual condition')
                        actual.append(brain.last_counts.copy()); drives.append(b.drive[allowed].copy())
                        rows.append({'phase': phase, 'bin': step, 'duration_ms': duration,
                                     'native_count_array_index': len(actual)-1, 'receipt': receipt})
                        report.update(status='running', phase=label + '/' + phase + '/' + str(step+1),
                                      active_rows=rows, updated_at=time.time())
                        atomic_json(path, report)
                counts = np.array(actual); del actual
                pre = counts[:3]; cue = counts[3:33]; tail = counts[-5:].sum(0)
                history_equal = digest(pre) == pre_history if pre_history is not None and visual else None
                if visual:
                    if pre_history is None: pre_history = digest(pre)
                    elif not history_equal: raise AssertionError('Matched visual conditions have different prehistory')
                result = {'label': label, 'frame': frame, 'retinal_input': visual,
                    'wash_retinal_input': wash_visual, 'rows': rows,
                    'prehistory_equal_to_first_visual_condition': history_equal,
                    'pre_counts_sha256': digest(pre), 'cue_summary': summaries(cue.sum(0), groups),
                    'wash_last500ms_summary': summaries(tail, groups),
                    'cue_motor_counts': {k: cue[:, ix].sum(0).tolist() for k, ix in brain.groups.items()},
                    'full_cue_spikes': int(cue.sum()), 'full_tail_spikes': int(tail.sum()),
                    'all_weights_unchanged': digest(b.weight) == brain.initial_all_weights_sha256}
                if not result['all_weights_unchanged']:
                    raise AssertionError('Visual propagation audit changed graph weights')
                filename = f'condition-{index:02d}.npz'
                result['native_arrays'] = {'file': filename, 'sha256': immutable_arrays(args.output / filename,
                    full_counts=counts, allowed_external_drive=np.array(drives), final_external_drive=b.drive,
                    actual_rgb=frames[frame])}
                if checkpoint is not None:
                    result['checkpoint'] = checkpoint
                    report['replay_expected'] = {'file': 'replay-expected.npz',
                        'sha256': immutable_arrays(args.output / 'replay-expected.npz',
                            full_counts=counts[33:36], allowed_external_drive=np.array(drives)[33:36],
                            rgb=frames['neutral_gray'])}
                if label in ('food-A', 'food-B', 'head-rotated', 'head-as-body',
                             'food-A-retinal-off', 'food-B-retinal-off', 'food-A-repeat'):
                    cue_records[label] = cue[:, selected].copy()
                    cue_hashes[label] = digest(cue)
                report['rows'].append(result); report.pop('active_rows', None)
                atomic_json(path, report)
                print(json.dumps({'condition': label, 'cue_motor': result['cue_motor_counts'],
                    'cue_KC': result['cue_summary']['all_KC']['spikes'],
                    'tail_KC': result['wash_last500ms_summary']['all_KC']['spikes'],
                    'tail_full': result['full_tail_spikes']}), flush=True)
                del counts, drives
            report['native_coding'] = {name: {group: cue_windows(cue_records[a], cue_records[c], ix)
                for group, ix in compact_groups.items()}
                for name, a, c in (
                    ('food_position', 'food-A', 'food-B'),
                    ('head_orientation', 'food-A', 'head-rotated'),
                    ('head_vs_body', 'food-A', 'head-as-body'))}
            for name, a, c in (('food_position', 'food-A', 'food-B'),
                              ('head_orientation', 'food-A', 'head-rotated'),
                              ('head_vs_body', 'food-A', 'head-as-body')):
                report['native_coding'][name]['changed_cells_in_300ms'] = {
                    group: int(np.count_nonzero(cue_records[a][:, ix].sum(0) != cue_records[c][:, ix].sum(0)))
                    for group, ix in compact_groups.items()}
            report['gates'] = {
                'only_declared_visual_and_fixed_motor_external_currents': True,
                'ORN_external_current_zero': True,
                'all_graph_weights_frozen': all(r['all_weights_unchanged'] for r in report['rows']),
                'retinal_off_food_positions_have_identical_full_neural_counts':
                    cue_hashes['food-A-retinal-off'] == cue_hashes['food-B-retinal-off'],
                'same_RGB_repeated_has_identical_full_neural_counts':
                    cue_hashes['food-A'] == cue_hashes['food-A-repeat'],
                'food_pixels_change_receptor_counts': any(
                    report['native_coding']['food_position']['changed_cells_in_300ms'][g] > 0
                    for g in ('R1_R6', 'R8')),
                'food_pixels_change_motor_counts':
                    report['native_coding']['food_position']['changed_cells_in_300ms']['motor'] > 0,
                'behavioral_learning_proven': False,
            }
            report.update(status='complete', completed_at=time.time(),
                interpretation='RGB-only native propagation and matched retinal-off controls. '
                    'Neural distinctions alone do not demonstrate spatial understanding, correct actions or learning. '
                    'Gamma1 readout relevance and biological visual dynamics remain unvalidated.')
            atomic_json(path, report)
            print(json.dumps({'status': 'complete', 'gates': report['gates']}), flush=True)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__ == '__main__':
    main()
