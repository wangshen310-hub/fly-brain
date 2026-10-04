"""Frozen complete-graph RGB assay with actual delivered-edge and voltage logs.

No learning, coordinates or action labels enter the brain. A within-process
passthrough replay checks every native state field before the recorder is used.
Previous count-only visual evidence and all original sources are preserved.
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.visual_apl import VisualAPLBrain
from flysnake.research.visual_stimuli import fixed_geometry, audit_frames
from flysnake.research.visual_relay_trace import (
    VERSION, EVENT_COLUMNS, build_trace, RelayRecorder, dynamic_hashes)
from flysnake.research.apl_local import file_hash
from flysnake.research.paired import digest, writer_lock
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import immutable_arrays


def anatomy(brain):
    b = brain.b
    a = feather.read_table(ROOT / 'data/malecns_v1/annotations.feather',
        columns=['bodyId', 'type', 'class']).to_pandas().set_index('bodyId').loc[b.ids]
    types = a.type.fillna('').to_numpy(str)
    classes = a['class'].fillna('').to_numpy(str)
    sensory = np.zeros(b.n, bool)
    sensory[np.r_[b.retina, b.r8]] = True
    groups = {'R1_R6': b.retina, 'R8': b.r8, 'lamina': b.lamina,
        'visual_annotated': np.flatnonzero(classes == 'visual'),
        'nonreceptors': np.flatnonzero(~sensory),
        'all_KC': brain.memory_readouts['KC'], 'gamma1_KC': np.unique(brain.pre),
        'MBON11': brain.memory_readouts['gamma1_MBON11'],
        'PPL101': brain.memory_readouts['PPL101'],
        'motor': np.r_[brain.groups['left'], brain.groups['forward'], brain.groups['right']]}
    for cell_type in ('L1', 'L2', 'L3', 'L5', 'aMe12', 'LoVC23', 'aMe22', 'aMe24'):
        groups[cell_type] = np.flatnonzero(types == cell_type)
    # Anatomy only: original outgoing edges of the actual mapped receptors.
    graph_rows = []
    for name, sources in (('R1_R6', b.retina), ('R8', b.r8)):
        buckets = {}
        for i in sources:
            edges = np.arange(b.ptr[i], b.ptr[i+1])
            for target_type in np.unique(types[b.post[edges]]):
                selected = edges[types[b.post[edges]] == target_type]
                w = b.weight[selected]
                row = buckets.setdefault(str(target_type), {'pairs': 0, 'positive_pairs': 0,
                    'negative_pairs': 0, 'positive_weight': 0., 'negative_weight_magnitude': 0.})
                row['pairs'] += len(selected)
                row['positive_pairs'] += int((w > 0).sum())
                row['negative_pairs'] += int((w < 0).sum())
                row['positive_weight'] += float(w[w > 0].sum(dtype=np.float64))
                row['negative_weight_magnitude'] -= float(w[w < 0].sum(dtype=np.float64))
        graph_rows.append({'source_group': name, 'cells': len(sources),
            'target_types': {k: buckets[k] for k in sorted(buckets)},
            'scope': 'Original retained pair topology, current frozen model signs; not receptor physiology'})
    return types, groups, graph_rows


def warm(brain, gray, enabled):
    brain.set_visual_input(enabled)
    brain.reset_trial(gray)
    pre = []
    for _ in range(3):
        brain.integrate(gray, 100)
        pre.append(digest(brain.last_counts))
    return pre


def group_summary(brain, recorder, ix):
    b = brain.b
    return {'spikes': int(brain.last_counts[ix].sum()),
        'active_spiking_cells': int(np.count_nonzero(brain.last_counts[ix])),
        'V_end_mV_quantiles': np.quantile(b.v[ix], [0, .5, .9, 1]).tolist(),
        'V_native_evaluation_min_mV': float(recorder.minimum[ix].min()),
        'V_native_evaluation_max_mV': float(recorder.maximum[ix].max()),
        'external_drive_nonzero': int(np.count_nonzero(b.drive[ix])),
        'fast_edge_delivery_budgets': {name: float(recorder.events[c, ix].sum())
                                     for c, name in enumerate(EVENT_COLUMNS)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output / 'status.json'
        if path.exists():
            raise ValueError('Preserve previous relay evidence; use a new output')
        old = json.loads((args.reference / 'status.json').read_text(encoding='utf-8'))
        if old['status'] != 'complete':
            raise ValueError('Completed frozen RGB evidence required')
        sources = dict(old['sources'])
        for name in ('scripts/audit_visual_relay.py', 'flysnake/research/visual_relay_trace.py',
                     'flysnake/research/kernels/visual_relay_trace_hooks.hpp'):
            sources[name] = file_hash(ROOT / name)
        for name, sha in sources.items():
            if file_hash(ROOT / name) != sha:
                raise ValueError('Locked source changed: ' + name)
            target = args.output / 'sources' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write((ROOT / name).read_bytes())
        settings = [('food-A', 'food_A', True), ('food-B', 'food_B', True),
            ('head-rotated', 'head_rotated', True), ('head-as-body', 'head_as_body', True),
            ('food-A-retinal-off', 'food_A', False), ('food-B-retinal-off', 'food_B', False),
            ('food-A-repeat', 'food_A', True)]
        report = {'protocol': VERSION, 'status': 'building', 'process_id': os.getpid(),
            'started_at': time.time(), 'sources': sources, 'rows': [],
            'reference_status_sha256': file_hash(args.reference / 'status.json'),
            'settings': {'conditions': settings, 'warm_ms': 100, 'pre_ms': 300,
                'cue_ms': 300, 'bin_ms': 10, 'learning': False, 'olfactory_input': False,
                'game_played': False, 'action_policy': False,
                'voltage_sampling': 'every native evolve evaluation plus 10ms boundaries',
                'event_budgets': 'actual delayed fast-edge deliveries accepted/rejected by refractory state; '
                    'not spike-count times weight, not charge or conductance',
                'event_columns': EVENT_COLUMNS},
            'behavioral_learning_proven': False}
        atomic_json(path, report)
        try:
            library, build = build_trace(args.output / 'build')
            report.update(status='loading', build=build)
            atomic_json(path, report)
            brain = VisualAPLBrain(args.bundle, attachment=args.bundle / 'geometry')
            b = brain.b
            frames = audit_frames(fixed_geometry(b, brain.aperture))
            types, groups, graph_rows = anatomy(brain)
            report.update(brain=brain.identity.copy(), anatomical_edge_audit=graph_rows,
                          groups={name: len(ix) for name, ix in groups.items()})
            immutable_arrays(args.output / 'anatomy.npz', cell_ids=b.ids, cell_types=types,
                cell_rest_mV=b.rest, external_allowed=brain.external_current_mask,
                **{name + '_indices': ix for name, ix in groups.items()})
            report['anatomy_sha256'] = file_hash(args.output / 'anatomy.npz')
            # One model; replay the exact original dynamics after swapping only
            # the observation function. No second full brain is needed in RAM.
            report.update(status='running', phase='original-kernel-passthrough-reference')
            atomic_json(path, report)
            pre_expected = warm(brain, frames['neutral_gray'], True)
            expected = []
            expected_counts = []
            for step in range(30):
                brain.integrate(frames['food_A'], 10)
                expected_counts.append(brain.last_counts.copy())
                expected.append(dynamic_hashes(brain))
            with np.load(args.reference / 'condition-03.npz', allow_pickle=False) as z:
                matches_previous = np.array_equal(np.asarray(expected_counts), z['full_counts'][3:33])
            if not matches_previous:
                raise AssertionError('The unmodified native brain differs from its locked visual reference')
            del expected_counts
            recorder = RelayRecorder(brain, library)
            recorder.install()
            report['passthrough_reference'] = expected
            report['original_counts_match_previous_frozen_assay'] = matches_previous
            pre_on = None
            condition_hashes = {}
            for condition_index, (label, frame, visual) in enumerate(settings):
                recorder.disable()
                pre = warm(brain, frames['neutral_gray'], visual)
                if visual:
                    if pre_on is None:
                        pre_on = pre
                    if pre != pre_on or pre != pre_expected:
                        raise AssertionError('Matched visual conditions have unequal neural prehistory')
                bins = []
                series = []
                for step in range(30):
                    recorder.begin_bin()
                    receipt = brain.integrate(frames[frame], 10)
                    if condition_index == 0:
                        state = dynamic_hashes(brain)
                        if state != expected[step]:
                            raise AssertionError('Recorder altered complete native state at bin ' + str(step))
                    filename = f'condition-{condition_index:02d}-bin-{step:02d}.npz'
                    sha = immutable_arrays(args.output / filename,
                        full_counts=brain.last_counts, voltage_mV=b.v, synaptic_g=b.g,
                        native_evaluation_min_mV=recorder.minimum,
                        native_evaluation_max_mV=recorder.maximum,
                        fast_event_budgets=recorder.events, external_drive=b.drive,
                        refractory=b.refractory, last_evaluation_tick=b.last)
                    summaries = {name: group_summary(brain, recorder, ix) for name, ix in groups.items()}
                    bins.append({'bin': step, 'cursor': int(b.cursor), 'receipt': receipt,
                        'arrays': {'file': filename, 'sha256': sha}, 'summary': summaries})
                    series.append({'v': digest(b.v), 'g': digest(b.g),
                        'counts': digest(brain.last_counts), 'events': digest(recorder.events),
                        'min_v': digest(recorder.minimum), 'max_v': digest(recorder.maximum)})
                    report.update(phase=label + '/' + str(step+1), active_bins=bins,
                                  updated_at=time.time())
                    atomic_json(path, report)
                recorder.disable()
                if digest(b.weight) != brain.initial_all_weights_sha256:
                    raise AssertionError('Frozen relay audit changed a weight')
                row = {'label': label, 'frame': frame, 'retinal_input': visual,
                    'RGB_sha256': digest(frames[frame]), 'bins': bins,
                    'all_weights_unchanged': True}
                report['rows'].append(row)
                report.pop('active_bins', None)
                condition_hashes[label] = series
                print(json.dumps({'condition': label,
                    'lamina_V_min': min(r['summary']['lamina']['V_native_evaluation_min_mV'] for r in bins),
                    'lamina_V_max': max(r['summary']['lamina']['V_native_evaluation_max_mV'] for r in bins),
                    'KC_spikes': sum(r['summary']['all_KC']['spikes'] for r in bins)}), flush=True)
                atomic_json(path, report)
            report['gates'] = {
                'original_counts_match_previous_frozen_assay': matches_previous,
                'instrumented_complete_native_state_equals_original_at_all_30_boundaries': True,
                'retinal_off_images_have_identical_full_voltage_counts_and_delivery_budgets':
                    condition_hashes['food-A-retinal-off'] == condition_hashes['food-B-retinal-off'],
                'repeated_image_has_identical_full_voltage_counts_and_delivery_budgets':
                    condition_hashes['food-A'] == condition_hashes['food-A-repeat'],
                'all_weights_frozen': True, 'behavioral_learning_proven': False}
            report.update(status='complete', completed_at=time.time(),
                interpretation='Observation-only full graph diagnosis; candidate repair is not implemented by this audit.')
            atomic_json(path, report)
        except Exception:
            report.update(status='error', error=traceback.format_exc())
            atomic_json(path, report)
            raise


if __name__ == '__main__':
    main()
