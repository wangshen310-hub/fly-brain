"""Fresh-OS-process native replay of the read-only visual relay recorder."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.visual_apl import VisualAPLBrain
from flysnake.research.visual_stimuli import audit_frames, fixed_geometry
from flysnake.research.visual_relay_trace import build_trace, RelayRecorder, dynamic_hashes
from flysnake.research.apl_local import file_hash
from flysnake.lab.experiment import atomic_json
from audit_visual_relay import warm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous verification')
    status = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise ValueError('Completed frozen relay audit required')
    if os.getpid() == status['process_id']:
        raise ValueError('An independent OS process is required')
    for name, sha in status['sources'].items():
        if file_hash(ROOT / name) != sha or file_hash(args.run / 'sources' / name) != sha:
            raise ValueError('Current/snapshotted source differs: ' + name)
    row = next(row for row in status['rows'] if row['label'] == 'food-A')
    for item in row['bins']:
        if file_hash(args.run / item['arrays']['file']) != item['arrays']['sha256']:
            raise ValueError('Saved native bin changed')
    library, build = build_trace(args.run / 'build')
    brain = VisualAPLBrain(args.bundle, attachment=args.bundle / 'geometry')
    frames = audit_frames(fixed_geometry(brain.b, brain.aperture))
    recorder = RelayRecorder(brain, library)
    recorder.install()
    warm(brain, frames['neutral_gray'], True)
    checks = []
    for step, item in enumerate(row['bins']):
        recorder.begin_bin()
        brain.integrate(frames['food_A'], 10)
        values = {'full_counts': brain.last_counts, 'voltage_mV': brain.b.v,
            'synaptic_g': brain.b.g, 'native_evaluation_min_mV': recorder.minimum,
            'native_evaluation_max_mV': recorder.maximum, 'fast_event_budgets': recorder.events,
            'external_drive': brain.b.drive, 'refractory': brain.b.refractory,
            'last_evaluation_tick': brain.b.last}
        with np.load(args.run / item['arrays']['file'], allow_pickle=False) as z:
            matches = {name: np.array_equal(value, z[name]) for name, value in values.items()}
        complete_state_equal = dynamic_hashes(brain) == status['passthrough_reference'][step]
        if not all(matches.values()) or not complete_state_equal:
            raise AssertionError('Fresh native replay differs at bin ' + str(step))
        checks.append({'bin': step, 'actual_native_arrays_equal': matches,
                       'all_native_fields_equal_original_kernel': complete_state_equal})
    recorder.disable()
    report = {'status': 'complete', 'process_id': os.getpid(),
        'experiment_process_id': status['process_id'], 'completed_at': time.time(),
        'verification_source_sha256': file_hash(__file__),
        'run_status_sha256': file_hash(args.run / 'status.json'),
        'sources_and_snapshots_equal': True, 'binary_sha256': build['binary_sha256'],
        'replayed_cue_ms': 300, 'warm_and_pre_ms': 400, 'checks': checks,
        'scope': 'Fresh process food-A replay, with all complete native state fields compared to '
            'the original kernel at 30 boundaries, and every saved recorder array compared exactly. '
            'No learning or old game-memory migration tested.', 'behavioral_learning_proven': False}
    atomic_json(args.output, report)
    print(json.dumps({'status': 'complete', 'replayed_ms': 300, 'exact_bins': len(checks)}))


if __name__ == '__main__':
    main()
