"""Fresh-process nonzero graded relay checkpoint continuation verification."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.graded_lamina import GradedLaminaBrain
from flysnake.research.visual_stimuli import fixed_geometry, audit_frames
from flysnake.research.visual_relay_trace import dynamic_hashes
from flysnake.research.apl_local import file_hash
from flysnake.lab.experiment import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing verification')
    status = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete' or status['process_id'] == os.getpid():
        raise ValueError('Complete experiment and a new actual process required')
    for name, sha in status['sources'].items():
        if file_hash(ROOT / name) != sha or file_hash(args.run / 'sources' / name) != sha:
            raise ValueError('Current or snapshotted source differs')
    row = next(r for r in status['rows'] if r['label'] == 'food-A')
    checkpoint = row['checkpoint']
    brain = GradedLaminaBrain(args.bundle, attachment=args.bundle / 'geometry',
        build_directory=args.run / 'build', enabled=True)
    frames = audit_frames(fixed_geometry(brain.b, brain.aperture))
    brain.load_state(args.run / checkpoint['file'], checkpoint['sha256'])
    nonzero_before = {'pending_positive': int((brain.graded_positive_queue != 0).sum()),
        'pending_negative': int((brain.graded_negative_queue != 0).sum()),
        'release_rates': int((brain.graded_rates != 0).sum())}
    if not nonzero_before['release_rates'] or not sum(nonzero_before.values()):
        raise AssertionError('Checkpoint does not exercise nonzero graded state')
    checks = []
    for step, expected in enumerate(status['checkpoint_expected_continuation']):
        brain.integrate(frames['neutral_gray'], 10)
        if dynamic_hashes(brain) != expected:
            raise AssertionError('Restored full native/graded continuation differs at bin ' + str(step))
        checks.append(True)
    report = {'status': 'complete', 'source_sha256': file_hash(__file__),
        'process_id': os.getpid(), 'experiment_process_id': status['process_id'],
        'run_status_sha256': file_hash(args.run / 'status.json'), 'completed_at': time.time(),
        'checkpoint_sha256': checkpoint['sha256'], 'nonzero_saved_graded_state': nonzero_before,
        'replayed_ms': 100, 'complete_native_and_graded_states_equal_at_10_boundaries': checks,
        'behavioral_learning_proven': False,
        'scope': 'Exact fresh-process 100ms continuation of a frozen nonzero graded relay; '
            'neither game training nor old stored memories are migrated.'}
    atomic_json(args.output, report)
    print(json.dumps({'status': 'complete', 'replayed_ms': 100, 'exact_bins': len(checks)}))


if __name__ == '__main__':
    main()
