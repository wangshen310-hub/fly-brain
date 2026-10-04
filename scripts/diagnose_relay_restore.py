"""Logged fresh-process continuation for independent graded relay protocols."""
from pathlib import Path
import argparse
import faulthandler
import json
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.apl_local import file_hash
from flysnake.research.visual_relay_trace import dynamic_hashes
from flysnake.research.visual_stimuli import fixed_geometry, audit_frames
from flysnake.lab.experiment import atomic_json


def main():
    faulthandler.enable(file=sys.stderr, all_threads=True)
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['graded', 'conductance'], required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new verification output')
    status = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete' or status['process_id'] == os.getpid():
        raise ValueError('Completed run and fresh OS process required')
    for name, sha in status['sources'].items():
        if file_hash(ROOT / name) != sha or file_hash(args.run / 'sources' / name) != sha:
            raise ValueError('Locked source differs: ' + name)
    report = {'status': 'loading', 'process_id': os.getpid(),
        'experiment_process_id': status['process_id'], 'kind': args.kind,
        'source_sha256': file_hash(__file__),
        'run_status_sha256': file_hash(args.run / 'status.json'), 'checks': []}
    atomic_json(args.output, report)
    if args.kind == 'graded':
        from flysnake.research.graded_lamina import GradedLaminaBrain as Brain
    else:
        from flysnake.research.conductance_lamina import ConductanceLaminaBrain as Brain
    print('Constructing independent complete brain', flush=True)
    brain = Brain(args.bundle, attachment=args.bundle / 'geometry',
                  build_directory=args.run / 'build', enabled=True)
    print('Complete brain constructed', flush=True)
    frames = audit_frames(fixed_geometry(brain.b, brain.aperture))
    row = next(r for r in status['rows'] if r['label'] == 'food-A')
    cp = row['checkpoint']
    report.update(status='loading-checkpoint', checkpoint_sha256=cp['sha256'])
    atomic_json(args.output, report)
    print('Loading checkpoint', flush=True)
    brain.load_state(args.run / cp['file'], cp['sha256'])
    print('Checkpoint loaded', flush=True)
    report.update(status='replaying', nonzero_graded_release_cells=int((brain.graded_rates > 0).sum()),
        nonzero_pending_positive_cells=int((brain.graded_positive_queue != 0).sum()),
        nonzero_pending_negative_cells=int((brain.graded_negative_queue != 0).sum()))
    atomic_json(args.output, report)
    for step, expected in enumerate(status['checkpoint_expected_continuation']):
        print('Native replay bin ' + str(step), flush=True)
        brain.integrate(frames['neutral_gray'], 10)
        equal = dynamic_hashes(brain) == expected
        if not equal:
            report.update(status='mismatch', failed_bin=step)
            atomic_json(args.output, report)
            raise AssertionError('Complete native/graded state differs')
        report['checks'].append(equal)
        atomic_json(args.output, report)
    report.update(status='complete', replayed_ms=len(report['checks']) * 10,
        completed_at=time.time(), behavioral_learning_proven=False,
        scope='Fresh process exact nonzero checkpoint continuation; no learning or old game-memory migration')
    atomic_json(args.output, report)
    print(json.dumps({'status': 'complete', 'replayed_ms': report['replayed_ms']}), flush=True)


if __name__ == '__main__':
    main()
