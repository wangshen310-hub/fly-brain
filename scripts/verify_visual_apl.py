"""Read actual saved arrays and replay RGB state in a distinct native process."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.visual_apl import VisualAPLBrain
from flysnake.research.apl_local import file_hash
from flysnake.research.raw_turn import decode_raw_turn
from flysnake.research.paired import digest
from flysnake.lab.experiment import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous independent RGB audit')
    status_path = args.run / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if status['status'] != 'complete' or status['process_id'] == os.getpid():
        raise ValueError('A completed assay in a distinct OS process is required')
    report = {'status': 'verifying', 'source_sha256': file_hash(__file__),
        'process_id': os.getpid(), 'assay_process_id': status['process_id'],
        'assay_status_sha256': file_hash(status_path), 'started_at': time.time()}
    atomic_json(args.output, report)
    try:
        for relative, sha in status['sources'].items():
            if file_hash(ROOT / relative) != sha or file_hash(args.run / 'sources' / relative) != sha:
                raise ValueError('RGB assay source differs: ' + relative)
        if file_hash(args.run / 'anatomy.npz') != status['anatomy_sha256']:
            raise ValueError('Saved anatomical indices changed')
        with np.load(args.run / 'anatomy.npz', allow_pickle=False) as z:
            anatomy = {k: z[k].copy() for k in z.files}
        groups = {k[:-8]: value for k, value in anatomy.items() if k.endswith('_indices')
                  and k != 'external_drive_allowed_indices'}
        allowed = anatomy['external_drive_allowed_indices']
        allowed_mask = np.zeros(len(anatomy['cell_ids']), bool); allowed_mask[allowed] = True
        motor_indices = {name: np.array([np.flatnonzero(anatomy['cell_ids'] == cell)[0]
            for cell in ids]) for name, ids in (
                ('left', [523769]), ('forward', [10162, 10059]), ('right', [10360]))}
        cue_digest = {}; verified = []
        for condition in status['rows']:
            record = condition['native_arrays']; path = args.run / record['file']
            if file_hash(path) != record['sha256']:
                raise ValueError('Actual native arrays changed')
            with np.load(path, allow_pickle=False) as z:
                counts = z['full_counts']; final_drive = z['final_external_drive']
                if np.any(final_drive[~allowed_mask]) or np.any(final_drive[groups['ORN']]):
                    raise AssertionError('Actual saved external drive contains a nonvisual current')
                rgb = np.asarray(Image.open(args.run / (condition['frame'] + '.png')))
                if not np.array_equal(rgb, z['actual_rgb']):
                    raise AssertionError('Saved visual image differs from actual input pixels')
                for row in condition['rows']:
                    actual = counts[row['native_count_array_index']]; receipt = row['receipt']
                    if digest(actual) != receipt['full_counts_sha256']:
                        raise AssertionError('Receipt is not from the saved complete native counts')
                    motor = {name: actual[ix].tolist() for name, ix in motor_indices.items()}
                    decoded = decode_raw_turn(motor, row['duration_ms'])
                    expected_rgb = rgb if row['phase'] == 'cue' else np.full((180, 320, 3), 85, np.uint8)
                    if digest(expected_rgb) != receipt['observation_sha256']:
                        raise AssertionError('Receipt is for different visual pixels')
                    if decoded['action'] != receipt['action'] or motor != receipt['counts']:
                        raise AssertionError('Action is not the declared native pulse readout')
                cue = counts[3:33].sum(0); tail = counts[-5:].sum(0)
                for phase, actual in (('cue_summary', cue), ('wash_last500ms_summary', tail)):
                    for name, ix in groups.items():
                        expected = condition[phase][name]
                        if (int(actual[ix].sum()) != expected['spikes'] or
                                int(np.count_nonzero(actual[ix])) != expected['active_cells']):
                            raise AssertionError('Summary differs from actual frozen native counts')
                cue_digest[condition['label']] = digest(counts[3:33])
                verified.append(condition['label'])
        if (cue_digest['food-A-retinal-off'] != cue_digest['food-B-retinal-off'] or
                cue_digest['food-A'] != cue_digest['food-A-repeat']):
            raise AssertionError('Full-count input controls failed')
        brain = VisualAPLBrain(args.bundle, attachment=args.bundle / 'geometry')
        checkpoint = next(r['checkpoint'] for r in status['rows'] if r['label'] == 'food-A')
        metadata = brain.load_state(args.run / checkpoint['file'], checkpoint['sha256'])
        record = status['replay_expected']
        if file_hash(args.run / record['file']) != record['sha256']:
            raise ValueError('Expected replay data changed')
        with np.load(args.run / record['file'], allow_pickle=False) as z:
            expected = z['full_counts']; drives = z['allowed_external_drive']; rgb = z['rgb']
            hashes = []
            for index in range(len(expected)):
                receipt = brain.integrate(rgb, 100)
                if not np.array_equal(brain.last_counts, expected[index]):
                    raise AssertionError('Fresh-process visual replay changed native spikes')
                if not np.array_equal(brain.b.drive[allowed], drives[index]):
                    raise AssertionError('Fresh-process visual replay changed external currents')
                hashes.append(receipt['full_counts_sha256'])
        if digest(brain.b.weight) != metadata['all_weights_sha256']:
            raise AssertionError('Frozen native replay changed weights')
        if file_hash(status_path) != report['assay_status_sha256']:
            raise ValueError('Assay was modified during verification')
        report.update(status='complete', completed_at=time.time(), verified_conditions=verified,
            sources_and_actual_array_hashes_match=True, receipts_match_full_native_counts=True,
            raw_motor_decoder_verified=True, sensory_off_control_verified=True,
            exact_repeat_control_verified=True, ORN_external_current_zero=True,
            fresh_process_replay={'duration_ms': 300, 'full_native_counts_equal': True,
                'external_currents_equal': True, 'native_count_sha256': hashes,
                'checkpoint_sha256': checkpoint['sha256']},
            scope='Frozen RGB pathway audit and 300ms complete native replay. No learning or game performance proof.')
        atomic_json(args.output, report)
        print(json.dumps(report), flush=True)
    except Exception:
        report.update(status='error', error=traceback.format_exc())
        atomic_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
