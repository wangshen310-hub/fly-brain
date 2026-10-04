"""Read frozen actual RGB evidence; never construct or control a neural model."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pyarrow.feather as feather
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flysnake.research.apl_local import file_hash
from flysnake.research.paired import digest
from flysnake.lab.experiment import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier visual pathway analysis')
    status_path = args.run / 'status.json'
    r = json.loads(status_path.read_text(encoding='utf-8'))
    if r['status'] != 'complete':
        raise ValueError('Completed native visual evidence required')
    with np.load(args.run / 'anatomy.npz', allow_pickle=False) as z:
        ids = z['cell_ids'].copy()
        receptors = np.r_[z['R1_R6_indices'], z['R8_indices']]
    receptor_mask = np.zeros(len(ids), bool); receptor_mask[receptors] = True
    annotation = feather.read_table(ROOT / 'data/malecns_v1/annotations.feather',
        columns=['bodyId', 'type']).to_pandas().set_index('bodyId').loc[ids]
    types = annotation.type.fillna('unresolved').to_numpy(str); del annotation
    by_label = {row['label']: row for row in r['rows']}

    def counts(label):
        row = by_label[label]; record = row['native_arrays']; path = args.run / record['file']
        if file_hash(path) != record['sha256']:
            raise ValueError('Actual saved counts changed')
        with np.load(path, allow_pickle=False) as z:
            return z['full_counts'][3:33].copy()

    reference = counts('food-A')
    result = {'status': 'complete', 'analysis_source_sha256': file_hash(__file__),
        'native_status_sha256': file_hash(status_path), 'comparisons': {},
        'scope': 'Actual 10ms counts in the declared 300ms visual window after matched neutral adaptation. '
            'Subthreshold states and within-bin spike timing were not saved by this audit. '
            'This is not an exhaustive board or learning audit.'}
    for other in ('food-B', 'head-rotated', 'head-as-body', 'pose-moved-food-fixed',
                  'board-no-food', 'food-A-retinal-off'):
        candidate = counts(other); diff = reference - candidate
        total_diff = diff.sum(0)
        changed = np.flatnonzero(np.any(diff != 0, axis=0))
        downstream = changed[~receptor_mask[changed]]
        result['comparisons']['food-A_vs_' + other] = {
            'all_changed_cells_in_10ms_bins': len(changed),
            'non_receptor_changed_cells_in_10ms_bins': len(downstream),
            'non_receptor_changed_cells_in_300ms_totals': int(np.count_nonzero(total_diff[~receptor_mask])),
            'changed_non_receptor_ids_types': [[int(ids[i]), str(types[i])] for i in downstream],
            'full_binwise_counts_sha256': [digest(reference), digest(candidate)],
        }
        del candidate, diff
    active = np.flatnonzero(reference.sum(0) > 0)
    active = active[~receptor_mask[active]]
    result['food_A_active_non_receptor_cells'] = [
        {'body_id': int(ids[i]), 'type': str(types[i]), 'spikes': int(reference[:, i].sum())}
        for i in active]
    result['non_receptor_active_cells'] = len(active)
    result['interpretation'] = ('Locate task-dependent native count differences before changing any learning rule. '
        'Identical downstream counts do not imply that anatomical edges are absent or that subthreshold states are identical.')
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('food_A_active_non_receptor_cells',)}))


if __name__ == '__main__':
    main()
