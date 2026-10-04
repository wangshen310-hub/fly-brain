"""Read saved native 10ms bins; no classifier, fitting or action selection."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    d = json.loads((args.run / 'status.json').read_text(encoding='utf-8'))
    item = d['native_spikes']; source = args.run / item['file']
    if d['status'] != 'complete' or hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('Completed immutable actual count arrays required')
    with np.load(source, allow_pickle=False) as z:
        selected = z['selected_indices']; pre = z['gamma1_KC_indices']
        counts = z['selected_counts'][:, :, np.searchsorted(selected, pre)]
    pairs = {}
    for i, row in enumerate(d['rows']): pairs.setdefault((row['mode'], row['seed']), {})[row['cue']] = i
    rows = []
    windows = [('cue_0_50ms', 0, 5), ('cue_0_100ms', 0, 10),
               ('cue_100_300ms', 10, 30), ('cue_0_300ms', 0, 30),
               ('wash_100_300ms', 40, 60)]
    for (mode, seed), indices in pairs.items():
        values = []
        for label, start, end in windows:
            x = counts[indices['DM1'], start:end].sum(axis=0).astype(float)
            y = counts[indices['DM4'], start:end].sum(axis=0).astype(float)
            norm = np.linalg.norm(x) * np.linalg.norm(y)
            union = np.count_nonzero((x > 0) | (y > 0)); shared = np.count_nonzero((x > 0) & (y > 0))
            values.append({'window': label, 'active_gamma1_KC': [int(np.count_nonzero(x)), int(np.count_nonzero(y))],
                           'count_vector_cosine_similarity': float(np.dot(x, y)/norm) if norm else None,
                           'active_set_jaccard': shared/union if union else None})
        rows.append({'mode': mode, 'seed': seed, 'windows': values})
    result = {'generated_at': time.time(), 'source_count_arrays_sha256': item['sha256'], 'rows': rows,
              'scope': 'Native count vectors in explicit windows. Cosine is not a percentage of aliased cells '
                       'or proof that exact spike timing cannot distinguish cues. No learned classifier or game output.'}
    if args.output.exists(): raise ValueError('Preserve previous timecourse analysis')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'conditions': len(rows)}))


if __name__ == '__main__': main()
