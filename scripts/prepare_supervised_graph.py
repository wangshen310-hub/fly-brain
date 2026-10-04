"""Export an independently identified complete graph; never alter legacy data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'data/malecns_v1'
    prepared = json.loads((source / 'prepared.json').read_text(encoding='utf-8'))
    if not prepared['verified_arrays'] or sha(source / 'graph.npz') != prepared['graph_sha256']:
        raise ValueError('Verified source graph required')
    print('Loading pinned graph and existing optical projection', flush=True)
    os.environ['FLYSNAKE_DATA'] = str(source)
    sys.path.insert(0, str(ROOT / 'vendor'))
    from stonkfly.neural.visual import VisualMemoryBrain
    from stonkfly.neural.common import annotations
    b = VisualMemoryBrain()
    a = annotations(b.ids)
    if (b.n, len(b.post), len(b.retina), len(b.r8)) != (166700, 25582938, 3335, 811):
        raise ValueError('Unexpected retained full graph')
    pre = np.repeat(np.arange(b.n, dtype=np.int32), np.diff(b.ptr))
    post = b.post
    # Existing fast signs include the already declared R8->aMe12 proxy.
    # Modulatory edges with zero fast efficacy stay in the anatomical arrays.
    signed = b.weight.copy()
    original_sign = np.sign(signed).astype(np.int8)
    # In the original event kernel pure modulators use a separate channel.
    # Their existing anatomical edges must not silently become fast excitation.
    signed[b.modulation_mask[pre].astype(bool)] = 0.
    total = np.bincount(post, weights=np.abs(signed), minlength=b.n).astype(np.float32)
    norm = np.abs(signed) / np.maximum(total[post], 1e-12)
    arrays = {'ids': b.ids, 'pre': pre, 'post': post.copy(),
        'signed_base': np.sign(signed) * norm,
        'original_sign': original_sign, 'modulation_mask': b.modulation_mask.copy(),
        'rest': ((b.rest + 60.) / 20.).astype(np.float32),
        'retina': b.retina, 'uv': b.uv, 'r8': b.r8, 'r8_uv': b.r8_uv,
        'r8_channel': b.r8_channel,
        'KC': np.flatnonzero(a.type.fillna('').str.startswith('KC')).astype(np.int32),
        'graded_lamina': np.flatnonzero(a.type.fillna('').isin(['L1', 'L2', 'L3', 'L4', 'L5'])).astype(np.int32)}
    for label, typ, side in [('left', 'DNa02', 'L'), ('right', 'DNa02', 'R'),
                              ('forward_left', 'DNp20', 'L'), ('forward_right', 'DNp20', 'R')]:
        ix = np.flatnonzero(a.type.eq(typ) & a.somaSide.eq(side)).astype(np.int32)
        if len(ix) != 1:
            raise ValueError('Unresolved fixed motor identity: ' + label)
        arrays[label] = ix
    if {label: b.ids[arrays[label]].tolist() for label in ['left', 'right']} != {
            'left': [523769], 'right': [10360]}:
        raise ValueError('Fixed DNa02 identity drift')
    print('Constructing sparse delivery and transpose arrays', flush=True)
    for label, select in [('exc', signed > 0), ('inh', signed < 0)]:
        indices = np.flatnonzero(select)
        order = indices[np.argsort(post[indices], kind='stable')]
        counts = np.bincount(post[order], minlength=b.n)
        arrays[label + '_crow'] = np.r_[0, np.cumsum(counts)].astype(np.int32)
        arrays[label + '_col'] = pre[order]
        arrays[label + '_value'] = norm[order]
        # Source graph was already grouped by pre. No dense NxN allocation.
        counts_t = np.bincount(pre[indices], minlength=b.n)
        arrays[label + '_tcrow'] = np.r_[0, np.cumsum(counts_t)].astype(np.int32)
        arrays[label + '_tcol'] = post[indices]
        arrays[label + '_tvalue'] = norm[indices]
    graph = out / 'graph.npz'
    with graph.open('xb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    metadata = {'version': 'complete-signed-MaleCNS-supervised-export-v1',
        'neurons': b.n, 'anatomical_edges': len(post),
        'fast_exc_edges': int((signed > 0).sum()), 'fast_inh_edges': int((signed < 0).sum()),
        'nonfast_edges_retained': int((signed == 0).sum()),
        'source_graph_sha256': prepared['graph_sha256'],
        'source_fast_signed_weight_sha256': digest(signed),
        'source_prepared_sha256': sha(source / 'prepared.json'),
        'array_sha256': {k: digest(v) for k, v in arrays.items()},
        'graph_sha256': sha(graph), 'export_source_sha256': sha(__file__),
        'motor_ids': {k: b.ids[arrays[k]].tolist() for k in
                      ['left', 'right', 'forward_left', 'forward_right']},
        'normalization': 'absolute incoming fast budget, preserving each existing sign and endpoint',
        'assumptions': 'Sparse engineering conductance model; pure modulatory edges have no fast event. '
            'No physiological fit or successful game learning claimed.'}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ['neurons', 'anatomical_edges',
        'fast_exc_edges', 'fast_inh_edges', 'nonfast_edges_retained', 'graph_sha256', 'motor_ids']}), flush=True)


if __name__ == '__main__':
    main()
