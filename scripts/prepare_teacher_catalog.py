"""Export offline annotations for signal audits; never a neural input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', required=True)
    p.add_argument('--catalog', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with np.load(Path(args.graph) / 'graph.npz', allow_pickle=False) as graph:
        ids = graph['ids'].copy()
    catalog = pd.read_feather(args.catalog).set_index('source_id')
    if not catalog.index.is_unique or not np.all(np.isin(ids, catalog.index)):
        raise ValueError('Complete unique annotation identity required')
    rows = catalog.loc[ids]
    if not rows.retained.all():
        raise ValueError('Annotations include excluded objects')
    np.savez_compressed(out / 'catalog.npz', ids=ids,
                        cell_type=rows.cell_type.fillna('').to_numpy(dtype=str),
                        superclass=rows.superclass.fillna('').to_numpy(dtype=str))
    result = {'catalog_sha256': sha(out / 'catalog.npz'), 'neurons': len(ids),
              'original_catalog_sha256': sha(args.catalog), 'source_sha256': sha(__file__),
              'role': 'Offline interpretation only; not imported by a neural model or game controller.'}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
