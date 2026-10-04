"""Canonical CSR columns; permutation only, never reconnects or changes weights."""
import numpy as np


def canonical_columns(crow, col, values, n):
    crow=np.asarray(crow);col=np.asarray(col);values=np.asarray(values)
    if (crow.shape!=(n+1,) or crow[0]!=0 or crow[-1]!=len(col)
            or values.shape!=col.shape or (np.diff(crow)<0).any()
            or (col<0).any() or (col>=n).any() or not np.isfinite(values).all()):
        raise ValueError('Invalid CSR arrays')
    rows=np.repeat(np.arange(n,dtype=np.int32),np.diff(crow))
    keys=rows.astype(np.uint64)*np.uint64(n)+col.astype(np.uint64)
    order=np.argsort(keys,kind='stable')
    ordered=keys[order]
    if (ordered[1:]<=ordered[:-1]).any():
        raise ValueError('Parallel CSR entries require an explicit coalescing protocol')
    return col[order].copy(),values[order].copy()
