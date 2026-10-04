"""Create a new immutable export with verified sorted transpose CSR columns."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flysnake.research.csr_layout import canonical_columns


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if sha(source/'graph.npz')!=manifest['graph_sha256']:raise ValueError('Parent graph integrity mismatch')
    with np.load(source/'graph.npz',allow_pickle=False) as loaded:
        arrays={k:loaded[k] for k in loaded.files}
    n=len(arrays['ids'])
    for name in ('exc','inh'):
        # Forward CSR is already sorted and unique. Recreate its mathematical
        # transpose ordering independently to check the original transpose.
        rows=np.repeat(np.arange(n,dtype=np.int32),np.diff(arrays[name+'_crow']))
        pre=arrays[name+'_col'];expected_order=np.lexsort((rows,pre))
        expected_col=rows[expected_order];expected_value=arrays[name+'_value'][expected_order]
        col,value=canonical_columns(arrays[name+'_tcrow'],arrays[name+'_tcol'],arrays[name+'_tvalue'],n)
        if not np.array_equal(col,expected_col) or not np.array_equal(value,expected_value):
            raise ValueError('Transpose does not represent the identical existing weighted graph')
        arrays[name+'_tcol']=col;arrays[name+'_tvalue']=value
        print(json.dumps({'matrix':name,'entries':len(col),'exact_transpose':True}),flush=True)
        del rows,pre,expected_order,expected_col,expected_value,col,value
    with (out/'graph.npz').open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    manifest.update(version='complete-signed-MaleCNS-canonical-CSR-export-v2',
        parent_export_sha256=manifest['graph_sha256'],parent_manifest_sha256=sha(source/'manifest.json'),
        graph_sha256=sha(out/'graph.npz'),array_sha256={k:digest(v) for k,v in arrays.items()},
        csr_canonicalizer_sha256=sha(__file__),csr_layout_source_sha256=sha(ROOT/'flysnake/research/csr_layout.py'),
        csr_change='Sort transpose columns and their paired weights only; forward, anatomical arrays and all other arrays unchanged.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'graph_sha256':manifest['graph_sha256'],'neurons':n,'anatomical_edges':manifest['anatomical_edges']}),flush=True)


if __name__=='__main__':main()
