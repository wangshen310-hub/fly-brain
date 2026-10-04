"""Anatomy-only complete pixel-grid search; no live protocol is changed."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.research.viewport_candidates import find_viewport
from flysnake.lab.experiment import atomic_json


if __name__=='__main__':
    path=ROOT/'diagnostics/retinal-geometry-20260930.npz'
    z=np.load(path,allow_pickle=False);uv=np.r_[z['uv'],z['r8_uv']]
    report={'used_by_current_training':False,
        'interpretation':'Spatial coverage only. R8 tuning, native pulses and gameplay require separate tests.',
        'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256((ROOT/'flysnake/research/viewport_candidates.py').read_bytes()).hexdigest(),
        'rows':[]}
    for size in (6,8,12):
        g=find_viewport(uv,size,pixel_stride=1,tile_stride=1)
        cells=np.asarray(g['cell_sample_counts']);q=np.asarray(g['quarter_sample_counts'])
        row={'size':size,'geometry':g,'blind_cells':int((cells==0).sum()),
             'blind_quarters':int((q==0).sum()),'minimum_cell_samples':int(cells.min())}
        report['rows'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='geometry'}),flush=True)
    atomic_json(ROOT/'diagnostics/dense-combined-retinal-viewport-20260930-v2.json',report)
