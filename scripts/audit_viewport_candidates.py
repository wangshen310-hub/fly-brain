from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.research.viewport_candidates import find_viewport
from flysnake.research.projection import coverage_geometry
from flysnake.lab.experiment import atomic_json


def stats(g):
    cells=np.asarray(g['cell_sample_counts']);q=np.asarray(g['quarter_sample_counts'])
    return {'geometry':g,'blind_cells':int((cells==0).sum()),'minimum_cell_samples':int(cells.min()),
            'blind_quarters':int((q==0).sum()),'minimum_quarter_samples':int(q.min())}


if __name__=='__main__':
    z=np.load(ROOT/'diagnostics/retinal-geometry-20260930.npz',allow_pickle=False)
    report={'interpretation':'Static input-coverage candidate only; no native behavioral or learning result.',
            'used_by_current_training':False,'sizes':{}}
    for size in (4,6,8,12):
        original=stats(coverage_geometry(z['uv'],size));candidate=stats(find_viewport(z['uv'],size))
        report['sizes'][str(size)]={'current':original,'candidate':candidate}
        print(json.dumps({'size':size,'old_blind':original['blind_cells'],
                          'candidate_blind':candidate['blind_cells'],
                          'candidate_quarters_blind':candidate['blind_quarters'],
                          'tile':candidate['geometry']['tile_pixels']}),flush=True)
    atomic_json(ROOT/'diagnostics/fixed-viewport-candidates-20260930-v1.json',report)
