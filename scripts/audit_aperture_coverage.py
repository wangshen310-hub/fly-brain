"""Anatomy-only display calibration, then actual color-input alias checks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.research.optical_sampling import ApertureConfig,sample_aperture_linear,coverage_uv,identity
from flysnake.research.viewport_candidates import find_viewport
from flysnake.research.vision import PALETTE
from flysnake.game import DIRECTIONS


def sensory(rgb,z,config):
    retina=sample_aperture_linear(rgb,z['uv'],config)@np.array([.2126,.7152,.0722])
    r8=sample_aperture_linear(rgb,z['r8_uv'],config)
    return np.r_[retina,r8[np.arange(len(r8)),z['r8_channel']]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--radius',type=float,default=.75)
    p.add_argument('--body-color',type=int,nargs=3,default=list(PALETTE['body']))
    p.add_argument('--sizes',type=int,nargs='+',default=[6,8,12]);p.add_argument('--output',type=Path,default=ROOT/'diagnostics/aperture-coverage-20260930-v1.json')
    args=p.parse_args()
    if not all(0<=c<=255 for c in args.body_color):raise ValueError('Valid rendered RGB body required')
    path=ROOT/'diagnostics/retinal-geometry-20260930.npz'
    z=np.load(path,allow_pickle=False);config=ApertureConfig(radius_pixels=args.radius)
    # The field and viewport calibration see only static sensor locations,
    # no food or snake. Counterexample images below audit the locked result.
    combined=np.r_[coverage_uv(z['uv'],config),coverage_uv(z['r8_uv'],config)]
    report={'status':'running','used_by_live_controller':False,
            'body_color':args.body_color,
            'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'optics':identity(np.r_[z['uv'],z['r8_uv']],config),'sizes':{},
            'interpretation':'Sensory input distinguishability only; actual native spikes and learning remain untested.'}
    output=args.output
    if output.exists():raise ValueError('Preserve existing optical audit; choose a new output')
    atomic_json(output,report)
    for size in args.sizes:
        geometry=find_viewport(combined,size,pixel_stride=1,tile_stride=1)
        tile=geometry['tile_pixels'];ox,oy=geometry['board_origin']
        base=np.empty((180,320,3),dtype=np.uint8);base[:]=PALETTE['background']
        base[oy-tile:oy+(size+1)*tile,ox-tile:ox+(size+1)*tile]=PALETTE['wall']
        base[oy:oy+size*tile,ox:ox+size*tile]=PALETTE['background']
        baseline=sensory(base,z,config);rows=[]
        for y in range(size):
            for x in range(size):
                sl=(slice(oy+y*tile,oy+(y+1)*tile),slice(ox+x*tile,ox+(x+1)*tile))
                rgb=base.copy();rgb[sl]=PALETTE['food'];food=sensory(rgb,z,config)
                rgb[sl]=args.body_color;body=sensory(rgb,z,config)
                heads=[]
                for dx,dy in DIRECTIONS:
                    rgb[sl]=PALETTE['head'];region=rgb[sl];mid=tile//2
                    if dy<0:region[mid:]=(235,115,30)
                    elif dy>0:region[:mid]=(235,115,30)
                    elif dx>0:region[:,:mid]=(235,115,30)
                    else:region[:,mid:]=(235,115,30)
                    heads.append(sensory(rgb,z,config))
                aliases=[(i,j) for i in range(4) for j in range(i+1,4)
                         if np.max(abs(heads[i]-heads[j]))<1e-10]
                rows.append({'cell':[x,y],'food_background_max_linear_difference':float(np.max(abs(food-baseline))),
                    'head_body_min_heading_max_difference':float(min(np.max(abs(h-body)) for h in heads)),
                    'heading_alias_pairs':aliases})
        report['sizes'][str(size)]={'geometry':geometry,'rows':rows,
            'food_alias_cells':sum(r['food_background_max_linear_difference']<1e-10 for r in rows),
            'head_body_alias_cells':sum(r['head_body_min_heading_max_difference']<1e-10 for r in rows),
            'heading_alias_cells':sum(bool(r['heading_alias_pairs']) for r in rows),
            'weakest_food_linear_difference':min(r['food_background_max_linear_difference'] for r in rows),
            'weakest_head_body_linear_difference':min(r['head_body_min_heading_max_difference'] for r in rows)}
        atomic_json(output,report)
        print(json.dumps({'size':size,**{k:v for k,v in report['sizes'][str(size)].items() if k not in ('geometry','rows')}}),flush=True)
    report['status']='complete';atomic_json(output,report)


if __name__=='__main__':main()
