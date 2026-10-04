"""Does the weakest static optical signal change actual native receptor spikes?

No game is controlled and no reward is given. This is perception evidence only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.research.aperture_native import NativeAperturePerceptionBrain
from flysnake.research.aperture_contrast import NativeContrastApertureBrain,BODY_COLOR
from flysnake.research.optical_sampling import sample_aperture_linear,ApertureConfig
from flysnake.research.paired import digest,writer_lock
from flysnake.research.vision import PALETTE
from flysnake.game import DIRECTIONS


def audit_images(geometry,cell):
    size=geometry['board_cells'];tile=geometry['tile_pixels'];ox,oy=geometry['board_origin']
    base=np.empty((180,320,3),np.uint8);base[:]=PALETTE['background']
    base[oy-tile:oy+(size+1)*tile,ox-tile:ox+(size+1)*tile]=PALETTE['wall']
    base[oy:oy+size*tile,ox:ox+size*tile]=PALETTE['background']
    x,y=cell;sl=(slice(oy+y*tile,oy+(y+1)*tile),slice(ox+x*tile,ox+(x+1)*tile))
    frames={'background':base.copy()}
    for name in ('food','body'):
        rgb=base.copy();rgb[sl]=PALETTE[name];frames[name]=rgb
    for heading,(dx,dy) in enumerate(DIRECTIONS):
        rgb=base.copy();rgb[sl]=PALETTE['head'];region=rgb[sl];mid=tile//2
        if dy<0:region[mid:]=(235,115,30)
        elif dy>0:region[:mid]=(235,115,30)
        elif dx>0:region[:,:mid]=(235,115,30)
        else:region[:,mid:]=(235,115,30)
        frames[f'head_{heading}']=rgb
    return frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--static-audit',type=Path,default=Path('diagnostics/aperture-coverage-20260930-radius15-v1.json'))
    p.add_argument('--cells',type=int,default=3);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():raise ValueError('Preserve earlier native optical evidence')
        static=json.loads(args.static_audit.read_text(encoding='utf-8'));size=static['sizes']['12']
        cells=[r['cell'] for r in sorted(size['rows'],key=lambda r:r['head_body_min_heading_max_difference'])[:args.cells]]
        report={'status':'loading','protocol':'native-aperture-weak-cell-audit-v1','cells':cells,'rows':[],
                'static_source_sha256':hashlib.sha256(args.static_audit.read_bytes()).hexdigest(),
                'interpretation':'Weakest-cell native sensation audit only; not exhaustive spike distinguishability or learning proof'}
        atomic_json(path,report)
        (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
        try:
            color=tuple(static.get('body_color',PALETTE['body']))
            factory=NativeContrastApertureBrain if color==BODY_COLOR else NativeAperturePerceptionBrain
            if color not in (BODY_COLOR,PALETTE['body']):raise ValueError('Use an implemented game display palette')
            brain=factory(size=12,aperture=ApertureConfig(**static['optics']['config']))
            if brain.geometry!=size['geometry']:raise RuntimeError('Locked optical geometry differs')
            report['brain']=brain.identity;atomic_json(path,report)
            receptors=np.r_[brain.brain.b.retina,brain.brain.b.r8]
            for cell in cells:
                row={'cell':cell,'frames':{}}
                actual={}
                frames=audit_images(brain.geometry,cell)
                if color!=PALETTE['body']:
                    geometry=brain.geometry;tile=geometry['tile_pixels'];ox,oy=geometry['board_origin'];x,y=cell
                    frames['body'][oy+y*tile:oy+(y+1)*tile,ox+x*tile:ox+(x+1)*tile]=color
                for name,rgb in frames.items():
                    brain.reset_trial();receipt=brain.observe(rgb,[0.,0.]);counts=brain.brain.last['counts'].copy()
                    actual[name]=counts
                    row['frames'][name]={'pixels':receipt['observation_sha256'],'receptor_spikes_sha256':digest(counts[receptors]),
                        'all_spikes_sha256':digest(counts),'motor_counts':receipt['counts'],'total_spikes':int(counts.sum())}
                    report.update(status='running',active_cell=cell,active_frame=name);atomic_json(path,report)
                    print(json.dumps({'cell':cell,'frame':name,'counts':receipt['counts']}),flush=True)
                row['food_changed_receptors']=int(np.count_nonzero(actual['food'][receptors]!=actual['background'][receptors]))
                row['head_body_changed_receptors']={h:int(np.count_nonzero(actual[f'head_{h}'][receptors]!=actual['body'][receptors])) for h in range(4)}
                row['heading_native_receptor_aliases']=[[i,j] for i in range(4) for j in range(i+1,4)
                    if np.array_equal(actual[f'head_{i}'][receptors],actual[f'head_{j}'][receptors])]
                row['heading_full_native_aliases']=[[i,j] for i in range(4) for j in range(i+1,4)
                    if np.array_equal(actual[f'head_{i}'],actual[f'head_{j}'])]
                report['rows'].append(row);atomic_json(path,report)
            report.update(status='complete',completed_at=time.time());atomic_json(path,report)
            print(json.dumps({'status':'complete','rows':[{k:v for k,v in r.items() if k!='frames'} for r in report['rows']]}),flush=True)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
