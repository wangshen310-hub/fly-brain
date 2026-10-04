"""Full native model audit of actual receptor spikes for board color classes."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.lab.brain import LabBrain
from flysnake.research.vision import RateRetinaBrain, PALETTE

OLD = {'background':(85,85,85),'wall':(3,3,3),'head':(250,160,20),
       'body':(25,210,45),'food':(180,200,255)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): raise ValueError('Use a new audit output')
    brain = RateRetinaBrain(); b = brain.b
    rows=[]; start=time.time()
    for name, integrate, palette in (('old_linear_current',LabBrain.integrate,OLD),
                                     ('rate_calibrated_new_palette',RateRetinaBrain.integrate,PALETTE)):
        for item in ('wall','background','body','food','head'):
            b.reset(keep_memory=True); b.weights_frozen=True
            brain.retinal_light.fill(0); brain.r8_light.fill(0)
            integrate(brain,np.full((180,320,3),85,dtype=np.uint8),100)
            rgb=np.zeros((180,320,3),dtype=np.uint8); rgb[:]=palette[item]
            counts=integrate(brain,rgb,300)
            photo=counts[b.retina]; r8=counts[b.r8]
            record={'encoding':name,'item':item,'rgb':palette[item],
                    'retina_median_spikes':float(np.median(photo)), 'retina_silent_fraction':float(np.mean(photo==0)),
                    'retina_mean_hz':float(photo.mean()/0.3),
                    'r8_green_median_spikes':float(np.median(r8[b.r8_channel==1])),
                    'r8_blue_median_spikes':float(np.median(r8[b.r8_channel==2])),
                    'total_spikes':int(counts.sum()),'elapsed':time.time()-start}
            rows.append(record); print(json.dumps(record),flush=True)
            args.output.write_text(json.dumps({'status':'running','rows':rows},indent=2),encoding='utf-8')
    args.output.write_text(json.dumps({'status':'complete','rows':rows,
        'interpretation':'Full-field sensory engineering audit only. Color and encoder changed together; does not isolate their individual contributions or prove learning.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
