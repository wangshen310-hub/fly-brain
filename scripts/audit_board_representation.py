"""Frozen full-network information audit, never an alternate action controller.

Both sensors receive identical world states. Predictors diagnose availability
of heading-relative food information in real native DNa02 afferents; predictor
coefficients are discarded and no predictor output enters any game action.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from types import MethodType
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.lab.brain import LabBrain
from flysnake.research.vision import RateRetinaBrain, ReadableGame
from flysnake.research.operating import apply_operating_point,OperatingPoint
from pilot_online_native_snake import NearbyFoodGame
from flysnake.game import DIRECTIONS
from probe_logged_snake_information import predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes',type=int,default=64)
    p.add_argument('--headroom',type=float)
    p.add_argument('--new-only',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'status.json').exists():raise ValueError('Use a new output')
    scenes=[];seen=set()
    for seed in range(51000,61000):
        game=NearbyFoodGame(seed=seed,size=4)
        identity=(tuple(game.snake),game.direction,game.food)
        if identity in seen:continue
        seen.add(identity)
        hx,hy=game.snake[0]
        direction=DIRECTIONS.index((game.food[0]-hx,game.food[1]-hy))
        label={3:0,0:1,1:2}[(direction-game.direction)%4]
        scenes.append({'seed':seed,'snake':game.snake,'direction':game.direction,'food':game.food,
                       'label':label,'body_group':str((tuple(game.snake),game.direction))})
        if len(scenes)==args.scenes:break
    brain=RateRetinaBrain();b=brain.b
    operating=apply_operating_point(brain,OperatingPoint(args.headroom)) if args.headroom is not None else None
    targets=np.array([np.flatnonzero(b.ids==cell)[0] for cell in (523769,10360)])
    b.tonic[targets]=12.
    edges=np.flatnonzero(np.isin(b.post,targets)&(b.weight!=0))
    pre=np.searchsorted(b.ptr,edges,side='right')-1
    fast=b.modulation_mask[pre]==0;pre=np.unique(pre[fast])
    report={'protocol':'frozen-board-representation-v1','status':'running','scenes':scenes,
            'operating_point':operating,
            'feature_cells':len(pre),'results':{},'completed':{},
            'inference_use':'None. This script never applies an action or saves fitted predictor weights.'}
    modes=[('rate_calibrated',RateRetinaBrain.integrate)] if args.new_only else [('old',LabBrain.integrate),('rate_calibrated',RateRetinaBrain.integrate)]
    for mode,method in modes:
        brain.integrate=MethodType(method,brain)
        features=[]; hashes=[]; motors=[]
        for i,scene in enumerate(scenes):
            game=(NearbyFoodGame if mode=='old' else ReadableGame)(seed=scene['seed'],size=4)
            game.snake=[tuple(p) for p in scene['snake']];game.direction=scene['direction'];game.food=tuple(scene['food'])
            rgb=game.render_observation()
            brain.observe(rgb,learning=False)
            counts=brain.last['counts']
            features.append(counts[pre].copy());motors.append(counts[targets].copy())
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            report['completed'][mode]=i+1
            if (i+1)%8==0:
                (args.output/'status.json').write_text(json.dumps(report),encoding='utf-8')
                print(json.dumps({'mode':mode,'through':i+1}),flush=True)
        x=np.array(features,dtype=float)
        labels=np.array([s['label'] for s in scenes]);groups=np.array([s['body_group'] for s in scenes])
        x_unique=np.unique(x.T,axis=0).T
        x_unique=x_unique[:,np.ptp(x_unique,axis=0)>0]
        result={'distinct_nonconstant_features':x_unique.shape[1], 'active_afferents':int(np.any(x>0,axis=0).sum()),
                'class_counts':np.bincount(labels,minlength=3).tolist(),'predictions':{}}
        for kind in ('linear','rbf'):
            guess,majority=predictions(x_unique,labels,groups,kind)
            result['predictions'][kind]={'accuracy':float(np.mean(guess==labels)),
                'majority_accuracy':float(np.mean(majority==labels)),
                'correct':int((guess==labels).sum()),'n':len(labels)}
        np.savez_compressed(args.output/f'{mode}-activity.npz',counts=x,pre_ids=b.ids[pre],
                            motor_counts=np.array(motors),labels=labels,groups=groups,image_hashes=hashes)
        report['results'][mode]=result
        (args.output/'status.json').write_text(json.dumps(report),encoding='utf-8')
        print(json.dumps({'mode':mode,'results':result}),flush=True)
    report.update(status='complete',interpretation='Frozen anatomical input information only. New color palette and rate transfer tested together, not isolated effects. No game learning claim.')
    (args.output/'status.json').write_text(json.dumps(report),encoding='utf-8')


if __name__=='__main__':main()
