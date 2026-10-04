"""Unrewarded raw motor readout and bilateral tonic-current audit.

No game is played or scored. Neutral images have no food; the experiment only
measures actual native pulse responses to fixed intrinsic drive conditions.
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
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.research.raw_turn import RawTurnBrain,RawTurnGame,VERSION
from flysnake.research.paired import digest,writer_lock
from flysnake.lab.experiment import atomic_json
from train_actor_credit import prepare_baseline
from train_sensory_basis import pose_split


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():raise ValueError('Preserve earlier native audit')
        sources={relative:hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() for relative in
                 ('scripts/audit_raw_motor_neutrality.py','flysnake/research/raw_turn.py')}
        report={'protocol':VERSION,'status':'loading','sources':sources,'rows':[],'calibration_reward':False,
                'actions_used_to_fit':False,'interpretation':'Unrewarded motor working-point audit only, not a learning result'}
        atomic_json(path,report)
        for relative in sources:
            target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/relative).read_bytes())
        try:
            prepare_baseline(args.baseline_run,args.output,4);brain=RawTurnBrain(args.output)
            poses,_=pose_split(4)
            frames=[]
            for heading in range(4):
                pose=next(p for p in poses if p['heading']==heading)
                game=RawTurnGame(seed=930801+heading,size=4,stage='uniform_food',geometry=brain.geometry)
                game.snake=[tuple(p) for p in pose['body']];game.direction=heading;game.food=None
                frames.append(game.render_observation())
            motors=[brain.brain.native.groups[name][0] for name in ('left','right')]
            report.update(brain=brain.identity,motor_rest_mV=brain.brain.b.rest[motors].tolist(),
                          baseline_tonic_mV=brain.base_tonic[motors].tolist())
            # Independent bilateral current perturbations, including common
            # drive. None can use food positions or desired steering labels.
            conditions=[(12.,12.),(14.,12.),(12.,14.),(14.,14.),(16.,16.)]
            for condition in conditions:
                brain.base_tonic[motors]=condition
                for index,rgb in enumerate(frames):
                    brain.reset_trial();receipt,_=brain.observe(rgb,[0.,0.])
                    row={'tonic_mV':list(condition),'fixed_image':index,
                         'pixels_sha256':receipt['observation_sha256'],'counts':receipt['counts'],
                         'raw_action':receipt['action'],'full_native_counts_sha256':digest(brain.brain.last['counts'])}
                    report['rows'].append(row);report.update(status='running',updated_at=time.time())
                    atomic_json(path,report);print(json.dumps(row),flush=True)
            report.update(status='complete',completed_at=time.time());atomic_json(path,report)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
