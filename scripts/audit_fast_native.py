"""Full-state bit audit and CPU benchmark of an independent kernel candidate.

One complete model is used sequentially; no extra brain is created for the
candidate. Every ten-ms boundary, all dynamic arrays and actual counts are
hashed, including subthreshold and queue state. The original kernel is restored.
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
from flysnake.lab.experiment import atomic_json
from flysnake.research.actor_credit import NativeActorCreditBrain,ActorCreditConfig
from flysnake.research.fast_native_candidate import build,bind
from flysnake.research.paired import digest,writer_lock
from flysnake.research.olfaction import odor_samples
from train_actor_credit import pose_split,make_batch,SensoryGame

DYNAMIC=('v','g','refractory','drive','previous_drive','queue','queue_count','counts',
         'active','active_flag','nactive','last','eligibility','eligibility_last','modulation',
         'modulation_last','adaptation','luminance','rate_kc','rate_dan','memory_u','memory_w')


def state_hashes(b):
    result={name:digest(getattr(b,name)) for name in DYNAMIC}
    result.update(cursor=b.cursor,sim_ms=b.sim_ms,total_spikes=b.total_spikes)
    return result


def replay(brain,advance,rgb,concentration,noise,kind):
    b=brain.brain.b;b.advance=advance;boundaries=[]
    original_step=b._neural_step
    def traced(*args,**kwargs):
        c,t=original_step(*args,**kwargs)
        boundaries.append(state_hashes(b));return c,t
    b._neural_step=traced
    start=time.process_time();wall=time.perf_counter()
    try:
        brain.reset_trial();r,t=brain.observe(rgb,concentration,noise)
        feedback=brain.feedback(kind)
        second,trace=brain.observe(rgb,concentration,-np.asarray(noise))
        result={'boundaries':boundaries,'counts':[r['counts'],second['counts']],
                'actions':[r['action'],second['action']],
                'eligibility':[digest(t),digest(trace)],'feedback':feedback['delivered_spikes'],
                'weights':brain.weight_hash(),'final_state':state_hashes(b),'retinal_state':digest(brain.brain.retinal_light),
                'r8_state':digest(brain.brain.r8_light)}
        cpu=time.process_time()-start;elapsed=time.perf_counter()-wall
    finally:b._neural_step=original_step
    return result,cpu,elapsed


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',type=int,default=3)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():raise ValueError('Preserve old kernel audit')
        report={'status':'compiling','protocol':'native-entire-state-tick-audit-v1','rows':[],
                'dt_ms':.1,'retained_neurons':166700,'retained_edges':25582938,
                'used_by_trainer':False,'interpretation':'Finite native bit audit and CPU benchmark only; not a universal proof or learning result'}
        atomic_json(path,report)
        try:
            library,record=build(args.output/'build');report['build']=record
            report['status']='loading';atomic_json(path,report)
            brain=NativeActorCreditBrain(args.baseline);b=brain.brain.b;original=b.advance
            loaded,fast=bind(b,library);report['brain']=brain.identity
            training,_=pose_split(brain.geometry['board_cells'])
            batch=make_batch(np.random.default_rng(930603),training,brain.geometry['board_cells'],'uniform_food',args.scenes)
            rng=np.random.default_rng(930603)
            for i,scenario in enumerate(batch):
                # Test zero and real independent signed native synaptic gains.
                theta=np.zeros_like(brain.coefficients) if i==0 else rng.normal(0,.1,len(brain.coefficients))
                brain.set_parameters(theta)
                game=SensoryGame(seed=scenario['seed'],size=brain.geometry['board_cells'],stage='uniform_food',geometry=brain.geometry)
                game.snake=[tuple(x) for x in scenario['pose']['body']];game.direction=scenario['pose']['heading'];game.food=tuple(scenario['food'])
                rgb=game.render_observation();sense=odor_samples(game.snake[0],game.direction,game.food,brain.odor)
                noise=rng.normal(0,3,2);kind='reward' if i%2 else 'aversive'
                first,cpu_old,wall_old=replay(brain,original,rgb,sense['concentration'],noise,kind)
                second,cpu_fast,wall_fast=replay(brain,fast,rgb,sense['concentration'],noise,kind)
                equal=first==second
                row={'seed':scenario['seed'],'kind':kind,'pixels':hashlib.sha256(rgb.tobytes()).hexdigest(),
                     'native_bit_equal':equal,'boundaries':len(first['boundaries']),'cpu_old':cpu_old,
                     'cpu_candidate':cpu_fast,'cpu_speedup':cpu_old/cpu_fast,'wall_old':wall_old,'wall_candidate':wall_fast,
                     'old_receipt_sha256':hashlib.sha256(json.dumps(first,sort_keys=True).encode()).hexdigest(),
                     'candidate_receipt_sha256':hashlib.sha256(json.dumps(second,sort_keys=True).encode()).hexdigest()}
                if not equal:row.update(old=first,candidate=second)
                report['rows'].append(row);report.update(status='running',active_scene=i+1);atomic_json(path,report)
                print(json.dumps(row),flush=True)
                if not equal:raise RuntimeError('Independent kernel changed actual native state')
            b.advance=original
            report.update(status='complete',all_boundaries_bit_equal=True,
                          aggregate_cpu_speedup=sum(r['cpu_old'] for r in report['rows'])/sum(r['cpu_candidate'] for r in report['rows']),
                          completed_at=time.time());atomic_json(path,report)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
