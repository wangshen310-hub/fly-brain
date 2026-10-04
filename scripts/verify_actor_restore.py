"""Two independent native reloads, compared to committed continuous trials.

Only the first K exploration episodes per arm of the next cycle are replayed.
This is a genuine native memory test, not a full coordinator restart proof.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.lab.experiment import atomic_json
from flysnake.research.actor_credit import NativeActorCreditBrain,ActorCreditConfig
from flysnake.research.paired import digest
from train_actor_credit import ARMS,restore_memory,sources,make_batch,episode,pose_split,trial_signature


def replay(run,report,cycle_index,k):
    settings=report['settings'];brain=NativeActorCreditBrain(run,ActorCreditConfig(**settings['actor_config']))
    memory,rng=restore_memory(run,report,brain)
    training,_=pose_split(settings['size'])
    batch=make_batch(rng,training,settings['size'],settings['stage'],settings['batch_size'])
    noise=rng.normal(0,brain.actor_config.noise_sigma,(len(batch),settings['limit'],2))
    result={'batch':batch,'noise_sha256':digest(noise),'next_rng':rng.bit_generator.state,'arms':{}}
    for arm in ARMS:
        brain.set_parameters(memory[arm]);rows=[]
        for i,scenario in enumerate(batch[:k]):
            row,trace,returns=episode(brain,scenario,stage=settings['stage'],size=settings['size'],limit=settings['limit'],noise=noise[i])
            rows.append({'signature':trial_signature(row),'trace':digest(trace),'returns':returns.tolist(),
                         'weights':brain.weight_hash(),'correction':digest(brain.correction)})
        result['arms'][arm]=rows
    del brain;gc.collect();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-cycle',type=int,required=True)
    p.add_argument('--trials-per-arm',type=int,default=2);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier native restore evidence')
    original=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if original['sources']!=sources():raise ValueError('Experiment source changed')
    cp=[c for c in original['checkpoints'] if c['cycles_completed']==args.checkpoint_cycle]
    if not cp:raise ValueError('Requested committed memory missing')
    if not 1<=args.trials_per_arm<=original['settings']['batch_size']:raise ValueError('Bounded replay trial count required')
    selected=copy.deepcopy(original);selected['checkpoints']=[cp[-1]]
    result={'status':'running','checkpoint':cp[-1],'continuous_match':None,
            'interpretation':'Independent complete-brain memory reload; subset replay only, not full coordinator restart or learning evidence'}
    atomic_json(args.output,result)
    first=replay(args.run,copy.deepcopy(selected),args.checkpoint_cycle,args.trials_per_arm)
    second=replay(args.run,copy.deepcopy(selected),args.checkpoint_cycle,args.trials_per_arm)
    if first!=second:
        result.update(status='error',first=first,second=second);atomic_json(args.output,result)
        raise RuntimeError('Native reloads differ')
    latest=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if len(latest['cycles'])>args.checkpoint_cycle:
        continuous=latest['cycles'][args.checkpoint_cycle]
        if json.loads(json.dumps(first['batch']))!=continuous['batch'] or first['noise_sha256']!=continuous['noise_sha256']:
            raise RuntimeError('Restored next cycle world/noise differs from continuous run')
        for arm in ARMS:
            expected=[trial_signature(r) for r in continuous['arms'][arm]['experience'][:args.trials_per_arm]]
            actual=[r['signature'] for r in first['arms'][arm]]
            if actual!=expected:raise RuntimeError('Restored native trials differ from continuous run')
        result['continuous_match']=True
    result.update(status='complete',independent_reloads_equal=True,native_trials_per_reload=args.trials_per_arm*len(ARMS),
                  checkpoint_has_nonzero_learning=cp[-1]['parameters_sha256']['learning']!=cp[-1]['parameters_sha256']['frozen'],
                  replay=first,completed_at=time.time())
    atomic_json(args.output,result);print(json.dumps({k:result[k] for k in ('status','continuous_match','native_trials_per_reload','checkpoint_has_nonzero_learning')}),flush=True)


if __name__=='__main__':main()
