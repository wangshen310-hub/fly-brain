"""Two independent native reloads replay actual next candidate comparisons."""
from pathlib import Path
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.lab.experiment import atomic_json
from flysnake.research.olfaction import OdorConfig
from flysnake.research.sensory_basis import NativeSensoryBasisBrain,BasisConfig,choose_reward_candidate
from flysnake.research.paired import digest
from train_sensory_basis import ARMS,pose_split,food_sites,episode,restore_memory,sources


def replay(run,report,latest):
    settings=report['settings'];brain=NativeSensoryBasisBrain(size=settings['size'],
        config=BasisConfig(**settings['config']),odor=OdorConfig(**settings['odor']))
    brain.restore_basis(run/report['basis']['file'],report['basis']['sha256'])
    memory,rng=restore_memory(run,report,brain)
    training,_=pose_split(settings['size']);coordinate=int(rng.integers(len(brain.coefficients)))
    # Generate the exact full next batch using the checkpoint RNG. To bound
    # replay cost, only its first two worlds are integrated for each proposal.
    batch=[]
    for i in range(settings['batch_size']):
        pose=training[int(rng.integers(len(training)))];food=food_sites(pose,settings['size'],settings['stage'])
        batch.append({'seed':int(rng.integers(1,2**31)),'pose':pose,'food':food[int(rng.integers(len(food)))]})
    result={'coordinate':coordinate,'next_rng':rng.bit_generator.state,'arms':{}}
    for arm in ARMS:
        candidates=np.tile(memory[arm],(3,1));candidates[1,coordinate]+=settings['proposal_step']
        candidates[2,coordinate]-=settings['proposal_step']
        candidates=np.clip(candidates,-brain.config.coefficient_bound,brain.config.coefficient_bound)
        if arm=='frozen':candidates[:]=memory[arm]
        actual=[];totals=[]
        for candidate,coefficients in enumerate(candidates):
            brain.set_parameters(coefficients);rows=[]
            for scenario in batch[:2]:
                row=episode(brain,scenario,stage=settings['stage'],size=settings['size'],limit=settings['limit'])
                rows.append({'seed':row['seed'],'food':row['food'],'return':row['return'],'reason':row['reason'],
                    'counts':[r['receipt']['counts'] for r in row['rows']],
                    'actions':[r['receipt']['action'] for r in row['rows']],
                    'pixels':[r['receipt']['observation_sha256'] for r in row['rows']],
                    'odor':[r['receipt']['olfactory_concentration'] for r in row['rows']],
                    'receptor_spikes':[r['receipt']['olfactory_receptor_spikes'] for r in row['rows']],
                    'feedback':[r['feedback']['delivered_spikes'] if r['feedback'] else None for r in row['rows']],
                    'deadline':row['deadline']['feedback']['delivered_spikes'] if row['deadline'] else None,
                    'weights':brain.weight_hash(),'correction':digest(brain.correction)})
            actual.append({'candidate':candidate,'rows':rows});totals.append(np.mean([r['return'] for r in rows]))
        updated,update=choose_reward_candidate(candidates,totals,arm=arm,
            shuffle_seed=settings['seed']+10000+latest['cycles_completed'])
        result['arms'][arm]={'native_replays':actual,'update':update,'updated_parameters':digest(updated)}
    del brain;gc.collect()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    original=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if original['sources']!=sources():raise ValueError('Saved experiment source changed')
    latest=original['checkpoints'][-1]
    result={'status':'running','checkpoint':latest,
        'interpretation':'Native restoration reproducibility; no claim of behavioral learning.'}
    atomic_json(args.output,result)
    first=replay(args.run,json.loads(json.dumps(original)),latest)
    second=replay(args.run,json.loads(json.dumps(original)),latest)
    if first!=second:
        result.update(status='error',first=first,second=second);atomic_json(args.output,result)
        raise RuntimeError('Independent native restoration differed')
    result.update(status='complete',replays_equal=True,native_episodes_per_reload=18,replay=first,completed_at=time.time())
    atomic_json(args.output,result);print(json.dumps({'status':'complete','native_episodes_per_reload':18}),flush=True)


if __name__=='__main__':main()
