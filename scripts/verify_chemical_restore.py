"""Independent full-native reloads of a committed chemical synaptic memory.

Subset replays check actual candidates, native spikes and DAN feedback against
continuous trials. They do not prove a full coordinator restart or learning.
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
import train_chemical_controls as trainer
from flysnake.research.chemical_basis import NativeChemicalBasisBrain,ChemicalConfig,synaptic_candidates,VERSION
from flysnake.research.paired import digest
from flysnake.lab.experiment import atomic_json


def load_brain(run,report):
    brain=NativeChemicalBasisBrain(run,chemical=ChemicalConfig(**report['settings']['chemical_config']))
    saved=report['chemical_basis']
    if Path(saved['file']).name!=saved['file']:raise ValueError('Unsafe chemical baseline file')
    brain.restore_chemical_basis(run/saved['file'],saved['sha256'])
    return brain


def replay(run,report,k):
    settings=report['settings'];brain=load_brain(run,report)
    memory,rng=trainer.engine.restore_memory(run,report,brain)
    poses,_=trainer.engine.pose_split(settings['size'])
    batch=trainer.grouped_worlds(rng,poses,settings['size'],settings['stage'],settings['poses_per_heading'])
    result={'batch':batch,'next_rng':rng.bit_generator.state,'arms':{}}
    for arm in trainer.engine.ARMS:
        candidates,proposal=synaptic_candidates(memory[arm],report['cycles_completed'],settings['synaptic_step'],
                                                brain.chemical.coefficient_bound)
        if arm=='frozen':candidates[:]=memory[arm]
        groups=[]
        for theta in candidates:
            brain.set_parameters(theta);rows=[]
            for scenario in batch[:k]:
                row,traces,returns=trainer.engine.episode(brain,scenario,stage=settings['stage'],
                    size=settings['size'],limit=settings['limit'])
                rows.append({'signature':trainer.engine.trial_signature(row),'traces':digest(traces),
                    'returns':returns.tolist(),'weights':brain.weight_hash(),'correction':digest(brain.correction)})
            groups.append({'parameters':digest(theta),'rows':rows})
        result['arms'][arm]={'proposal':proposal,'groups':groups}
    del brain;gc.collect();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-cycle',type=int,required=True)
    p.add_argument('--trials-per-candidate',type=int,default=2);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier native evidence')
    original=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if original['protocol']!=VERSION or original['sources']!=trainer.sources():raise ValueError('Protocol/source changed')
    cps=[cp for cp in original['checkpoints'] if cp['cycles_completed']==args.checkpoint_cycle]
    if not cps:raise ValueError('Requested committed native memory missing')
    if not 1<=args.trials_per_candidate<=4:raise ValueError('Bounded native replay count required')
    selected=copy.deepcopy(original);selected['checkpoints']=[cps[-1]]
    result={'status':'running','checkpoint':cps[-1],'protocol':VERSION,'continuous_match':None,
        'verification_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation':'Two independent full-native loads, subset candidate replay; not full coordinator recovery or learning evidence'}
    atomic_json(args.output,result)
    first=replay(args.run,copy.deepcopy(selected),args.trials_per_candidate)
    second=replay(args.run,copy.deepcopy(selected),args.trials_per_candidate)
    if first!=second:
        result.update(status='error',first=first,second=second);atomic_json(args.output,result)
        raise RuntimeError('Independent native reloads differ')
    latest=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if len(latest['cycles'])>args.checkpoint_cycle:
        continuous=latest['cycles'][args.checkpoint_cycle]
        if json.loads(json.dumps(first['batch']))!=continuous['batch']:raise RuntimeError('Next physical batch differs')
        for arm in trainer.engine.ARMS:
            for replayed,actual in zip(first['arms'][arm]['groups'],continuous['arms'][arm]['groups']):
                if replayed['parameters']!=actual['parameters_sha256']:raise RuntimeError('Native candidate differs')
                expected=[trainer.engine.trial_signature(row) for row in actual['episodes'][:args.trials_per_candidate]]
                if expected!=[row['signature'] for row in replayed['rows']]:
                    raise RuntimeError('Restored spikes/actions/feedback differ from continuous trials')
        cp_next=[cp for cp in latest['checkpoints'] if cp['cycles_completed']==args.checkpoint_cycle+1]
        if not cp_next:raise RuntimeError('Next continuous cycle lacks committed memory')
        with np.load(args.run/cp_next[-1]['file'],allow_pickle=False) as z:
            if json.loads(str(z['metadata']))['rng']!=first['next_rng']:
                raise RuntimeError('Next continuous RNG differs')
        result['continuous_match']=True
    result.update(status='complete',replay=first,independent_reloads_equal=True,
                  native_trials_per_reload=3*3*args.trials_per_candidate,
                  checkpoint_has_nonzero_learning=cps[-1]['parameters_sha256']['learning']!=cps[-1]['parameters_sha256']['frozen'],
                  completed_at=time.time())
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','continuous_match','native_trials_per_reload','checkpoint_has_nonzero_learning')}),flush=True)


if __name__=='__main__':main()
