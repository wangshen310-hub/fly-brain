"""Paired held-out analysis; diagnostics never feed a controller or optimizer."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.game import DIRECTIONS
from flysnake.lab.experiment import atomic_json


def paired_clusters(normal,control,seed=930599):
    if len(normal)!=len(control):raise ValueError('Unmatched held-out episodes')
    clusters={}
    for a,b in zip(normal,control):
        keys=('seed','pose','food_start','stage','size','limit')
        if any(a[k]!=b[k] for k in keys):raise ValueError('Control worlds differ')
        if a['rows'][0]['receipt']['observation_sha256']!=b['rows'][0]['receipt']['observation_sha256']:
            raise ValueError('Control initial pixels differ')
        key=json.dumps(a['pose'],sort_keys=True)
        cluster=clusters.setdefault(key,{'difference':0.,'n':0})
        cluster['difference']+=a['food']-b['food'];cluster['n']+=1
    values=np.array([[c['difference'],c['n']] for c in clusters.values()])
    rng=np.random.default_rng(seed)
    indices=rng.integers(len(values),size=(20000,len(values)))
    draws=values[indices].sum(1)
    ratios=draws[:,0]/draws[:,1]
    return {'food_gain_per_episode':float(values[:,0].sum()/values[:,1].sum()),
        'pose_cluster_bootstrap_95':np.quantile(ratios,[.025,.975]).tolist(),
        'pose_clusters':len(values),'n':len(normal),
        'both_have_food':sum(a['food']>0 and b['food']>0 for a,b in zip(normal,control)),
        'normal_only_food':sum(a['food']>0 and b['food']==0 for a,b in zip(normal,control)),
        'control_only_food':sum(a['food']==0 and b['food']>0 for a,b in zip(normal,control)),
        'interpretation':'Paired differences resampled by initial body pose, not treating all neighboring food choices as independent.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    path=args.run/'status.json';raw=path.read_bytes();d=json.loads(raw)
    if d['status']!='complete' or set(d['after'])!={'learning','frozen','shuffled'}:
        raise ValueError('Wait for all final frozen assessments; incomplete results are not a learning test')
    result={'status':'complete','source_report_sha256':hashlib.sha256(raw).hexdigest(),
            'protocol':d['protocol'],
            'stage':d['settings']['stage'],'size':d['settings']['size'],
            'cycles':d['cycles_completed'],'evaluations':{},'controls':{},
            'ordinary_snake_goal_complete':False,'interpretation':'Single-seed experimental analysis. Does not satisfy independent repetitions or ordinary 12x12 acceptance.'}
    initial=d['before']['episodes'];held_hash={e['rows'][0]['receipt']['observation_sha256'] for e in initial}
    calibration={x['rgb'] for x in d['brain']['baseline_inputs']}
    if held_hash&calibration:raise ValueError('Label-free calibration overlaps held-out first images')
    for arm,ev in [('before',d['before']),*d['after'].items()]:
        rows=ev['episodes'];actions=Counter(r['receipt']['action'] for e in rows for r in e['rows'])
        for e in rows:
            for row in e['rows']:
                if row['receipt']['memory_sha256']!=ev['memory_sha256']:
                    raise ValueError('Frozen evaluation changed effective memory')
                if row['feedback'] and row['feedback']['delivered_spikes']<=0:
                    raise ValueError('Native feedback not actually delivered')
        result['evaluations'][arm]={'food':ev['food'],'n':ev['n'],
            'success_episodes':ev['success_episodes'],'actions':dict(actions),
            'memory_sha256':ev['memory_sha256']}
    for arm in ('frozen','shuffled'):
        result['controls'][arm]=paired_clusters(d['after']['learning']['episodes'],d['after'][arm]['episodes'])
    for a,b in zip(initial,d['after']['frozen']['episodes']):
        if (a['food']!=b['food'] or a['return']!=b['return'] or a['reason']!=b['reason']
                or [r['receipt']['counts'] for r in a['rows']]!=[r['receipt']['counts'] for r in b['rows']]):
            raise ValueError('Frozen baseline failed to reproduce native behavior')
    if d['settings']['stage']=='adjacent_one_step':
        targets=Counter()
        for row in initial:
            h=row['pose']['heading'];head=row['pose']['body'][0];food=row['food_start']
            goal=next(a for a,t in [('left',-1),('forward',0),('right',1)]
                if [head[0]+DIRECTIONS[(h+t)%4][0],head[1]+DIRECTIONS[(h+t)%4][1]]==food)
            targets[goal]+=1
        result['constant_action_analysis']={'goal_counts':dict(targets),'best_constant_food':max(targets.values()),
            'labels_reach_learner':False}
    # Equal score VALUES cannot tell whether the candidate INDEX permutation
    # was the identity: flat candidates retain the same values under any order.
    permutations={}
    for c in d['cycles']:
        update=c['arms']['shuffled']['update']
        if 'feedback_permutation' in update:perm=update['feedback_permutation']
        else:
            indices=np.arange(3)
            np.random.default_rng(d['settings']['seed']+10000+c['cycle']-1).shuffle(indices)
            perm=indices.tolist()
        permutations[str(c['cycle'])]=perm
    result['feedback_control_audit']={'candidate_index_permutations':permutations,
        'identity_index_permutation_cycles':[int(c) for c,p in permutations.items() if p==list(range(len(p)))],
        'reward_values_unchanged_cycles':[
        c['cycle'] for c in d['cycles'] if c['arms']['shuffled']['update']['actual_returns']==
        c['arms']['shuffled']['update']['assigned_returns']],
        'same_normal_shuffled_final_parameters':d['checkpoints'][-1]['parameters_sha256']['learning']==
                                              d['checkpoints'][-1]['parameters_sha256']['shuffled']}
    if d['protocol'] in ('native-self-action-credit-verified-reward-v1','native-pulse-balanced-self-action-credit-v1'):
        for c in d['cycles']:
            proposal=c['arms']['shuffled']['proposal'];perm=proposal['feedback_permutation']
            if any(i==j for i,j in enumerate(perm)):raise ValueError('Wrongly matched shuffled feedback trial')
            if any(i==j for i,j in enumerate(permutations[str(c['cycle'])])):
                raise ValueError('Wrongly matched shuffled candidate verification')
        result['feedback_control_audit']['all_shuffled_indices_displaced']=True
    result['pilot_paired_improvement']=all(v['pose_cluster_bootstrap_95'][0]>0 for v in result['controls'].values())
    if result.get('constant_action_analysis'):
        result['pilot_paired_improvement'] &= d['after']['learning']['food']>max(targets.values())
    atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='evaluations'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
