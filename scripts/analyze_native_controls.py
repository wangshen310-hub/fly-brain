"""Offline paired behavioral analysis; this module never enters the actor."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flysnake.lab.experiment import atomic_json
from flysnake.game import DIRECTIONS


def world_key(row):
    return (row['seed'],row['pose']['heading'],tuple(map(tuple,row['pose']['body'])),tuple(row['food_start']))


def comparison(left,right,replicates=10000):
    if [world_key(r) for r in left]!=[world_key(r) for r in right]:raise ValueError('Matched evaluation worlds required')
    difference=np.array([a['food']-b['food'] for a,b in zip(left,right)],float)
    groups={}
    for i,row in enumerate(left):
        key=(row['pose']['heading'],tuple(map(tuple,row['pose']['body'])))
        groups.setdefault(key,[]).append(i)
    blocks=list(groups.values());rng=np.random.default_rng(930730)
    samples=[]
    for _ in range(replicates):
        indices=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        samples.append(float(difference[indices].mean()))
    return {'mean_food_difference':float(difference.mean()),'cluster_bootstrap_95_ci':np.quantile(samples,[.025,.975]).tolist(),
            'n_worlds':len(left),'independent_pose_groups':len(groups),'bootstrap_replicates':replicates,
            'paired_positive':int(np.sum(difference>0)),'paired_negative':int(np.sum(difference<0)),
            'paired_unchanged':int(np.sum(difference==0))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    s=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if s['status']!='complete' or set(s['after'])!={'learning','frozen','shuffled'}:
        raise ValueError('Complete three-arm frozen assessments required')
    evaluations={'before':s['before'],**s['after']}
    for label,ev in evaluations.items():
        if not ev['parameters_frozen'] or not ev['exploration_off']:raise ValueError('Frozen assessment required')
        if sum(r['food'] for r in ev['episodes'])!=ev['food']:raise ValueError('Food summary differs from actual episodes')
        for row in ev['episodes']:
            food_events=sum(bool(r['event'] and r['event']['kind']=='reward') for r in row['rows'])
            if food_events!=row['food']:raise ValueError('Game reward events differ from food score')
            for step in row['rows']:
                if step['feedback'] and step['feedback']['delivered_spikes']<=0:
                    raise ValueError('Native DAN feedback missing')
                if step['receipt']['motor_current_noise']!=[0.,0.]:raise ValueError('Exploration in frozen assessment')
    results={control:comparison(s['after']['learning']['episodes'],s['after'][control]['episodes'])
             for control in ('frozen','shuffled')}
    best_constant=None
    if s['settings']['stage']=='adjacent_one_step':
        targets={a:0 for a in ('left','forward','right')}
        for row in s['before']['episodes']:
            x,y=row['pose']['body'][0];fx,fy=row['food_start'];heading=row['pose']['heading']
            absolute=DIRECTIONS.index((fx-x,fy-y));relative=(absolute-heading)%4
            if relative==2:raise ValueError('Food overlaps the initial body')
            targets[{3:'left',0:'forward',1:'right'}[relative]]+=1
        best_constant={'offline_only':True,'target_distribution':targets,'food_upper_bound':max(targets.values())}
    result={'protocol':s['protocol'],'run':str(args.run.resolve()),'stage':s['settings']['stage'],
        'evaluations':{k:{'food':v['food'],'n':v['n'],'return':v['return']} for k,v in evaluations.items()},
        'comparisons':results,'best_constant_action_analysis':best_constant,
        'reward_association_supported_in_this_pilot':all(r['cluster_bootstrap_95_ci'][0]>0 for r in results.values()),
        'ordinary_snake_goal_complete':False,
        'interpretation':'Developmental pilot only. Requires independent training seeds, ordinary multi-step growth and independent frozen evaluation.'}
    atomic_json(args.output,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
