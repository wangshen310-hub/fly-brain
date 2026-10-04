"""Actual native gamma1 conditioning, including cue-only and wrong-time controls.

This is a memory-mechanism diagnostic, not a game or behavioral success test.
The event-driven kernel is allowed to change only existing gamma1 KC->MBON11
edges. Actual receptor and DAN currents are declared physical interventions.
No score optimizer, action labels, direction lookup or synthetic neural counts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.research.input_isolation import IsolatedInputBrain,InputConfig
from flysnake.research.paired import writer_lock,digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as parent_sources,inputs

VERSION='native-gamma1-pairing-cue-only-wrong-time-controls-v1'
ARMS=('paired','frozen','wrong_time','cue_only')


def sources():
    result=parent_sources()
    for name in ('scripts/pilot_gamma1_conditioning.py','flysnake/research/input_isolation.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def assessment(brain):
    rows=[]
    for cue,rgb,concentration in inputs(brain):
        brain.reset_trial();receipt=brain.integrate(rgb,300,concentration,learning=False)
        rows.append({'cue':cue,'receipt':receipt})
    return rows


def trial(brain,arm,cue):
    rgb=np.full((180,320,3),85,np.uint8)
    concentrations=dict((name,concentration) for name,_,concentration in inputs(brain))
    zero=np.zeros((2,2));stim=[(brain.memory_readouts['PPL101'],brain.config.feedback_current_mV)]
    brain.reset_trial();before=brain.b.weight[brain.edges].copy()
    # Every arm receives 1.0s of the cue in a 1.5s trial. Frozen uses the
    # identical paired stimulus. Wrong-time US ends 300ms before cue onset.
    reinforce=cue=='DM1' and arm!='cue_only'
    if arm=='wrong_time' and reinforce:
        sequence=[('US_before_cue',200,zero,stim),('gap',300,zero,None),('cue',1000,concentrations[cue],None)]
    else:
        sequence=[('gap',300,zero,None),('cue_before_US',500,concentrations[cue],None),
                  ('cue_US' if reinforce else 'cue_no_US',200,concentrations[cue],stim if reinforce else None),
                  ('cue_after_US',300,concentrations[cue],None),('gap_end',200,zero,None)]
    rows=[]
    for label,duration,physical,stimulation in sequence:
        previous=brain.b.weight[brain.edges].copy()
        receipt=brain.integrate(rgb,duration,physical,stimulation=stimulation,learning=arm!='frozen')
        changed=brain.b.weight[brain.edges]-previous
        rows.append({'phase':label,'receipt':receipt,'changed_edges':int(np.count_nonzero(changed)),
                     'mean_fraction_change':float((changed/brain.base).mean())})
    fraction=brain.b.weight[brain.edges]/brain.base
    return {'arm':arm,'cue':cue,'rows':rows,'weight_before_sha256':digest(before),
            'weight_after_sha256':brain.weight_hash(),'changed_edges':int(np.count_nonzero(brain.b.weight[brain.edges]-before)),
            'mean_fraction':float(fraction.mean()),'minimum_fraction':float(fraction.min())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=1);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if not 1<=args.rounds<=8:raise ValueError('Finite mechanism pilot only')
    args.output.mkdir(parents=True,exist_ok=True)
    settings={'rounds':args.rounds,'input_config':{'vision_input':False,'lamina_bias_mV':0.,'zero_odor_is_silent':True},
              'paired_cue':'DM1','cue_ms':1000,'US_ms':200,'wrong_time_gap_ms':300,
              'retention_ms':1000,'order':[(a,c) for a in ARMS for _ in range(args.rounds) for c in ('DM1','DM4')]}
    settings=json.loads(json.dumps(settings))
    with writer_lock(args.output):
        path=args.output/'status.json';source=sources()
        if path.exists():
            if not args.resume:raise ValueError('Preserve old mechanism pilot; use --resume')
            report=json.loads(path.read_text(encoding='utf-8'))
            if report['sources']!=source or report['settings']!=settings:raise ValueError('Source or settings changed')
            if report['status']=='complete':print('already_complete');return
        else:
            report={'protocol':VERSION,'status':'loading','sources':source,'settings':settings,
                    'arms':{},'before':None,'after':{},'retained':{},'started_at':time.time(),
                    'behavioral_learning_proven':False,'ordinary_snake_goal_complete':False}
            for relative in source:
                target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/relative).read_bytes())
            atomic_json(path,report)
        report['process_id']=os.getpid()
        try:
            brain=IsolatedInputBrain(inputs=InputConfig(vision_input=False,lamina_bias_mV=0.,zero_odor_is_silent=True))
            report['brain']=brain.identity
            if report['before'] is None:
                report['before']=assessment(brain);atomic_json(path,report)
            for arm in ARMS:
                record=report['arms'].setdefault(arm,{'trials':[],'checkpoints':[]})
                if arm in report['retained']:continue
                brain.reset_memory()
                if record['checkpoints']:
                    cp=record['checkpoints'][-1]
                    meta=brain.restore(args.output/cp['file'],cp['sha256'])
                    if meta['state']!={'arm':arm,'completed_trials':len(record['trials']),'sources':source}:
                        raise ValueError('Trial boundary state mismatch')
                order=[cue for _ in range(args.rounds) for cue in ('DM1','DM4')]
                for index in range(len(record['trials']),len(order)):
                    report.update(status='running',phase='native_conditioning',active_arm=arm,active_trial=index+1)
                    atomic_json(path,report)
                    row=trial(brain,arm,order[index]);record['trials'].append(row)
                    name=f'memory-{arm}-{index+1:04d}-{time.time_ns()}.npz'
                    sha=brain.checkpoint(args.output/name,{'arm':arm,'completed_trials':index+1,'sources':source})
                    record['checkpoints'].append({'file':name,'sha256':sha,'memory_sha256':brain.weight_hash()})
                    atomic_json(path,report)
                    print(json.dumps({'arm':arm,'trial':index+1,'cue':order[index],
                        'changed_edges':row['changed_edges'],'mean_fraction':row['mean_fraction'],
                        'minimum_fraction':row['minimum_fraction']}),flush=True)
                report['after'][arm]=assessment(brain)
                brain.reset_trial();before=brain.weight_hash()
                brain.integrate(np.full((180,320,3),85,np.uint8),1000,learning=False)
                if before!=brain.weight_hash():raise RuntimeError('Retained native memory changed during frozen quiet period')
                report['retained'][arm]=assessment(brain)
                record['final_memory_sha256']=brain.weight_hash()
                atomic_json(path,report)
            if report['after']['frozen']!=report['before']:
                # Absolute experiment timestamps differ; compare actual biology.
                for old,new in zip(report['before'],report['after']['frozen']):
                    for key in ('counts','full_counts_sha256','memory_after','readouts'):
                        if old['receipt'][key]!=new['receipt'][key]:raise RuntimeError('Frozen native reference changed: '+key)
            report.update(status='complete',phase='complete',completed_at=time.time(),
                interpretation='Native LTD and stimulus-specificity diagnostic only; not learned Snake behavior.')
            atomic_json(path,report);print(json.dumps({'status':'complete'}),flush=True)
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report);raise


if __name__=='__main__':main()
