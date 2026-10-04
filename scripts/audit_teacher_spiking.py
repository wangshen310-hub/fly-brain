"""Actual complete-graph causality/timing/numerical audit before training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import SpikingConnectome, Dynamics, file_hash, STATE_ATOL


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def states(state):
    return {k:tensor_hash(v) if isinstance(v,torch.Tensor) else v for k,v in state.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--window-ms',type=int,default=100)
    p.add_argument('--trained-run',help='Optional already selected frozen parameter checkpoint')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    status={'phase':'loading','status':'running','process_id':os.getpid(),'game_played':False,'learning':False}
    def save():
        q=out/'status.json';t=q.with_suffix('.partial');t.write_text(json.dumps(status,indent=2),encoding='utf-8');t.replace(q)
    save();torch.set_num_threads(4)
    start=time.perf_counter();b=SpikingConnectome.load_graph(args.graph,device=args.device,config=Dynamics(window_ms=args.window_ms))
    if args.trained_run:
        trained=json.loads((Path(args.trained_run)/'status.json').read_text(encoding='utf-8'))
        if trained['status']!='complete':raise ValueError('Use a completed selected training run')
        checkpoint=trained['best_checkpoint'];b.restore_checkpoint(checkpoint['path'],checkpoint['sha256'])
        status['frozen_trained_checkpoint']=checkpoint
    status.update(brain=b.identity,neurons=b.n,anatomical_edges=b.manifest['anatomical_edges'],
        environment={'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None},load_seconds=time.perf_counter()-start)
    print(json.dumps({k:status[k] for k in ['neurons','anatomical_edges','load_seconds','environment']}),flush=True)
    d=np.load(Path(args.dataset)/'dataset.npz',allow_pickle=False)
    meta=json.loads((Path(args.dataset)/'manifest.json').read_text(encoding='utf-8'))
    if file_hash(Path(args.dataset)/'dataset.npz')!=meta['dataset_sha256']:
        raise ValueError('Dataset integrity mismatch')
    rows=meta['rows']['train'];pose=rows[0]['pose_key']
    ix=[i for i,row in enumerate(rows) if row['pose_key']==pose and row['adjacent_eat']]
    if len(ix)<2:raise ValueError('Physical same-pose food counterfactuals required')
    frames=[d['train_rgb'][i] for i in ix]
    results=[];state_records=[];array_records={}
    for label,frame,enabled in [(f'food_{k}',v,True) for k,v in enumerate(frames)]+[
            ('same_image_repeat',frames[0],True),('visual_off_A',frames[0],False),('visual_off_B',frames[1],False)]:
        t=time.perf_counter();receipt,s=b.act(frame,visual_enabled=enabled)
        results.append(dict(condition=label,visual_enabled=enabled,wall_seconds=time.perf_counter()-t,**receipt))
        state_records.append(states(s))
        for name,value in s.items():
            if isinstance(value,torch.Tensor):array_records[label+'_'+name]=value.cpu().numpy()
        status.update(phase='conditions',rows=results);save()
        print(json.dumps({'condition':label,'action':receipt['action'],'counts':receipt['counts'],
            'total_spikes':receipt['actual_total_spikes'],'voltage_min':receipt['voltage_min'],
            'voltage_max':receipt['voltage_max'],'wall_seconds':results[-1]['wall_seconds']}),flush=True)
    repeat=len(frames);off=repeat+1
    equality=lambda a,z:state_records[a]==state_records[z] and results[a]['full_counts_sha256']==results[z]['full_counts_sha256']
    def numeric(a,z):
        names=('v','ge','gi','adaptation','refractory','release_queue')
        diff={name:float(np.max(np.abs(array_records[results[a]['condition']+'_'+name]-array_records[results[z]['condition']+'_'+name]))) for name in names}
        return {'max_abs':diff,'integer_spikes_exact':results[a]['full_counts_sha256']==results[z]['full_counts_sha256'],
                'float_state_within_preregistered_atol':all(v<=STATE_ATOL for k,v in diff.items() if k!='refractory'),
                'refractory_exact':diff['refractory']==0,'float_state_atol':STATE_ATOL}
    repeat_numeric=numeric(0,repeat);off_numeric=numeric(off,off+1)
    motor=b.motor.cpu().numpy()
    voltage_diff=max(float(np.max(np.abs(array_records[f'food_{i}_v'][motor]-array_records['food_0_v'][motor]))) for i in range(1,len(frames)))
    pulse_diff=any(r['counts']!=results[0]['counts'] for r in results[1:len(frames)])
    status.update(status='complete',phase='complete',state_hashes=state_records,
        checks={'same_image_exact':equality(0,repeat),'visual_off_exact':equality(off,off+1),
            'same_image_numeric_replay':repeat_numeric,'visual_off_numeric_replay':off_numeric,
            'finite_voltage_all':all(np.isfinite(v).all() for v in array_records.values()),
            'food_changes_motor_voltage':voltage_diff>1e-7,'food_changes_motor_counts':pulse_diff},
        motor_maximum_voltage_difference=voltage_diff,parameter_sha256=b.parameter_hash(),
        wall_seconds=time.perf_counter()-start,
        interpretation='Frozen engineering dynamics audit only; no training or ordinary Snake success.')
    np.savez_compressed(out/'actual_arrays.npz',**array_records)
    status['arrays_sha256']=file_hash(out/'actual_arrays.npz');save()
    print(json.dumps(status['checks']),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'status.json'
            if path.exists():
                value=json.loads(path.read_text(encoding='utf-8'))
                value.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
                path.write_text(json.dumps(value,indent=2),encoding='utf-8')
        raise
