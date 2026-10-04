"""Bounded existing-afferent three-scene overfit sanity probe, never a game-learning claim.

Only training images of one physical pose are used. Pairwise motor constraints
cannot be improved by uniformly reducing both left/right pulse counts. A
nonidentity label permutation provides the matching false-teacher experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import Dynamics,file_hash,STATE_ATOL,ACTIONS
from flysnake.research.teacher_afferent import PlasticAfferentBrain as SpikingConnectome


def atomic(path,value):
    temp=path.with_suffix('.partial')
    with temp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    for attempt in range(10):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05*(attempt+1))


def observe(b,frames,physical,assigned):
    rows=[];pred=[]
    for frame in frames:
        receipt,_=b.act(frame);rows.append(receipt);pred.append(ACTIONS.index(receipt['action']) if receipt['action'] else -1)
    return {'receipts':rows,'assignment_correct':int(np.sum(np.array(pred)==assigned)),
        'physical_direction_correct':int(np.sum(np.array(pred)==physical)),
        'cases':3,'actions':[r['action'] for r in rows]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--warm-run',required=True);p.add_argument('--run',required=True)
    p.add_argument('--steps',type=int,default=60);p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--mismatch',action='store_true');args=p.parse_args()
    out=Path(args.run);out.mkdir(parents=True,exist_ok=False)
    checkpoint=json.loads((Path(args.warm_run)/'committed.json').read_text(encoding='utf-8'))['checkpoint']
    payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
    torch.set_num_threads(4);torch.manual_seed(1003)
    b=SpikingConnectome.load_graph(args.graph,config=Dynamics(**payload['identity']['dynamics']))
    b.warm_start_checkpoint(checkpoint['path'],checkpoint['sha256'])
    meta=json.loads((Path(args.dataset)/'manifest.json').read_text(encoding='utf-8'))
    if file_hash(Path(args.dataset)/'dataset.npz')!=meta['dataset_sha256']:raise ValueError('Dataset changed')
    data=np.load(Path(args.dataset)/'dataset.npz',allow_pickle=False)
    pose=meta['rows']['train'][0]['pose_key'];rows=meta['rows']['train']
    ids=[i for i,r in enumerate(rows) if r['pose_key']==pose and r['adjacent_eat']]
    ids=sorted(ids,key=lambda i:int(data['train_labels'][i]))
    physical=data['train_labels'][ids]
    if not np.array_equal(physical,[0,1,2]):raise ValueError('Same-pose left/forward/right physical triplet required')
    frames=data['train_rgb'][ids];assigned=np.roll(physical,1) if args.mismatch else physical.copy()
    order=np.argsort(assigned);light=b.encode(frames);target=torch.as_tensor(assigned,device=light.device)
    status={'status':'running','pid':os.getpid(),'source_sha256':file_hash(__file__),'config':vars(args),
        'brain':b.identity,'warm_checkpoint':checkpoint,'frame_sha256':[hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
        'train_indices':ids,'assigned_labels':assigned.tolist(),'physical_labels':physical.tolist(),
        'training_cases':3,'heldout_cases':0,'ordinary_game_learning_validated':False,
        'interpretation':'Optimization/conditional-output sanity check on TRAINING images only; fitting them is overfitting, not generalization.',
        'progress':[]}
    sources=out/'sources';sources.mkdir()
    for path in [Path(__file__),ROOT/'flysnake/research/teacher_spiking.py',ROOT/'flysnake/research/optical_sampling.py',ROOT/'flysnake/research/teacher_afferent.py']:
        shutil.copyfile(path,sources/path.name)
    initial=observe(b,frames,physical,assigned);status['initial']=initial;best=initial['assignment_correct']
    best_checkpoint=b.save_checkpoint(out/'initial.pt');trainable=[v for v in b.parameters() if v.requires_grad];optimizer=torch.optim.Adam(trainable,lr=args.lr,eps=1e-10)
    original={k:v.detach().clone() for k,v in b.named_parameters()};atomic(out/'status.json',status)
    for step in range(1,args.steps+1):
        optimizer.zero_grad(set_to_none=True);_,motor,_=b.integrate_encoded(light)
        difference=motor[:,1]-motor[:,0]
        # All three pixels differ only in food, so common-mode shifts cancel.
        rank=torch.nn.functional.softplus(2.-(difference[order[1]]-difference[order[0]]))+torch.nn.functional.softplus(2.-(difference[order[2]]-difference[order[1]]))
        ce=torch.nn.functional.cross_entropy(b.logits(motor),target)
        regularizer=sum((v-original[k]).square().mean() for k,v in b.named_parameters())*1e-5
        loss=rank+.1*ce+regularizer
        if step==1:
            grad=torch.autograd.grad(rank,trainable,retain_graph=True)
            status['rank_gradient']={k:{'norm':float(g.norm()),'finite':bool(torch.isfinite(g).all()),'nonzero':int(torch.count_nonzero(g))}
                for (k,v),g in zip([(k,v) for k,v in b.named_parameters() if v.requires_grad],grad)}
            if not all(v['finite'] for v in status['rank_gradient'].values()):raise FloatingPointError('Nonfinite conditional gradient')
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
        entry={'step':step,'pair_loss':float(rank.detach()),'CE':float(ce.detach())}
        if step%10==0 or step==args.steps:
            result=observe(b,frames,physical,assigned);entry['frozen_three_scene_evaluation']=result
            if result['assignment_correct']>best:
                best=result['assignment_correct'];best_checkpoint=b.save_checkpoint(out/f'best-{step:04d}.pt')
            point=b.save_checkpoint(out/f'memory-{step:04d}.pt',step=step,optimizer=optimizer.state_dict(),config=vars(args),
                protocol_source_sha256=file_hash(__file__),best_checkpoint=best_checkpoint)
            atomic(out/'committed.json',{'checkpoint':point,'step':step})
            print(json.dumps({'step':step,'pair_loss':entry['pair_loss'],'CE':entry['CE'],
                'assignment_correct':result['assignment_correct'],'physical_correct':result['physical_direction_correct'],'actions':result['actions']}),flush=True)
        status['progress'].append(entry);status['step']=step;atomic(out/'status.json',status)
        del loss,ce,rank,regularizer,motor
    b.restore_checkpoint(best_checkpoint['path'],best_checkpoint['sha256']);final=observe(b,frames,physical,assigned)
    off=[b.act(frame,visual_enabled=False)[0] for frame in frames]
    status.update(status='complete',best_checkpoint=best_checkpoint,final=final,visual_off=off,
        trained_parameter_sha256=b.parameter_hash(),fits_all_three_training_cases=final['assignment_correct']==3,
        ordinary_game_learning_validated=False)
    atomic(out/'status.json',status)
    # Finite nonzero-state continuation, separate from the overfit score.
    state=None
    for frame in frames[:2]:_,state=b.act(frame,state)
    point=b.save_checkpoint(out/'replay.pt',state=state)
    continuation=frames[[2,1]];expected=[];record={}
    for i,frame in enumerate(continuation):
        receipt,state=b.act(frame,state)
        hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() if isinstance(v,torch.Tensor) else v for k,v in state.items()}
        expected.append({'receipt':receipt,'state':hashes})
        for k,v in state.items():
            if isinstance(v,torch.Tensor):record[f'{i}_{k}']=v.cpu().numpy()
    np.savez_compressed(out/'replay-frames.npz',frames=continuation);np.savez_compressed(out/'replay-states.npz',**record)
    atomic(out/'replay.json',{'checkpoint':point,'expected':expected,'float_state_atol':STATE_ATOL,
        'frames_sha256':file_hash(out/'replay-frames.npz'),'expected_states_sha256':file_hash(out/'replay-states.npz')})
    print(json.dumps({'status':'complete','fits_all_three_training_cases':status['fits_all_three_training_cases'],
        'final_assignment_correct':final['assignment_correct'],'final_physical_correct':final['physical_direction_correct'],
        'heldout_test':False,'ordinary_game_learning_validated':False}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--run' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--run')+1])/'status.json'
            if path.exists():
                with path.open(encoding='utf-8') as f:status=json.load(f)
                status.update(status='failed',error={'type':type(error).__name__,'message':str(error)});atomic(path,status)
        raise
