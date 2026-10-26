"""Finite teacher fit of INTERNAL full-graph parameters; never a live policy."""
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
from flysnake.research.teacher_spiking import SpikingConnectome, Dynamics, decode_spikes, file_hash, ACTIONS, STATE_ATOL


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def metrics(motor,labels,rows,margin=.2):
    """Score actual pulses; missing classes are unassessed, never a perfect BA."""
    motor=np.asarray(motor);labels=np.asarray(labels)
    if (labels.ndim!=1 or not len(labels) or not np.issubdtype(labels.dtype,np.integer)
            or not np.isin(labels,[0,1,2]).all() or motor.shape!=(len(labels),4)
            or len(rows)!=len(labels)):
        raise ValueError('Aligned nonempty motor counts, rows and three-class integer labels required')
    if (not np.issubdtype(motor.dtype,np.number) or np.iscomplexobj(motor)
            or not np.isfinite(motor).all() or (motor<0).any()
            or (motor>=2**63).any() or not np.equal(motor,np.floor(motor)).all()):
        raise ValueError('Actual finite nonnegative integer motor counts required; no rounding')
    predictions=[];receipts=[];eaten=0;adjacent=0
    for values,row in zip(motor,rows):
        counts={'left':[int(values[0])],'right':[int(values[1])],'forward':[int(v) for v in values[2:]]}
        receipt=decode_spikes(counts,margin);action=receipt['action']
        predictions.append(ACTIONS.index(action) if action is not None else -1)
        receipts.append(receipt)
        if row['adjacent_eat']:
            adjacent+=1;eaten+=int(action==row['teacher'])
    predictions=np.array(predictions)
    recall=[float(np.mean(predictions[labels==k]==k)) if np.any(labels==k) else None for k in range(3)]
    return {'accuracy':float(np.mean(predictions==labels)),
        'balanced_accuracy':float(np.mean(recall)) if all(x is not None for x in recall) else None,
        'class_coverage_complete':all(x is not None for x in recall),
        'recall':recall,'class_counts':np.bincount(labels,minlength=3).tolist(),
        'predicted_actions':[int(np.sum(predictions==k)) for k in range(3)],
        'silent':int(np.sum(predictions<0)),'adjacent_food_choices':{'eaten':eaten,'cases':adjacent},
        'best_constant_action_accuracy':float(np.bincount(labels,minlength=3).max()/len(labels)),
        'motor_counts':motor.tolist(),'receipts':receipts,
        'scope':'isolated short scenes; not ordinary-game score or multi-step learning'}


def two_window_gate(windows,window_ms):
    """Necessary numeric gate only, not proof of learning/generalization."""
    complete=(len(windows)==2 and all(m['class_coverage_complete'] for m in windows))
    duration_ok=type(window_ms) is int and window_ms==100
    passed=(complete and duration_ok
        and all(m['balanced_accuracy']>=.8 and all(r>=.75 for r in m['recall']) for m in windows))
    return {'passed':bool(passed),'required_windows':2,'required_window_ms':100,
        'minimum_balanced_accuracy_each_window':.8,'minimum_recall_each_class_each_window':.75,
        'all_classes_observed_each_window':bool(complete),
        'window_duration_matches':duration_ok,'full_brain_learning_confirmed':False,
        'scope':'Necessary two-window numeric gate only; no ordinary-game or generalization claim'}


def evaluate(b,encoded,labels,rows,batch=12,visual=True):
    """Frozen same-image two-window audit, with all returned state passed through.

    Top-level metrics remain the cold window for historical callers. The new
    report cannot turn a cold-only score into a passed continuous-learning gate.
    Training loss and checkpoint selection remain cold-window protocols.
    """
    if type(batch) is not int or batch<1 or encoded.ndim!=2 or encoded.shape[1]<1:
        raise ValueError('Positive batch and nonempty encoded scenes required')
    if len(labels)!=encoded.shape[1] or len(rows)!=encoded.shape[1]:
        raise ValueError('Encoded scenes, labels and metadata rows must align')
    result=[[],[]]
    with torch.no_grad():
        for start in range(0,encoded.shape[1],batch):
            light=encoded[:,start:start+batch];state=None
            for window in range(2):
                _,motor,state=b.integrate_encoded(light,state,visual_enabled=visual)
                b.validate_state(state)
                if state['ticks']!=(window+1)*b.config.window_ms:
                    raise ValueError('Continuous evaluation clock mismatch')
                result[window].append(motor.cpu().numpy().copy())
    windows=[metrics(np.concatenate(values),labels,rows,b.config.margin) for values in result]
    for i,m in enumerate(windows):
        m.update(start_tick=i*b.config.window_ms,end_tick=(i+1)*b.config.window_ms)
    return {**windows[0],'continuous_two_window':{
        'windows':windows,'gate':two_window_gate(windows,b.config.window_ms),
        'same_input_each_window':True,'state_reset_between_windows':False,
        'training_objective':'unchanged cold-window loss',
        'checkpoint_selection':'unchanged cold-window balanced accuracy'}}


def checked_optimizer_step(optimizer,parameters,max_norm=1.):
    """Reject a nonfinite gradient on EVERY update before optimizer mutation."""
    parameters=tuple(parameters)
    if not parameters or not any(p.grad is not None for p in parameters):
        raise ValueError('An optimizer update requires parameter gradients')
    if not np.isfinite(max_norm) or max_norm<=0:
        raise ValueError('Finite positive gradient norm bound required')
    norm=torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def compact(m):
    if isinstance(m,dict):return {k:compact(v) for k,v in m.items() if k not in ('motor_counts','receipts')}
    if isinstance(m,list):return [compact(v) for v in m]
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--run',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=1003)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--batch',type=int,default=3)
    p.add_argument('--lr',type=float,default=.03);p.add_argument('--window-ms',type=int,default=100)
    p.add_argument('--eval-every',type=int,default=25);p.add_argument('--shuffle-labels',action='store_true')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.batch%3 or args.batch<3:raise ValueError('Class-balanced batch must be a multiple of three')
    out=Path(args.run).resolve();out.mkdir(parents=True,exist_ok=args.resume)
    lock=(out/'writer.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous_status=None
    if args.resume:
        previous_status=json.loads((out/'status.json').read_text(encoding='utf-8'))
        if previous_status['status']=='complete':
            print(json.dumps({'status':'already_complete','summary':previous_status['summary']}),flush=True)
            return
    torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    config={k:v for k,v in vars(args).items() if k not in ('run','resume')}
    status={'status':'running','phase':'loading','process_id':os.getpid(),'config':config,
        'ordinary_game_learning_validated':False,'expert_available_at_inference':False,
        'source_sha256':file_hash(__file__),'progress':[]}
    if not args.resume:atomic_json(out/'status.json',status)
    start=time.perf_counter();b=SpikingConnectome.load_graph(args.graph,device=args.device,
        seed=args.seed,config=Dynamics(window_ms=args.window_ms))
    meta=json.loads((Path(args.dataset)/'manifest.json').read_text(encoding='utf-8'))
    data=np.load(Path(args.dataset)/'dataset.npz',allow_pickle=False)
    if file_hash(Path(args.dataset)/'dataset.npz')!=meta['dataset_sha256']:
        raise ValueError('Dataset integrity mismatch')
    status.update(brain=b.identity,graph_manifest=b.manifest,dataset_sha256=meta['dataset_sha256'],
        environment={'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None})
    snapshots=out/'sources';snapshots.mkdir(exist_ok=args.resume)
    for path in [Path(__file__), ROOT/'flysnake/research/teacher_spiking.py',
                 ROOT/'flysnake/research/optical_sampling.py', ROOT/'flysnake/research/teacher_data.py']:
        target=snapshots/path.name
        if target.exists() and file_hash(target)!=file_hash(path):raise ValueError('Preserve locked experiment source')
        if not target.exists():shutil.copyfile(path,target)
    train_light=b.encode(data['train_rgb']);validation_light=b.encode(data['validation_rgb'])
    labels=data['train_labels'].copy();validation_labels=data['validation_labels']
    if args.shuffle_labels:
        # Nonidentity reassignment is fixed and contains no correct labels at
        # training time. Validation never uses its labels as gradient input.
        before=labels.copy();rng.shuffle(labels)
        if np.array_equal(labels,before):labels=np.roll(labels,1)
        status['training_label_mismatch_count']=int(np.sum(labels!=before))
    strata=[np.flatnonzero(labels==i) for i in range(3)]
    if any(len(x)==0 for x in strata):raise ValueError('All three teacher action classes required')
    initial_parameters={k:v.detach().clone() for k,v in b.named_parameters()}
    initial_hash=b.parameter_hash()
    initial_validation=evaluate(b,validation_light,validation_labels,meta['rows']['validation'])
    if initial_validation['balanced_accuracy'] is None:
        raise ValueError('All three validation classes are required for checkpoint selection')
    if not args.resume:atomic_json(out/'initial-validation.json',initial_validation)
    status.update(initial_parameter_sha256=initial_hash,initial_validation=compact(initial_validation),
        parameter_count=sum(x.numel() for x in b.parameters()),phase='training')
    optimizer=torch.optim.Adam(b.parameters(),lr=args.lr)
    best=initial_validation['balanced_accuracy']
    best_checkpoint=({'path':str(out/'initial.pt'),'sha256':file_hash(out/'initial.pt'),
        'parameter_sha256':initial_hash} if args.resume else b.save_checkpoint(out/'initial.pt'))
    last_step=0
    if args.resume:
        if (previous_status['source_sha256']!=file_hash(__file__)
                or previous_status['initial_parameter_sha256']!=initial_hash
                or previous_status['brain']!=b.identity):
            raise ValueError('Training source/initial state changed; preserve saved memory')
        previous=json.loads((out/'committed.json').read_text(encoding='utf-8'))
        _,extra=b.restore_checkpoint(previous['checkpoint']['path'],previous['checkpoint']['sha256'])
        if extra['config']!=config or extra['dataset_sha256']!=meta['dataset_sha256']:
            raise ValueError('Incompatible training resume')
        optimizer.load_state_dict(extra['optimizer']);rng.bit_generator.state=json.loads(extra['rng_json'])
        last_step=extra['step'];best=extra['best'];best_checkpoint=extra['best_checkpoint']
        status['progress']=previous['progress']
    atomic_json(out/'status.json',status)
    print(json.dumps({'stage':'initial','seed':args.seed,'initial_validation':compact(initial_validation),
        'parameter_count':status['parameter_count']}),flush=True)
    for step in range(last_step+1,args.steps+1):
        tick=time.perf_counter();idx=np.concatenate([rng.choice(x,args.batch//3,replace=True) for x in strata]);rng.shuffle(idx)
        optimizer.zero_grad(set_to_none=True)
        _,motor,_=b.integrate_encoded(train_light[:,idx])
        target=torch.as_tensor(labels[idx],device=motor.device)
        ce=torch.nn.functional.cross_entropy(b.logits(motor),target)
        silent=.05*torch.nn.functional.softplus(2.-motor.sum(dim=1)).mean()
        regularizer=sum((value-initial_parameters[name]).square().mean() for name,value in b.named_parameters())*1e-5
        loss=ce+silent+regularizer
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-graph teacher loss')
        loss.backward()
        if step==last_step+1:
            status['first_gradient']={name:{'nonzero':int(torch.count_nonzero(value.grad)),
                'finite':bool(torch.isfinite(value.grad).all()),'norm':float(value.grad.norm())} for name,value in b.named_parameters()}
            if not all(v['finite'] for v in status['first_gradient'].values()):raise FloatingPointError('Nonfinite internal gradient')
        gradient_norm=checked_optimizer_step(optimizer,b.parameters())
        entry={'step':step,'loss':float(loss.detach()),'teacher_loss':float(ce.detach()),
            'gradient_norm_before_clip':gradient_norm,
            'wall_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-start}
        status['progress'].append(entry)
        if step%args.eval_every==0 or step==args.steps:
            validation=evaluate(b,validation_light,validation_labels,meta['rows']['validation'])
            entry['validation']=compact(validation)
            if validation['balanced_accuracy']>best:
                best=validation['balanced_accuracy'];best_checkpoint=b.save_checkpoint(out/f'best-{step:04d}.pt')
                atomic_json(out/f'validation-{step:04d}.json',validation)
            extra={'step':step,'config':config,'dataset_sha256':meta['dataset_sha256'],
                'optimizer':optimizer.state_dict(),'rng_json':json.dumps(rng.bit_generator.state),
                'best':best,'best_checkpoint':best_checkpoint}
            checkpoint=b.save_checkpoint(out/f'memory-{step:04d}.pt',**extra)
            atomic_json(out/'committed.json',{'checkpoint':checkpoint,'progress':status['progress']})
        status.update(step=step,best_validation_balanced_accuracy=best,best_checkpoint=best_checkpoint,
            current_parameter_sha256=b.parameter_hash())
        atomic_json(out/'status.json',status)
        print(json.dumps({'stage':'training','seed':args.seed,**entry}),flush=True)
        del motor,loss,ce,silent,regularizer
    # The heldout test pixels and labels are first used only after training and
    # validation-based checkpoint selection have finished.
    b.restore_checkpoint(best_checkpoint['path'],best_checkpoint['sha256'])
    test_light=b.encode(data['test_rgb']);test_labels=data['test_labels']
    trained=evaluate(b,test_light,test_labels,meta['rows']['test'])
    vision_off=evaluate(b,test_light,test_labels,meta['rows']['test'],visual=False)
    selected_hash=b.parameter_hash()
    with torch.no_grad():
        for name,value in b.named_parameters():value.copy_(initial_parameters[name])
    before=evaluate(b,test_light,test_labels,meta['rows']['test'])
    b.restore_checkpoint(best_checkpoint['path'],best_checkpoint['sha256'])
    summary={'trained':compact(trained),'initial_or_parameter_reset':compact(before),'vision_off':compact(vision_off),
        'selected_parameter_sha256':selected_hash,'initial_parameter_sha256':initial_hash,
        'short_scene_improvement':(trained['balanced_accuracy']-before['balanced_accuracy']
            if trained['balanced_accuracy'] is not None and before['balanced_accuracy'] is not None else None),
        'ordinary_game_learning_validated':False,'training_seed_count':1,
        'interpretation':'Short-scene supervised fit through complete graph, not dopamine learning or normal-game success.'}
    for label,m in [('test-trained',trained),('test-reset',before),('test-vision-off',vision_off)]:atomic_json(out/(label+'.json'),m)
    atomic_json(out/'summary.json',summary)
    # Persist a nonzero state and its exact continuation for an independent
    # inference-only OS process; the verifier never imports this trainer.
    sequence=data['validation_rgb'][:4]
    state=None
    for frame in sequence[:2]:_,state=b.act(frame,state)
    replay=b.save_checkpoint(out/'replay.pt',state=state)
    expected=[];replay_states={}
    for frame in sequence[2:]:
        receipt,state=b.act(frame,state)
        hashes={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                if isinstance(v,torch.Tensor) else v for k,v in state.items()}
        expected.append({'receipt':receipt,'state':hashes})
        for name,value in state.items():
            if isinstance(value,torch.Tensor):replay_states[f'{len(expected)-1}_{name}']=value.cpu().numpy()
    np.savez_compressed(out/'replay-frames.npz',frames=sequence[2:])
    np.savez_compressed(out/'replay-states.npz',**replay_states)
    atomic_json(out/'replay.json',{'checkpoint':replay,'expected':expected,
        'expected_states_sha256':file_hash(out/'replay-states.npz'),
        'frames_sha256':file_hash(out/'replay-frames.npz'),'float_state_atol':STATE_ATOL,
        'scope':'Parameters byte exact; spike counts/timing exact; CUDA float continuation uses pre-training 1e-6 absolute tolerance.'})
    status.update(status='complete',phase='complete',summary=summary,wall_seconds=time.perf_counter()-start,
        parameter_sha256=b.parameter_hash(),peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    atomic_json(out/'status.json',status);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--run' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--run')+1])
            p=folder/'status.json'
            if p.exists():
                s=json.loads(p.read_text(encoding='utf-8'))
                s.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
                atomic_json(p,s)
        raise
