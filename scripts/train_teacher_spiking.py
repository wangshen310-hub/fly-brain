"""Finite teacher fit of INTERNAL full-graph parameters; never a live policy."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flysnake.research.teacher_spiking import SpikingConnectome, Dynamics, decode_spikes, file_hash, ACTIONS, STATE_ATOL

LEGACY_PROTOCOL='legacy-cold-v1'
CONTINUOUS_PROTOCOL='continuous-two-window-v1'
GATE_ATOL=1e-12  # Floating representation tolerance, not percentage rounding.


def protocol_details(protocol,window_ms):
    if protocol not in (LEGACY_PROTOCOL,CONTINUOUS_PROTOCOL):
        raise ValueError('Unknown teacher training protocol')
    continuous=protocol==CONTINUOUS_PROTOCOL
    if continuous and (type(window_ms) is not int or window_ms!=100):
        raise ValueError('Continuous protocol requires exactly two 100 ms windows')
    return {'name':protocol,'training_windows':2 if continuous else 1,
        'training_objective':('mean of two continuous-window CE and silence losses' if continuous
            else 'legacy cold-window CE and silence loss'),
        'checkpoint_selection':('development-validation two-window gate; worst BA, minimum recall, mean BA'
            if continuous else 'legacy cold-window balanced accuracy'),
        'reset_between_training_batches':True,'detach_between_windows':False,
        'full_brain_learning_confirmed':False}


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


def _valid_three_class_metric(metric):
    """Selection inputs fail closed, including NaN and absent/inconsistent fields."""
    if not isinstance(metric,dict) or metric.get('class_coverage_complete') is not True:return False
    support=metric.get('class_counts');recall=metric.get('recall');ba=metric.get('balanced_accuracy')
    if (not isinstance(support,(list,tuple)) or len(support)!=3
            or any(type(v) is not int or v<=0 for v in support)
            or not isinstance(recall,(list,tuple)) or len(recall)!=3):return False
    values=[ba,*recall]
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):return False
    return math.isclose(ba,sum(recall)/3.,rel_tol=0.,abs_tol=1e-12)


def two_window_gate(windows,window_ms):
    """Necessary numeric gate only, not proof of learning/generalization."""
    complete=(isinstance(windows,(list,tuple)) and len(windows)==2
        and all(_valid_three_class_metric(m) for m in windows)
        and windows[0]['class_counts']==windows[1]['class_counts'])
    duration_ok=type(window_ms) is int and window_ms==100
    passed=(complete and duration_ok
        and all(m['balanced_accuracy']+GATE_ATOL>=.8
            and all(r+GATE_ATOL>=.75 for r in m['recall']) for m in windows))
    return {'passed':bool(passed),'required_windows':2,'required_window_ms':100,
        'minimum_balanced_accuracy_each_window':.8,'minimum_recall_each_class_each_window':.75,
        'floating_comparison_atol':GATE_ATOL,
        'all_classes_observed_each_window':bool(complete),
        'window_duration_matches':duration_ok,'full_brain_learning_confirmed':False,
        'scope':'Necessary two-window numeric gate only; no ordinary-game or generalization claim'}


def checkpoint_score(evaluation,protocol,window_ms):
    """Only gate-eligible continuous candidates get a selection rank.

    Development validation is used for selection; it is not a sealed test.
    None means unqualified, not a low score that may still be selected.
    """
    protocol_details(protocol,window_ms)
    if protocol==LEGACY_PROTOCOL:
        return evaluation['balanced_accuracy'] if _valid_three_class_metric(evaluation) else None
    if not isinstance(evaluation,dict):return None
    report=evaluation.get('continuous_two_window')
    if (not isinstance(report,dict) or report.get('same_input_each_window') is not True
            or report.get('state_reset_between_windows') is not False):return None
    windows=report.get('windows')
    if not two_window_gate(windows,window_ms)['passed']:return None
    if any(type(m.get('start_tick')) is not int or type(m.get('end_tick')) is not int
            or (m['start_tick'],m['end_tick'])!=(i*100,(i+1)*100) for i,m in enumerate(windows)):
        return None
    bas=[m['balanced_accuracy'] for m in windows]
    return (min(bas),min(r for m in windows for r in m['recall']),sum(bas)/2.)


def training_objective(b,light,labels,protocol=LEGACY_PROTOCOL):
    """Original losses, optionally over an untruncated differentiable 200 ms.

    State is passed directly: no detach, cloning, reset or fixed-boundary replay
    is inserted between windows. Labels enter only this training-only loss.
    """
    spec=protocol_details(protocol,b.config.window_ms)
    state=None;ces=[];silents=[]
    for window in range(spec['training_windows']):
        _,motor,state=b.integrate_encoded(light,state)
        b.validate_state(state)
        if state['ticks']!=(window+1)*b.config.window_ms:
            raise ValueError('Continuous training clock mismatch')
        ces.append(torch.nn.functional.cross_entropy(b.logits(motor),labels))
        silents.append(.05*torch.nn.functional.softplus(2.-motor.sum(dim=1)).mean())
    return {'teacher_loss':ces[0] if len(ces)==1 else torch.stack(ces).mean(),
        'silence_loss':silents[0] if len(silents)==1 else torch.stack(silents).mean(),
        'window_teacher_losses':ces,'window_silence_losses':silents}


def validate_training_resume(extra,config,dataset_sha256,source_sha256):
    if (extra.get('config')!=config or extra.get('dataset_sha256')!=dataset_sha256
            or extra.get('protocol_source_sha256')!=source_sha256):
        raise ValueError('Incompatible training resume protocol/source/dataset')
    protocol_details(config['protocol'],config['window_ms'])


def evaluate(b,encoded,labels,rows,batch=12,visual=True,protocol=LEGACY_PROTOCOL):
    """Frozen same-image two-window audit, with all returned state passed through.

    Top-level metrics remain the cold window for historical callers. The new
    report cannot turn a cold-only score into a passed continuous-learning gate.
    The training/selection protocol is explicit; this function never trains.
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
        'protocol':protocol_details(protocol,b.config.window_ms)}}


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
    p.add_argument('--protocol',choices=(LEGACY_PROTOCOL,CONTINUOUS_PROTOCOL),default=LEGACY_PROTOCOL,
        help='Continuous mode retains a 200 ms autograd graph; full-brain memory/runtime are not measured here')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    spec=protocol_details(args.protocol,args.window_ms)
    if args.batch%3 or args.batch<3:raise ValueError('Class-balanced batch must be a multiple of three')
    if args.protocol==CONTINUOUS_PROTOCOL and (args.steps<1 or args.eval_every<1):
        raise ValueError('Continuous protocol requires positive steps and evaluation interval')
    config={k:v for k,v in vars(args).items() if k not in ('run','resume')}
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
        if (previous_status.get('config')!=config
                or previous_status.get('source_sha256')!=file_hash(__file__)):
            raise ValueError('Incompatible training resume protocol/config/source')
        if previous_status['status']=='complete':
            print(json.dumps({'status':'already_complete','summary':previous_status['summary']}),flush=True)
            return
    torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    status={'status':'running','phase':'loading','process_id':os.getpid(),'config':config,
        'ordinary_game_learning_validated':False,'expert_available_at_inference':False,
        'source_sha256':file_hash(__file__),'protocol':spec,'progress':[]}
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
    initial_validation=evaluate(b,validation_light,validation_labels,meta['rows']['validation'],protocol=args.protocol)
    if initial_validation['balanced_accuracy'] is None:
        raise ValueError('All three validation classes are required for checkpoint selection')
    if not args.resume:atomic_json(out/'initial-validation.json',initial_validation)
    status.update(initial_parameter_sha256=initial_hash,initial_validation=compact(initial_validation),
        parameter_count=sum(x.numel() for x in b.parameters()),phase='training')
    optimizer=torch.optim.Adam(b.parameters(),lr=args.lr)
    best=checkpoint_score(initial_validation,args.protocol,args.window_ms)
    initial_checkpoint=({'path':str(out/'initial.pt'),'sha256':file_hash(out/'initial.pt'),
        'parameter_sha256':initial_hash} if args.resume else b.save_checkpoint(out/'initial.pt'))
    best_checkpoint=initial_checkpoint if best is not None else None
    selected_step=0 if best is not None else None
    last_step=0;last_memory_checkpoint=None
    if args.resume:
        if (previous_status['source_sha256']!=file_hash(__file__)
                or previous_status['initial_parameter_sha256']!=initial_hash
                or previous_status['brain']!=b.identity):
            raise ValueError('Training source/initial state changed; preserve saved memory')
        previous=json.loads((out/'committed.json').read_text(encoding='utf-8'))
        _,extra=b.restore_checkpoint(previous['checkpoint']['path'],previous['checkpoint']['sha256'])
        validate_training_resume(extra,config,meta['dataset_sha256'],file_hash(__file__))
        optimizer.load_state_dict(extra['optimizer']);rng.bit_generator.state=json.loads(extra['rng_json'])
        last_step=extra['step'];best=extra['best'];best_checkpoint=extra['best_checkpoint']
        selected_step=extra['selected_step'];last_memory_checkpoint=previous['checkpoint']
        status['progress']=previous['progress']
    atomic_json(out/'status.json',status)
    print(json.dumps({'stage':'initial','seed':args.seed,'initial_validation':compact(initial_validation),
        'parameter_count':status['parameter_count']}),flush=True)
    for step in range(last_step+1,args.steps+1):
        tick=time.perf_counter();idx=np.concatenate([rng.choice(x,args.batch//3,replace=True) for x in strata]);rng.shuffle(idx)
        optimizer.zero_grad(set_to_none=True)
        target=torch.as_tensor(labels[idx],device=train_light.device)
        terms=training_objective(b,train_light[:,idx],target,args.protocol)
        ce=terms['teacher_loss'];silent=terms['silence_loss']
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
            'window_teacher_losses':[float(v.detach()) for v in terms['window_teacher_losses']],
            'window_silence_losses':[float(v.detach()) for v in terms['window_silence_losses']],
            'gradient_norm_before_clip':gradient_norm,
            'wall_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-start}
        status['progress'].append(entry)
        if step%args.eval_every==0 or step==args.steps:
            validation=evaluate(b,validation_light,validation_labels,meta['rows']['validation'],protocol=args.protocol)
            entry['validation']=compact(validation)
            candidate=checkpoint_score(validation,args.protocol,args.window_ms)
            entry['checkpoint_eligible']=candidate is not None
            entry['checkpoint_score']=candidate
            if candidate is not None and (best is None or candidate>best):
                best=candidate;best_checkpoint=b.save_checkpoint(out/f'best-{step:04d}.pt');selected_step=step
                atomic_json(out/f'validation-{step:04d}.json',validation)
            extra={'step':step,'config':config,'dataset_sha256':meta['dataset_sha256'],
                'protocol_source_sha256':file_hash(__file__),'selected_step':selected_step,
                'optimizer':optimizer.state_dict(),'rng_json':json.dumps(rng.bit_generator.state),
                'best':best,'best_checkpoint':best_checkpoint}
            checkpoint=b.save_checkpoint(out/f'memory-{step:04d}.pt',**extra)
            last_memory_checkpoint=checkpoint
            atomic_json(out/'committed.json',{'checkpoint':checkpoint,'progress':status['progress']})
        status.update(step=step,best_validation_balanced_accuracy=best if args.protocol==LEGACY_PROTOCOL else None,
            best_validation_score=best,best_checkpoint=best_checkpoint,selected_step=selected_step,
            current_parameter_sha256=b.parameter_hash())
        atomic_json(out/'status.json',status)
        print(json.dumps({'stage':'training','seed':args.seed,**entry}),flush=True)
        del terms,loss,ce,silent,regularizer
    if best_checkpoint is None:
        # Keep the last recoverable update, optimizer/RNG and all diagnostics.
        # Never consume sealed test arrays for an unqualified candidate.
        summary={'protocol':spec,'selection_state':'no_eligible_checkpoint',
            'best_checkpoint':None,'selected_checkpoint_origin':None,'eligible_checkpoint_found':False,
            'last_memory_checkpoint':last_memory_checkpoint,'heldout_test_evaluated':False,
            'ordinary_game_learning_validated':False,'full_brain_learning_confirmed':False,
            'interpretation':'No development-validation candidate passed both windows and all six recalls.'}
        atomic_json(out/'summary.json',summary)
        status.update(status='complete',phase='complete',summary=summary,wall_seconds=time.perf_counter()-start,
            parameter_sha256=b.parameter_hash())
        atomic_json(out/'status.json',status);print(json.dumps(summary),flush=True)
        return
    # The heldout test pixels and labels are first used only after training and
    # validation-based checkpoint selection have finished.
    b.restore_checkpoint(best_checkpoint['path'],best_checkpoint['sha256'])
    test_light=b.encode(data['test_rgb']);test_labels=data['test_labels']
    trained=evaluate(b,test_light,test_labels,meta['rows']['test'],protocol=args.protocol)
    vision_off=evaluate(b,test_light,test_labels,meta['rows']['test'],visual=False,protocol=args.protocol)
    selected_hash=b.parameter_hash()
    with torch.no_grad():
        for name,value in b.named_parameters():value.copy_(initial_parameters[name])
    before=evaluate(b,test_light,test_labels,meta['rows']['test'],protocol=args.protocol)
    b.restore_checkpoint(best_checkpoint['path'],best_checkpoint['sha256'])
    summary={'trained':compact(trained),'initial_or_parameter_reset':compact(before),'vision_off':compact(vision_off),
        'protocol':spec,'eligible_checkpoint_found':True,'selected_training_step':selected_step,
        'selected_checkpoint_origin':'unchanged_initial' if selected_step==0 else 'trained_update',
        'selected_checkpoint_evaluation':compact(trained),
        'trained_field_is_legacy_alias_for_selected_checkpoint':True,
        'qualified_updated_checkpoint_selected':args.protocol==CONTINUOUS_PROTOCOL and selected_step>0,
        'heldout_test_evaluated':True,'full_brain_learning_confirmed':False,
        'selected_parameter_sha256':selected_hash,'initial_parameter_sha256':initial_hash,
        'short_scene_improvement':(trained['balanced_accuracy']-before['balanced_accuracy']
            if trained['balanced_accuracy'] is not None and before['balanced_accuracy'] is not None else None),
        'ordinary_game_learning_validated':False,'training_seed_count':1,
        'interpretation':('Unchanged initial checkpoint selected; no improved trained checkpoint was selected. '
            'No learning or ordinary-game success claim.' if selected_step==0 else
            'Selected supervised update; development-validation selection is not unbiased generalization '
            'or dopamine learning or normal-game success.')}
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
