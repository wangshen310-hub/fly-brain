"""Reward-verified self-action eligibility in the full native connectome.

Exploration changes real motor-cell current. It never samples game actions.
Candidate validation uses fresh native games without exploration. Short-course
and ordinary Snake results are kept separate. Committed memories are atomic.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flysnake.lab.experiment import atomic_json
from flysnake.research.actor_credit import (VERSION,ActorCreditConfig,NativeActorCreditBrain,
    reward_proposal,accept_game_candidate,surrogate_probabilities)
from flysnake.research.paired import digest,writer_lock
from flysnake.research.olfaction import odor_samples
from train_sensory_basis import pose_split,food_sites,SensoryGame,png,sources as parent_sources

ARMS=('learning','frozen','shuffled')


def sources():
    result=parent_sources()
    for relative in ('scripts/train_actor_credit.py','flysnake/research/actor_credit.py'):
        result[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    return result


def prepare_baseline(source,output,size):
    """Copy only the unrewarded perception snapshot, never trained weights."""
    path=output/'perception-baseline.json'
    report=json.loads((source/'status.json').read_text(encoding='utf-8'))
    if report['settings']['size']!=size:raise ValueError('Unrewarded baseline board size differs')
    basis=report['basis'];file=source/basis['file']
    if Path(basis['file']).name!=basis['file'] or hashlib.sha256(file.read_bytes()).hexdigest()!=basis['sha256']:
        raise ValueError('Unrewarded baseline file/hash mismatch')
    with np.load(file,allow_pickle=False) as z:
        identity=json.loads(str(z['identity']))
        if not identity['basis_fitted'] or len(identity['baseline_inputs'])<16:
            raise ValueError('Unrewarded native perception calibration missing')
    manifest={'size':size,'config':report['settings']['config'],'odor':report['settings']['odor'],
        'file':basis['file'],'sha256':basis['sha256'],
        'copied_from':str(source.resolve()),'trained_parameters_imported':False,
        'calibration_reward':False,'calibration_action_labels':False}
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Existing baseline provenance differs')
        if hashlib.sha256((output/manifest['file']).read_bytes()).hexdigest()!=manifest['sha256']:
            raise ValueError('Saved perception baseline corrupted')
    else:
        shutil.copyfile(file,output/manifest['file']);atomic_json(path,manifest)
    return report


def episode(brain,scenario,*,stage,size,limit,noise=None,publish=None):
    game=SensoryGame(seed=scenario['seed'],size=size,stage=stage,geometry=brain.geometry,
                     calibration=brain.brain.calibration)
    game.snake=[tuple(p) for p in scenario['pose']['body']]
    game.direction=scenario['pose']['heading'];game.food=tuple(scenario['food'])
    noise=np.zeros((limit,2)) if noise is None else np.asarray(noise,float)
    if noise.shape!=(limit,2) or not np.isfinite(noise).all():raise ValueError('Matched native exploration required')
    brain.reset_trial();rows=[];traces=[];rewards=[];no_output=0
    for window in range(limit):
        rgb=game.render_observation();sense=odor_samples(game.snake[0],game.direction,game.food,brain.odor)
        if publish:publish(phase='neural_integration',image=png(rgb),game=game.snapshot(),sense=sense,
                           receipt=None,event=None,feedback=None)
        receipt,trace=brain.observe(rgb,sense['concentration'],noise[window])
        if receipt['observation_sha256']!=hashlib.sha256(rgb.tobytes()).hexdigest():
            raise RuntimeError('Actual native pixels differ')
        receipt.update(episode_id=game.episode_id,step_id=game.step_id)
        event=None;feedback=None;reward=0.;kind=None
        if receipt['action'] is None:
            no_output+=1;reward=-.1;kind='aversive'
        else:
            event=game.apply_receipt(receipt)
            if not event['accepted']:raise RuntimeError('Native receipt rejected: '+str(event))
            if event['kind']!='none':
                kind=event['kind'];reward=1. if kind=='reward' else -1.
        # A single-decision development course fails when food wasn't eaten.
        # Ordinary time truncation is NOT a collision or a fictitious death.
        if stage=='adjacent_one_step' and not game.terminal and game.score==0:
            reward-=1.;kind='aversive'
        if kind is not None:
            feedback=brain.feedback(kind)
            if event is not None and event['kind']!='none':game.complete_feedback(event['event_id'])
        rows.append({'window':window,'receipt':receipt,'event':event,'feedback':feedback,
                     'sense':sense,'reward':reward})
        traces.append(trace);rewards.append(reward)
        if publish:publish(phase='native_action',image=png(rgb),game=game.snapshot(),sense=sense,
                           receipt=receipt,event=event,feedback=feedback)
        if game.terminal or stage=='adjacent_one_step':break
    returns=discounted_returns(rewards,brain.actor_config.discount)
    row={'seed':scenario['seed'],'pose':scenario['pose'],'food_start':scenario['food'],
         'stage':stage,'size':size,'limit':limit,'food':game.score,'steps':game.step_id,
         'windows':len(rows),'no_output':no_output,'return':float(sum(rewards)),
         'reason':game.death_reason or ('food' if stage=='adjacent_one_step' and game.score else 'window_expired'),
         'rows':rows,'returns_to_go':returns.tolist(),'qualification':'actual native spikes; no game-level action fallback'}
    return row,np.asarray(traces),returns


def discounted_returns(rewards,discount):
    values=np.asarray(rewards,float)
    if values.ndim!=1 or not np.isfinite(values).all() or not 0<discount<=1:
        raise ValueError('Actual finite rewards and bounded discount required')
    result=np.empty_like(values);value=0.
    for i in range(len(values)-1,-1,-1):
        value=values[i]+discount*value;result[i]=value
    return result


def assessment(brain,parameters,scenarios,args,publish=None):
    brain.set_parameters(parameters);before=brain.weight_hash();rows=[]
    for scenario in scenarios:
        row,_,_=episode(brain,scenario,stage=args.stage,size=args.size,limit=args.limit,publish=publish)
        rows.append(row)
    if before!=brain.weight_hash():raise RuntimeError('Frozen evaluation mutated weights')
    return {'food':sum(r['food'] for r in rows),'n':len(rows),'return':sum(r['return'] for r in rows),
            'success_episodes':sum(r['food']>0 for r in rows),'memory_sha256':before,'episodes':rows,
            'exploration_off':True,'parameters_frozen':True}


def make_batch(rng,poses,size,stage,n):
    batch=[]
    for i in range(n):
        pose=poses[int(rng.integers(len(poses)))];sites=food_sites(pose,size,stage)
        batch.append({'seed':int(rng.integers(1,2**31)),'pose':pose,
                      'food':sites[int(rng.integers(len(sites)))]})
    return batch


def save_memory(output,report,memory,rng,brain):
    name=f'memory-{report["cycles_completed"]:04d}-{time.time_ns()}.npz';weights={}
    for arm,theta in memory.items():brain.set_parameters(theta);weights[arm]=brain.weight_hash()
    metadata={'version':VERSION,'identity':brain.identity,'sources':report['sources'],
              'settings':report['settings'],'cycles_completed':report['cycles_completed'],
              'rng':rng.bit_generator.state,'weights':weights}
    with (output/name).open('xb') as stream:
        np.savez_compressed(stream,metadata=json.dumps(metadata,sort_keys=True),**memory)
        stream.flush();os.fsync(stream.fileno())
    cp={'file':name,'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest(),
        'cycles_completed':report['cycles_completed'],'parameters_sha256':{a:digest(v) for a,v in memory.items()},
        'weights_sha256':weights}
    report['checkpoints'].append(cp);atomic_json(output/'status.json',report)


def restore_memory(output,report,brain):
    cp=report['checkpoints'][-1]
    if Path(cp['file']).name!=cp['file']:raise ValueError('Unsafe saved checkpoint path')
    path=output/cp['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=cp['sha256']:raise ValueError('Checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        expected=json.loads(json.dumps({'version':VERSION,'identity':brain.identity,'sources':report['sources'],
                                       'settings':report['settings'],'cycles_completed':cp['cycles_completed']}))
        if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint source or graph/configuration differs')
        memory={arm:z[arm].copy() for arm in ARMS}
    rng=np.random.default_rng();rng.bit_generator.state=meta['rng']
    for arm,theta in memory.items():
        if digest(theta)!=cp['parameters_sha256'][arm]:raise ValueError('Saved parameters differ')
        brain.set_parameters(theta)
        if brain.weight_hash()!=meta['weights'][arm] or brain.weight_hash()!=cp['weights_sha256'][arm]:
            raise ValueError('Saved effective weights differ')
    committed=meta['cycles_completed'];report['cycles']=report['cycles'][:committed]
    report['cycles_completed']=committed
    return memory,rng


def trial_signature(row):
    return {k:row[k] for k in ('seed','food','steps','windows','no_output','return','reason')}|{
        'counts':[r['receipt']['counts'] for r in row['rows']],
        'actions':[r['receipt']['action'] for r in row['rows']],
        'noise':[r['receipt']['motor_current_noise'] for r in row['rows']],
        'pixels':[r['receipt']['observation_sha256'] for r in row['rows']],
        'orn':[r['receipt']['olfactory_receptor_spikes'] for r in row['rows']],
        'eligibility':[r['receipt']['eligibility_sha256'] for r in row['rows']],
        'feedback':[r['feedback']['delivered_spikes'] if r['feedback'] else None for r in row['rows']]}


def cycle(brain,memory,rng,args,training,heldout_hashes,cycle_index,publish=None):
    batch=make_batch(rng,training,args.size,args.stage,args.batch_size)
    noise=rng.normal(0,brain.actor_config.noise_sigma,(len(batch),args.limit,2))
    record={'cycle':cycle_index+1,'batch':batch,'noise_sha256':digest(noise),'arms':{}}
    for arm in ARMS:
        brain.set_parameters(memory[arm]);experience=[];traces=[];returns=[]
        for i,scenario in enumerate(batch):
            if publish:publish(experiment_phase='exploration',cycle=cycle_index+1,arm=arm,candidate='体验',trial=i+1)
            row,t,g=episode(brain,scenario,stage=args.stage,size=args.size,limit=args.limit,noise=noise[i],publish=publish)
            if row['rows'][0]['receipt']['observation_sha256'] in heldout_hashes:
                raise RuntimeError('Training and held-out first-decision pixels overlap')
            experience.append(row);traces.extend(t);returns.extend(g)
        candidates,proposal=reward_proposal(memory[arm],traces,returns,arm=arm,
            seed=args.seed+100000+cycle_index,config=brain.actor_config)
        groups=[]
        for candidate,theta in enumerate(candidates):
            brain.set_parameters(theta);rows=[]
            for i,scenario in enumerate(batch):
                if publish:publish(experiment_phase='reward_verification',cycle=cycle_index+1,arm=arm,candidate=candidate,trial=i+1)
                row,_,_=episode(brain,scenario,stage=args.stage,size=args.size,limit=args.limit,publish=publish)
                rows.append(row)
            groups.append({'candidate':candidate,'parameters_sha256':digest(theta),
                          'food':sum(r['food'] for r in rows),'return':float(np.mean([r['return'] for r in rows])),
                          'episodes':rows})
        memory[arm],update=accept_game_candidate(candidates,[g['return'] for g in groups],arm=arm,
                                               seed=args.seed+200000+cycle_index)
        record['arms'][arm]={'experience':experience,'proposal':proposal,'groups':groups,'update':update}
        print(json.dumps({'cycle':cycle_index+1,'arm':arm,'experience_food':sum(r['food'] for r in experience),
                          'proposal':proposal,'update':update}),flush=True)
    return record


def native_credit_audit(brain,batch,amplitude=.12,publish=None):
    rows=[];zero=np.zeros_like(brain.coefficients)
    for scenario in batch:
        game=SensoryGame(seed=scenario['seed'],size=brain.geometry['board_cells'],stage='adjacent_one_step',geometry=brain.geometry)
        game.snake=[tuple(p) for p in scenario['pose']['body']];game.direction=scenario['pose']['heading'];game.food=tuple(scenario['food'])
        rgb=game.render_observation();sense=odor_samples(game.snake[0],game.direction,game.food,brain.odor)
        def probe(theta):
            brain.set_parameters(theta);brain.reset_trial();return brain.observe(rgb,sense['concentration'])
        baseline,trace=probe(zero);scale=float(np.max(abs(trace)))
        row={'seed':scenario['seed'],'pixels':baseline['observation_sha256'],'self_action':baseline['action'],
             'baseline_counts':baseline['counts'],'trace_nonzero':int(np.count_nonzero(trace))}
        if scale>0:
            vector=trace*amplitude/scale;plus,_=probe(vector);minus,_=probe(-vector)
            contrast=(plus['scores']['right']-plus['scores']['left'])-(minus['scores']['right']-minus['scores']['left'])
            desired=-1 if baseline['action']=='left' else 1 if baseline['action']=='right' else -np.sign(baseline['scores']['right']-baseline['scores']['left'])
            row.update(positive_counts=plus['counts'],negative_counts=minus['counts'],contrast=float(contrast),
                       self_action_slope_aligned=bool(contrast*desired>0),direction_sha256=digest(vector))
        rows.append(row)
        print(json.dumps({'phase':'credit_audit',**row}),flush=True)
    brain.set_parameters(zero)
    return {'rows':rows,'amplitude':amplitude,'aligned':sum(r.get('self_action_slope_aligned',False) for r in rows),
            'n':len(rows),'interpretation':'Local native causal audit; not a learning result or exact recurrent gradient'}


def parser():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--seed',type=int,default=930601)
    p.add_argument('--size',type=int,choices=(4,6,8,12),default=4)
    p.add_argument('--stage',choices=('adjacent_one_step','nearby_food','uniform_food'),default='adjacent_one_step')
    p.add_argument('--limit',type=int,default=1);p.add_argument('--cycles',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=6);p.add_argument('--evaluation-n',type=int,default=24)
    p.add_argument('--noise-sigma',type=float,default=3.);p.add_argument('--edge-step',type=float,default=.12)
    p.add_argument('--resume',action='store_true');return p


def main():
    args=parser().parse_args()
    if min(args.limit,args.cycles,args.batch_size,args.evaluation_n,args.noise_sigma,args.edge_step)<=0 or args.batch_size<2:
        raise ValueError('Positive experimental configuration required')
    if args.stage=='adjacent_one_step' and args.limit!=1:raise ValueError('Single-decision course requires limit=1')
    config=ActorCreditConfig(noise_sigma=args.noise_sigma,maximum_edge_step=args.edge_step)
    settings={k:v for k,v in vars(args).items() if k not in ('output','resume','baseline_run')}
    settings.update(actor_config=asdict(config),baseline_run=str(args.baseline_run.resolve()))
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json';source=sources();old=prepare_baseline(args.baseline_run,args.output,args.size)
        if path.exists():
            if not args.resume:raise ValueError('Preserve existing memory; use --resume')
            report=json.loads(path.read_text(encoding='utf-8'))
            if report['sources']!=source or report['settings']!=settings or report['protocol']!=VERSION:
                raise ValueError('Saved source/settings differ')
            if report['status']=='complete':print('already_complete',flush=True);return
        else:
            report={'protocol':VERSION,'status':'loading','sources':source,'settings':settings,'cycles_completed':0,
                    'cycles':[],'checkpoints':[],'before':None,'after':{},'credit_audit':None,'started_at':time.time(),
                    'behavioral_learning_proven':False,'ordinary_snake_goal_complete':False}
            for relative in source:
                target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
            atomic_json(path,report)
        public={'protocol':VERSION,'settings':settings,'learning_proven':False}
        def publish(**fields):
            public.update(fields);public['updated_at']=time.time();public['process_id']=os.getpid()
            atomic_json(args.output/'current.json',public)
        publish(phase='loading',message='加载原生连接组和无奖励感官基线')
        try:
            brain=NativeActorCreditBrain(args.output,config=config);report['brain']=brain.identity
            training,heldout=pose_split(args.size)
            if report['checkpoints']:memory,rng=restore_memory(args.output,report,brain)
            else:memory={a:np.zeros_like(brain.coefficients) for a in ARMS};rng=np.random.default_rng(args.seed)
            evaluation_rng=np.random.default_rng(args.seed+777000)
            if args.stage=='adjacent_one_step':
                scenarios=[{'seed':args.seed+888000+i,'pose':pose,'food':food} for i,(pose,food) in
                    enumerate((p,f) for p in heldout for f in food_sites(p,args.size,args.stage))]
            else:scenarios=make_batch(evaluation_rng,heldout,args.size,args.stage,args.evaluation_n)
            if report['credit_audit'] is None:
                batch=make_batch(np.random.default_rng(930602),training,args.size,args.stage,3)
                report['credit_audit']=native_credit_audit(brain,batch);atomic_json(path,report)
            if report['before'] is None:
                publish(experiment_phase='initial_frozen_assessment');report.update(status='running',phase='initial_frozen_assessment')
                report['before']=assessment(brain,memory['frozen'],scenarios,args,publish)
                if args.stage=='adjacent_one_step' and old['settings']['stage']==args.stage:
                    signatures=lambda ev:[(r['pose'],list(r['food_start']),r['food'],r['rows'][0]['receipt']['counts'],r['rows'][0]['receipt']['action']) for r in ev['episodes']]
                    if json.loads(json.dumps(signatures(report['before'])))!=json.loads(json.dumps(signatures(old['before']))):
                        raise RuntimeError('Untrained native baseline behavior changed')
                    report['native_baseline_matches_old_untrained']=True
                save_memory(args.output,report,memory,rng,brain)
                print(json.dumps({'phase':'before','food':report['before']['food'],'n':report['before']['n']}),flush=True)
            heldout_hashes={r['rows'][0]['receipt']['observation_sha256'] for r in report['before']['episodes']}
            if heldout_hashes&{x['rgb'] for x in brain.identity['baseline_inputs']}:raise RuntimeError('Unrewarded calibration overlaps heldout')
            for index in range(report['cycles_completed'],args.cycles):
                report.update(status='running',phase='actual_game_comparison',active_cycle=index+1);atomic_json(path,report)
                record=cycle(brain,memory,rng,args,training,heldout_hashes,index,publish)
                report['cycles'].append(record);report['cycles_completed']=index+1;save_memory(args.output,report,memory,rng,brain)
                publish(cycles_completed=index+1,memory_saved=True,checkpoints=len(report['checkpoints']))
                if (args.output/'STOP_AFTER_CYCLE').exists():
                    report.update(status='stopped',phase='committed_boundary');atomic_json(path,report)
                    publish(phase='stopped',message='完整比较已保存，后台训练已结束');return
            for arm in ARMS:
                publish(experiment_phase='final_frozen_assessment',arm=arm)
                report.update(phase='final_frozen_assessment',active_arm=arm);atomic_json(path,report)
                report['after'][arm]=assessment(brain,memory[arm],scenarios,args,publish);atomic_json(path,report)
                print(json.dumps({'phase':'after','arm':arm,'food':report['after'][arm]['food'],'n':report['after'][arm]['n']}),flush=True)
            if [trial_signature(r) for r in report['before']['episodes']]!=[trial_signature(r) for r in report['after']['frozen']['episodes']]:
                raise RuntimeError('Frozen native behavior changed')
            report.update(status='complete',phase='complete',completed_at=time.time());atomic_json(path,report)
            publish(phase='complete',message='有限机制实验完成；独立冻结成绩才是学习证据')
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report)
            publish(phase='error',message='实验出错，已提交记忆保留',error=report['error']);raise


if __name__=='__main__':main()
