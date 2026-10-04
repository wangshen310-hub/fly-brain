"""Actual-reward comparisons of static native synaptic candidates.

All proposals are evaluated by full-brain spiking and genuine Snake events.
No action label is supplied to the learner. Adjacent one-step is a development
course; ordinary multi-step games and larger boards remain separate gates.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import base64
import hashlib
from io import BytesIO
import json
import os
import sys
import time
import traceback
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flysnake.game import DIRECTIONS
from flysnake.lab.experiment import atomic_json
from flysnake.research.paired import writer_lock,digest
from flysnake.research.projection import CoveredGame
from flysnake.research.olfaction import OdorConfig,odor_samples
from flysnake.research.sensory_basis import VERSION,BasisConfig,NativeSensoryBasisBrain,choose_reward_candidate

ARMS = ('learning','frozen','shuffled')


def pose_split(size):
    training,heldout = [],[]
    for heading,(dx,dy) in enumerate(DIRECTIONS):
        poses = []
        for x in range(size):
            for y in range(size):
                body = [(x-i*dx,y-i*dy) for i in range(3)]
                if all(0<=a<size and 0<=b<size for a,b in body):
                    poses.append({'heading':heading,'body':body})
        indices = np.random.default_rng(930401+heading).permutation(len(poses))
        n = max(2,len(poses)//4)
        heldout.extend(poses[i] for i in indices[:n])
        training.extend(poses[i] for i in indices[n:])
    return training,heldout


def food_sites(pose,size,stage):
    x,y = pose['body'][0]
    available = ([(x+dx,y+dy) for dx,dy in DIRECTIONS] if stage!='uniform_food'
                 else [(a,b) for b in range(size) for a in range(size)])
    return [p for p in available if 0<=p[0]<size and 0<=p[1]<size and p not in pose['body']]


class SensoryGame(CoveredGame):
    def __init__(self,*,stage,**kwargs):
        self.stage = stage
        super().__init__(**kwargs)

    def _sample_food(self):
        if self.stage!='uniform_food':
            available = food_sites({'heading':self.direction,'body':self.snake},self.size,self.stage)
            if available:return self._rng.choice(available)
        return super()._sample_food()


def png(rgb):
    stream=BytesIO();Image.fromarray(rgb).save(stream,'PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def episode(brain,scenario,*,stage,size,limit,publish=None):
    game = SensoryGame(seed=scenario['seed'],size=size,stage=stage,geometry=brain.geometry,
                       calibration=brain.brain.calibration)
    game.snake=[tuple(p) for p in scenario['pose']['body']]
    game.direction=scenario['pose']['heading'];game.food=tuple(scenario['food'])
    brain.reset_trial()
    rows=[];value=0.;no_output=0
    for window in range(limit):
        rgb=game.render_observation()
        sense=odor_samples(game.snake[0],game.direction,game.food,brain.odor)
        if publish:
            publish(phase='neural_integration',image=png(rgb),game=game.snapshot(),sense=sense,
                    receipt=None,event=None,feedback=None)
        receipt=brain.observe(rgb,sense['concentration'])
        if receipt['observation_sha256']!=hashlib.sha256(rgb.tobytes()).hexdigest():
            raise RuntimeError('Actual neural pixels changed')
        receipt.update(episode_id=game.episode_id,step_id=game.step_id)
        event=feedback=None
        if receipt['action'] is None:
            no_output+=1;value-=.1;feedback=brain.feedback('aversive')
        else:
            event=game.apply_receipt(receipt)
            if not event['accepted']:
                raise RuntimeError('Actual native receipt rejected: '+str(event))
            if event['kind']!='none':
                feedback=brain.feedback(event['kind'])
                value+=1. if event['kind']=='reward' else -1.
                game.complete_feedback(event['event_id'])
        row={'receipt':receipt,'event':event,'feedback':feedback,'sense':sense,'window':window}
        rows.append(row)
        if publish:
            publish(phase='native_action',image=png(rgb),game=game.snapshot(),sense=sense,
                    receipt=receipt,event=event,feedback=feedback)
        if game.terminal or (stage=='adjacent_one_step' and game.score>0):break
    deadline=None
    if not game.terminal and not (stage=='adjacent_one_step' and game.score>0):
        value-=1.
        # A deadline is an experiment outcome, not a collision/death. Deliver
        # it exactly once. A previous food/no-output stimulus consumed last.
        if brain.brain.last is None:
            rgb=game.render_observation()
            sense=odor_samples(game.snake[0],game.direction,game.food,brain.odor)
            # This additional window exists only to receive feedback. It is
            # explicitly nonmotor and no action is applied to the environment.
            feedback_receipt=brain.observe(rgb,sense['concentration'])
            deadline={'receipt':feedback_receipt,'motor_applied':False}
        else:deadline={}
        deadline['feedback']=brain.feedback('aversive')
    return {'seed':scenario['seed'],'pose':scenario['pose'],'food_start':scenario['food'],
        'size':size,'stage':stage,'limit':limit,'food':game.score,'steps':game.step_id,
        'windows':len(rows),'no_output':no_output,'return':value,
        'reason':game.death_reason or ('food' if stage=='adjacent_one_step' and game.score else 'window_expired'),
        'rows':rows,'deadline':deadline}


def assessment(brain,coefficients,scenarios,args,publish):
    brain.set_parameters(coefficients)
    before=brain.weight_hash()
    rows=[episode(brain,s,stage=args.stage,size=args.size,limit=args.limit,publish=publish) for s in scenarios]
    if brain.weight_hash()!=before:raise RuntimeError('Frozen evaluation changed native synapses')
    return {'food':sum(r['food'] for r in rows),'n':len(rows),'return':sum(r['return'] for r in rows),
        'success_episodes':sum(r['food']>0 for r in rows),'memory_sha256':before,'episodes':rows}


def sources():
    paths=[Path(__file__),ROOT/'flysnake/research/sensory_basis.py',ROOT/'flysnake/research/olfaction.py',
           ROOT/'flysnake/research/projection.py',ROOT/'flysnake/research/paired.py',
           ROOT/'flysnake/research/vision.py',ROOT/'flysnake/research/operating.py',
           ROOT/'flysnake/lab/brain.py',ROOT/'flysnake/lab/environment.py',ROOT/'flysnake/lab/protocol.py',
           ROOT/'flysnake/game.py',ROOT/'flysnake/neural.py',ROOT/'vendor/stonkfly/neural/kernel.cpp']
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save_memory(output,report,memory,rng,brain):
    name=f'memory-{report["cycles_completed"]:04d}-{time.time_ns()}.npz'
    weights={}
    for arm,coefficients in memory.items():
        brain.set_parameters(coefficients);weights[arm]=brain.weight_hash()
    meta={'version':VERSION,'identity':brain.identity,'sources':report['sources'],
          'settings':report['settings'],'rng':rng.bit_generator.state,'cycles_completed':report['cycles_completed'],
          'weights':weights}
    with (output/name).open('xb') as stream:
        np.savez_compressed(stream,metadata=json.dumps(meta,sort_keys=True),**memory)
        stream.flush();os.fsync(stream.fileno())
    report['checkpoints'].append({'file':name,'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest(),
        'cycles_completed':report['cycles_completed'],'parameters_sha256':{a:digest(c) for a,c in memory.items()},
        'weights_sha256':weights})
    atomic_json(output/'status.json',report)


def restore_memory(output,report,brain):
    latest=report['checkpoints'][-1];path=output/latest['file']
    if Path(latest['file']).name!=latest['file'] or hashlib.sha256(path.read_bytes()).hexdigest()!=latest['sha256']:
        raise ValueError('Memory checkpoint filename/hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        if (meta['version']!=VERSION or meta['identity']!=json.loads(json.dumps(brain.identity))
                or meta['sources']!=report['sources'] or meta['settings']!=report['settings']
                or meta['cycles_completed']!=latest['cycles_completed']):
            raise ValueError('Memory graph, sensory basis, source or configuration mismatch')
        memory={arm:z[arm].copy() for arm in ARMS}
    rng=np.random.default_rng();rng.bit_generator.state=meta['rng']
    for arm,coefficients in memory.items():
        if digest(coefficients)!=latest['parameters_sha256'][arm]:
            raise ValueError('Parameter hash mismatch')
        brain.set_parameters(coefficients)
        if brain.weight_hash()!=meta['weights'][arm] or brain.weight_hash()!=latest['weights_sha256'][arm]:
            raise ValueError('Effective native weight hash mismatch')
    # Work after an interrupted cycle must be regenerated from the committed
    # RNG and parameters, rather than treating in-progress rows as committed.
    committed=meta['cycles_completed']
    report['cycles']=report['cycles'][:committed];report['cycles_completed']=committed
    return memory,rng


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=930501)
    p.add_argument('--size',type=int,choices=(4,6,8,12),default=4)
    p.add_argument('--stage',choices=('adjacent_one_step','nearby_food','uniform_food'),default='adjacent_one_step')
    p.add_argument('--limit',type=int,default=1)
    p.add_argument('--cycles',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=6)
    p.add_argument('--evaluation-n',type=int,default=12)
    p.add_argument('--proposal-step',type=float,default=1.)
    p.add_argument('--rank',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.limit,args.cycles,args.batch_size,args.evaluation_n,args.proposal_step)<=0:
        raise ValueError('Positive experiment sizes required')
    if args.stage=='adjacent_one_step' and args.limit!=1:
        raise ValueError('Explicit one-decision curriculum requires limit=1')
    config=BasisConfig(rank=args.rank);odor=OdorConfig()
    settings={k:v for k,v in vars(args).items() if k not in ('output','resume')}
    settings.update(config=asdict(config),odor=asdict(odor))
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        source=sources()
        if path.exists():
            if not args.resume:raise ValueError('Existing memory preserved; use --resume or another directory')
            report=json.loads(path.read_text(encoding='utf-8'))
            if report['sources']!=source or report['settings']!=settings or report['protocol']!=VERSION:
                raise ValueError('Cannot resume changed source or settings')
            if report['status']=='complete':
                print('already_complete',flush=True);return
        else:
            report={'protocol':VERSION,'status':'loading','sources':source,'settings':settings,
                    'cycles_completed':0,'cycles':[],'checkpoints':[],'before':None,'after':{},
                    'basis':None,'started_at':time.time(),'behavioral_learning_proven':False}
            for relative in source:
                target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/relative).read_bytes())
            atomic_json(path,report)
        public={'protocol':VERSION,'settings':settings,'learning_proven':False}
        def publish(**fields):
            public.update(fields);public['updated_at']=time.time();public['process_id']=os.getpid()
            atomic_json(args.output/'current.json',public)
        publish(phase='loading',message='加载完整连接组与静态感官接口')
        try:
            brain=NativeSensoryBasisBrain(size=args.size,config=config,odor=odor)
            training,heldout=pose_split(args.size)
            evaluation_rng=np.random.default_rng(args.seed+777000)
            scenarios=[]
            if args.stage=='adjacent_one_step':
                for pose in heldout:
                    for food in food_sites(pose,args.size,args.stage):
                        scenarios.append({'seed':args.seed+888000+len(scenarios),'pose':pose,'food':food})
            else:
                for i in range(args.evaluation_n):
                    pose=heldout[int(evaluation_rng.integers(len(heldout)))]
                    food=food_sites(pose,args.size,args.stage)
                    scenarios.append({'seed':args.seed+888000+i,'pose':pose,'food':food[int(evaluation_rng.integers(len(food)))]})
            if report['basis'] is None:
                publish(phase='label_free_calibration',message='仅测量感官输入引起的实际神经活动，无奖惩或动作标签')
                baseline_rng=np.random.default_rng(930460)
                frames=[]
                for i in range(32):
                    pose=training[int(baseline_rng.integers(len(training)))]
                    game=SensoryGame(seed=930460+i,size=args.size,stage=args.stage,geometry=brain.geometry)
                    game.snake=pose['body'].copy();game.direction=pose['heading']
                    food=food_sites(pose,args.size,args.stage)
                    game.food=food[int(baseline_rng.integers(len(food)))]
                    sense=odor_samples(game.snake[0],game.direction,game.food,odor)
                    frames.append((game.render_observation(),sense['concentration']))
                measured=brain.fit_basis(frames)
                name=f'basis-{time.time_ns()}.npz';sha=brain.save_basis(args.output/name)
                report['basis']={'file':name,'sha256':sha,'measured':measured}
                report['brain']=brain.identity;atomic_json(path,report)
            else:
                basis=report['basis'];brain.restore_basis(args.output/basis['file'],basis['sha256'])
            if report['checkpoints']:
                memory,rng=restore_memory(args.output,report,brain)
            else:
                memory={arm:np.zeros_like(brain.coefficients) for arm in ARMS}
                rng=np.random.default_rng(args.seed)
            if report['before'] is None:
                report.update(status='running',phase='initial_frozen_assessment');atomic_json(path,report)
                publish(experiment_phase='initial_frozen_assessment')
                report['before']=assessment(brain,memory['frozen'],scenarios,args,publish)
                save_memory(args.output,report,memory,rng,brain)
                print(json.dumps({'phase':'before','food':report['before']['food'],'n':report['before']['n']}),flush=True)
            heldout_hashes={r['rows'][0]['receipt']['observation_sha256'] for r in report['before']['episodes']}
            if heldout_hashes&{x['rgb'] for x in brain.identity['baseline_inputs']}:
                raise RuntimeError('Calibration and initial held-out poses overlap')
            for cycle in range(report['cycles_completed'],args.cycles):
                coordinate=int(rng.integers(len(brain.coefficients)))
                batch=[]
                for i in range(args.batch_size):
                    pose=training[int(rng.integers(len(training)))];food=food_sites(pose,args.size,args.stage)
                    batch.append({'seed':int(rng.integers(1,2**31)),'pose':pose,'food':food[int(rng.integers(len(food)))]})
                record={'cycle':cycle+1,'coordinate':coordinate,'arms':{}}
                for arm in ARMS:
                    candidates=np.tile(memory[arm],(3,1))
                    candidates[1,coordinate]+=args.proposal_step;candidates[2,coordinate]-=args.proposal_step
                    candidates=np.clip(candidates,-config.coefficient_bound,config.coefficient_bound)
                    if arm=='frozen':candidates[:]=memory[arm]
                    groups=[]
                    for candidate,coefficients in enumerate(candidates):
                        brain.set_parameters(coefficients);rows=[]
                        for i,scenario in enumerate(batch):
                            report.update(status='running',phase='actual_game_comparison',active_cycle=cycle+1,
                                          active_arm=arm,active_candidate=candidate,active_trial=i+1)
                            atomic_json(path,report)
                            publish(experiment_phase='training',cycle=cycle+1,arm=arm,candidate=candidate,
                                    trial=i+1,cycles_completed=report['cycles_completed'],
                                    memory_saved=True,checkpoints=len(report['checkpoints']))
                            row=episode(brain,scenario,stage=args.stage,size=args.size,limit=args.limit,publish=publish)
                            if args.stage=='adjacent_one_step' and row['rows'][0]['receipt']['observation_sha256'] in heldout_hashes:
                                raise RuntimeError('Training/held-out first-decision pixels overlap')
                            rows.append(row)
                        groups.append({'candidate':candidate,'coefficients':coefficients.tolist(),
                            'food':sum(r['food'] for r in rows),'return':float(np.mean([r['return'] for r in rows])),
                            'episodes':rows})
                    memory[arm],update=choose_reward_candidate(candidates,[g['return'] for g in groups],
                        arm=arm,shuffle_seed=args.seed+10000+cycle)
                    record['arms'][arm]={'groups':groups,'update':update}
                    print(json.dumps({'cycle':cycle+1,'arm':arm,'update':update}),flush=True)
                report['cycles'].append(record);report['cycles_completed']=cycle+1
                save_memory(args.output,report,memory,rng,brain)
                if (args.output/'STOP_AFTER_CYCLE').exists():
                    report.update(status='stopped',phase='committed_boundary');atomic_json(path,report)
                    publish(phase='stopped',message='已在完整比较边界保存记忆');return
            for arm in ARMS:
                report.update(phase='final_frozen_assessment',active_arm=arm);atomic_json(path,report)
                publish(experiment_phase='final_frozen_assessment',arm=arm)
                report['after'][arm]=assessment(brain,memory[arm],scenarios,args,publish)
                atomic_json(path,report)
                print(json.dumps({'phase':'after','arm':arm,'food':report['after'][arm]['food'],'n':report['after'][arm]['n']}),flush=True)
            before=report['before']['episodes'];frozen=report['after']['frozen']['episodes']
            for a,b in zip(before,frozen):
                if (any(a[k]!=b[k] for k in ('food','return','reason','steps'))
                        or [r['receipt']['counts'] for r in a['rows']]!=[r['receipt']['counts'] for r in b['rows']]):
                    raise RuntimeError('Frozen control did not reproduce initial native behavior')
            report.update(status='complete',phase='complete',completed_at=time.time())
            atomic_json(path,report);publish(phase='complete',message='本次有限实验结束；学习结论须看冻结对照')
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report)
            publish(phase='error',message='实验出现错误，已提交记忆保留',error=report['error']);raise


if __name__=='__main__':
    main()
