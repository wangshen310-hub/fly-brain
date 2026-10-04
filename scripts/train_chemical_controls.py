"""Scalar-reward search of native chemical-response synaptic coordinates.

Each pose is experienced with every legal adjacent food placement. The learner
receives real game returns, not food directions, optimal actions or paths.
Ordinary multi-step Snake remains a separate stage and validation gate.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_actor_credit as engine
from flysnake.research.chemical_basis import VERSION,ChemicalConfig,NativeChemicalBasisBrain,synaptic_candidates
from flysnake.research.actor_credit import accept_game_candidate
from flysnake.research.pulse_balance import PulseBalancedGame
from flysnake.research.paired import digest,writer_lock
from flysnake.lab.experiment import atomic_json

engine.VERSION=VERSION
engine.SensoryGame=PulseBalancedGame


def sources():
    result=engine.sources()
    for relative in ('scripts/train_chemical_controls.py','flysnake/research/chemical_basis.py',
                     'flysnake/research/pulse_balance.py'):
        result[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    return result


def grouped_worlds(rng,poses,size,stage,per_heading):
    """Environment construction only; no labels enter synaptic proposals."""
    if type(per_heading) is not int or per_heading<1:raise ValueError('Positive pose count required')
    batch=[]
    for heading in range(4):
        pool=[p for p in poses if p['heading']==heading]
        if len(pool)<per_heading:raise ValueError('Insufficient independent training poses')
        for index in rng.choice(len(pool),per_heading,replace=False):
            pose=pool[int(index)]
            for food in engine.food_sites(pose,size,stage):
                batch.append({'seed':int(rng.integers(1,2**31)),'pose':pose,'food':food})
    return batch


def calibration_frames(brain,poses):
    frames=[]
    for heading in range(4):
        pose=next(p for p in poses if p['heading']==heading)
        game=PulseBalancedGame(seed=930701+heading,size=brain.geometry['board_cells'],
                               stage='adjacent_one_step',geometry=brain.geometry)
        game.snake=[tuple(p) for p in pose['body']];game.direction=heading;game.food=None
        frames.append(game.render_observation())
    return frames


def cycle(brain,memory,rng,args,poses,heldout_hashes,index,publish=None):
    batch=grouped_worlds(rng,poses,args.size,args.stage,args.poses_per_heading)
    record={'cycle':index+1,'batch':batch,'arms':{},'motor_exploration':False}
    for arm in engine.ARMS:
        candidates,proposal=synaptic_candidates(memory[arm],index,args.synaptic_step,
                                                brain.chemical.coefficient_bound)
        if arm=='frozen':candidates[:]=memory[arm]
        groups=[]
        for candidate,theta in enumerate(candidates):
            brain.set_parameters(theta);rows=[]
            for i,scenario in enumerate(batch):
                if publish:publish(experiment_phase='reward_verification',cycle=index+1,
                                   arm=arm,candidate=candidate,trial=i+1,trials=len(batch))
                row,_,_=engine.episode(brain,scenario,stage=args.stage,size=args.size,
                                       limit=args.limit,publish=publish)
                if row['rows'][0]['receipt']['observation_sha256'] in heldout_hashes:
                    raise RuntimeError('Training and heldout first-decision pixels overlap')
                rows.append(row)
            groups.append({'candidate':candidate,'parameters_sha256':digest(theta),
                          'weights_sha256':brain.weight_hash(),
                          'food':sum(r['food'] for r in rows),
                          'return':float(np.mean([r['return'] for r in rows])),
                          'episodes':rows})
        memory[arm],update=accept_game_candidate(candidates,[g['return'] for g in groups],
                                               arm=arm,seed=args.seed+200000+index)
        record['arms'][arm]={'proposal':proposal,'groups':groups,'update':update}
        print(json.dumps({'cycle':index+1,'arm':arm,'worlds':len(batch),'update':update}),flush=True)
    return record


def parser():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--seed',type=int,default=930701)
    p.add_argument('--size',type=int,choices=(4,6,8,12),default=4)
    p.add_argument('--stage',choices=('adjacent_one_step','nearby_food','uniform_food'),default='adjacent_one_step')
    p.add_argument('--limit',type=int,default=1);p.add_argument('--cycles',type=int,default=16)
    p.add_argument('--poses-per-heading',type=int,default=1)
    p.add_argument('--synaptic-step',type=float,default=1.)
    p.add_argument('--evaluation-n',type=int,default=24);p.add_argument('--resume',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if (min(args.limit,args.cycles,args.poses_per_heading,args.evaluation_n)<=0
            or not np.isfinite(args.synaptic_step) or args.synaptic_step<=0):
        raise ValueError('Positive finite experimental configuration required')
    if args.stage=='adjacent_one_step' and args.limit!=1:raise ValueError('Single-decision course requires limit=1')
    chemical=ChemicalConfig()
    settings={k:v for k,v in vars(args).items() if k not in ('output','resume','baseline_run')}
    settings.update(chemical_config=asdict(chemical),baseline_run=str(args.baseline_run.resolve()),
                    raw_pulse_balance=True,grouped_food_worlds=True,motor_exploration=False)
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json';source=sources()
        engine.prepare_baseline(args.baseline_run,args.output,args.size)
        if path.exists():
            if not args.resume:raise ValueError('Existing memories preserved; use --resume')
            report=json.loads(path.read_text(encoding='utf-8'))
            if report['sources']!=source or report['settings']!=settings or report['protocol']!=VERSION:
                raise ValueError('Saved protocol/source/settings differ')
            if report['status']=='complete':print('already_complete',flush=True);return
        else:
            report={'protocol':VERSION,'status':'loading','sources':source,'settings':settings,
                    'cycles_completed':0,'cycles':[],'checkpoints':[],'before':None,'after':{},
                    'chemical_basis':None,'started_at':time.time(),
                    'behavioral_learning_proven':False,'ordinary_snake_goal_complete':False}
            for relative in source:
                target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/relative).read_bytes())
            atomic_json(path,report)
        public={'protocol':VERSION,'settings':settings,'learning_proven':False}
        def publish(**fields):
            public.update(fields);public['updated_at']=time.time();public['process_id']=os.getpid()
            atomic_json(args.output/'current.json',public)
        publish(phase='loading',message='加载完整连接组与原生化学因果通路')
        try:
            brain=NativeChemicalBasisBrain(args.output,chemical=chemical)
            poses,heldout=engine.pose_split(args.size)
            if report['chemical_basis']:
                saved=report['chemical_basis']
                if Path(saved['file']).name!=saved['file']:raise ValueError('Unsafe chemical baseline path')
                brain.restore_chemical_basis(args.output/saved['file'],saved['sha256'])
            else:
                publish(phase='chemical_calibration',message='固定 RGB，测量双侧原生 ORN 的因果响应')
                measured=brain.fit_chemical_basis(calibration_frames(brain,poses))
                name=f'chemical-baseline-{time.time_ns()}.npz'
                sha=brain.save_chemical_basis(args.output/name)
                report['chemical_basis']={'file':name,'sha256':sha,'measurement':measured}
                report['brain']=brain.identity;atomic_json(path,report)
                print(json.dumps({'phase':'chemical_calibration','measurement':measured}),flush=True)
            report['brain']=brain.identity
            if report['checkpoints']:memory,rng=engine.restore_memory(args.output,report,brain)
            else:memory={arm:np.zeros(4) for arm in engine.ARMS};rng=np.random.default_rng(args.seed)
            if args.stage=='adjacent_one_step':
                scenarios=[{'seed':args.seed+888000+i,'pose':p,'food':f} for i,(p,f) in
                           enumerate((p,f) for p in heldout for f in engine.food_sites(p,args.size,args.stage))]
            else:scenarios=engine.make_batch(np.random.default_rng(args.seed+777000),heldout,
                                             args.size,args.stage,args.evaluation_n)
            if report['before'] is None:
                publish(experiment_phase='initial_frozen_assessment')
                report.update(status='running',phase='initial_frozen_assessment')
                report['before']=engine.assessment(brain,memory['frozen'],scenarios,args,publish)
                engine.save_memory(args.output,report,memory,rng,brain)
                print(json.dumps({'phase':'before','food':report['before']['food'],'n':report['before']['n']}),flush=True)
            hashes={r['rows'][0]['receipt']['observation_sha256'] for r in report['before']['episodes']}
            calibration_hashes={x['rgb'] for x in brain.identity['baseline_inputs']}|{
                               x['rgb'] for x in brain.identity['chemical_calibration_inputs']}
            if hashes&calibration_hashes:raise RuntimeError('Unrewarded calibration overlaps heldout pixels')
            for index in range(report['cycles_completed'],args.cycles):
                report.update(status='running',phase='actual_game_comparison',active_cycle=index+1)
                atomic_json(path,report)
                report['cycles'].append(cycle(brain,memory,rng,args,poses,hashes,index,publish))
                report['cycles_completed']=index+1
                engine.save_memory(args.output,report,memory,rng,brain)
                publish(cycles_completed=index+1,memory_saved=True,checkpoints=len(report['checkpoints']))
                if (args.output/'STOP_AFTER_CYCLE').exists():
                    report.update(status='stopped',phase='committed_boundary');atomic_json(path,report)
                    publish(phase='stopped',message='完整比较边界已保存');return
            for arm in engine.ARMS:
                publish(experiment_phase='final_frozen_assessment',arm=arm)
                report.update(phase='final_frozen_assessment',active_arm=arm);atomic_json(path,report)
                report['after'][arm]=engine.assessment(brain,memory[arm],scenarios,args,publish)
                atomic_json(path,report)
                print(json.dumps({'phase':'after','arm':arm,'food':report['after'][arm]['food'],
                                  'n':report['after'][arm]['n']}),flush=True)
            if [engine.trial_signature(r) for r in report['before']['episodes']]!=[
                    engine.trial_signature(r) for r in report['after']['frozen']['episodes']]:
                raise RuntimeError('Frozen native behavior changed')
            report.update(status='complete',phase='complete',completed_at=time.time());atomic_json(path,report)
            publish(phase='complete',message='有限化学因果实验完成，须按冻结与错配结果判断学习')
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report)
            publish(phase='error',message='实验出错，已提交记忆保留',error=report['error']);raise


if __name__=='__main__':main()
