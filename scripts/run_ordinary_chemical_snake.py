"""Continuous ordinary Snake observation with frozen native chemical memory.

No curriculum timeout, nearby-food generator or policy controls this game.
This runner evaluates a committed brain. DAN feedback is delivered, but weights
are frozen: its score is not a claim of new online reward learning.
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
from verify_chemical_restore import load_brain
import train_chemical_controls as trainer
from flysnake.research.pulse_balance import PulseBalancedGame
from flysnake.research.olfaction import odor_samples
from flysnake.research.continuous_state import VERSION,save_window,restore_window
from flysnake.research.paired import writer_lock
from flysnake.lab.experiment import atomic_json


def sources(parent):
    result=parent['sources'].copy()
    for relative in ('scripts/run_ordinary_chemical_snake.py','scripts/verify_chemical_restore.py',
                     'flysnake/research/continuous_state.py'):
        result[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    return result


def advance(brain,game,publish=None):
    rgb=game.render_observation();sense=odor_samples(game.snake[0],game.direction,game.food,brain.odor)
    if publish:publish(phase='neural_integration',image=trainer.engine.png(rgb),game=game.snapshot(),
                       sense=sense,receipt=None,event=None,feedback=None)
    receipt,_=brain.observe(rgb,sense['concentration'])
    if receipt['observation_sha256']!=hashlib.sha256(rgb.tobytes()).hexdigest():raise RuntimeError('Actual ordinary pixels differ')
    receipt.update(episode_id=game.episode_id,step_id=game.step_id)
    reward=0.;feedback=event=None
    if receipt['action'] is None:
        reward=-.1;feedback=brain.feedback('aversive')
    else:
        event=game.apply_receipt(receipt)
        if not event['accepted']:raise RuntimeError('Ordinary native receipt rejected: '+str(event))
        if event['kind']!='none':
            reward=1. if event['kind']=='reward' else -1.
            feedback=brain.feedback(event['kind']);game.complete_feedback(event['event_id'])
    result={'receipt':receipt,'sense':sense,'event':event,'feedback':feedback,'reward':reward,
            'game':game.snapshot(),'exploration_off':True,'synaptic_parameters_frozen':True}
    if publish:publish(phase='native_action',image=trainer.engine.png(rgb),**result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-cycle',type=int,required=True)
    p.add_argument('--arm',choices=trainer.engine.ARMS,default='learning')
    p.add_argument('--seed',type=int,default=930901);p.add_argument('--max-windows',type=int,default=0)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.max_windows<0:raise ValueError('Nonnegative window budget required; zero means continuous')
    parent=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if parent['sources']!=trainer.sources():raise ValueError('Parent native source changed')
    selected=[cp for cp in parent['checkpoints'] if cp['cycles_completed']==args.checkpoint_cycle]
    if not selected:raise ValueError('Requested committed chemical memory missing')
    parent['checkpoints']=[selected[-1]]
    settings={'parent_run':str(args.run.resolve()),'checkpoint_cycle':args.checkpoint_cycle,'arm':args.arm,
              'seed':args.seed,'size':parent['settings']['size'],'stage':'uniform_food','continuous':True,
              'max_windows':args.max_windows,'raw_pulse_balance':True,'parameters_frozen':True,
              'training_feedback_imported':False}
    source=sources(parent);args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json'
        if path.exists():
            if not args.resume:raise ValueError('Preserve ordinary progress; use --resume')
            report=json.loads(path.read_text(encoding='utf-8'))
            if report['protocol']!=VERSION or report['sources']!=source or report['settings']!=settings:
                raise ValueError('Saved ordinary protocol/source/settings differ')
            if report['status']=='complete':print('bounded_evaluation_already_complete',flush=True);return
        else:
            report={'protocol':VERSION,'status':'loading','sources':source,'settings':settings,'cycles':[],
                    'cycles_completed':args.checkpoint_cycle,'windows_completed':0,'checkpoints':[],
                    'totals':{'food':0,'deaths':0,'wall_deaths':0,'self_deaths':0,'episodes_completed':0,
                              'no_output':0,'return':0.},'parent_checkpoint':selected[-1],
                    'behavioral_learning_proven':False,'ordinary_snake_goal_complete':False,'started_at':time.time()}
            for relative in source:
                target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/relative).read_bytes())
            atomic_json(path,report)
        public={'protocol':VERSION,'settings':settings,'learning_proven':False}
        def publish(**fields):
            public.update(fields);public.update(updated_at=time.time(),process_id=os.getpid())
            atomic_json(args.output/'current.json',public)
        publish(phase='loading',message='加载已提交脑记忆，普通游戏参数冻结')
        try:
            brain=load_brain(args.run,parent)
            memory,_=trainer.engine.restore_memory(args.run,parent,brain)
            brain.set_parameters(memory[args.arm]);report['brain']=brain.identity
            game=PulseBalancedGame(seed=args.seed,size=settings['size'],stage='uniform_food',
                geometry=brain.geometry,calibration=brain.brain.calibration)
            if report['checkpoints']:restore_window(args.output,report,brain,game)
            else:brain.reset_trial();save_window(args.output,report,brain,game)
            frozen=brain.weight_hash();report.update(status='running');atomic_json(path,report)
            while not args.max_windows or report['windows_completed']<args.max_windows:
                if game.terminal:game.new_episode();brain.reset_trial()
                result=advance(brain,game,publish)
                if brain.weight_hash()!=frozen:raise RuntimeError('Frozen ordinary evaluation modified native weights')
                totals=report['totals'];totals['return']+=result['reward']
                if result['receipt']['action'] is None:totals['no_output']+=1
                event=result['event']
                if event and event['kind']=='reward':totals['food']+=1
                if game.terminal:
                    totals['episodes_completed']+=1
                    if game.death_reason:
                        totals['deaths']+=1;totals[game.death_reason+'_deaths']+=1
                report['windows_completed']+=1;report['latest_result']=result
                cp=save_window(args.output,report,brain,game)
                publish(windows_completed=report['windows_completed'],totals=totals,memory_saved=True,
                        latest_checkpoint=cp)
                print(json.dumps({'window':report['windows_completed'],'action':result['receipt']['action'],
                    'counts':result['receipt']['counts'],'event':event['kind'] if event else 'no_output',
                    'totals':totals,'checkpoint':cp['file']}),flush=True)
                if (args.output/'STOP_AFTER_WINDOW').exists():
                    report.update(status='stopped');atomic_json(path,report)
                    publish(phase='stopped',message='普通游戏、原生动态状态与记忆已保存');return
            report.update(status='complete',completed_at=time.time());atomic_json(path,report)
            publish(phase='complete',message='普通游戏冻结评估窗口已完成；不代表在线学习')
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report)
            publish(phase='error',error=report['error']);raise


if __name__=='__main__':main()
