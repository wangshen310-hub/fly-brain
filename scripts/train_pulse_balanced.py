"""Independent experiment with the requested raw-pulse forward balance rule.

Reuses frozen native actor mechanics, without editing its saved dependencies.
Only the decoder and receipt gate change. All actions remain native receipts.
"""
from pathlib import Path
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
from flysnake.research.actor_credit import ActorCreditConfig
from flysnake.research.pulse_balance import LEARNING_VERSION,PulseBalancedBrain,PulseBalancedGame
from flysnake.lab.experiment import atomic_json
from flysnake.research.paired import writer_lock

# Shared transaction functions use this NEW protocol within this process only;
# retained source and other running processes are unchanged. The bridge source
# and decoder are included in provenance. This is not a game action adapter.
engine.VERSION=LEARNING_VERSION
engine.SensoryGame=PulseBalancedGame


def sources():
    result=engine.sources()
    for relative in ('scripts/train_pulse_balanced.py','flysnake/research/pulse_balance.py'):
        result[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    return result


def main():
    args=engine.parser().parse_args()
    if min(args.limit,args.cycles,args.batch_size,args.evaluation_n,args.noise_sigma,args.edge_step)<=0 or args.batch_size<2:
        raise ValueError('Positive experiment configuration required')
    if args.stage=='adjacent_one_step' and args.limit!=1:raise ValueError('Single-decision course requires limit=1')
    config=ActorCreditConfig(noise_sigma=args.noise_sigma,maximum_edge_step=args.edge_step)
    settings={k:v for k,v in vars(args).items() if k not in ('output','resume','baseline_run')}
    settings.update(actor_config=asdict(config),baseline_run=str(args.baseline_run.resolve()),raw_pulse_balance=True)
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output):
        path=args.output/'status.json';source=sources();old=engine.prepare_baseline(args.baseline_run,args.output,args.size)
        if path.exists():
            if not args.resume:raise ValueError('Preserve existing memory; use --resume')
            report=json.loads(path.read_text(encoding='utf-8'))
            if report['sources']!=source or report['settings']!=settings or report['protocol']!=LEARNING_VERSION:
                raise ValueError('Saved source/settings differ')
            if report['status']=='complete':print('already_complete',flush=True);return
        else:
            report={'protocol':LEARNING_VERSION,'status':'loading','sources':source,'settings':settings,
                    'cycles_completed':0,'cycles':[],'checkpoints':[],'before':None,'after':{},'credit_audit':None,
                    'started_at':time.time(),'behavioral_learning_proven':False,'ordinary_snake_goal_complete':False}
            for relative in source:
                target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
            atomic_json(path,report)
        public={'protocol':LEARNING_VERSION,'settings':settings,'learning_proven':False}
        def publish(**fields):
            public.update(fields);public['updated_at']=time.time();public['process_id']=os.getpid()
            atomic_json(args.output/'current.json',public)
        publish(phase='loading',message='加载原生连接组与原始脉冲平衡门')
        try:
            brain=PulseBalancedBrain(args.output,config=config);report['brain']=brain.identity
            training,heldout=engine.pose_split(args.size)
            if report['checkpoints']:memory,rng=engine.restore_memory(args.output,report,brain)
            else:memory={a:np.zeros_like(brain.coefficients) for a in engine.ARMS};rng=np.random.default_rng(args.seed)
            ev_rng=np.random.default_rng(args.seed+777000)
            if args.stage=='adjacent_one_step':
                scenarios=[{'seed':args.seed+888000+i,'pose':p,'food':f} for i,(p,f) in
                    enumerate((p,f) for p in heldout for f in engine.food_sites(p,args.size,args.stage))]
            else:scenarios=engine.make_batch(ev_rng,heldout,args.size,args.stage,args.evaluation_n)
            if report['credit_audit'] is None:
                batch=engine.make_batch(np.random.default_rng(930602),training,args.size,args.stage,3)
                report['credit_audit']=engine.native_credit_audit(brain,batch);atomic_json(path,report)
            if report['before'] is None:
                publish(experiment_phase='initial_frozen_assessment');report.update(status='running',phase='initial_frozen_assessment')
                report['before']=engine.assessment(brain,memory['frozen'],scenarios,args,publish)
                if args.stage=='adjacent_one_step' and old['settings']['stage']==args.stage:
                    old_rows=old['before']['episodes'];new_rows=report['before']['episodes']
                    if len(old_rows)!=len(new_rows):raise RuntimeError('Unrewarded comparison worlds differ')
                    for a,b in zip(old_rows,new_rows):
                        ar,br=a['rows'][0]['receipt'],b['rows'][0]['receipt']
                        if ar['observation_sha256']!=br['observation_sha256'] or ar['counts']!=br['counts']:
                            raise RuntimeError('Decoder unexpectedly changed untrained native pixels/spikes')
                    report['native_baseline_spikes_match_old']=True
                    report['native_baseline_actions_changed_only_by_decoder']=sum(a['rows'][0]['receipt']['action']!=b['rows'][0]['receipt']['action'] for a,b in zip(old_rows,new_rows))
                engine.save_memory(args.output,report,memory,rng,brain)
                print(json.dumps({'phase':'before','food':report['before']['food'],'n':report['before']['n'],
                                  'raw_balance_action_changes':report.get('native_baseline_actions_changed_only_by_decoder')}),flush=True)
            heldout_hashes={r['rows'][0]['receipt']['observation_sha256'] for r in report['before']['episodes']}
            if heldout_hashes&{x['rgb'] for x in brain.identity['baseline_inputs']}:raise RuntimeError('Calibration overlaps heldout')
            for index in range(report['cycles_completed'],args.cycles):
                report.update(status='running',phase='actual_game_comparison',active_cycle=index+1);atomic_json(path,report)
                record=engine.cycle(brain,memory,rng,args,training,heldout_hashes,index,publish)
                report['cycles'].append(record);report['cycles_completed']=index+1
                engine.save_memory(args.output,report,memory,rng,brain)
                publish(cycles_completed=index+1,memory_saved=True,checkpoints=len(report['checkpoints']))
                if (args.output/'STOP_AFTER_CYCLE').exists():
                    report.update(status='stopped',phase='committed_boundary');atomic_json(path,report)
                    publish(phase='stopped',message='完整更新边界已保存');return
            for arm in engine.ARMS:
                publish(experiment_phase='final_frozen_assessment',arm=arm)
                report.update(phase='final_frozen_assessment',active_arm=arm);atomic_json(path,report)
                report['after'][arm]=engine.assessment(brain,memory[arm],scenarios,args,publish);atomic_json(path,report)
                print(json.dumps({'phase':'after','arm':arm,'food':report['after'][arm]['food'],'n':report['after'][arm]['n']}),flush=True)
            if [engine.trial_signature(r) for r in report['before']['episodes']]!=[engine.trial_signature(r) for r in report['after']['frozen']['episodes']]:
                raise RuntimeError('Frozen native behavior changed')
            report.update(status='complete',phase='complete',completed_at=time.time());atomic_json(path,report)
            publish(phase='complete',message='有限实验完成；学习结论须看独立冻结结果')
        except Exception:
            report.update(status='error',error=traceback.format_exc());atomic_json(path,report)
            publish(phase='error',message='实验出错，已提交记忆保留',error=report['error']);raise


if __name__=='__main__':main()
