"""Reproducible assays, matched controls, atomic trial checkpoints, curriculum.

The coordinator owns scenes/outcomes. The spawned brain process receives no
target direction or environment state. All actions come from fresh neural RGB
responses. Test layouts are disjoint from training; inference is frozen.
"""
import base64
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image
from .brain import worker
from .environment import LabGame, cue_scene, choice_scene, neutral_scene
from .protocol import VERSION

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_STORAGE=ROOT/'runtime/lab-v4'
ARMS=('learning','frozen','shuffled')
SEEDS=(1701,2903)
TRAIN_N=24
TEST_N=16


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    with temporary.open('w',encoding='utf-8') as out:
        json.dump(value,out,ensure_ascii=False,allow_nan=False)
        out.flush()
        import os
        os.fsync(out.fileno())
    # Windows readers/antivirus can briefly hold a handle without FILE_SHARE_DELETE.
    # Keep the old committed file intact and retry the atomic rename, never unlink.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt==7: raise
            time.sleep(.01*2**min(attempt,5))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png_data(rgb):
    stream=io.BytesIO(); Image.fromarray(rgb).save(stream,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


class BrainProcess:
    def __init__(self):
        self.connection,child=mp.get_context('spawn').Pipe()
        self.process=mp.get_context('spawn').Process(target=worker,args=(child,))
        self.process.start(); child.close()
        self.initial=self.receive()

    def receive(self):
        while not self.connection.poll(1):
            if not self.process.is_alive(): raise RuntimeError('Neural worker exited unexpectedly')
        result=self.connection.recv()
        if 'error' in result: raise RuntimeError(result['error']+'\n'+result.get('traceback',''))
        return result['result']

    def rpc(self,operation,**values):
        self.connection.send({'operation':operation,**values})
        return self.receive()

    def close(self):
        if self.process.is_alive():
            self.connection.send({'operation':'close'})
            self.process.join(15)
        self.connection.close()
        if self.process.is_alive(): raise RuntimeError('Neural worker did not exit cleanly')


def wilson(successes,n):
    if not n: return [0.,1.]
    p=successes/n; z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def summarize(rows):
    output={}
    for phase in ('test','reversal_test'):
        groups={}
        for arm in ARMS:
            selected=[x for x in rows if x['phase']==phase and x['arm']==arm]
            correct=sum(x['correct'] for x in selected)
            groups[arm]={'n':len(selected),'correct':correct,
                'accuracy':correct/len(selected) if selected else None,
                'wilson95':wilson(correct,len(selected)),
                'per_seed':{str(seed):[sum(x['correct'] for x in selected if x['seed']==seed),
                                      sum(x['seed']==seed for x in selected)] for seed in SEEDS}}
        output[phase]=groups
    # Conservative engineering go/no-go, not a publication significance test.
    passed=True
    for phase in output.values():
        actual=phase['learning']
        control=max((phase[a]['accuracy'] or 0) for a in ('frozen','shuffled'))
        passed &= (all(phase[a]['n']==len(SEEDS)*TEST_N for a in ARMS) and actual['accuracy']>=.75
                   and actual['accuracy']-control>=.15
                   and actual['wilson95'][0]>.5
                   and all(n and s/n>=.625 for s,n in actual['per_seed'].values()))
    return {'phases':output,'passed':bool(passed),
            'criterion':'Both tests: >=75%, >=15 percentage points over both controls, Wilson lower >50%, each seed >=62.5%; only a screening gate, not a general learning claim'}


def trial_plan():
    tasks=[]
    # Scene order and initial memory are matched across arms. Shuffled rewards
    # have independently fixed balanced signs, not access to the desired move.
    for seed in SEEDS:
        for arm in ARMS:
            for phase,n in [('train',TRAIN_N),('test',TEST_N),('reversal',TRAIN_N),('reversal_test',TEST_N)]:
                rng=np.random.default_rng(seed+{'train':0,'test':10000,'reversal':20000,'reversal_test':30000}[phase])
                sides=['left','right']*(n//2); rng.shuffle(sides)
                fake=['reward','aversive']*(n//2); np.random.default_rng(seed+9183).shuffle(fake)
                for i in range(n):
                    tasks.append({'seed':seed,'arm':arm,'phase':phase,'i':i,'side':sides[i],
                        'initial_mapping':'same' if seed==SEEDS[0] else 'opposite',
                        'scene_seed':seed*100000+{'train':0,'test':10000,'reversal':20000,'reversal_test':30000}[phase]+i,
                        'shuffled_kind':fake[i],'start_arm':phase=='train' and i==0})
    return tasks


class Experiment:
    def __init__(self,storage=DEFAULT_STORAGE,notify=None,stop=None):
        self.storage=Path(storage).resolve(); self.storage.mkdir(parents=True,exist_ok=True)
        self.notify=notify or (lambda _:None)
        self.stop=stop or (lambda:False)
        self.brain=None
        self.state={'version':VERSION,'phase':'starting','message':'准备真实连接组',
            'gates':{},'rows':[],'cursor':0,'stage':'cue','next_stage':'cue','image':None,
            'legacy_memory_untouched':True,'training_complete':False}

    def publish(self,**values):
        self.state.update(values)
        self.state['updated_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(self.storage/'status.json',self.state)
        self.notify(copy.deepcopy(self.state))

    def save(self):
        generation=self.storage/'checkpoints'/uuid.uuid4().hex
        generation.mkdir(parents=True)
        metadata=self.brain.rpc('checkpoint',path=str(generation/'memory.npz'))
        self.state['brain']=self.brain.rpc('metadata')
        self.state['saved_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(generation/'state.json',self.state)
        manifest={'version':VERSION,'memory':'memory.npz','state':'state.json',
                  'hashes':{name:digest(generation/name) for name in ('memory.npz','state.json')},
                  'memory_sha256':metadata['memory_sha256']}
        atomic_json(generation/'commit.json',manifest)
        atomic_json(self.storage/'latest.json',{'generation':generation.name,'commit_sha256':digest(generation/'commit.json')})
        self.publish()

    def restore(self):
        pointer=self.storage/'latest.json'
        if not pointer.exists(): return False
        data=json.loads(pointer.read_text(encoding='utf-8'))
        generation=(self.storage/'checkpoints'/data['generation']).resolve()
        if generation.parent!=(self.storage/'checkpoints').resolve(): raise ValueError('Invalid checkpoint path')
        if digest(generation/'commit.json')!=data['commit_sha256']: raise ValueError('Incomplete checkpoint commit')
        m=json.loads((generation/'commit.json').read_text(encoding='utf-8'))
        if m['version']!=VERSION or set(m['hashes'])!={'memory.npz','state.json'}: raise ValueError('Wrong checkpoint format')
        for name,expected in m['hashes'].items():
            if digest(generation/name)!=expected: raise ValueError('Corrupt checkpoint; preserved for inspection')
        self.brain.rpc('restore',path=str(generation/'memory.npz'))
        self.state=json.loads((generation/'state.json').read_text(encoding='utf-8'))
        return True

    def observe(self,rgb,**kwargs):
        self.publish(image=png_data(rgb),last_decision=None,last_feedback=None)
        receipt=self.brain.rpc('observe',rgb=rgb,**kwargs)
        # High-dimensional afferent features are retained only for sensory assay,
        # never passed to an external action model.
        self.publish(last_decision={k:v for k,v in receipt.items() if k!='afferent_counts'})
        return receipt

    def basic_assays(self):
        self.publish(phase='calibration',message='用无食物画面校准活动基线；不提供正确方向')
        self.brain.rpc('calibrate',frames=[neutral_scene(i+901) for i in range(8)])
        self.publish(phase='sensory',message='检查不同位置的视觉信号能否到达运动输入')
        samples=[]
        for seed in (31,47,59,71):
            for side in ('left','right'):
                r=self.observe(cue_scene(seed,side),learning=False)
                samples.append({'seed':seed,'side':side,'features':r.pop('afferent_counts'),'receipt':r})
        correct=0
        for x in samples:
            others=[y for y in samples if y['seed']!=x['seed']]
            nearest=min(others,key=lambda y:np.linalg.norm(np.array(x['features'])-y['features']))
            correct+=nearest['side']==x['side']
        sensory={'passed':correct>=6,'correct':correct,'n':8,
            'method':'Leave-one-layout-out nearest-neighbor on motor-afferent spike counts; diagnostic only, never controls actions'}
        self.state['gates']['sensory']=sensory
        self.publish(phase='motor',message='刺激左右运动神经群，检验因果方向')
        scene=neutral_scene(951)
        self.publish(image=png_data(scene),last_decision=None,last_feedback=None)
        left=self.brain.rpc('intervention',rgb=scene,group='left',current=30.)
        right=self.brain.rpc('intervention',rgb=scene,group='right',current=30.)
        self.state['gates']['motor']={'passed':left['action']=='left' and right['action']=='right','left':left,'right':right}
        self.publish(phase='memory_path',message='改变已有运动输入突触，检验其对神经输出的因果作用')
        meta=self.brain.rpc('metadata'); n=meta['candidate_plastic_edges']
        # Broad efficacy interventions test the candidate locus without changing
        # graph topology. No intervention weights survive into training.
        before=self.observe(scene,learning=False)
        self.brain.rpc('set_fraction',fraction=np.ones(n)*.5)
        lower=self.observe(scene,learning=False)
        self.brain.rpc('set_fraction',fraction=np.ones(n)*1.5)
        higher=self.observe(scene,learning=False)
        self.brain.rpc('set_fraction',fraction=np.ones(n))
        difference=lower['counts']!=higher['counts']
        self.state['gates']['memory_path']={'passed':difference,'baseline':before['counts'],'half':lower['counts'],'one_and_half':higher['counts'],
            'limitation':'Broad weight intervention establishes sensitivity, not successful learning or natural plasticity'}
        self.state['assay_samples']=samples
        self.state['basic_assays_complete']=True
        self.save()

    def learning_assay(self,stage):
        plan=trial_plan()
        start=self.state.get('cursor',0)
        for index in range(start,len(plan)):
            if self.stop():
                self.publish(phase='paused',message='实验已保存，停在完整试次边界'); self.save(); return False
            task=plan[index]
            if task['start_arm']:
                initial=self.state.get(stage+'_initial')
                if initial:
                    self.brain.rpc('restore',path=initial)
                else:
                    n=self.brain.rpc('metadata')['candidate_plastic_edges']
                    self.brain.rpc('set_fraction',fraction=np.ones(n))
            phase=task['phase']; evaluation=phase in ('test','reversal_test')
            frame=(cue_scene if stage=='cue' else choice_scene)(task['scene_seed'],task['side'])
            self.publish(phase=phase,stage=stage,message=f"{stage} · {task['arm']} · 种子 {task['seed']} · {index+1}/{len(plan)}",
                         total_trials=len(plan))
            learning=not evaluation and task['arm']!='frozen'
            r=self.observe(frame,task='choice',learning=learning)
            target=task['side']
            opposite=(task['initial_mapping']=='opposite') != (phase in ('reversal','reversal_test'))
            if opposite: target='right' if target=='left' else 'left'
            correct=r['action']==target
            feedback=None
            if not evaluation:
                kind='reward' if correct else 'aversive'
                if task['arm']=='shuffled': kind=task['shuffled_kind']
                feedback=self.brain.rpc('feedback',kind=kind,learning=learning)
                if feedback['delivered_spikes']<=0: raise RuntimeError('Feedback circuit failed to respond')
            row={**task,'stage':stage,'correct':correct,'action':r['action'],
                 'counts':r['counts'],'memory_sha256':r['memory_sha256'],'feedback':feedback}
            self.state['rows'].append(row)
            if phase=='train' and task['i']==TRAIN_N-1 and task['arm']=='learning':
                endpoint=self.storage/'endpoints'/f'{stage}-{task["seed"]}-{uuid.uuid4().hex}.npz'
                self.brain.rpc('checkpoint',path=str(endpoint))
                self.state.setdefault('learned_endpoints',{})[f'{stage}-{task["seed"]}']=str(endpoint)
            self.state['cursor']=index+1
            self.publish(last_feedback=feedback,results=summarize([x for x in self.state['rows'] if x['stage']==stage]))
            # Every completed feedback and evaluation trial is durable. A crash
            # during a trial resumes its pre-state, so no stimulus is applied twice.
            self.save()
        report=summarize([x for x in self.state['rows'] if x['stage']==stage])
        self.state['gates'][stage+'_learning']=report
        self.state['cursor']=0
        self.state['next_stage']=('body_choice' if stage=='cue' else 'snake') if report['passed'] else stage
        if not report['passed']:
            self.state.update(phase='needs_revision',message=f'{stage} 学习／反转未通过对照，停止自动升级')
        if report['passed']:
            endpoint=self.state['learned_endpoints'][f'{stage}-{SEEDS[0]}']
            self.state[self.state['next_stage']+'_initial']=endpoint
        self.save()
        return report['passed']

    def snake_curriculum(self):
        """Reached only after sensory/motor/cue/body-choice learning gates pass.

        Bounded batches. No advance for survival alone; food efficiency must
        improve over matched frozen baseline. State commits at episode ends.
        """
        self.publish(phase='snake',message='基础学习检验通过，开始分级贪吃蛇对照')
        meta=self.brain.rpc('metadata')
        # Preserve the learned endpoint as its own immutable comparator.
        start=self.storage/'snake-initial.npz'
        if not start.exists():
            self.brain.rpc('restore',path=self.state['snake_initial'])
            self.brain.rpc('checkpoint',path=str(start))
        rows=self.state.setdefault('snake_rows',[])
        for size in (4,6,8,12):
            for arm in ('frozen','learning'):
                done=[r for r in rows if r['size']==size and r['arm']==arm]
                if not done: self.brain.rpc('restore',path=str(start))
                for episode in range(len(done),12):
                    if self.stop():
                        self.publish(phase='paused',message='已保存完整棋局实验'); self.save(); return
                    game=LabGame(seed=800000+size*100+episode,size=size,calibration=meta['calibration'])
                    food=0; collisions=0; no_action=0
                    for attempt in range(100):
                        rgb=game.render_observation()
                        r=self.observe(rgb,task='snake',learning=arm=='learning',reset=attempt==0)
                        r.update(episode_id=game.episode_id,step_id=game.step_id)
                        if r['action'] is None:
                            no_action+=1
                            self.brain.rpc('feedback',kind='aversive',learning=arm=='learning')
                            continue
                        outcome=game.apply_receipt(r)
                        if not outcome['accepted']: raise RuntimeError('Invalid brain receipt: '+str(outcome))
                        if outcome['kind']!='none':
                            f=self.brain.rpc('feedback',kind=outcome['kind'],learning=arm=='learning')
                            if f['delivered_spikes']<=0: raise RuntimeError('Missing feedback spikes')
                            game.complete_feedback(outcome['event_id'])
                            food+=outcome['kind']=='reward'; collisions+=outcome.get('death_reason') is not None
                        self.publish(game=game.snapshot(),last_feedback=f if outcome['kind']!='none' else None)
                        if game.terminal: break
                    rows.append({'size':size,'arm':arm,'episode':episode,'food':food,'steps':game.step_id,
                                 'collisions':collisions,'no_action':no_action,'truncated':not game.terminal})
                    self.save()
            trained=[r for r in rows if r['size']==size and r['arm']=='learning']
            frozen=[r for r in rows if r['size']==size and r['arm']=='frozen']
            rate=lambda rs:sum(r['food'] for r in rs)/max(1,sum(r['steps'] for r in rs))*100
            passed=rate(trained)>rate(frozen)+1 and sum(r['food'] for r in trained)>=6
            self.state['gates'][f'snake_{size}']={'passed':passed,'trained_food_per_100_steps':rate(trained),'frozen_food_per_100_steps':rate(frozen),
                'criterion':'Exploratory curriculum gate: >=6 foods and >1 extra food/100 steps; not proof of generalization'}
            self.save()
            if not passed:
                self.publish(phase='needs_revision',message=f'{size}×{size} 课程未超过冻结对照，停止自动升级'); self.save(); return
        self.publish(phase='complete',training_complete=True,message='分级实验完成；请查看对照结果与局限'); self.save()

    def run(self):
        # One writer across CLI/dashboard instances; lock is released on crash.
        import os
        lock=(self.storage/'experiment.lock').open('a+b')
        try:
            lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            self._run()
        finally:
            lock.close()

    def _run(self):
        try:
            self.publish(phase='loading',message='载入完整 MaleCNS 连接组；不会载入或改写旧版本记忆')
            self.brain=BrainProcess()
            resumed=self.restore()
            if resumed and self.state['phase'] in ('needs_revision','complete'):
                self.publish(message='已恢复实验记录；未通过的验证门不会被继续按钮绕过'); return
            if not self.state.get('basic_assays_complete'):
                self.basic_assays()
            if not all(self.state['gates'][name]['passed'] for name in ('sensory','motor','memory_path')):
                self.publish(phase='needs_revision',message='基础回路验证未通过，未进入训练；保留测量供修订'); self.save(); return
            if any(not value['passed'] for name,value in self.state['gates'].items() if name.endswith('_learning')):
                self.publish(phase='needs_revision',message='已提交的学习验证未通过，禁止越过验证门'); self.save(); return
            for stage in ('cue','body_choice'):
                if self.state.get('next_stage','cue') not in (stage,):
                    continue
                if not self.learning_assay(stage):
                    if self.state['phase']!='paused':
                        self.publish(phase='needs_revision',message=f'{stage} 学习／反转没有通过对照，不进入完整游戏'); self.save()
                    return
            if self.state.get('next_stage')=='snake': self.snake_curriculum()
        except Exception as error:
            import traceback
            self.publish(phase='error',message=str(error),error_traceback=traceback.format_exc())
            raise
        finally:
            if self.brain is not None: self.brain.close()


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--storage',type=Path,default=DEFAULT_STORAGE)
    args=p.parse_args()
    def notify(s):
        print(json.dumps({k:s.get(k) for k in ('phase','message','cursor')},ensure_ascii=False),flush=True)
    Experiment(args.storage,notify).run()


if __name__=='__main__':
    mp.freeze_support(); main()
