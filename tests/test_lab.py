"""Rule fixtures only; actual neural assays are run separately on MaleCNS."""
import copy
import json
import numpy as np
import pytest
from flysnake.lab.protocol import Config, decode, sample_linear, eligibility_update, plastic_update
from flysnake.lab.environment import LabGame, choice_scene, cue_scene
from flysnake.lab.experiment import Experiment, trial_plan, summarize, digest, atomic_json

CAL={'baseline_hz':[10.,10.,10.],'scale_hz':[10.,10.,10.]}


def test_silence_does_not_generate_motion():
    assert decode({'left':[0],'forward':[0,0],'right':[0]},300,CAL)['action'] is None


def test_balanced_choice_not_random_but_snake_forward():
    counts={'left':[5],'forward':[0,0],'right':[5]}
    assert decode(counts,300,CAL,task='choice')['action'] is None
    assert decode(counts,300,CAL,task='snake')['action']=='forward'


def test_neutral_calibration_removes_channel_bias():
    counts={'left':[6],'forward':[3,3],'right':[9]}
    cal={'baseline_hz':[10.,10.,30.],'scale_hz':[10.,10.,10.]}
    assert decode(counts,300,cal,task='choice')['action']=='left'


@pytest.mark.parametrize('counts',[{'left':[-1],'forward':[0,0],'right':[2]},
    {'left':[np.nan],'forward':[0,0],'right':[2]}, {'left':[1.2],'forward':[0,0],'right':[2]}])
def test_invalid_neural_data_rejected(counts):
    with pytest.raises(ValueError): decode(counts,300,CAL)


def test_linear_drive_retains_brightness_difference():
    uv=np.array([[0.,0.],[1.,1.]])
    frame=np.full((180,320,3),85,dtype=np.uint8); frame[-1,-1]=200
    light=sample_linear(frame,uv)
    assert light[1].mean()>5*light[0].mean()


def test_rule_requires_eligibility_and_reward_sign_changes_effect():
    cfg=Config(); f=np.ones(3); e=np.array([1.,-.5,0.])
    assert np.array_equal(plastic_update(f,np.zeros(3),1,cfg),f)
    reward=plastic_update(f,e,1,cfg); punishment=plastic_update(f,e,-1,cfg)
    assert reward[0]>f[0]>punishment[0] and reward[1]<f[1]<punishment[1]
    assert reward[2]==punishment[2]==1


def test_local_eligibility_depends_on_pre_and_post_activity():
    args=(np.zeros(3),np.array([20.,10.,0.]),np.ones(3)*10,np.ones(3)*10)
    e=eligibility_update(*args,[2.,0.,-2.],np.array([0,1,2]),.3,1.)
    assert e[0]>0 and e[1]==0 and e[2]>0
    balanced=eligibility_update(*args,[1.,1.,1.],np.array([0,1,2]),.3,1.)
    assert not np.any(balanced)


def test_bounds_preserve_sign():
    value=plastic_update(np.ones(3),np.array([10000.,-10000.,0]),1,Config())
    assert value.tolist()==[2.,.25,1.]


def test_fixed_camera_heading_and_food_pixels():
    g=LabGame(seed=1,size=6,calibration=CAL)
    g.snake=[(3,3),(3,4)];g.direction=0;g.food=(1,1)
    a=g.render_observation();g.direction=1;b=g.render_observation()
    assert not np.array_equal(a,b)
    food=(a==[180,200,255]).all(axis=2)
    assert food.sum()>0
    assert np.array_equal(food,(b==[180,200,255]).all(axis=2))
    g.snake=[(4,3),(3,3)];c=g.render_observation()
    assert np.array_equal(food,(c==[180,200,255]).all(axis=2))


def receipt(g,counts,start=100):
    return {**decode(counts,300,CAL), 'calibration':CAL,'neural_time_start':start,'neural_time_end':start+300,
            'observation_sha256':g.observation_hash(),'episode_id':g.episode_id,'step_id':g.step_id}


def test_lab_receipt_does_not_avoid_wall_and_cannot_replay():
    g=LabGame(seed=1,size=6,calibration=CAL);g.snake=[(0,2),(0,3)];g.direction=0;g.food=(5,5)
    r=receipt(g,{'left':[12],'forward':[0,0],'right':[0]})
    result=g.apply_receipt(r)
    assert result['accepted'] and result['death_reason']=='wall'
    assert not g.apply_receipt(r)['accepted']


def test_lab_rejects_fabricated_action_or_mismatched_image():
    g=LabGame(seed=1,size=6,calibration=CAL)
    r=receipt(g,{'left':[12],'forward':[0,0],'right':[0]})
    r['action']='right'
    assert not g.apply_receipt(r)['accepted']
    r['action']='left';r['observation_sha256']='wrong'
    assert not g.apply_receipt(r)['accepted']


def test_choice_generators_reproducible():
    assert np.array_equal(choice_scene(123,'left'),choice_scene(123,'left'))
    assert not np.array_equal(cue_scene(123,'left'),cue_scene(123,'right'))


def test_controls_share_scenes_and_training_test_do_not_overlap():
    tasks=trial_plan()
    for seed in {t['seed'] for t in tasks}:
        arms=[]
        for arm in ('learning','frozen','shuffled'):
            rows=[t for t in tasks if t['seed']==seed and t['arm']==arm]
            arms.append([(r['scene_seed'],r['side'],r['phase']) for r in rows])
            train={r['scene_seed'] for r in rows if r['phase']=='train'}
            test={r['scene_seed'] for r in rows if r['phase']=='test'}
            assert not train&test
        assert arms[0]==arms[1]==arms[2]
    assert {t['initial_mapping'] for t in tasks}=={'same','opposite'}


def test_gate_cannot_pass_without_controls_and_reversal():
    assert not summarize([])['passed']
    rows=[dict(t,correct=True) for t in trial_plan() if t['arm']=='learning']
    # Perfect learner without any controls must never be a pass.
    assert not summarize(rows)['passed']


class MemoryFixture:
    def rpc(self,op,**kwargs):
        if op=='checkpoint':
            from pathlib import Path
            Path(kwargs['path']).parent.mkdir(parents=True,exist_ok=True)
            with open(kwargs['path'],'xb') as f: f.write(b'fixture')
            return {'memory_sha256':'fixture'}
        if op=='metadata': return {'memory_sha256':'fixture'}
        if op=='restore': return {'restored':True}
        raise AssertionError(op)


def test_atomic_bundle_restore_and_corruption_rejection(tmp_path):
    e=Experiment(tmp_path);e.brain=MemoryFixture();e.state['cursor']=12;e.save()
    restored=Experiment(tmp_path);restored.brain=MemoryFixture()
    assert restored.restore() and restored.state['cursor']==12
    latest=json.loads((tmp_path/'latest.json').read_text())
    path=tmp_path/'checkpoints'/latest['generation']/'memory.npz'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'): restored.restore()


def test_atomic_write_retries_windows_reader_without_deleting_commit(tmp_path,monkeypatch):
    from pathlib import Path
    p=tmp_path/'state.json';p.write_text('{"cursor":1}')
    real_replace=Path.replace;calls=[]
    def intermittent(path,target):
        calls.append(1)
        if len(calls)<=2:
            assert json.loads(p.read_text())['cursor']==1
            raise PermissionError('simulated Windows sharing violation')
        return real_replace(path,target)
    monkeypatch.setattr(Path,'replace',intermittent)
    atomic_json(p,{'cursor':2})
    assert len(calls)==3 and json.loads(p.read_text())['cursor']==2


def test_runner_freezes_evaluation_and_never_sends_target_action(tmp_path,monkeypatch):
    import flysnake.lab.experiment as module
    monkeypatch.setattr(module,'TRAIN_N',2);monkeypatch.setattr(module,'TEST_N',2)
    class AssayFixture(MemoryFixture):
        def __init__(self):self.calls=[]
        def rpc(self,op,**kwargs):
            self.calls.append((op,kwargs))
            if op=='set_fraction':return None
            if op=='observe':
                assert set(kwargs)=={'rgb','task','learning'}
                return {'action':'left','counts':{'left':[1],'forward':[0,0],'right':[0]},
                        'memory_sha256':'fixture','afferent_counts':[]}
            if op=='feedback':
                assert set(kwargs)=={'kind','learning'}
                return {'delivered_spikes':1,'memory_before':'fixture','memory_after':'fixture'}
            if op=='metadata':return {'candidate_plastic_edges':1,'memory_sha256':'fixture'}
            return super().rpc(op,**kwargs)
    e=Experiment(tmp_path);e.brain=AssayFixture()
    assert not e.learning_assay('cue')
    assert e.state['phase']=='needs_revision' and e.state['next_stage']=='cue'
    rows=e.state['rows'];observations=[c[1] for c in e.brain.calls if c[0]=='observe']
    assert len(rows)==len(observations)
    for row,request in zip(rows,observations):
        if row['phase'] in ('test','reversal_test'):
            assert request['learning'] is False and row['feedback'] is None
        if row['arm']=='frozen':assert request['learning'] is False


def test_crash_before_trial_commit_resumes_last_committed_cursor(tmp_path):
    e=Experiment(tmp_path);e.brain=MemoryFixture();e.state['cursor']=8;e.save()
    e.state['cursor']=9
    e.state['rows'].append({'uncommitted':True})
    recovered=Experiment(tmp_path);recovered.brain=MemoryFixture();recovered.restore()
    assert recovered.state['cursor']==8 and recovered.state['rows']==[]
