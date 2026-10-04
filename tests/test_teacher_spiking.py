import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from flysnake.research.teacher_spiking import SpikingConnectome, Dynamics, decode_spikes, _FixedMM, _Spike


def fixture():
    n=7
    edges=[(1,2,1.),(2,3,.7),(2,4,.3),(2,5,1.),(2,6,1.),(0,1,-1.)]
    arrays={'ids':np.arange(n),'rest':np.full(n,.4,np.float32),
        'retina':np.array([0]),'uv':np.array([[.3,.5]]),
        'r8':np.array([],int),'r8_uv':np.empty((0,2)),'r8_channel':np.array([],int),
        'graded_lamina':np.array([1]),'KC':np.array([2]),
        'left':np.array([3]),'right':np.array([4]),'forward_left':np.array([5]),'forward_right':np.array([6])}
    for prefix,sgn in [('exc',1),('inh',-1)]:
        for marker,flip in [('',False),('t',True)]:
            selected=sorted([(pre,post,abs(w)) if flip else (post,pre,abs(w)) for pre,post,w in edges if np.sign(w)==sgn])
            rows=[a for a,b,w in selected];col=[b for a,b,w in selected];value=[w for a,b,w in selected]
            arrays[prefix+'_'+marker+'crow']=np.r_[0,np.cumsum(np.bincount(rows,minlength=n))].astype(np.int32)
            arrays[prefix+'_'+marker+'col']=np.array(col,np.int32)
            arrays[prefix+'_'+marker+'value']=np.array(value,np.float32)
    return arrays


def brain(seed=0):
    return SpikingConnectome(fixture(),{},device='cpu',seed=seed,config=Dynamics(window_ms=20))


def test_actual_pulses_gate_turns_and_silence():
    assert decode_spikes({'left':[10],'right':[9],'forward':[0,0]})['action']=='forward'
    assert decode_spikes({'left':[10],'right':[7],'forward':[0,0]})['action']=='left'
    assert decode_spikes({'left':[2],'right':[8],'forward':[0,0]})['action']=='right'
    assert decode_spikes({'left':[0],'right':[0],'forward':[0,0]})['action'] is None
    with pytest.raises(ValueError):decode_spikes({'left':[1.5],'right':[1],'forward':[0,0]})


def test_sparse_state_derivative_matches_dense_without_edge_gradient():
    b=brain();x=torch.randn(7,2,requires_grad=True);out=_FixedMM.apply(x,b.exc_matrix,b.exc_transpose)
    target=torch.randn_like(out);(out*target).sum().backward()
    assert torch.allclose(x.grad,b.exc_matrix.to_dense().T@target,atol=1e-7)
    assert not b.exc_matrix.requires_grad


def test_surrogate_does_not_change_binary_forward():
    x=torch.tensor([-.3,0.,.3],requires_grad=True)
    y=_Spike.apply(x,.15)
    assert y.tolist()==[0.,1.,1.]
    y.sum().backward();assert torch.isfinite(x.grad).all() and (x.grad>0).all()


def test_split_window_keeps_delays_refractory_and_voltage():
    b=brain();light=torch.ones(1,1)*.3
    with torch.no_grad():
        whole,_,state=b.integrate_encoded(light,steps=20)
        first,_,s=b.integrate_encoded(light,steps=7)
        second,_,s=b.integrate_encoded(light,s,steps=13)
    assert torch.equal(whole,first+second)
    for k,v in state.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,s[k]),k
        else:assert v==s[k]
    assert torch.equal(whole,whole.round())


def test_inhibitory_and_adaptation_conductance_bounds_and_signs():
    b=brain();fixed=b.inh_matrix.values().clone()
    with torch.no_grad():
        b.rest_delta.fill_(-100);b.gain_raw.fill_(100)
        counts,_,s=b.integrate_encoded(torch.ones(1,1),steps=80)
    assert torch.isfinite(s['v']).all()
    assert s['v'].min()>=b.config.inh_reversal-1e-6
    assert s['v'].max()<=b.config.exc_reversal+1e-6
    assert (s['ge']>=0).all() and (s['gi']>=0).all() and (s['adaptation']>=0).all()
    assert torch.equal(fixed,b.inh_matrix.values())


def test_checkpoint_continuation_and_integrity(tmp_path):
    b=brain();light=torch.ones(1,1)*.4
    with torch.no_grad():_,_,state=b.integrate_encoded(light,steps=9)
    c=b.save_checkpoint(tmp_path/'state.pt',state=state)
    clone=brain();restored,_=clone.restore_checkpoint(c['path'],c['sha256'])
    with torch.no_grad():
        counts,_,a=b.integrate_encoded(light,state,steps=11)
        other,_,z=clone.integrate_encoded(light,restored,steps=11)
    assert torch.equal(counts,other)
    for k,v in a.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,z[k]),k
        else:assert v==z[k]
    with pytest.raises(ValueError):clone.restore_checkpoint(c['path'],'0'*64)


def test_rgb_boundary_and_sensory_off_counterfactual():
    b=brain();white=np.full((180,320,3),255,np.uint8);black=np.zeros_like(white)
    first,s=b.act(white,visual_enabled=False);second,t=b.act(black,visual_enabled=False)
    assert first['full_counts_sha256']==second['full_counts_sha256']
    for k,v in s.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,t[k])
    with pytest.raises(ValueError):b.encode(np.zeros((1,6,6),dtype=np.uint8))
    assert 'teacher_data' not in Path(importlib.util.find_spec('flysnake.research.teacher_spiking').origin).read_text(encoding='utf-8')


def test_training_updates_only_internal_parameters():
    b=brain();initial=b.parameter_hash();original=b.exc_matrix.values().clone()
    optimizer=torch.optim.Adam(b.parameters(),lr=.01)
    _,motor,_=b.integrate_encoded(torch.ones(1,1)*.3,steps=20)
    loss=torch.nn.functional.cross_entropy(b.logits(motor),torch.tensor([0]));loss.backward();optimizer.step()
    assert b.parameter_hash()!=initial
    assert torch.equal(original,b.exc_matrix.values())
    assert set(dict(b.named_parameters()))=={'rest_delta','gain_raw','tau_raw'}


def test_invalid_restore_cannot_partly_overwrite_memory(tmp_path):
    b=brain();initial=b.parameter_hash();c=b.save_checkpoint(tmp_path/'valid.pt')
    payload=torch.load(c['path'],weights_only=True)
    payload['parameters']['rest_delta'].fill_(.1)
    payload['parameters']['tau_raw'][0]=float('nan')
    bad=tmp_path/'invalid.pt';torch.save(payload,bad)
    sha=hashlib.sha256(bad.read_bytes()).hexdigest()
    with pytest.raises(ValueError):b.restore_checkpoint(bad,sha)
    assert b.parameter_hash()==initial


def test_invalid_neural_state_is_rejected_before_use():
    b=brain();state=b.zero_state(1);state['cursor']=1
    with pytest.raises(ValueError):b.integrate_encoded(torch.ones(1,1),state)
    with pytest.raises(ValueError):Dynamics(delay_steps=1.5)
