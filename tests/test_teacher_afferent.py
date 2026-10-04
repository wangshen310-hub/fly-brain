import numpy as np
import pytest
torch=pytest.importorskip('torch')
from flysnake.research.teacher_spiking import SpikingConnectome,Dynamics
from flysnake.research.teacher_afferent import PlasticAfferentBrain
from test_teacher_spiking import fixture


def arrays():
    a=fixture();a['pre']=np.array([1,2,2,2,2,0]);a['post']=np.array([2,3,4,5,6,1])
    a['signed_base']=np.array([1.,.7,.3,1.,1.,-1.],np.float32)
    return a


def test_zero_plasticity_is_original_neural_dynamics():
    cfg=Dynamics(window_ms=20);base=SpikingConnectome(arrays(),{},device='cpu',config=cfg)
    plastic=PlasticAfferentBrain(arrays(),{},device='cpu',config=cfg)
    with torch.no_grad():
        c,_,s=base.integrate_encoded(torch.ones(1,1)*.3)
        z,_,t=plastic.integrate_encoded(torch.ones(1,1)*.3)
    assert torch.equal(c,z)
    for k,v in s.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,t[k])
        else:assert v==t[k]


def test_plastic_delivery_changes_only_existing_signed_afferents():
    b=PlasticAfferentBrain(arrays(),{},device='cpu',config=Dynamics(window_ms=20));x=torch.rand(7,2)
    before=b.exc_matrix.to_dense().clone()
    with torch.no_grad():b.afferent_log_gain.copy_(torch.tensor([-.2,.3]))
    actual=b.delivery(x,'exc');expected=before.clone()
    for pre,post,w,p in zip(b.aff_pre,b.aff_post,b.aff_abs,b.afferent_log_gain):expected[post,pre]=w*torch.exp(4*torch.tanh(p))
    assert torch.allclose(actual,expected@x,atol=1e-6)
    assert torch.equal(b.exc_matrix.to_dense(),before)
    actual.sum().backward()
    assert torch.isfinite(b.afferent_log_gain.grad).all() and (b.afferent_log_gain.grad!=0).all()
    assert [k for k,p in b.named_parameters() if p.requires_grad]==['afferent_log_gain']
