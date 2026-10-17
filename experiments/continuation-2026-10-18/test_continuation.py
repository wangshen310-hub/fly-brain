"""Reproducibility and test-isolation checks for the conditional experiment kit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import pytest
from continue_conditional_experiment import bind, restore, objective, safe_record, ranking
from sweep_continuation import rank as balanced_rank

ROOT=Path(__file__).resolve().parent

@pytest.fixture(scope='module')
def current():
    torch.set_num_threads(2)
    Replay,loss,Dynamics=bind(ROOT)
    train,a,_=restore('train',ROOT,Replay,Dynamics)
    valid,b,_=restore('validation',ROOT,Replay,Dynamics)
    return train,a,valid,b,loss


def test_native_boundary_baseline_matches_200ms_complete_brain(current):
    train,a,val,b,_=current
    for model in (train,val):
        v=model.verify_reference()
        assert v['parent_millisecond_spike_timing_exact']
        assert v['parent_internal_arrivals_exact']
        assert max(v['native_boundary_max_abs_error'].values()) == 0


def test_split_identity_and_readonly_rgb_sources(current):
    train,a,val,b,_=current
    assert train.batch==48 and val.batch==24
    assert not set(a['rgb_sha256'].tolist()) & set(b['rgb_sha256'].tolist())
    # Pose overlap is recorded transparently; different RGB is not a fully new pose.
    assert set(a['pose_id'].tolist()) & set(b['pose_id'].tolist())
    for k in ('baseline_afferent_raw','baseline_relay_raw','baseline_relay_rest_raw','baseline_node_gain_raw','baseline_node_tau_raw'):
        assert torch.equal(getattr(train,k),getattr(val,k))


def test_fixed_integer_decoder_rejects_tie_and_silence_as_turn():
    from flysnake.research.teacher_spiking import decode_spikes
    assert decode_spikes({'left':[2],'right':[1],'forward':[0,0]})['action']=='left'
    assert decode_spikes({'left':[1],'right':[2],'forward':[0,0]})['action']=='right'
    assert decode_spikes({'left':[2],'right':[2],'forward':[0,0]})['action']=='forward'
    assert decode_spikes({'left':[0],'right':[0],'forward':[0,0]})['action'] is None


def test_temporal_penalty_is_zero_for_matching_differences():
    a=torch.tensor([[1.,2.],[0.,3.]],requires_grad=True)
    loss=(a[1]-a[0]).square().mean()
    assert loss.item()>0
    loss.backward()
    assert torch.isfinite(a.grad).all()
    assert (a.grad.abs()>0).any()
    zeros=torch.zeros(2,2,requires_grad=True)
    assert (zeros[1]-zeros[0]).square().mean().item()==0


def test_every_export_reproduces_saved_actions_without_new_training(current):
    train,a,val,b,loss=current
    for folder, key in [('experiment-01','selected_validation'),('experiment-02','selected_validation'),('sweep-01','validation_selected')]:
        path=ROOT/'results'/folder
        manifest=json.loads((path/'summary.json').read_text())
        d=np.load(path/'conditional-selected-parameters.npz',allow_pickle=False)
        names=('afferent_log_gain','relay_edge_log_gain','relay_rest_raw','node_gain_raw','node_tau_raw')
        parameters=[torch.from_numpy(d[k].copy()) for k in names]
        with torch.no_grad():
            _,result=objective(val,parameters,loss)
        assert safe_record(result)['class_recall'] == manifest[key]['class_recall']
        assert safe_record(result)['balanced_accuracy'] == manifest[key]['balanced_accuracy']
        assert manifest.get('learning_gate_passed',False) is False


def test_all_exported_arrays_are_finite_bounded_and_original_shape(current):
    train,a,val,b,loss=current
    for folder in ('experiment-01','experiment-02','sweep-01'):
        path=ROOT/'results'/folder/'conditional-selected-parameters.npz'
        with np.load(path,allow_pickle=False) as d:
            for name,source in [('afferent_log_gain','baseline_afferent_raw'),('relay_edge_log_gain','baseline_relay_raw')]:
                x=d[name]
                assert x.dtype==np.float32 and x.shape==tuple(getattr(train,source).shape)
                assert np.isfinite(x).all()
                assert float(np.abs(x-getattr(train,source).numpy()).max()) <= 0.120001


def test_model_selection_scores_class_accuracy_not_only_soft_loss():
    a={'group_recall':[[.5,1,0],[.5,.5,1]],'loss':.25}
    b={'group_recall':[[.5,.5,.5],[.5,.5,.5]],'loss':1.0}
    assert ranking(b)>ranking(a)
    assert balanced_rank(b)>balanced_rank(a)
