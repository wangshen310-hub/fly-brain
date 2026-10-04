"""Check sensory geometry and outcome association; no learning-success claims."""
import numpy as np
import pytest
from flysnake.research.olfaction import odor_samples
from flysnake.research.sensory_basis import activity_basis,choose_reward_candidate


def test_antennal_concentration_uses_a_stationary_source_and_rotates_with_body():
    # Food is physically left of an upward-facing head; rotation puts it right.
    up=odor_samples((4,4),0,(3,4))
    down=odor_samples((4,4),2,(3,4))
    assert up['concentration'][0]>up['concentration'][1]
    assert down['concentration'][0]<down['concentration'][1]
    assert np.allclose(up['concentration'],down['concentration'][::-1])
    assert np.allclose(odor_samples((4,4),0,(4,2))['concentration'][0],
                       odor_samples((4,4),0,(4,2))['concentration'][1])
    assert odor_samples((4,4),0,None)['concentration']==[0.,0.]


def test_field_translation_does_not_change_physical_samples():
    a=odor_samples((3,4),1,(6,2))['concentration']
    b=odor_samples((13,14),1,(16,12))['concentration']
    assert np.allclose(a,b)


def test_static_basis_never_crosses_motor_side_and_contains_no_scene_encoder():
    counts=np.random.default_rng(123).poisson(4,(32,20))
    base=np.r_[np.ones(10),-np.ones(10)]
    side=np.arange(20)%2
    basis,mean,singular=activity_basis(counts,base,side,rank=3)
    assert basis.shape==(20,8) and mean.shape==(20,)
    assert np.all(basis[side==0,4:]==0) and np.all(basis[side==1,:4]==0)
    assert np.max(abs(basis))<=1
    assert np.allclose(mean,counts.mean(0)) and len(singular)==20


def test_actual_reward_acceptance_and_frozen_control():
    candidates=np.array([[0.,0.],[1.,0.],[-1.,0.]])
    learned,record=choose_reward_candidate(candidates,[-1.,1.,-1.],arm='learning',shuffle_seed=7)
    assert np.array_equal(learned,candidates[1]) and record['accepted']
    frozen,_=choose_reward_candidate(candidates,[-1.,1.,-1.],arm='frozen',shuffle_seed=7)
    assert np.array_equal(frozen,candidates[0])
    tied,record=choose_reward_candidate(candidates,[0.,0.,0.],arm='learning',shuffle_seed=7)
    assert np.array_equal(tied,candidates[0]) and not record['accepted']


def test_mismatched_feedback_really_breaks_candidate_association():
    candidates=np.array([[0.],[1.],[-1.]])
    winners=[]
    for seed in range(20):
        _,r=choose_reward_candidate(candidates,[-1.,1.,-1.],arm='shuffled',shuffle_seed=seed)
        assert sorted(r['assigned_returns'])==[-1.,-1.,1.]
        winners.append(r['winner'])
    assert set(winners)=={0,1,2}
    with pytest.raises(ValueError):
        choose_reward_candidate(candidates,[np.nan,0.,1.],arm='learning',shuffle_seed=1)
