"""Verify learning-only derivatives and feedback displacement, not learning."""
import numpy as np
from flysnake.research.actor_credit import (surrogate_probabilities,self_action_credit,
    ActorCreditConfig,reward_proposal,accept_game_candidate,displaced_feedback)


def test_self_action_surrogate_derivative_matches_finite_difference():
    for gap in (-2.,-.1,0.,.3,2.):
        p,d=surrogate_probabilities(gap,.2,.5)
        hi,_=surrogate_probabilities(gap+1e-5,.2,.5)
        lo,_=surrogate_probabilities(gap-1e-5,.2,.5)
        assert abs(sum(p.values())-1)<1e-12
        for action in p:
            assert p[action]>0
            assert np.isclose(d[action],(hi[action]-lo[action])/2e-5,atol=1e-8)


def test_reward_trace_uses_native_self_action_and_signed_existing_synapses():
    trace=self_action_credit(np.array([8,8,8,8]),np.array([4.,4.,4.,4.]),
        np.array([1.,-1.,1.,-1.]),np.array([0,0,1,1]),
        {'action':'left','scores':{'left':1.,'forward':0.,'right':0.}},
        {'scale_hz':[1.,1.,1.]})
    # A rewarded left self action increases left excitation/decreases left
    # inhibition, and does the opposite to right afferents. No target label.
    assert np.sign(trace).tolist()==[1.,-1.,-1.,1.]
    silent=self_action_credit(np.zeros(4),np.ones(4),np.ones(4),np.array([0,0,1,1]),
        {'action':None,'scores':{}},{'scale_hz':[1.,1.,1.]})
    assert not silent.any()


def test_displaced_feedback_never_pairs_a_trial_with_itself():
    for n in (3,6,16):
        for seed in range(20):
            assigned,perm=displaced_feedback(np.arange(n,dtype=float),seed)
            assert sorted(perm)==list(range(n))
            assert all(i!=p for i,p in enumerate(perm))
            assert np.array_equal(assigned,np.asarray(perm))


def test_actual_game_validation_can_reject_both_or_choose_reverse_credit():
    theta=np.zeros(4);traces=np.eye(4);rewards=np.array([1.,-1.,1.,-1.])
    candidates,proposal=reward_proposal(theta,traces,rewards,arm='learning',seed=1)
    rejected,r=accept_game_candidate(candidates,[1.,0.,-1.],arm='learning',seed=1)
    assert np.array_equal(rejected,theta) and not r['accepted']
    reverse,r=accept_game_candidate(candidates,[-1.,0.,1.],arm='learning',seed=1)
    assert np.array_equal(reverse,candidates[2]) and r['accepted']
    frozen,p=reward_proposal(theta,traces,rewards,arm='frozen',seed=1)
    assert not frozen.any() and p['step_rms']==0


def test_saturated_self_action_still_receives_punishment_credit():
    trace=self_action_credit(np.array([8,8]),np.array([4.,4.]),np.ones(2),np.array([0,1]),
        {'action':'left','scores':{'left':100.,'forward':0.,'right':0.}},
        {'scale_hz':[1.,1.,1.]})
    assert trace[0]>0 and trace[1]<0
    candidates,_=reward_proposal(np.zeros(2),[trace,trace],[-1.,-1.],arm='learning',seed=5)
    assert candidates[1,0]<0 and candidates[1,1]>0
