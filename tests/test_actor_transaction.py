"""Reward timing and committed transaction tests, not native learning proof."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_actor_credit import discounted_returns,save_memory,restore_memory,ARMS
from flysnake.research.paired import digest


def test_reward_sign_is_not_squared_or_applied_twice():
    assert np.array_equal(discounted_returns([0,0,-1],1),[-1,-1,-1])
    assert np.allclose(discounted_returns([0,1,-1],.5),[.25,.5,-1])


class NativeFixture:
    identity={'graph':'test fixture','plasticity':'self action'}
    def set_parameters(self,theta):
        theta=np.asarray(theta)
        if theta.shape!=(3,) or not np.isfinite(theta).all() or (abs(theta)>2).any():
            raise ValueError('Invalid fixture parameter')
        self.theta=theta.copy()
    def weight_hash(self):return digest(np.array([1.,-2.,3.])*np.exp(self.theta))


def test_committed_memory_restores_all_arms_and_exact_rng_after_interruption(tmp_path):
    brain=NativeFixture();memory={a:np.ones(3)*.1*i for i,a in enumerate(ARMS)}
    report={'cycles_completed':2,'cycles':[{'cycle':1},{'cycle':2}],
            'sources':{'source':'fixture'},'settings':{'seed':123},'checkpoints':[]}
    rng=np.random.default_rng(123);rng.normal(size=12);save_memory(tmp_path,report,memory,rng,brain)
    report['cycles'].append({'cycle':3});report['cycles_completed']=3
    actual,restored=restore_memory(tmp_path,report,brain)
    assert report['cycles_completed']==2 and len(report['cycles'])==2
    assert all(np.array_equal(actual[a],memory[a]) for a in ARMS)
    assert np.array_equal(rng.normal(size=(12,2)),restored.normal(size=(12,2)))
    report['sources']['source']='changed'
    with pytest.raises(ValueError,match='source'):restore_memory(tmp_path,report,brain)
