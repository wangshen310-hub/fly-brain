import numpy as np
import pytest
from flysnake.research.covered_learning import CoveredLearningConfig, reward_update


def test_reward_correlation_and_frozen_control():
    config = CoveredLearningConfig()
    theta = np.zeros(2)
    traces = [[1., 0.], [-1., 0.]]
    updated, report = reward_update(theta, traces, [1., -1.],
                                    arm='learning', config=config, shuffle_seed=1)
    assert updated[0]>0 and updated[1]==0
    frozen, _ = reward_update(theta, traces, [1., -1.],
                              arm='frozen', config=config, shuffle_seed=1)
    assert np.array_equal(frozen, theta)
    flat, report = reward_update(updated, traces, [-1., -1.],
                                 arm='learning', config=config, shuffle_seed=1)
    assert np.array_equal(flat, updated) and report['step_rms']==0


def test_wrong_credit_shape_and_nonfinite_are_rejected():
    for traces in ([[1.]], [[np.nan, 0.], [0., 0.]]):
        with pytest.raises(ValueError):
            reward_update(np.zeros(2), traces, [1., -1.], arm='learning',
                          config=CoveredLearningConfig(), shuffle_seed=1)
