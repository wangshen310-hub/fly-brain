"""Current-conservation checks, not evidence of learned behavior."""
import numpy as np
from flysnake.research.centered_learning import mean_current_correction


def test_mean_current_is_cancelled_with_signed_and_converging_connections():
    # Excitatory and inhibitory changes into one cell, with a second target.
    changed = np.array([2., -1., 3.])
    pre = np.array([0, 1, 0]); post = np.array([2, 2, 3])
    counts = np.array([6., 3., 0., 0.])
    correction = mean_current_correction(changed, pre, post, counts, 4, 300., 3.)
    delivered_mean = np.zeros(4)
    for w, source, target in zip(changed, pre, post):
        delivered_mean[target] += w*counts[source]*5./300.
    assert np.allclose(correction+delivered_mean, 0.)
    assert np.array_equal(correction[:2], np.zeros(2))


def test_intrinsic_correction_cannot_exceed_declared_cap():
    correction = mean_current_correction(np.array([1000.]), np.array([0]),
        np.array([1]), np.array([100., 0.]), 2, 300., 3.)
    assert correction[1]==-3.
