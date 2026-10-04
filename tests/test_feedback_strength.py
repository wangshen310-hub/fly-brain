"""Verify feedback parameters reach integration; TEST-ONLY neural fixture."""
from types import SimpleNamespace

import numpy as np
import pytest

from flysnake.neural import RealFlyBrain


@pytest.mark.parametrize("kind,targets", [("reward", [0, 1]), ("aversive", [2])])
def test_strength_reaches_selected_circuit_and_matches_receipt(kind, targets):
    brain = object.__new__(RealFlyBrain)
    brain.brain = SimpleNamespace(circuit={"reward": [0, 1], "aversive": [2], "kc": [3]},
                                  ids=np.array([10, 11, 12, 13]))
    brain.total_neural_ms = 0.0
    brain.feedback_events = 0
    brain.memory = lambda: {"fixture": True}
    frame = np.full((180, 320, 3), 128, dtype=np.uint8)

    def integrate(rgb, duration, *, stimulation):
        np.testing.assert_array_equal(rgb, frame)
        assert duration == 200.0
        assert stimulation == (targets, 30.0)
        brain.total_neural_ms += duration
        return np.array([1, 2, 3, 4]), 0.001

    brain._integrate = integrate
    receipt = brain.feedback(kind, frame)
    assert receipt["stimulus_current_mV_equivalent"] == 30.0
    assert receipt["stimulus_ms"] == 200.0
    assert receipt["action"] is None
    assert receipt["delivered_cells"] == [str(brain.brain.ids[i]) for i in targets]
    assert brain.feedback_events == 1
