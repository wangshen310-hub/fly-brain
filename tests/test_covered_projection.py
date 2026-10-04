"""Sensory coverage checks; these are not behavioral learning evidence."""
from pathlib import Path
import hashlib
import numpy as np
import pytest

from flysnake.lab.protocol import sample_linear
from flysnake.research.projection import coverage_geometry, CoveredGame


def actual_uv():
    # Read-only geometry from the already retained anatomical graph.
    path = Path(__file__).resolve().parents[1]/'data/malecns_v1/graph.npz'
    if not path.exists():
        # Portable test fixture contains only the exact retained coordinates.
        # This is never a substitute graph for neuronal inference.
        path = Path(__file__).with_name('fixtures')/'retinal-uv.npz'
    with np.load(path, allow_pickle=False) as graph:
        uv = graph['uv'].copy()
    assert hashlib.sha256(uv.tobytes()).hexdigest() == '51f7dafd4355a06176418e22373ffcdf5bc066f75050da6dcbaaaccbad55a714'
    return uv


def test_every_cell_and_heading_is_observable_at_actual_receptors():
    uv = actual_uv()
    geometry = coverage_geometry(uv, 4)
    assert np.min(geometry['quarter_sample_counts']) > 0
    game = CoveredGame(seed=1, size=4, geometry=geometry)
    food_signals = []
    for y in range(4):
        for x in range(4):
            game.snake = [(x, y)]; game.food = None
            headings = []
            for direction in range(4):
                game.direction = direction
                headings.append(sample_linear(game.render_observation(), uv).tobytes())
                assert game.observation_geometry() == geometry
            assert len(set(headings)) == 4
            # Compare each food tile with an otherwise identical background.
            game.snake = [((x+1)%4, y)]
            without = sample_linear(game.render_observation(), uv)
            game.food = (x, y)
            with_food = sample_linear(game.render_observation(), uv)
            assert np.any(with_food != without)
            food_signals.append((with_food-without).tobytes())
    assert len(set(food_signals)) == 16


@pytest.mark.parametrize('uv', [np.array([[np.nan, 0.]]), np.array([[1.1, 0.]]),
                                  np.empty((0, 2)), np.zeros((2, 3))])
def test_invalid_anatomical_geometry_is_rejected(uv):
    with pytest.raises(ValueError):
        coverage_geometry(uv)
