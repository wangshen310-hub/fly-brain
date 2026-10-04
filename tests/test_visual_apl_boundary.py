"""Input-boundary fixtures; full native evidence is produced by the assay."""
from types import SimpleNamespace
import numpy as np
import pytest
from flysnake.research.visual_apl import VisualAPLBrain
from flysnake.research.visual_stimuli import VisualBoardGame, audit_frames, BODY_COLOR


def geometry():
    return {'projection': 'fixed_world', 'tile_pixels': 8, 'board_cells': 12,
            'board_origin': [214, 28], 'image_size': [320, 180], 'wall_cells': 1}


def test_nonvisual_arguments_are_rejected_by_the_interface():
    brain = object.__new__(VisualAPLBrain)
    rgb = np.zeros((180, 320, 3), np.uint8)
    for kwargs in ({'concentrations': [[1., 0.], [0., 0.]]},
                   {'stimulation': [(np.array([0]), 30.)]}, {'learning': True},
                   {'food_coordinates': [5, 6]}):
        with pytest.raises(TypeError):
            brain.integrate(rgb, 10, **kwargs)
    with pytest.raises(TypeError):
        brain.integrate(rgb, 10, [1., 0.])


def test_undeclared_tonic_rejected_before_native_mutation():
    brain = object.__new__(VisualAPLBrain)
    brain.b = SimpleNamespace(tonic=np.array([0., 1.]))
    brain.motor_tonic = np.array([0., 0.])
    with pytest.raises(ValueError, match='Undeclared tonic'):
        brain.integrate(np.zeros((180, 320, 3), np.uint8), 10)


def test_food_and_walls_do_not_follow_the_head():
    frames = audit_frames(geometry())
    a, b = frames['food_A'], frames['pose_moved_food_fixed']
    # World food at (5,6); all its original pixels remain identical.
    np.testing.assert_array_equal(a[76:84, 254:262], b[76:84, 254:262])
    np.testing.assert_array_equal(a[20:28, 206:318], b[20:28, 206:318])
    assert not np.array_equal(a, b)


def test_head_body_and_heading_enter_as_different_actual_pixels():
    frames = audit_frames(geometry())
    assert not np.array_equal(frames['food_A'], frames['head_as_body'])
    assert not np.array_equal(frames['food_A'], frames['head_rotated'])
    head = frames['food_A'][76:84, 262:270]
    body = frames['food_A'][84:92, 262:270]
    assert len(np.unique(head.reshape(-1, 3), axis=0)) == 2
    assert np.array_equal(np.unique(body.reshape(-1, 3), axis=0), [BODY_COLOR])


def test_feedback_keeps_the_exact_before_action_visual_observation():
    game = VisualBoardGame(seed=2, size=12, geometry=geometry())
    before = game.render_observation()
    game._feedback_observation = before.copy()
    game.snake = [(1, 1), (1, 2)]
    game.food = (8, 8)
    np.testing.assert_array_equal(game.render_observation(), before)
