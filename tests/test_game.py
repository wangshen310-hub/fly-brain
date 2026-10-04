"""Handmade spike/action fixtures exercise rules; they are NOT a brain demo."""
import copy

import numpy as np
import pytest

from flysnake.decoder import decode_counts
from flysnake.game import BACKGROUND, BODY_COLOR, FOOD_COLOR, HEAD_COLOR, IMAGE_SIZE, SOLID, SnakeGame


def fixture_game(snake=None, direction=0, food=(10, 10)):
    """Explicit rule-test fixture, not a production initialization path."""
    game = SnakeGame(seed=101)
    game.snake = list(snake or [(5, 5), (5, 6), (5, 7)])
    game.direction = direction
    game.food = food
    return game


def fixture_receipt(game, action="forward", start=None, counts=None):
    """Fabricated neuronal records exclusively for deterministic unit tests."""
    counts = counts or {"left": [0], "forward": [0, 0], "right": [0]}
    if all(not any(c) for c in counts.values()):
        counts[action] = [3] * len(counts[action])
    start = game.step_id * .1 if start is None else start
    end = start + .1
    decision = decode_counts(counts, end - start)
    return {**decision, "episode_id": game.episode_id, "step_id": game.step_id,
            "observation_hash": game.observation_hash(), "neural_time_start": start,
            "neural_time_end": end, "brain_checkpoint_id": "RULE_TEST_FIXTURE_NOT_A_BRAIN"}


def test_seeded_initialization_is_legal_and_reproducible():
    a, b = SnakeGame(7), SnakeGame(7)
    for _ in range(8):
        assert a.snapshot() == b.snapshot()
        assert len(a.snake) == 3 and len(set(a.snake)) == 3
        assert all(a._inside(p) for p in a.snake)
        assert a.food not in a.snake and a._inside(a.food)
        a.new_episode(); b.new_episode()


def test_no_receipt_no_motion_and_snapshot_is_detached():
    game = fixture_game()
    before = game.snapshot()
    assert not game.apply_receipt(None)["accepted"]
    assert game.snapshot() == before
    external = game.snapshot()
    external["snake"][0][0] = -900
    assert game.snapshot() == before


@pytest.mark.parametrize("action,head,direction", [("left", (4, 5), 3), ("forward", (5, 4), 0), ("right", (6, 5), 1)])
def test_each_relative_action_moves_exactly_one_cell(action, head, direction):
    game = fixture_game()
    receipt = fixture_receipt(game, action)
    event = game.apply_receipt(receipt)
    assert event["accepted"] and event["kind"] == "none"
    assert game.snake[0] == head and game.direction == direction
    assert game.step_id == 1 and len(game.snake) == 3
    before = game.snapshot()
    assert not game.apply_receipt(receipt)["accepted"]
    assert game.snapshot() == before


@pytest.mark.parametrize("field,bad_value", [
    ("episode_id", 999), ("step_id", 999), ("observation_hash", "old-frame"),
    ("fixed_decoder_version", "other"), ("selected_cell_ids", {}),
    ("decoder_config", {}), ("brain_checkpoint_id", ""),
    ("neural_time_start", float("nan")), ("neural_time_end", float("inf")),
    ("action", "left"), ("scores", {"left": 0., "forward": 999., "right": 0.}),
])
def test_invalid_receipt_is_atomic(field, bad_value):
    game = fixture_game()
    receipt = fixture_receipt(game)
    receipt[field] = bad_value
    before = game.snapshot()
    assert not game.apply_receipt(receipt)["accepted"]
    assert game.snapshot() == before


def test_reusing_spike_window_for_new_step_is_rejected():
    game = fixture_game()
    assert game.apply_receipt(fixture_receipt(game, start=1.0))["accepted"]
    before = game.snapshot()
    assert game.apply_receipt(fixture_receipt(game, start=1.0))["reason"] == "reused_neural_window"
    assert game.snapshot() == before


@pytest.mark.parametrize("seconds", [.01, .15, .6, 100.])
def test_receipt_must_use_the_declared_100_to_500ms_sampling_windows(seconds):
    game = fixture_game()
    receipt = fixture_receipt(game)
    receipt["neural_time_end"] = receipt["neural_time_start"] + seconds
    before = game.snapshot()
    assert game.apply_receipt(receipt)["reason"] == "invalid_sampling_window"
    assert game.snapshot() == before


@pytest.mark.parametrize("counts", [
    {"left": [0], "forward": [0, 0], "right": [0]},
    {"left": [1], "forward": [1, 1], "right": [1]},
    {"left": [float("nan")], "forward": [0, 0], "right": [0]},
])
def test_zero_tie_nan_cannot_be_forged_into_a_forward_move(counts):
    game = fixture_game()
    receipt = fixture_receipt(game)
    receipt["new_spike_counts"] = counts
    before = game.snapshot()
    assert not game.apply_receipt(receipt)["accepted"]
    assert game.snapshot() == before


def test_lethal_neural_action_is_not_safety_filtered_and_feedback_is_required():
    game = fixture_game(snake=[(0, 4), (0, 5), (0, 6)])
    pre_image = game.render_observation()
    event = game.apply_receipt(fixture_receipt(game, "left"))
    assert event["accepted"] and event["kind"] == "aversive" and event["terminal"]
    assert event["death_reason"] == "wall" and event["collision_at"] == [-1, 4]
    assert np.array_equal(game.render_observation(), pre_image)
    with pytest.raises(RuntimeError):
        game.new_episode()
    assert game.complete_feedback(event["event_id"])["feedback_complete"]
    assert not game.complete_feedback(event["event_id"])["accepted"]
    assert not game.apply_receipt(fixture_receipt(game))["accepted"]
    assert game.new_episode()["alive"]


def test_self_collision_and_vacating_tail_have_distinct_rules():
    safe = fixture_game(snake=[(3, 3), (3, 4), (2, 4), (2, 3)])
    event = safe.apply_receipt(fixture_receipt(safe, "left"))
    assert not event["terminal"] and safe.snake[0] == (2, 3)
    lethal = fixture_game(snake=[(3, 3), (3, 4), (2, 4), (2, 3), (2, 2)])
    event = lethal.apply_receipt(fixture_receipt(lethal, "left"))
    assert event["terminal"] and event["death_reason"] == "self"
    assert event["kind"] == "aversive" and event["score_delta"] == 0
    assert event["collision_at"] == [2, 3] and not lethal.alive


@pytest.mark.parametrize("action,collision", [
    ("left", [4, 5]), ("forward", [5, 4]), ("right", [6, 5]),
])
def test_head_entering_body_is_terminal_and_punished_for_every_action(action, collision):
    # A contiguous coiled snake has occupied, non-tail cells on all three sides.
    # These fabricated counts exercise the rule; no production action is forced.
    snake = [(5, 5), (5, 6), (6, 6), (6, 5), (6, 4),
             (5, 4), (4, 4), (4, 5), (4, 6)]
    game = fixture_game(snake=snake)
    before_pixels = game.render_observation()
    receipt = fixture_receipt(game, action)
    event = game.apply_receipt(receipt)
    assert event["accepted"] and event["action"] == action
    assert event["kind"] == "aversive" and event["terminal"]
    assert event["death_reason"] == "self" and event["outcome"] == "death"
    assert event["collision_at"] == collision and event["score_delta"] == 0
    assert not game.alive and game.step_id == 1 and game.snake == snake
    assert game.pending_feedback["event_id"] == event["event_id"]
    assert np.array_equal(game.render_observation(), before_pixels)
    frozen = game.snapshot()
    assert game.apply_receipt(receipt)["reason"] == "feedback_pending"
    assert game.snapshot() == frozen
    with pytest.raises(RuntimeError, match="feedback"):
        game.new_episode()


def test_food_feedback_holds_pre_action_image_and_delays_rng_sampling():
    game = fixture_game(food=(5, 4))
    before_image = game.render_observation()
    rng_before = game._rng.getstate()
    event = game.apply_receipt(fixture_receipt(game))
    assert event["kind"] == "reward" and event["score_delta"] == 1
    assert len(game.snake) == 4 and game.score == 1 and game.food is None
    assert game._rng.getstate() == rng_before
    assert np.array_equal(game.render_observation(), before_image)
    assert game.apply_receipt(fixture_receipt(game))["reason"] == "feedback_pending"
    assert not game.complete_feedback("wrong-id")["accepted"]
    assert game.food is None
    assert game.complete_feedback(event["event_id"])["accepted"]
    assert game.food not in game.snake and game.food is not None
    state = game.snapshot(), game._rng.getstate()
    assert not game.complete_feedback(event["event_id"])["accepted"]
    assert (game.snapshot(), game._rng.getstate()) == state


def test_last_food_is_a_win_reward_without_death_punishment():
    game = SnakeGame(5, size=3)
    game.snake = [(1, 0), (0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (1, 2), (0, 2)]
    game.direction, game.food = 1, (2, 0)
    event = game.apply_receipt(fixture_receipt(game))
    assert event["kind"] == "reward" and event["terminal"] and event["outcome"] == "win"
    assert event["death_reason"] is None
    game.complete_feedback(event["event_id"])
    assert game.food is None


def test_no_hud_or_score_leak_and_rgb_output_is_detached():
    game = fixture_game()
    pixels = game.render_observation()
    assert pixels.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    assert pixels.dtype == np.uint8
    game.score, game.death_reason = 9999, "DO NOT RENDER"
    assert np.array_equal(pixels, game.render_observation())
    pixels[:] = 0
    assert game.render_observation().any()


@pytest.mark.parametrize("direction", range(4))
def test_fixed_camera_keeps_food_and_walls_still_when_head_moves_or_turns(direction):
    game = fixture_game(snake=[(5, 4), (5, 5), (4, 5)], direction=0, food=(7, 2))
    before = game.render_observation()
    food_before = np.all(before == FOOD_COLOR, axis=2)
    game.direction = direction
    assert np.array_equal(game.render_observation(), before)

    game.snake = [(x + 1, y + 1) for x, y in game.snake]
    after = game.render_observation()
    assert np.array_equal(np.all(after == FOOD_COLOR, axis=2), food_before)
    # Everything outside the fixed inner board includes wall and margin only.
    outside_board = np.ones(before.shape[:2], dtype=bool)
    outside_board[18:162, 88:232] = False
    assert np.array_equal(before[outside_board], after[outside_board])
    assert np.all(before[66:78, 148:160] == HEAD_COLOR)  # World (5,4).
    assert np.all(after[78:90, 160:172] == HEAD_COLOR)  # World (6,5).
    assert np.all(after[66:78, 148:160] == BACKGROUND)
    assert food_before.sum() == 144
    assert np.all(after[42:54, 172:184] == FOOD_COLOR)  # World (7,2).


@pytest.mark.parametrize("direction", range(4))
def test_actual_sensory_pixels_distinguish_head_body_and_food(direction):
    dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
    snake = [(5 - i * dx, 5 - i * dy) for i in range(3)]
    game = fixture_game(snake=snake, direction=direction, food=(10, 10))
    frame = game.render_observation()
    assert np.all(frame[78:90, 148:160] == HEAD_COLOR)  # Fixed world (5,5).
    for x, y in snake[1:]:
        assert np.all(frame[18 + y * 12:30 + y * 12, 88 + x * 12:100 + x * 12] == BODY_COLOR)
    assert BODY_COLOR == (70, 160, 110)
    assert BODY_COLOR not in (SOLID, HEAD_COLOR, FOOD_COLOR, BACKGROUND)
    assert np.all(frame[6:18, 76:244] == SOLID)  # Walls retain the dark sensory RGB.
    assert np.count_nonzero(np.all(frame == BODY_COLOR, axis=2)) == 288
    assert np.count_nonzero(np.all(frame == HEAD_COLOR, axis=2)) == 144
    assert np.count_nonzero(np.all(frame == FOOD_COLOR, axis=2)) == 144
    assert np.all(frame[138:150, 208:220] == FOOD_COLOR)  # Fixed world (10,10).
    assert game.snake == snake


@pytest.mark.parametrize("size,origin", [(3, (142, 72)), (6, (124, 54)), (12, (88, 18))])
def test_fixed_board_has_uniform_square_cells_and_explicit_wall_perimeter(size, origin):
    game = SnakeGame(seed=7, size=size)
    frame = game.render_observation()
    x0, y0 = origin
    x1, y1 = x0 + 12 * size, y0 + 12 * size
    assert np.all(frame[y0 - 12:y0, x0 - 12:x1 + 12] == SOLID)
    assert np.all(frame[y1:y1 + 12, x0 - 12:x1 + 12] == SOLID)
    assert np.all(frame[y0:y1, x0 - 12:x0] == SOLID)
    assert np.all(frame[y0:y1, x1:x1 + 12] == SOLID)
    outside_wall = np.ones(frame.shape[:2], dtype=bool)
    outside_wall[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = False
    assert np.all(frame[outside_wall] == BACKGROUND)
    assert set(map(tuple, np.unique(frame.reshape(-1, 3), axis=0))) == {BACKGROUND, SOLID, HEAD_COLOR, BODY_COLOR, FOOD_COLOR}
    geometry = game.snapshot()["sensory_geometry"]
    assert geometry == {"projection": "fixed_world", "tile_pixels": 12,
                        "board_origin": list(origin), "board_cells": size,
                        "wall_cells": 1, "image_size": [320, 180]}
    assert game.snapshot()["sensory_version"] == "fixed-world-distinct-body-v4"


def test_old_episode_receipt_and_conflicting_score_alias_are_rejected():
    game = fixture_game()
    receipt = fixture_receipt(game)
    game.new_episode()
    before = game.snapshot()
    assert not game.apply_receipt(receipt)["accepted"]
    assert game.snapshot() == before
    receipt = fixture_receipt(game)
    receipt["action_scores"] = copy.deepcopy(receipt["scores"])
    receipt["action_scores"]["forward"] += 1
    assert not game.apply_receipt(receipt)["accepted"]
    assert game.snapshot() == before
