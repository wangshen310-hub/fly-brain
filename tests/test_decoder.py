"""Synthetic spike fixtures test the interface only, never control a demo."""
import inspect
import math

import pytest

from flysnake.decoder import CELL_GROUPS, DecoderConfig, decode_counts


def test_mean_rates_do_not_double_count_the_two_cell_forward_group():
    result = decode_counts({"left": [3], "forward": [2, 2], "right": [1]}, .1)
    assert result["action"] == "left"
    assert result["rates"] == {"left": 30., "forward": 20., "right": 10.}
    assert result["selected_cell_ids"] == {k: list(v) for k, v in CELL_GROUPS.items()}


@pytest.mark.parametrize("counts,reason", [
    ({"left": [0], "forward": [0, 0], "right": [0]}, "silent"),
    ({"left": [math.nan], "forward": [0, 0], "right": [0]}, "nonfinite_counts"),
    ({"left": [math.inf], "forward": [0, 0], "right": [0]}, "nonfinite_counts"),
    ({"left": [-1], "forward": [0, 0], "right": [0]}, "invalid_counts"),
    ({"left": [1.5], "forward": [0, 0], "right": [0]}, "invalid_counts"),
    ({"left": [True], "forward": [0, 0], "right": [0]}, "invalid_counts"),
    ({"left": ["3"], "forward": [0, 0], "right": [0]}, "invalid_counts"),
    ({"left": [3], "forward": [0], "right": [0]}, "invalid_counts"),
])
def test_silence_ties_and_invalid_values_never_choose_an_action(counts, reason):
    result = decode_counts(counts, .1)
    assert result["action"] is None
    assert result["reason"] == reason


@pytest.mark.parametrize("duration", [0, -1, math.nan, math.inf, None, True])
def test_invalid_neural_duration(duration):
    assert decode_counts({"left": [2], "forward": [0, 0], "right": [0]}, duration)["action"] is None


def test_fixed_scaling_and_threshold_without_per_frame_normalization():
    cfg = DecoderConfig(baselines=(10., 0., 0.), scales=(2., 1., 1.), minimum_score=15.)
    result = decode_counts({"left": [3], "forward": [1, 1], "right": [0]}, .1, cfg)
    assert result["scores"] == {"left": 10., "forward": 10., "right": 0.}
    assert result["action"] is None
    assert result["reason"] == "below_threshold"


def test_left_right_swap_swaps_action_without_other_state():
    original = decode_counts({"left": [4], "forward": [1, 1], "right": [2]}, .1)
    mirrored = decode_counts({"left": [2], "forward": [1, 1], "right": [4]}, .1)
    assert (original["action"], mirrored["action"]) == ("left", "right")


@pytest.mark.parametrize("counts", [
    {"left": [10], "forward": [0, 0], "right": [8]},
    {"left": [8], "forward": [0, 0], "right": [10]},
    {"left": [2], "forward": [2, 2], "right": [2]},
])
def test_close_turn_channels_choose_forward(counts):
    result = decode_counts(counts, .1)
    assert result["action"] == "forward"
    assert result["reason"] == "turn_channels_close"
    assert result["turn_relative_difference"] <= .20


def test_turn_channel_difference_above_margin_uses_stronger_turn():
    assert decode_counts({"left": [10], "forward": [0, 0], "right": [7]}, .1)["action"] == "left"
    assert decode_counts({"left": [7], "forward": [0, 0], "right": [10]}, .1)["action"] == "right"


def test_no_ema_or_previous_action_contamination():
    decode_counts({"left": [20], "forward": [0, 0], "right": [0]}, .1)
    assert decode_counts({"left": [0], "forward": [0, 0], "right": [0]}, .1)["reason"] == "silent"


def test_decoder_has_no_game_inputs_and_rejects_extra_group_metadata():
    assert tuple(inspect.signature(decode_counts).parameters) == ("group_counts", "duration_s", "config")
    counts = {"left": [3], "forward": [1, 1], "right": [0]}
    assert decode_counts(counts, .1) == decode_counts(counts, .1)
    assert decode_counts({**counts, "food": [4, 4]}, .1)["action"] is None
    with pytest.raises(TypeError):
        decode_counts(counts, .1, board="any", reward=1)


def test_config_rejects_nonfinite_or_unusable_constants():
    for kwargs in ({"scales": (1., 0., 1.)}, {"minimum_score": 0},
                   {"baselines": (math.nan, 0., 0.)}, {"baselines": (-1., 0., 0.)},
                   {"tie_tolerance": -1}, {"turn_relative_margin": -0.1},
                   {"turn_relative_margin": 1.0}):
        with pytest.raises(ValueError):
            DecoderConfig(**kwargs)


def test_calibration_arrays_cannot_be_mutated_after_locking():
    baselines = [0., 0., 0.]
    config = DecoderConfig(baselines=baselines)
    baselines[0] = 900.
    assert config.baselines == (0., 0., 0.)


def test_huge_counts_and_overflowing_rates_return_no_action():
    assert decode_counts({"left": [10**300], "forward": [0, 0], "right": [0]}, .1)["action"] is None
    assert decode_counts({"left": [1], "forward": [0, 0], "right": [0]}, 1e-320)["action"] is None
