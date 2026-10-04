"""Fixed neural readout. This module has no game-state or image inputs.

Counts are NEW spikes from one decision window; no EMA or previous action is
used. The mappings below are an engineered BCI, not natural snake functions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


ACTIONS = ("left", "forward", "right")
CELL_GROUPS = {"left": (10162,), "forward": (10527, 555871), "right": (10059,)}
DECODER_VERSION = "malecns-dnp20-dnpe017-turn-margin-v2"


@dataclass(frozen=True)
class DecoderConfig:
    """Locked engineering defaults; not claimed physiological calibration."""

    baselines: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    minimum_score: float = 1e-9
    tie_tolerance: float = 1e-9
    turn_relative_margin: float = 0.20

    def __post_init__(self):
        object.__setattr__(self, "baselines", tuple(float(x) for x in self.baselines))
        object.__setattr__(self, "scales", tuple(float(x) for x in self.scales))
        if len(self.baselines) != 3 or len(self.scales) != 3:
            raise ValueError("Exactly three locked baselines and scales are required")
        values = (*self.baselines, *self.scales, self.minimum_score,
                  self.tie_tolerance, self.turn_relative_margin)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Decoder parameters must be finite")
        if (any(x < 0 for x in self.baselines) or any(x <= 0 for x in self.scales)
                or self.minimum_score <= 0 or self.tie_tolerance < 0
                or not 0 <= self.turn_relative_margin < 1):
            raise ValueError("Baselines/tolerance must be nonnegative; scales/threshold positive")

    def as_dict(self) -> dict:
        return {"baselines": list(self.baselines), "scales": list(self.scales),
                "minimum_score": self.minimum_score, "tie_tolerance": self.tie_tolerance,
                "turn_relative_margin": self.turn_relative_margin}


DEFAULT_CONFIG = DecoderConfig()


def decode_counts(group_counts: Mapping[str, Sequence[int]], duration_s: float,
                  config: DecoderConfig | None = None) -> dict:
    """Return a JSON-compatible decision, including explicit NO_ACTION reasons.

    ``duration_s`` is neural seconds, not wall time. Each group contains counts
    in CELL_GROUPS order. Malformed/NaN data cannot produce an executable action.
    Sampling extension and pause policy belong to the neural runner, not here.
    """
    cfg = config or DEFAULT_CONFIG
    result = {"action": None, "reason": "invalid_counts", "rates": {}, "scores": {},
              "new_spike_counts": {}, "selected_cell_ids": {k: list(v) for k, v in CELL_GROUPS.items()},
              "fixed_decoder_version": DECODER_VERSION, "decoder_config": cfg.as_dict()}
    try:
        duration = float(duration_s)
    except (TypeError, ValueError, OverflowError):
        result["reason"] = "invalid_duration"
        return result
    if isinstance(duration_s, bool) or not math.isfinite(duration) or duration <= 0:
        result["reason"] = "invalid_duration"
        return result
    if not isinstance(group_counts, Mapping) or set(group_counts) != set(ACTIONS):
        return result
    parsed = {}
    for action in ACTIONS:
        try:
            counts = list(group_counts[action])
        except (TypeError, ValueError):
            return result
        if len(counts) != len(CELL_GROUPS[action]):
            return result
        parsed[action] = []
        for count in counts:
            try:
                value = float(count)
            except (TypeError, ValueError, OverflowError):
                return result
            if not math.isfinite(value):
                result["reason"] = "nonfinite_counts"
                return result
            if isinstance(count, (bool, str, bytes)) or value < 0 or value > 2**53 or not value.is_integer():
                return result
            parsed[action].append(int(value))
    result["new_spike_counts"] = parsed
    for index, action in enumerate(ACTIONS):
        try:
            rate = sum(parsed[action]) / len(parsed[action]) / duration
        except (OverflowError, ZeroDivisionError):
            result["reason"] = "nonfinite_scores"
            return result
        score = max(0.0, (rate - cfg.baselines[index]) / cfg.scales[index])
        if not math.isfinite(rate) or not math.isfinite(score):
            result["reason"] = "nonfinite_scores"
            return result
        result["rates"][action] = rate
        result["scores"][action] = score
    if not any(sum(counts) for counts in parsed.values()):
        result["reason"] = "silent"
        return result
    highest = max(result["scores"].values())
    if highest < cfg.minimum_score:
        result["reason"] = "below_threshold"
        return result
    # The paired left/right descending channels express a turn choice. When
    # neither side dominates by more than the locked relative margin, treat
    # their neural output as a request to continue forward. Silence remains
    # NO_ACTION and is handled by the coordinator's punishment protocol.
    turn_high = max(result["scores"]["left"], result["scores"]["right"])
    if turn_high >= cfg.minimum_score:
        turn_gap = abs(result["scores"]["left"] - result["scores"]["right"])
        result["turn_relative_difference"] = turn_gap / turn_high
        if result["turn_relative_difference"] <= cfg.turn_relative_margin:
            result.update(action="forward", reason="turn_channels_close")
            return result
    winners = [a for a in ACTIONS if highest - result["scores"][a] <= cfg.tie_tolerance]
    if len(winners) != 1:
        result["reason"] = "tie"
        return result
    result.update(action=winners[0], reason="ok")
    return result


# Descriptive alias for callers that prefer the verb used in the design.
decode = decode_counts
