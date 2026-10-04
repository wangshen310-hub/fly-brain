"""Snake rules, pixel-only sensation, and a once-per-step neural receipt gate.

There is intentionally no public move(action) or automatic clock. Receipts are
self-consistency checks, not cryptographic proof of their neuronal provenance;
the production runner must be the only writer of the action channel.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw

from .decoder import ACTIONS, CELL_GROUPS, DECODER_VERSION, DEFAULT_CONFIG, DecoderConfig, decode_counts


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
BACKGROUND = (242, 242, 230)
SOLID = (12, 12, 16)
FOOD_COLOR = (32, 110, 238)
HEAD_COLOR = (232, 94, 75)
BODY_COLOR = (70, 160, 110)
TILE_PIXELS = 12
IMAGE_SIZE = (320, 180)
SENSORY_VERSION = "fixed-world-distinct-body-v4"


class SnakeGame:
    def __init__(self, seed: int = 0, size: int = 12, decoder_config: DecoderConfig | None = None):
        if type(size) is not int or size < 3 or size > 12:
            raise ValueError("Board size must be 3..12; production default is 12")
        if type(seed) is not int:
            raise ValueError("An integer seed is required")
        self.seed, self.size = seed, size
        self.decoder_config = decoder_config or DEFAULT_CONFIG
        self._rng = random.Random(seed)
        self.episode_id = 0
        self._pending_feedback = None
        self._feedback_observation = None
        self._completed_feedback = set()
        self.new_episode()

    @property
    def alive(self):
        return not self.terminal

    @property
    def pending_feedback(self):
        return copy.deepcopy(self._pending_feedback)

    def new_episode(self) -> dict:
        """Reset only the game; neural state/reset/long-term memory is external.

        Explicit user resets are allowed, but never while feedback is pending.
        A manual reset is not a death and creates no punishment.
        """
        if self._pending_feedback is not None:
            raise RuntimeError("Complete pending feedback before starting another episode")
        self.episode_id += 1
        self.step_id = 0
        candidates = []
        for direction, (dx, dy) in enumerate(DIRECTIONS):
            for x in range(self.size):
                for y in range(self.size):
                    tail = (x - 2 * dx, y - 2 * dy)
                    if self._inside(tail):
                        candidates.append((x, y, direction))
        x, y, self.direction = self._rng.choice(candidates)
        dx, dy = DIRECTIONS[self.direction]
        self.snake = [(x - i * dx, y - i * dy) for i in range(3)]
        self.food = self._sample_food()
        self.score = 0
        self.terminal = False
        self.death_reason = None
        self.outcome = None
        self.last_event = None
        self._feedback_observation = None
        self._last_neural_end = None
        return self.snapshot()

    def _inside(self, point):
        return 0 <= point[0] < self.size and 0 <= point[1] < self.size

    def _sample_food(self):
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied]
        return self._rng.choice(free) if free else None

    def observation_geometry(self) -> dict:
        """Fixed world projection; geometry depends only on board size."""
        return {"projection": "fixed_world", "tile_pixels": TILE_PIXELS,
                "board_origin": [(IMAGE_SIZE[0] - self.size * TILE_PIXELS) // 2,
                                 (IMAGE_SIZE[1] - self.size * TILE_PIXELS) // 2],
                "board_cells": self.size, "wall_cells": 1,
                "image_size": list(IMAGE_SIZE)}

    def render_observation(self) -> np.ndarray:
        """320x180 RGB with a fixed world camera, square cells and no UI/HUD.

        Each world cell always occupies the same 12x12 pixel block. World +x
        points right and world +y points down, independent of head/heading.
        The board is centered and enclosed by a one-cell dark wall perimeter.
        Default size 12: board (88,18)..(231,161), wall outer bounds
        (76,6)..(243,173), with symmetric bright margins outside the wall.
        Head is coral, body green, food blue, and walls dark. Feedback returns the
        exact pre-action frame required by the neural stimulation protocol.
        """
        if self._feedback_observation is not None:
            return self._feedback_observation.copy()
        image = Image.new("RGB", IMAGE_SIZE, BACKGROUND)
        draw = ImageDraw.Draw(image)
        tile = TILE_PIXELS
        origin_x, origin_y = self.observation_geometry()["board_origin"]
        occupied = set(self.snake)
        for row in range(-1, self.size + 1):
            for col in range(-1, self.size + 1):
                world = (col, row)
                color = SOLID if not self._inside(world) else BACKGROUND
                if world in occupied:
                    color = BODY_COLOR
                if world == self.food:
                    color = FOOD_COLOR
                if world == self.snake[0]:
                    color = HEAD_COLOR
                x0, y0 = origin_x + col * tile, origin_y + row * tile
                draw.rectangle((x0, y0, x0 + tile - 1, y0 + tile - 1), fill=color)
        return np.asarray(image, dtype=np.uint8).copy()

    def observation_hash(self) -> str:
        pixels = self.render_observation()
        return hashlib.sha256(pixels.tobytes(order="C")).hexdigest()

    def snapshot(self) -> dict:
        return {"seed": self.seed, "size": self.size,
                "episode_id": self.episode_id, "episode": self.episode_id,
                "step_id": self.step_id, "step": self.step_id,
                "snake": [list(p) for p in self.snake],
                "direction": self.direction, "food": list(self.food) if self.food is not None else None,
                "alive": self.alive, "terminal": self.terminal, "score": self.score,
                "death_reason": self.death_reason, "outcome": self.outcome,
                "pending_feedback": self.pending_feedback,
                "phase": "feedback" if self._pending_feedback else ("terminal" if self.terminal else "waiting_neural"),
                "sensory_version": SENSORY_VERSION,
                "sensory_geometry": self.observation_geometry(),
                "observation_hash": self.observation_hash()}

    def _reject(self, reason) -> dict:
        return {"accepted": False, "reason": reason, "kind": "none", "terminal": self.terminal,
                "score": self.score, "episode_id": self.episode_id, "step_id": self.step_id}

    def _validate_receipt(self, receipt):
        if not isinstance(receipt, Mapping):
            return "missing_receipt"
        for key, expected in (("episode_id", self.episode_id), ("step_id", self.step_id)):
            if type(receipt.get(key)) is not int or receipt.get(key) != expected:
                return "wrong_" + key
        if receipt.get("observation_hash") != self.observation_hash():
            return "wrong_observation_hash"
        if receipt.get("fixed_decoder_version") != DECODER_VERSION:
            return "wrong_decoder_version"
        if receipt.get("selected_cell_ids") != {k: list(v) for k, v in CELL_GROUPS.items()}:
            return "wrong_selected_cell_ids"
        if receipt.get("decoder_config") != self.decoder_config.as_dict():
            return "wrong_decoder_config"
        if not isinstance(receipt.get("brain_checkpoint_id"), str) or not receipt["brain_checkpoint_id"].strip():
            return "missing_brain_checkpoint_id"
        start, end = receipt.get("neural_time_start"), receipt.get("neural_time_end")
        if isinstance(start, bool) or isinstance(end, bool):
            return "invalid_neural_time"
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError, OverflowError):
            return "invalid_neural_time"
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            return "invalid_neural_time"
        duration = end - start
        if (duration < .1 - 1e-9 or duration > .5 + 1e-9
                or not math.isclose(duration / .1, round(duration / .1), abs_tol=1e-8)):
            return "invalid_sampling_window"
        if self._last_neural_end is not None and start < self._last_neural_end - 1e-12:
            return "reused_neural_window"
        decision = decode_counts(receipt.get("new_spike_counts"), duration, self.decoder_config)
        if decision["action"] is None:
            return "neural_" + decision["reason"]
        if receipt.get("action") != decision["action"]:
            return "action_disagrees_with_neural_output"
        score_fields = [receipt[k] for k in ("scores", "action_scores") if k in receipt]
        if not score_fields:
            return "missing_action_scores"
        for scores in score_fields:
            if not isinstance(scores, Mapping) or set(scores) != set(ACTIONS):
                return "invalid_action_scores"
            for action in ACTIONS:
                try:
                    value = float(scores[action])
                except (TypeError, ValueError, OverflowError):
                    return "invalid_action_scores"
                if not math.isfinite(value) or not math.isclose(value, decision["scores"][action], rel_tol=1e-9, abs_tol=1e-9):
                    return "action_scores_disagree"
        return None

    def apply_receipt(self, receipt: Mapping | None) -> dict:
        """Consume exactly one valid, fresh receipt; never filter lethal actions."""
        if self._pending_feedback is not None:
            return self._reject("feedback_pending")
        if self.terminal:
            return self._reject("episode_terminal")
        invalid = self._validate_receipt(receipt)
        if invalid:
            return self._reject(invalid)
        before_pixels = self.render_observation()
        before_step = self.step_id
        action = receipt["action"]
        direction = (self.direction + {"left": -1, "forward": 0, "right": 1}[action]) % 4
        dx, dy = DIRECTIONS[direction]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        eating = head == self.food
        collision_body = self.snake if eating else self.snake[:-1]
        self.direction = direction
        self.step_id += 1
        self._last_neural_end = float(receipt["neural_time_end"])
        kind = "none"
        collision = None
        if not self._inside(head) or head in collision_body:
            self.terminal = True
            self.death_reason = "wall" if not self._inside(head) else "self"
            self.outcome = "death"
            kind, collision = "aversive", list(head)
        else:
            self.snake.insert(0, head)
            if eating:
                self.score += 1
                self.food = None  # A new picture/food is forbidden until feedback finishes.
                kind = "reward"
                if len(self.snake) == self.size * self.size:
                    self.terminal, self.outcome = True, "win"
            else:
                self.snake.pop()
        event = {"accepted": True, "reason": "ok", "kind": kind,
                 "event_id": f"{self.seed}:{self.episode_id}:{before_step}:{kind}",
                 "episode_id": self.episode_id, "step_id": before_step,
                 "next_step_id": self.step_id, "action": action,
                 "terminal": self.terminal, "score": self.score,
                 "score_delta": 1 if kind == "reward" else 0,
                 "death_reason": self.death_reason, "outcome": self.outcome,
                 "collision_at": collision, "feedback_complete": kind == "none",
                 "observation_hash": receipt["observation_hash"]}
        if kind != "none":
            self._pending_feedback = copy.deepcopy(event)
            self._feedback_observation = before_pixels
        self.last_event = copy.deepcopy(event)
        return event

    def complete_feedback(self, event_id: str) -> dict:
        """Called only after neural stimulus delivery and durable audit commit.

        This releases the game gate, not the stimulus itself. An idempotent
        duplicate returns accepted=False and never resamples food.
        """
        if event_id in self._completed_feedback:
            return self._reject("feedback_already_completed")
        if self._pending_feedback is None or event_id != self._pending_feedback["event_id"]:
            return self._reject("wrong_feedback_event")
        event = self._pending_feedback
        self._completed_feedback.add(event_id)
        self._pending_feedback = None
        self._feedback_observation = None
        if event["kind"] == "reward" and not self.terminal:
            self.food = self._sample_food()
        event["feedback_complete"] = True
        self.last_event = copy.deepcopy(event)
        return copy.deepcopy(event)
