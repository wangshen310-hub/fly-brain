"""Real retained MaleCNS dynamics behind an RGB-only snake interface.

No game state, navigation policy, fallback brain, or action replacement belongs
in this module. The public methods are intended to run in a dedicated process.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
VENDOR = PROJECT / "vendor"
UPSTREAM_COMMIT = "78ef3e05ab0fa086032098558d893667068944a0"
ADAPTER_VERSION = "flysnake-real-malecns-v1"
FEEDBACK_CURRENT_MV_EQUIVALENT = 30.0
DEFAULT_DATA = PROJECT / "data" / "malecns_v1"


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RealFlyBrain:
    """Only the verified biological graph can construct this adapter.

    `progress` receives small dictionaries suitable for a worker status queue.
    `learning=False` freezes both long-term u and w (not merely the reward).
    """

    def __init__(self, progress: Callable[[dict], None] | None = None, *, learning: bool = True):
        self.progress = progress
        self.learning = bool(learning)
        self.data = Path(os.environ.get("FLYSNAKE_DATA", DEFAULT_DATA)).resolve()
        receipt = self.data / "prepared.json"
        if not receipt.is_file():
            raise RuntimeError("Real MaleCNS data is not prepared. Run scripts/prepare_brain.py; no fallback exists.")
        self.prepared = json.loads(receipt.read_text(encoding="utf-8"))
        if self.prepared.get("upstream_commit") != UPSTREAM_COMMIT or self.prepared.get("verified_arrays") is not True:
            raise RuntimeError("Unverified or incompatible MaleCNS preparation receipt")
        with (self.data / "graph.npz").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != self.prepared.get("graph_sha256"):
                raise RuntimeError("Prepared graph bytes changed; refusing an unverified graph")
        os.environ["FLYSNAKE_DATA"] = str(self.data)
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        self._report("loading", "Loading verified MaleCNS graph and annotated circuits")
        from stonkfly.neural.common import annotations
        from stonkfly.neural.visual import VisualMemoryBrain

        self.brain = VisualMemoryBrain()
        self.brain.weights_frozen = not self.learning
        b = self.brain
        if (b.n, len(b.post), len(b.retina), len(b.r8), len(b.circuit["edges"])) != (166700, 25582938, 3335, 811, 7835):
            raise RuntimeError("The loaded graph/circuits differ from the pinned MaleCNS model")
        a = annotations(b.ids)
        kinds = a.type.fillna("")
        sides = a.somaSide.fillna("")
        self.groups = {
            "left": np.flatnonzero(kinds.eq("DNp20") & sides.eq("L")),
            "forward": np.flatnonzero(kinds.eq("DNpe017")),
            "right": np.flatnonzero(kinds.eq("DNp20") & sides.eq("R")),
        }
        self.ids = {key: [int(b.ids[i]) for i in indices] for key, indices in self.groups.items()}
        if self.ids != {"left": [10162], "forward": [10527, 555871], "right": [10059]}:
            raise RuntimeError(f"Unexpected annotated action cells: {self.ids}")
        del a, kinds, sides
        self.total_neural_ms = 0.0
        self.observations = 0
        self.feedback_events = 0
        self.episode_resets = 0
        self.last_result = None
        self._report("ready", "Real neural kernel loaded", neurons=b.n, edges=len(b.post))

    def _report(self, stage: str, message: str, **details) -> None:
        if self.progress is not None:
            self.progress({"stage": stage, "message": message, **details})

    @staticmethod
    def _rgb(rgb) -> np.ndarray:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
            raise ValueError("RGB input must be a nonempty uint8 H x W x 3 image")
        return np.ascontiguousarray(frame)

    def _integrate(self, frame: np.ndarray, duration_ms: float, *, stimulation=None, learning=None):
        counts, native_wall = self.brain.rgb_step(
            frame, duration_ms,
            learning=self.learning if learning is None else bool(learning),
            stimulation=stimulation,
        )
        if not np.isfinite(self.brain.v).all() or not np.isfinite(self.brain.g).all():
            raise FloatingPointError("Nonfinite neural state; execution stopped")
        if np.any(counts < 0):
            raise FloatingPointError("Invalid neural spike counts")
        self.total_neural_ms += duration_ms
        return counts, native_wall

    def observe(self, rgb) -> dict:
        """Accumulate fresh spikes for 100--500 ms; never break ties by policy."""
        from .decoder import decode_counts

        frame = self._rgb(rgb)
        start_wall = time.perf_counter()
        start_ms = self.total_neural_ms
        aggregate = np.zeros(self.brain.n, dtype=np.int32)
        native_wall = 0.0
        decision = None
        groups = {}
        elapsed_ms = 0.0
        for window in range(1, 6):
            counts, duration = self._integrate(frame, 100.0)
            aggregate += counts
            native_wall += duration
            elapsed_ms = float(window * 100)
            groups = {key: aggregate[indices].tolist() for key, indices in self.groups.items()}
            decision = decode_counts(groups, elapsed_ms / 1000.0)
            if decision["action"] is not None:
                break
            if decision["reason"] not in ("silent", "tie", "below_threshold"):
                raise FloatingPointError(f"Invalid neural decoding: {decision['reason']}")
            self._report("sampling", "No unique effective neural output; extending same image", neural_ms=elapsed_ms)
        assert decision is not None
        self.brain.counts[:] = aggregate
        self.observations += 1
        result = {
            **decision,
            "ids": self.ids,
            "counts": groups,
            "rates_hz": decision.get("rates", {}),
            "neural_ms": elapsed_ms,
            "neural_time_start": start_ms,
            "neural_time_end": self.total_neural_ms,
            "brain_ms": self.brain.sim_ms,
            "compute_seconds": time.perf_counter() - start_wall,
            "native_compute_seconds": native_wall,
            "observation_sha256": _digest(frame),
            "input_sha256": _digest(frame),
            "spike_sha256": _digest(aggregate),
            "total_spikes": int(aggregate.sum(dtype=np.int64)),
            "KC_spikes": int(aggregate[self.brain.circuit["kc"]].sum()),
            "memory": self.memory(),
            "adapter_version": ADAPTER_VERSION,
        }
        self.last_result = result
        return result

    def feedback(self, kind: str = "reward", rgb=None) -> dict:
        """Deliver all 200 ms before return; counts here never become an action."""
        if kind not in ("reward", "aversive"):
            raise ValueError("Feedback must be reward or aversive")
        frame = self._rgb(rgb)
        start_wall = time.perf_counter()
        start_ms = self.total_neural_ms
        before = self.memory()
        targets = self.brain.circuit[kind]
        counts, native_wall = self._integrate(frame, 200.0, stimulation=(targets, FEEDBACK_CURRENT_MV_EQUIVALENT))
        self.feedback_events += 1
        return {
            "stimulus": kind,
            "kind": kind,
            "stimulus_ms": 200.0,
            "stimulus_current_mV_equivalent": FEEDBACK_CURRENT_MV_EQUIVALENT,
            "delivered_cells": [str(self.brain.ids[i]) for i in targets],
            "reward_spikes": int(counts[self.brain.circuit["reward"]].sum()),
            "aversive_spikes": int(counts[self.brain.circuit["aversive"]].sum()),
            "KC_spikes": int(counts[self.brain.circuit["kc"]].sum()),
            "total_spikes": int(counts.sum(dtype=np.int64)),
            "spike_sha256": _digest(counts),
            "observation_sha256": _digest(frame),
            "neural_time_start": start_ms,
            "neural_time_end": self.total_neural_ms,
            "neural_ms": 200.0,
            "compute_seconds": time.perf_counter() - start_wall,
            "native_compute_seconds": native_wall,
            "memory_before": before,
            "memory_after": self.memory(),
            "memory": self.memory(),
            "action": None,
        }

    def memory(self) -> dict:
        b = self.brain
        return {
            **b.memory(),
            "u_sha256": _digest(b.memory_u),
            "w_sha256": _digest(b.memory_w),
            "u_nonzero": int(np.count_nonzero(b.memory_u)),
            "w_nonzero": int(np.count_nonzero(b.memory_w)),
            "u_min": float(b.memory_u.min()),
            "u_max": float(b.memory_u.max()),
            "w_min": float(b.memory_w.min()),
            "w_max": float(b.memory_w.max()),
            "learning_enabled": self.learning,
            "weights_frozen": bool(b.weights_frozen),
        }

    def reset_episode(self) -> dict:
        """Preserve effective weights AND u/w; reset short state, neutral warm-up."""
        b = self.brain
        before = self.memory()
        start = self.total_neural_ms
        b.reset(keep_memory=True)
        previously_frozen = b.weights_frozen
        b.weights_frozen = True
        neutral = np.full((180, 320, 3), 128, dtype=np.uint8)
        try:
            counts, _ = self._integrate(neutral, 100.0, learning=False)
        finally:
            b.weights_frozen = previously_frozen
        # Clear eligibility and rate histories only; retain the warmed physical
        # state and slow u/w. Last-update times match the post-warm-up cursor.
        b.eligibility.fill(0)
        b.eligibility_last.fill(b.cursor)
        b.rate_kc.fill(0)
        b.rate_dan.fill(0)
        b.counts.fill(0)
        after = self.memory()
        for name in ("sha256", "u_sha256", "w_sha256"):
            if before[name] != after[name]:
                raise RuntimeError("Episode reset changed retained long-term memory")
        self.episode_resets += 1
        self.last_result = None
        return {
            "protocol": "reset-short-state-neutral-100ms-frozen-clear-traces-v1",
            "warmup_ms": 100.0,
            "neural_time_start": start,
            "neural_time_end": self.total_neural_ms,
            "memory": after,
        }

    def checkpoint(self, path) -> dict:
        from .decoder import DECODER_VERSION, DEFAULT_CONFIG

        target = Path(path)
        self.brain.checkpoint(target)
        with target.open("rb") as stream:
            native_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {
            "adapter_version": ADAPTER_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "total_neural_ms": self.total_neural_ms,
            "observations": self.observations,
            "feedback_events": self.feedback_events,
            "feedback_current_mV_equivalent": FEEDBACK_CURRENT_MV_EQUIVALENT,
            "episode_resets": self.episode_resets,
            "learning": self.learning,
            "native_checkpoint_sha256": native_sha,
            "decoder_version": DECODER_VERSION,
            "decoder_config": DEFAULT_CONFIG.as_dict(),
            "ids": self.ids,
            "memory": self.memory(),
        }
        _atomic_json(target.with_suffix(target.suffix + ".adapter.json"), receipt)
        return {"path": str(target), **receipt}

    def restore(self, path) -> dict:
        from .decoder import DECODER_VERSION, DEFAULT_CONFIG

        target = Path(path)
        metadata = json.loads(target.with_suffix(target.suffix + ".adapter.json").read_text(encoding="utf-8"))
        if metadata.get("adapter_version") != ADAPTER_VERSION or metadata.get("upstream_commit") != UPSTREAM_COMMIT or metadata.get("ids") != self.ids:
            raise ValueError("Incompatible adapter checkpoint")
        legacy_decoder = (metadata.get("decoder_version") == "malecns-dnp20-dnpe017-mean-v1"
                          and metadata.get("decoder_config") == {
                              "baselines": [0.0, 0.0, 0.0], "scales": [1.0, 1.0, 1.0],
                              "minimum_score": 1e-09, "tie_tolerance": 1e-09})
        if not legacy_decoder and (metadata.get("decoder_version") != DECODER_VERSION
                                   or metadata.get("decoder_config") != DEFAULT_CONFIG.as_dict()):
            raise ValueError("Checkpoint decoder protocol differs")
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != metadata.get("native_checkpoint_sha256"):
                raise ValueError("Incomplete or mismatched checkpoint/adapter pair; refusing restore")
        self.brain.restore(target)
        self.total_neural_ms = float(metadata["total_neural_ms"])
        self.observations = int(metadata["observations"])
        self.feedback_events = int(metadata["feedback_events"])
        self.episode_resets = int(metadata["episode_resets"])
        self.learning = bool(metadata["learning"])
        self.last_result = None
        return {"restored": True, "path": str(target), "memory": self.memory()}

    def metadata(self) -> dict:
        return {
            "adapter_version": ADAPTER_VERSION,
            "model": "stonkfly-dual-compartment-v1",
            "dataset": "MaleCNS v1.0",
            "neurons": self.brain.n,
            "edges": len(self.brain.post),
            "synaptic_contacts": 124177617,
            "retina_inputs": len(self.brain.retina),
            "r8_inputs": len(self.brain.r8),
            "plastic_edges": len(self.brain.circuit["edges"]),
            "selected_cell_ids": self.ids,
            "ids": self.ids,
            "upstream_commit": UPSTREAM_COMMIT,
            "data_license": "CC-BY-4.0",
            "code_license": "MIT",
            "neural_dt_ms": 0.1,
            "total_neural_ms": self.total_neural_ms,
            "build": self.brain.build,
            "visual_model": self.brain.visual_report,
            "circuit": self.brain.circuit["report"],
            "memory": self.memory(),
            "validated_learning": False,
            "feedback_stimulation": {"reward_current_mV_equivalent": FEEDBACK_CURRENT_MV_EQUIVALENT,
                                     "aversive_current_mV_equivalent": FEEDBACK_CURRENT_MV_EQUIVALENT,
                                     "duration_ms": 200.0},
            "assumptions": [
                "LIF and RGB retinal proxies; lamina bias is not calibrated vision",
                "Three-action readout is a fixed engineering assignment",
                "PAM11/PPL101 stimulation and candidate plasticity do not establish learned play",
            ],
        }
