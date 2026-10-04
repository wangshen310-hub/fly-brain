"""Local viewer and strict neural-action coordinator. No game policy lives here."""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .brain_worker import brain_entry
from .game import SnakeGame

ROOT = Path(__file__).resolve().parents[1]
# Optional persistence override; web assets and neural data keep their own ROOT.
# None intentionally resolves ROOT at call time, including in test harnesses.
STORAGE_ROOT = None


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(serializable(payload), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Session:
    def __init__(self, seed=20260912, autorun=True, resume=True, storage_root=None):
        self.lock = threading.RLock()
        self.commands = queue.Queue()
        self.running = bool(autorun)
        self.resume = resume
        self.closed = False
        self.game = SnakeGame(seed=seed)
        self.status = "loading"
        self.message = "准备真实 MaleCNS 神经模型"
        self.brain = {"loaded": False}
        self.last_decision = None
        self.last_feedback = None
        self.events = collections.deque(maxlen=70)
        self.stats = {"episodes": 1, "total_food": 0, "deaths": 0, "steps": 0, "no_action": 0}
        self.checkpoint_id = "initial"
        self.memory_dirty = False
        self.last_saved_monotonic = time.monotonic()
        self.autosave = {"enabled": True, "interval_seconds": 60,
                         "last_saved_at": None, "last_reason": None, "saving": False}
        self.session_id = uuid.uuid4().hex
        self.storage_root = Path(storage_root or STORAGE_ROOT or ROOT).resolve()
        self.run_dir = self.storage_root / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + self.session_id[:8])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.latest_pointer = self.storage_root / "runs" / "latest_memory.json"
        self.pending_path = self.storage_root / "runs" / "pending_feedback.json"
        self.journal = self.run_dir / "events.jsonl"
        self.parent, child = mp.get_context("spawn").Pipe()
        self.process = mp.get_context("spawn").Process(target=brain_entry, args=(child,), daemon=True)
        self.observation = b""
        self.observation_version = 0
        self._refresh_observation()
        self.event("system", "等待真实数据和神经内核；没有替代控制器")
        self.process.start()
        child.close()
        self.worker = threading.Thread(target=self.loop, name="neural-coordinator", daemon=True)
        self.worker.start()

    def persistence_root(self):
        return Path(getattr(self, "storage_root", None) or STORAGE_ROOT or ROOT).resolve()

    def event(self, kind, message, **details):
        record = {"time": utc_now(), "kind": kind, "message": message, **serializable(details)}
        with self.lock:
            self.events.appendleft(record)
            with self.journal.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _publish_observation(self, rgb, *, kind, source=None, active=False,
                             feedback_kind=None, event_id=None, view="fixed_world_north_up"):
        """Publish pixels and provenance together; previous versions are immutable.

        This is observer metadata only. It never changes the pixels sent to the
        neural process, decoder inputs, or game actions.
        """
        frame = np.asarray(rgb)
        if frame.shape != (180, 320, 3) or frame.dtype != np.uint8:
            raise ValueError("Sensory snapshot must be 320x180 uint8 RGB")
        frame = np.ascontiguousarray(frame)
        output = io.BytesIO()
        Image.fromarray(frame).save(output, format="PNG")
        png = output.getvalue()
        labels = {"decision": "决策输入 · 整个棋盘位置与朝向固定",
                  "feedback": "奖惩画面 · 保持本次观察输入",
                  "ready": "下一次观察 · 等待送入神经模型",
                  "snapshot": "保留的感官快照 · 当前未输入",
                  "neutral_warmup": "中性预热输入 · 全灰 RGB 128"}
        if kind not in labels:
            raise ValueError("Unknown sensory frame kind")
        source = source or {}
        with self.lock:
            if not hasattr(self, "_observation_cache"):
                self._observation_cache = collections.OrderedDict()
            self.observation_version = getattr(self, "observation_version", 0) + 1
            self.observation = png  # Backward-compatible unversioned endpoint.
            self._observation_cache[self.observation_version] = png
            while len(self._observation_cache) > 64:
                self._observation_cache.popitem(last=False)
            self._sensory = {
                "version": self.observation_version,
                "data_url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                "rgb_sha256": hashlib.sha256(frame.tobytes(order="C")).hexdigest(),
                "width": 320, "height": 180, "kind": kind, "view": view,
                "source_episode_id": source.get("episode_id"),
                "source_step_id": source.get("step_id"),
                "source_heading": source.get("direction"),
                "neural_input_active": bool(active), "label": labels[kind],
                "feedback_kind": feedback_kind, "event_id": event_id,
            }

    def _refresh_observation(self):
        with self.lock:
            source = self.game.snapshot()
            kind = "ready" if self.running and not self.game.terminal else "snapshot"
            self._publish_observation(self.game.render_observation(), kind=kind, source=source)

    def _idle_observation(self):
        """Keep the last exact frame, but stop presenting it as active input."""
        with self.lock:
            if not hasattr(self, "_sensory"):
                self._refresh_observation()
                return
            old = self._sensory
            if old["kind"] == "snapshot" and not old["neural_input_active"]:
                return
            source = {"episode_id": old["source_episode_id"], "step_id": old["source_step_id"],
                      "direction": old["source_heading"]}
            rgb = np.asarray(Image.open(io.BytesIO(self.observation)).convert("RGB"), dtype=np.uint8)
            self._publish_observation(rgb, kind="snapshot", source=source, view=old["view"],
                                      feedback_kind=old["feedback_kind"], event_id=old["event_id"])

    def observation_frame(self, version=None):
        """A versioned URL never aliases newer bytes; at most 64 frames remain."""
        with self.lock:
            if version is None:
                return 200, self.observation
            if type(version) is not int or version < 1:
                return 404, {"error": "Unknown observation version"}
            cached = getattr(self, "_observation_cache", {}).get(version)
            if cached is not None:
                return 200, cached
            if version <= self.observation_version:
                return 410, {"error": "Observation version expired"}
            return 404, {"error": "Unknown observation version"}

    def _reset_brain_with_sensory(self):
        # Matches RealFlyBrain.reset_episode exactly: RGB128 for 100ms.
        neutral = np.full((180, 320, 3), 128, dtype=np.uint8)
        with self.lock:
            self.status = "warming"
            self.message = "神经短时状态复位，中性灰色输入预热 100 ms；保留长期记忆"
            self._publish_observation(neutral, kind="neutral_warmup", view="neutral", active=True)
        try:
            metadata = self.rpc("reset_episode")
        except Exception:
            self._idle_observation()
            raise
        self.update_metadata(metadata)
        with self.lock:
            self._publish_observation(neutral, kind="neutral_warmup", view="neutral", active=False)
        return metadata

    def snapshot(self):
        with self.lock:
            game = self.game.snapshot()
            game.setdefault("episode", game.get("episode_id"))
            game.setdefault("step", game.get("step_id"))
            return serializable({
                "status": self.status, "message": self.message,
                "running": self.running, "brain": self.brain, "game": game,
                "last_decision": self.last_decision, "last_feedback": self.last_feedback,
                "stats": self.stats, "events": list(self.events),
                "autosave": self.autosave,
                "sensory": self._sensory,
                "observation_url": f"/api/observation.png?v={self.observation_version}",
                "run_directory": str(self.run_dir), "learning": True,
                "control_source": "MaleCNS v1.0 neural process; no scripted action fallback",
            })

    def command(self, action):
        with self.lock:
            if action == "run":
                if self.status == "error":
                    raise ValueError("神经进程已停止，请检查日志后重启程序")
                self.running = True
                if self.brain.get("loaded"):
                    self.message = "等待下一次神经决策"
            elif action == "pause":
                self.running = False
                self.message = "正在暂停；已发生的奖惩会先完整交付"
                if not getattr(self, "_sensory", {}).get("neural_input_active", False):
                    self._idle_observation()
            elif action in ("save", "reset"):
                self.commands.put(action)
            else:
                raise ValueError("不支持的操作")
        self.event("command", {"run": "开始神经控制", "pause": "请求暂停", "save": "请求保存记忆", "reset": "请求新棋局（保留记忆）"}[action])
        return {"ok": True, "action": action}

    def receive(self):
        while not self.closed:
            if self.parent.poll(0.25):
                packet = self.parent.recv()
                if packet["type"] == "progress":
                    details = packet["message"]
                    message = details.get("message", str(details)) if isinstance(details, dict) else str(details)
                    with self.lock:
                        self.message = message
                    self.event("loading", message)
                    continue
                if packet["type"] == "error":
                    (self.run_dir / "neural-error.txt").write_text(packet.get("traceback", ""), encoding="utf-8")
                    raise RuntimeError(packet["message"])
                return packet["result"]
            if not self.process.is_alive():
                raise RuntimeError(f"真实神经进程退出（{self.process.exitcode}）；游戏已停止")
        raise RuntimeError("Session closed")

    def rpc(self, operation, **fields):
        self.parent.send({"operation": operation, **fields})
        return self.receive()

    def update_metadata(self, metadata):
        with self.lock:
            self.brain.update(serializable(metadata))
            self.brain["loaded"] = True
            self.brain.setdefault("edges", self.brain.get("connections"))
            self.brain.setdefault("contacts", self.brain.get("synaptic_contacts"))

    def save_memory(self, *, completed_event_id=None, make_latest=True, reason="manual"):
        checkpoint_id = f"brain-{time.time_ns()}"
        storage = self.persistence_root()
        folder = storage / "runs" / "checkpoints" / checkpoint_id
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "brain.npz"
        with self.lock:
            self.autosave["saving"] = True
        try:
            result = self.rpc("checkpoint", path=str(path))
        except Exception:
            with self.lock:
                self.autosave["saving"] = False
            raise
        files = [path, path.with_suffix(".npz.adapter.json")]
        record = {
            "checkpoint_id": checkpoint_id, "path": str(path.relative_to(storage)),
            "files": {str(p.relative_to(storage)): file_hash(p) for p in files},
            "completed_event_id": completed_event_id, "created_at": utc_now(),
            "memory": result.get("memory"), "reason": reason,
        }
        atomic_json(folder / "commit.json", record)
        if make_latest:
            atomic_json(self.latest_pointer, record)
            with self.lock:
                self.memory_dirty = False
                self.last_saved_monotonic = time.monotonic()
                self.autosave.update(last_saved_at=record["created_at"], last_reason=reason)
        self.checkpoint_id = checkpoint_id
        with self.lock:
            self.autosave["saving"] = False
        self.event("checkpoint", "记忆已自动保存" if reason in ("periodic", "paused") else "真实可塑状态已保存", checkpoint=record)
        return record

    def maybe_autosave(self, *, force=False):
        # Only the coordinator calls this, between complete neural operations.
        # A pending feedback transaction must commit before any unrelated save.
        if (not self.memory_dirty or self.pending_path.exists()
                or (not force and time.monotonic() - self.last_saved_monotonic < self.autosave["interval_seconds"])):
            return False
        self.save_memory(reason="paused" if force else "periodic")
        return True

    @staticmethod
    def checked_checkpoint(record, storage_root=None):
        storage = Path(storage_root or STORAGE_ROOT or ROOT).resolve()
        for relative, expected in record["files"].items():
            path = (storage / relative).resolve()
            if not path.is_relative_to((storage / "runs" / "checkpoints").resolve()) or file_hash(path) != expected:
                raise RuntimeError("已保存的神经检查点校验失败")
        checkpoint = (storage / record["path"]).resolve()
        if not checkpoint.is_relative_to((storage / "runs" / "checkpoints").resolve()):
            raise RuntimeError("Checkpoint path is outside its storage root")
        if str(checkpoint.relative_to(storage)) not in record["files"]:
            raise RuntimeError("Checkpoint path is not in its manifest")
        return checkpoint

    def restore_record(self, record):
        path = self.checked_checkpoint(record, storage_root=self.persistence_root())
        self.update_metadata(self.rpc("restore", path=str(path)))
        self.checkpoint_id = record["checkpoint_id"]
        with self.lock:
            self.memory_dirty = False
            self.last_saved_monotonic = time.monotonic()
            self.autosave.update(last_saved_at=record["created_at"], last_reason="restored")

    def schedule_feedback(self, outcome, rgb, source=None):
        # The pre-feedback state is immutable. If power/process loss interrupts
        # delivery, recovery starts here; it never stimulates an unknown state.
        checkpoint = self.save_memory(make_latest=False, reason="feedback_pre")
        event_id = outcome["event_id"]
        storage = self.persistence_root()
        frame_path = storage / "runs" / "checkpoints" / checkpoint["checkpoint_id"] / "feedback.png"
        Image.fromarray(rgb).save(frame_path)
        pending = {"event_id": event_id, "transaction_id": f"{self.session_id}:{event_id}", "kind": outcome["kind"], "pre_checkpoint": checkpoint,
                   "frame": str(frame_path.relative_to(storage)), "frame_sha256": file_hash(frame_path),
                   "outcome": outcome, "created_at": utc_now(),
                   "sensory_source": {key: (source or {}).get(key) for key in ("episode_id", "step_id", "direction")}}
        atomic_json(self.pending_path, pending)
        self.event("feedback_scheduled", "反馈事务已保存，准备真实刺激", pending=pending)

    @staticmethod
    def check_feedback(feedback, observation_hash):
        if feedback.get("observation_sha256") != observation_hash or feedback.get("stimulus_ms") != 200.0:
            raise RuntimeError("反馈刺激的感官帧或交付时长不符合协议")

    def finish_feedback_transaction(self, event_id):
        # Updating this single pointer is the commit boundary for both the
        # native NPZ and adapter metadata; neither file is overwritten in place.
        pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
        if pending["event_id"] != event_id:
            raise RuntimeError("Feedback transaction changed before commit")
        record = self.save_memory(completed_event_id=pending["transaction_id"], reason="feedback")
        self.event("feedback_committed", "反馈与神经记忆已提交", event_id=event_id, checkpoint_id=record["checkpoint_id"])
        self.pending_path.unlink(missing_ok=True)
        return record

    def recover_memory(self):
        latest = json.loads(self.latest_pointer.read_text(encoding="utf-8")) if self.latest_pointer.exists() else None
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text(encoding="utf-8"))
            if latest and latest.get("completed_event_id") == pending["transaction_id"]:
                self.restore_record(latest)
                self.pending_path.unlink()
                self.event("recovery", "反馈已提交，恢复时没有重复施加刺激")
            else:
                self.restore_record(pending["pre_checkpoint"])
                storage = self.persistence_root()
                frame = (storage / pending["frame"]).resolve()
                if not frame.is_relative_to((storage / "runs" / "checkpoints").resolve()) or file_hash(frame) != pending["frame_sha256"]:
                    raise RuntimeError("待恢复的反馈图像校验失败")
                rgb = np.asarray(Image.open(frame).convert("RGB"), dtype=np.uint8)
                source = pending.get("sensory_source") or {
                    "episode_id": pending.get("outcome", {}).get("episode_id"),
                    "step_id": pending.get("outcome", {}).get("step_id"), "direction": None}
                with self.lock:
                    self.status = "feedback"
                    self.message = "从已保存的动作前输入重放未提交奖惩"
                    self._publish_observation(rgb, kind="feedback", source=source, active=True,
                                              feedback_kind=pending["kind"], event_id=pending["event_id"])
                feedback = self.rpc("feedback", kind=pending["kind"], rgb=rgb)
                self.check_feedback(feedback, hashlib.sha256(rgb.tobytes(order="C")).hexdigest())
                with self.lock:
                    self._publish_observation(rgb, kind="feedback", source=source, active=False,
                                              feedback_kind=pending["kind"], event_id=pending["event_id"])
                self.last_feedback = {**serializable(feedback), "kind": pending["kind"],
                                      "reason": pending.get("outcome", {}).get("reason")}
                self.finish_feedback_transaction(pending["event_id"])
                self.event("recovery", "已从事件前状态重放未提交的反馈，保留记忆开始新局")
            self._reset_brain_with_sensory()
            self._refresh_observation()
        elif self.resume and latest:
            self.restore_record(latest)
            self._reset_brain_with_sensory()
            self._refresh_observation()
            self.event("checkpoint", "已恢复上次保存的真实脑记忆")

    @staticmethod
    def normalized_decision(raw):
        result = dict(raw)
        if result.get("action") is not None:
            result["action"] = str(result["action"]).lower()
        for source, target in [("counts", "new_spike_counts"), ("rates_hz", "rates"), ("ids", "selected_cell_ids")]:
            if target not in result and source in result:
                result[target] = result[source]
        for key in ("new_spike_counts", "rates", "scores", "selected_cell_ids"):
            if isinstance(result.get(key), dict):
                result[key] = {str(k).lower(): v for k, v in result[key].items()}
        return result

    def step(self):
        with self.lock:
            before = self.game.snapshot()
            rgb = self.game.render_observation()
            self.status = "thinking"
            self.message = "果蝇神经网络正在计算；棋盘等待新输出"
            self._publish_observation(rgb, kind="decision", source=before, active=True)
        self.memory_dirty = True
        raw = self.rpc("observe", rgb=rgb)
        if raw.get("observation_sha256") != before["observation_hash"] or raw.get("input_sha256") != before["observation_hash"]:
            raise RuntimeError("神经进程返回的感官图像哈希与当前观察不一致")
        decision = self.normalized_decision(raw)
        with self.lock:
            self.last_decision = serializable(decision)
            self.brain["neural_ms"] = raw.get("neural_time_end", raw.get("neural_ms"))
            self.brain["memory"] = serializable(raw.get("memory", {}))
            if not self.running:
                self.status = "paused"
                self.message = "已暂停，未执行刚完成的神经输出"
                self._idle_observation()
                self.event("discarded", "暂停期间的动作未交给游戏", decision=decision)
                return
            if decision.get("action") is None:
                # A failed neural decision is an experimental aversive event,
                # never a movement or a death. Each retry has its own durable
                # transaction even though the board step stays the same.
                self.status = "feedback"
                self.message = "没有唯一有效神经输出，交付惩罚后继续计算"
                self.stats["no_action"] += 1
                outcome = {"kind": "aversive", "reason": "no_action",
                           "event_id": "no-action:" + uuid.uuid4().hex,
                           "episode_id": before["episode_id"], "step_id": before["step_id"],
                           "terminal": False, "score_delta": 0}
                self.event("no_action", self.message, decision=decision, outcome=outcome)
            else:
                receipt = {
                    **decision, "episode_id": before["episode_id"], "step_id": before["step_id"],
                    "observation_hash": before["observation_hash"],
                    "brain_checkpoint_id": self.checkpoint_id,
                    "neural_time_start": float(raw["neural_time_start"]) / 1000,
                    "neural_time_end": float(raw["neural_time_end"]) / 1000,
                }
                outcome = self.game.apply_receipt(receipt)
                if outcome.get("accepted") is False:
                    raise RuntimeError(f"动作凭据被拒绝: {outcome}")
                self.stats["steps"] += 1
                self.event("action", f"神经输出：{decision['action']}", receipt=receipt, outcome=outcome)
            if outcome["kind"] in ("reward", "aversive"):
                self._publish_observation(rgb, kind="feedback", source=before,
                                          feedback_kind=outcome["kind"], event_id=outcome["event_id"])
            else:
                self._refresh_observation()

        if outcome["kind"] in ("reward", "aversive"):
            kind = outcome["kind"]
            reason = outcome.get("reason") or ("food" if kind == "reward" else "death")
            with self.lock:
                self.status = "feedback"
                self.message = ("没有唯一有效神经输出，交付惩罚后继续计算" if reason == "no_action" else
                                "吃到食物，交付奖励刺激" if kind == "reward" else "已死亡，先交付惩罚再重开")
            self.schedule_feedback(outcome, rgb, source=before)
            with self.lock:
                self._publish_observation(rgb, kind="feedback", source=before, active=True,
                                          feedback_kind=kind, event_id=outcome["event_id"])
            feedback = self.rpc("feedback", kind=kind, rgb=rgb)
            self.check_feedback(feedback, before["observation_hash"])
            with self.lock:
                self._publish_observation(rgb, kind="feedback", source=before, active=False,
                                          feedback_kind=kind, event_id=outcome["event_id"])
                self.last_feedback = {**serializable(feedback), "kind": kind, "reason": reason}
                self.brain["neural_ms"] = feedback.get("neural_time_end")
                self.brain["memory"] = serializable(feedback.get("memory_after", {}))
                if reason != "no_action":
                    self.stats["total_food" if kind == "reward" else "deaths"] += 1
                self.event("feedback", "无有效输出惩罚完成，继续计算" if reason == "no_action" else
                           "奖励刺激完成" if kind == "reward" else "惩罚刺激完成",
                           event_id=outcome.get("event_id"), reason=reason, feedback=feedback)
            self.finish_feedback_transaction(outcome["event_id"])
            with self.lock:
                if reason != "no_action":
                    self.game.complete_feedback(outcome["event_id"])
                self._refresh_observation()
        if outcome.get("terminal"):
            self.new_episode(automatic=True)
        with self.lock:
            self.status = "running" if self.running else "paused"
            self.message = "神经控制运行中" if self.running else "已暂停"

    def new_episode(self, automatic=False):
        self.memory_dirty = True
        self._reset_brain_with_sensory()
        with self.lock:
            self.game.new_episode()
            self.stats["episodes"] += 1
            self._refresh_observation()
            self.status = "running" if self.running else "paused"
            self.message = "神经控制运行中" if self.running else "已暂停"
            self.event("episode", "开始新棋局，保留长期可塑状态", automatic=automatic, episode=self.game.snapshot()["episode_id"])

    def loop(self):
        try:
            self.update_metadata(self.receive())
            self.recover_memory()
            self.event("ready", "真实连接组与内核已加载", model=self.brain)
            with self.lock:
                self.status = "ready"
                self.message = "真实神经模型就绪"
            while not self.closed:
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command == "save":
                    self.save_memory()
                elif command == "reset":
                    self.save_memory(reason="new_episode")
                    self.new_episode()
                elif self.running:
                    self.step()
                else:
                    with self.lock:
                        if self.status != "error":
                            self.status = "paused"
                    time.sleep(0.1)
                self.maybe_autosave(force=not self.running)
        except Exception as error:
            with self.lock:
                self.running = False
                self.status = "error"
                self.message = f"已停止：{error}"
                self.autosave["saving"] = False
                self._idle_observation()
            (self.run_dir / "server-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self.event("error", self.message)

    def close(self):
        self.closed = True
        self.running = False
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith("POST"):
                super().log_message(fmt, *args)

        def respond(self, body, content_type="application/json; charset=utf-8", status=200):
            if not isinstance(body, bytes):
                body = json.dumps(serializable(body), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                return self.respond(session.snapshot())
            if path == "/api/observation.png":
                versions = parse_qs(urlparse(self.path).query, keep_blank_values=True).get("v")
                if versions is None:
                    version = None
                elif len(versions) == 1 and len(versions[0]) <= 20 and versions[0].isascii() and versions[0].isdigit():
                    version = int(versions[0])
                else:
                    return self.respond({"error": "Unknown observation version"}, status=404)
                status, body = session.observation_frame(version)
                return self.respond(body, "image/png" if status == 200 else "application/json; charset=utf-8", status=status)
            files = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in files:
                name, content_type = files[path]
                return self.respond((ROOT / "web" / name).read_bytes(), content_type)
            return self.respond({"error": "Not found"}, status=404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.respond({"error": "Origin denied"}, status=403)
            if urlparse(self.path).path != "/api/command":
                return self.respond({"error": "Not found"}, status=404)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2048 or "application/json" not in self.headers.get("Content-Type", ""):
                    raise ValueError("Expected a small JSON command")
                payload = json.loads(self.rfile.read(size))
                self.respond(session.command(payload.get("action")))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.respond({"error": str(error)}, status=400)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Do not restore saved memory; does not delete it")
    parser.add_argument("--storage-root", type=Path, help="Separate directory for run logs and saved neural memory")
    args = parser.parse_args()
    session = Session(seed=args.seed, autorun=not args.paused, resume=not args.fresh, storage_root=args.storage_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"Fly Brain Snake http://127.0.0.1:{args.port} — run log {session.run_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
