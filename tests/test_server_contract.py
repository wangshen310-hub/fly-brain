"""Coordinator fault injection only; these fixtures never enter production."""
import collections
import hashlib
import json
import threading
from pathlib import Path

import pytest

from flysnake import server
from flysnake.decoder import decode_counts
from flysnake.game import SnakeGame


def harness(tmp_path, monkeypatch, *, wrong_image=False, feedback_failure=False):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    s = object.__new__(server.Session)
    s.lock = threading.RLock()
    s.running = True
    s.resume = True
    s.closed = False
    s.session_id = "test-session-a"
    s.game = SnakeGame(seed=17)
    s.game.snake = [(5, 5), (5, 6), (5, 7)]
    s.game.direction = 0
    s.game.food = (5, 4)
    s.status = "ready"
    s.message = ""
    s.brain = {"loaded": True}
    s.last_decision = None
    s.last_feedback = None
    s.events = collections.deque(maxlen=70)
    s.stats = {"episodes": 1, "total_food": 0, "deaths": 0, "steps": 0, "no_action": 0}
    s.checkpoint_id = "test-only-initial"
    s.memory_dirty = False
    s.last_saved_monotonic = 0.0
    s.autosave = {"enabled": True, "interval_seconds": 60,
                  "last_saved_at": None, "last_reason": None, "saving": False}
    s.run_dir = tmp_path / "runs" / "test"
    s.run_dir.mkdir(parents=True)
    s.journal = s.run_dir / "events.jsonl"
    s.latest_pointer = tmp_path / "runs" / "latest_memory.json"
    s.pending_path = tmp_path / "runs" / "pending_feedback.json"
    s.observation_version = 0
    s._refresh_observation()
    calls = []

    def rpc(operation, **fields):
        calls.append((operation, fields))
        if operation == "observe":
            image_hash = s.game.observation_hash()
            if wrong_image:
                image_hash = "different-frame"
            return {**decode_counts({"left": [0], "forward": [6, 6], "right": [0]}, 0.1),
                    "neural_time_start": 0, "neural_time_end": 100,
                    "observation_sha256": image_hash, "input_sha256": image_hash,
                    "memory": {"fixture": True}}
        if operation == "feedback":
            assert s.pending_path.exists(), "Delivery may only start after durable scheduling"
            if feedback_failure:
                raise RuntimeError("injected process interruption")
            import hashlib
            return {"observation_sha256": hashlib.sha256(fields["rgb"].tobytes()).hexdigest(),
                    "stimulus_ms": 200.0, "neural_time_end": 300,
                    "memory_after": {"fixture": True}}
        if operation == "checkpoint":
            path = Path(fields["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"TEST-ONLY-CHECKPOINT")
            path.with_suffix(".npz.adapter.json").write_text('{"fixture":true}')
            return {"memory": {"fixture": True}}
        if operation in ("restore", "reset_episode"):
            return {"memory": {"fixture": True}}
        raise AssertionError(operation)

    s.rpc = rpc
    return s, calls


def test_wrong_neural_image_cannot_be_relabelled_as_current_observation(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch, wrong_image=True)
    with pytest.raises(RuntimeError, match="哈希"):
        s.step()
    assert s.game.step_id == 0
    assert [c[0] for c in calls] == ["observe"]


def test_reward_delivered_and_committed_before_new_food(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch)
    s.step()
    assert [c[0] for c in calls] == ["observe", "checkpoint", "feedback", "checkpoint"]
    assert s.game.score == 1
    assert s.game.pending_feedback is None
    assert s.game.food is not None
    latest = json.loads(s.latest_pointer.read_text())
    assert latest["completed_event_id"].startswith("test-session-a:")
    assert not s.pending_path.exists()
    assert s.last_feedback["kind"] == "reward"


def test_lethal_neural_move_is_punished_before_neural_and_game_reset(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch)
    s.game.snake = [(5, 0), (5, 1), (5, 2)]
    s.game.food = (9, 9)
    s.step()
    assert [c[0] for c in calls] == ["observe", "checkpoint", "feedback", "checkpoint", "reset_episode"]
    assert calls[2][1]["kind"] == "aversive"
    assert s.stats["deaths"] == 1
    assert s.game.episode_id == 2


def test_failed_feedback_preserves_transaction_and_freezes_game(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch, feedback_failure=True)
    with pytest.raises(RuntimeError, match="injected"):
        s.step()
    assert s.pending_path.exists()
    assert s.game.pending_feedback is not None
    assert s.game.food is None
    assert not s.latest_pointer.exists()
    pending = json.loads(s.pending_path.read_text())
    server.Session.checked_checkpoint(pending["pre_checkpoint"])


def test_recovery_replays_only_uncommitted_feedback_from_pre_state(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch, feedback_failure=True)
    with pytest.raises(RuntimeError):
        s.step()
    old_rpc = s.rpc

    def restored_rpc(operation, **fields):
        if operation == "feedback":
            calls.append((operation, fields))
            return {"stimulus_ms": 200.0,
                    "observation_sha256": hashlib.sha256(fields["rgb"].tobytes()).hexdigest()}
        return old_rpc(operation, **fields)

    s.rpc = restored_rpc
    calls.clear()
    s.recover_memory()
    assert [c[0] for c in calls] == ["restore", "feedback", "checkpoint", "reset_episode"]
    assert not s.pending_path.exists()
    assert s.latest_pointer.exists()


def test_recovery_of_already_committed_event_never_repeats_stimulus(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch, feedback_failure=True)
    with pytest.raises(RuntimeError):
        s.step()
    pending = json.loads(s.pending_path.read_text())
    committed = s.save_memory(completed_event_id=pending["transaction_id"])
    calls.clear()
    s.recover_memory()
    assert [c[0] for c in calls] == ["restore", "reset_episode"]
    assert not s.pending_path.exists()
    assert s.checkpoint_id == committed["checkpoint_id"]


def test_same_game_event_in_different_run_is_not_already_committed(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch, feedback_failure=True)
    with pytest.raises(RuntimeError):
        s.step()
    pending = json.loads(s.pending_path.read_text())
    s.save_memory(completed_event_id="some-other-session:" + pending["event_id"])
    calls.clear()
    with pytest.raises(RuntimeError, match="injected"):
        s.recover_memory()
    assert [c[0] for c in calls] == ["restore", "feedback"]


def test_modified_checkpoint_is_rejected(tmp_path, monkeypatch):
    s, _ = harness(tmp_path, monkeypatch)
    record = s.save_memory()
    (tmp_path / record["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="校验失败"):
        s.restore_record(record)
