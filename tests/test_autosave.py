"""Memory safety at timer, pause and feedback boundaries; no real brain fixture."""
import json

import pytest

from flysnake import server
from test_server_contract import harness


def test_periodic_save_waits_for_deadline_and_never_rewrites_unchanged_state(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch)
    s.memory_dirty = True
    monkeypatch.setattr(server.time, "monotonic", lambda: 59.0)
    assert not s.maybe_autosave()
    assert not calls
    monkeypatch.setattr(server.time, "monotonic", lambda: 60.0)
    assert s.maybe_autosave()
    record = json.loads(s.latest_pointer.read_text())
    assert record["reason"] == "periodic"
    assert s.autosave["last_saved_at"] == record["created_at"]
    assert s.autosave["saving"] is False
    assert not s.memory_dirty
    monkeypatch.setattr(server.time, "monotonic", lambda: 300.0)
    assert not s.maybe_autosave()
    assert [c[0] for c in calls] == ["checkpoint"]


def test_pause_saves_neural_activity_even_when_completed_action_was_discarded(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch)
    s.running = False
    s.step()
    assert s.game.step_id == 0
    assert s.memory_dirty
    assert s.maybe_autosave(force=True)
    assert [c[0] for c in calls] == ["observe", "checkpoint"]
    assert s.autosave["last_reason"] == "paused"


def test_timer_cannot_commit_over_unfinished_feedback(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch, feedback_failure=True)
    with pytest.raises(RuntimeError, match="injected"):
        s.step()
    calls.clear()
    s.memory_dirty = True
    assert not s.maybe_autosave(force=True)
    assert not calls
    assert s.pending_path.exists()
    assert not s.latest_pointer.exists()


def test_failed_autosave_keeps_last_committed_memory_and_remains_dirty(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch)
    s.save_memory()
    committed = s.latest_pointer.read_bytes()
    s.memory_dirty = True

    def fail(operation, **fields):
        raise OSError("injected disk or worker failure")

    s.rpc = fail
    with pytest.raises(OSError, match="injected"):
        s.maybe_autosave(force=True)
    assert s.latest_pointer.read_bytes() == committed
    assert s.memory_dirty
    assert s.autosave["saving"] is False
