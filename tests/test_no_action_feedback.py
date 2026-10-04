"""TEST-ONLY neural fixtures for punishment and retry, never production actions."""
import hashlib
import json

import numpy as np
import pytest

from flysnake.decoder import decode_counts
from test_server_contract import harness


def test_no_output_punishes_repeatedly_then_only_neural_action_moves(tmp_path, monkeypatch):
    counts = {"left": [0], "forward": [0, 0], "right": [0]}
    s, calls = harness(tmp_path, monkeypatch)
    s.game.food = (9, 9)
    before = s.game.snapshot()
    before_rgb = s.game.render_observation()
    old_rpc = s.rpc
    observations = 0
    transaction_ids = []

    def rpc(operation, **fields):
        nonlocal observations
        if operation == "feedback":
            assert fields["kind"] == "aversive"
            assert s.running
            assert s.game.snapshot() == before
            assert np.array_equal(fields["rgb"], before_rgb)
            assert s._sensory["kind"] == "feedback" and s._sensory["neural_input_active"]
            assert s._sensory["source_step_id"] == 0
            pending = json.loads(s.pending_path.read_text())
            assert pending["outcome"]["reason"] == "no_action"
            transaction_ids.append(pending["transaction_id"])
        result = old_rpc(operation, **fields)
        if operation == "observe":
            observations += 1
            if observations <= 2:
                result.update(decode_counts(counts, .5))
            result.update(neural_time_start=(observations - 1) * 700,
                          neural_time_end=(observations - 1) * 700 + (500 if observations <= 2 else 100))
        return result

    s.rpc = rpc
    for attempt in range(2):
        s.step()
        assert s.running and s.status == "running"
        assert s.game.snapshot() == before
        assert s.stats["no_action"] == attempt + 1
        assert s.stats["deaths"] == s.stats["steps"] == s.stats["total_food"] == 0
        assert s.last_feedback["reason"] == "no_action"
        assert s.last_feedback["stimulus_ms"] == 200.0
        assert not s.pending_path.exists()
        assert json.loads(s.latest_pointer.read_text())["completed_event_id"] == transaction_ids[-1]
    assert len(set(transaction_ids)) == 2
    assert [c[0] for c in calls] == ["observe", "checkpoint", "feedback", "checkpoint"] * 2
    assert not any(e["kind"] == "action" for e in s.events)
    s.step()
    assert s.game.step_id == 1 and s.game.snake[0] == (5, 4)
    assert s.stats["steps"] == 1 and s.stats["no_action"] == 2
    assert calls[-1][0] == "observe"
    assert not any(c[0] == "reset_episode" for c in calls)


@pytest.mark.parametrize("committed", [False, True])
def test_no_output_punishment_recovery_replays_only_uncommitted_transaction(tmp_path, monkeypatch, committed):
    s, calls = harness(tmp_path, monkeypatch, feedback_failure=True)
    before = s.game.snapshot()
    old_rpc = s.rpc

    def silent_rpc(operation, **fields):
        result = old_rpc(operation, **fields)
        if operation == "observe":
            result.update(decode_counts({"left": [0], "forward": [0, 0], "right": [0]}, .5))
        return result

    s.rpc = silent_rpc
    with pytest.raises(RuntimeError, match="injected process interruption"):
        s.step()
    assert s.game.snapshot() == before
    pending = json.loads(s.pending_path.read_text())
    assert pending["outcome"]["reason"] == "no_action"
    assert not s.latest_pointer.exists()
    if committed:
        s.save_memory(completed_event_id=pending["transaction_id"])

    def recovered_rpc(operation, **fields):
        if operation == "feedback":
            calls.append((operation, fields))
            assert fields["kind"] == "aversive"
            return {"stimulus_ms": 200.0,
                    "observation_sha256": hashlib.sha256(fields["rgb"].tobytes()).hexdigest()}
        return old_rpc(operation, **fields)

    calls.clear()
    s.rpc = recovered_rpc
    s.recover_memory()
    assert [c[0] for c in calls] == (["restore", "reset_episode"] if committed else
                                   ["restore", "feedback", "checkpoint", "reset_episode"])
    assert not s.pending_path.exists()
    assert json.loads(s.latest_pointer.read_text())["completed_event_id"] == pending["transaction_id"]
    assert s.running
    if not committed:
        assert s.last_feedback["reason"] == "no_action"


@pytest.mark.parametrize("fault", ["frame", "duration"])
def test_no_output_bad_feedback_cannot_commit_or_retry(tmp_path, monkeypatch, fault):
    s, calls = harness(tmp_path, monkeypatch)
    before = s.game.snapshot()
    old_rpc = s.rpc

    def rpc(operation, **fields):
        result = old_rpc(operation, **fields)
        if operation == "observe":
            result.update(decode_counts({"left": [0], "forward": [0, 0], "right": [0]}, .5))
        if operation == "feedback":
            result["observation_sha256" if fault == "frame" else "stimulus_ms"] = "wrong" if fault == "frame" else 100.0
        return result

    s.rpc = rpc
    with pytest.raises(RuntimeError, match="反馈刺激"):
        s.step()
    assert s.game.snapshot() == before
    assert s.pending_path.exists() and not s.latest_pointer.exists()
    assert [c[0] for c in calls] == ["observe", "checkpoint", "feedback"]

    # A malformed receipt during recovery must also leave the transaction
    # pending, without updating the committed memory pointer.
    calls.clear()
    with pytest.raises(RuntimeError, match="反馈刺激"):
        s.recover_memory()
    assert s.pending_path.exists() and not s.latest_pointer.exists()
    assert [c[0] for c in calls] == ["restore", "feedback"]
