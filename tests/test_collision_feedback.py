"""Collision-to-feedback coordinator contracts using TEST-ONLY brain fixtures.

These exercise the real Session.step/game/persistence orchestration. RPC stubs
do not demonstrate biological spike responses or learned avoidance.
"""
import json

import numpy as np
import pytest

from flysnake.decoder import decode_counts
from test_server_contract import harness


COILED_BODY = [(5, 5), (5, 6), (6, 6), (6, 5), (6, 4),
               (5, 4), (4, 4), (4, 5), (4, 6)]


def test_head_body_collision_dispatches_aversive_and_commits_before_reset(tmp_path, monkeypatch):
    session, calls = harness(tmp_path, monkeypatch)
    session.game.snake = list(COILED_BODY)
    session.game.food = (10, 10)
    before_pixels = session.game.render_observation()
    old_rpc = session.rpc
    delivered = []

    def audited_rpc(operation, **fields):
        if operation == "feedback":
            assert fields["kind"] == "aversive"
            assert session.game.terminal and session.game.death_reason == "self"
            assert session.game.episode_id == 1
            assert np.array_equal(fields["rgb"], before_pixels)
            assert session.game.pending_feedback["kind"] == "aversive"
            pending = json.loads(session.pending_path.read_text())
            assert pending["kind"] == "aversive"
            assert pending["outcome"]["death_reason"] == "self"
            assert pending["outcome"]["collision_at"] == [5, 4]
            delivered.append(pending["transaction_id"])
        if operation == "reset_episode":
            assert len(delivered) == 1
            assert not session.pending_path.exists()
            assert session.game.pending_feedback is None
            latest = json.loads(session.latest_pointer.read_text())
            assert latest["completed_event_id"] == delivered[0]
            # Game reset follows the neural reset RPC, not the other way round.
            assert session.game.episode_id == 1 and session.game.terminal
        return old_rpc(operation, **fields)

    session.rpc = audited_rpc
    session.step()
    operations = [operation for operation, _ in calls]
    assert operations[:5] == ["observe", "checkpoint", "feedback", "checkpoint", "reset_episode"]
    assert operations.count("feedback") == 1
    assert session.last_feedback["kind"] == "aversive"
    assert session.last_feedback["stimulus_ms"] == 200.0
    assert session.stats["deaths"] == 1 and session.stats["total_food"] == 0
    assert session.game.episode_id == 2 and session.game.alive
    actions = [record for record in session.events if record["kind"] == "action"]
    assert actions[0]["outcome"]["death_reason"] == "self"
    assert actions[0]["outcome"]["score_delta"] == 0


def test_interrupted_self_collision_punishment_freezes_dead_board_for_recovery(tmp_path, monkeypatch):
    session, calls = harness(tmp_path, monkeypatch, feedback_failure=True)
    session.game.snake = list(COILED_BODY)
    session.game.food = (10, 10)
    before_pixels = session.game.render_observation()
    with pytest.raises(RuntimeError, match="injected process interruption"):
        session.step()
    assert [operation for operation, _ in calls] == ["observe", "checkpoint", "feedback"]
    assert calls[-1][1]["kind"] == "aversive"
    assert session.game.episode_id == 1 and not session.game.alive
    assert session.game.death_reason == "self" and session.game.pending_feedback is not None
    assert np.array_equal(session.game.render_observation(), before_pixels)
    assert session.pending_path.exists() and not session.latest_pointer.exists()
    pending = json.loads(session.pending_path.read_text())
    assert pending["outcome"]["death_reason"] == "self"
    assert pending["kind"] == "aversive"


def test_entering_tail_that_vacates_this_step_dispatches_no_punishment(tmp_path, monkeypatch):
    session, calls = harness(tmp_path, monkeypatch)
    session.game.snake = [(3, 3), (3, 4), (2, 4), (2, 3)]
    session.game.direction = 0
    session.game.food = (10, 10)
    old_rpc = session.rpc

    def left_fixture_rpc(operation, **fields):
        result = old_rpc(operation, **fields)
        if operation == "observe":
            # Explicit TEST-ONLY neuron counts choose the relative left action.
            result.update(decode_counts({"left": [6], "forward": [0, 0], "right": [0]}, .1))
        return result

    session.rpc = left_fixture_rpc
    session.step()
    assert all(operation not in ("feedback", "reset_episode") for operation, _ in calls)
    assert session.game.alive and session.game.episode_id == 1
    assert session.game.snake[0] == (2, 3) and len(session.game.snake) == 4
    assert session.stats["deaths"] == 0 and session.last_feedback is None
    assert not session.pending_path.exists()
