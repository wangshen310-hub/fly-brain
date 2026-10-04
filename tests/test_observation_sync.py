"""Sensory publication/transport contracts with TEST-ONLY neural RPC fixtures.

These tests exercise coordinator provenance and persistence, not brain biology.
"""
import base64
from contextlib import contextmanager
import hashlib
import http.client
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from flysnake import server
from test_server_contract import harness


def sensory_rgb(state):
    sensory = state["sensory"]
    prefix, encoded = sensory["data_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
    assert pixels.shape == (sensory["height"], sensory["width"], 3) == (180, 320, 3)
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == sensory["rgb_sha256"]
    return pixels


@contextmanager
def serving(session):
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(session))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    client = http.client.HTTPConnection(*httpd.server_address, timeout=3)
    try:
        yield client
    finally:
        client.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=3)


def request(client, path):
    client.request("GET", path)
    response = client.getresponse()
    return response.status, response.getheader("Content-Type"), response.read()


def test_state_contains_exact_frame_and_detached_provenance(tmp_path, monkeypatch):
    s, _ = harness(tmp_path, monkeypatch)
    old = s.snapshot()
    np.testing.assert_array_equal(sensory_rgb(old), s.game.render_observation())
    assert old["sensory"]["rgb_sha256"] == old["game"]["observation_hash"]
    assert old["sensory"]["kind"] == "ready"
    assert old["sensory"]["source_episode_id"] == old["game"]["episode_id"]
    assert old["sensory"]["source_step_id"] == old["game"]["step_id"]
    assert old["sensory"]["source_heading"] == old["game"]["direction"]
    assert old["sensory"]["neural_input_active"] is False

    s.game.direction = 1
    s.game.food = (8, 8)
    s._refresh_observation()
    current = s.snapshot()
    assert old["sensory"]["data_url"] != current["sensory"]["data_url"]
    assert old["sensory"]["source_heading"] == 0
    assert current["sensory"]["source_heading"] == 1
    current["sensory"]["source_heading"] = 999
    current["game"]["snake"][0][0] = 999
    assert s.snapshot()["sensory"]["source_heading"] == 1
    assert s.game.snake[0][0] == 5


def test_versioned_http_images_never_alias_latest_and_cache_is_bounded(tmp_path, monkeypatch):
    s, _ = harness(tmp_path, monkeypatch)
    old = s.snapshot()
    old_png = base64.b64decode(old["sensory"]["data_url"].split(",", 1)[1])
    s.game.direction = 1
    s.game.food = (8, 8)
    s._refresh_observation()
    with serving(s) as client:
        status, content_type, body = request(client, old["observation_url"])
        assert (status, content_type, body) == (200, "image/png", old_png)
        assert request(client, "/api/observation.png")[2] == s.observation
        for _ in range(64):
            s._refresh_observation()
        assert len(s._observation_cache) == 64
        assert request(client, old["observation_url"])[0] == 410
        assert request(client, f"/api/observation.png?v={s.observation_version + 1}")[0] == 404
        for invalid in ("", "0", "-1", "abc", "1&v=2", "9" * 5000):
            assert request(client, "/api/observation.png?v=" + invalid)[0] == 404
        state = json.loads(request(client, "/api/state")[2])
        assert request(client, state["observation_url"])[2] == base64.b64decode(state["sensory"]["data_url"].split(",", 1)[1])


def test_decision_is_exact_active_input_and_move_publishes_next_frame(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch)
    s.game.food = (9, 9)  # Ordinary move; no feedback transaction.
    old_rpc = s.rpc
    observed = []

    def inspect(operation, **fields):
        if operation == "observe":
            state = s.snapshot()
            observed.append(state)
            assert state["sensory"]["kind"] == "decision"
            assert state["sensory"]["neural_input_active"] is True
            assert state["sensory"]["source_step_id"] == state["game"]["step_id"] == 0
            np.testing.assert_array_equal(sensory_rgb(state), fields["rgb"])
        return old_rpc(operation, **fields)

    s.rpc = inspect
    s.step()
    assert [operation for operation, _ in calls] == ["observe"]
    after = s.snapshot()
    assert after["sensory"]["kind"] == "ready"
    assert after["sensory"]["neural_input_active"] is False
    assert after["sensory"]["source_step_id"] == after["game"]["step_id"] == 1
    assert observed[0]["sensory"]["rgb_sha256"] != after["sensory"]["rgb_sha256"]
    np.testing.assert_array_equal(sensory_rgb(after), s.game.render_observation())


def test_feedback_keeps_pre_action_pixels_and_source_until_commit(tmp_path, monkeypatch):
    s, _ = harness(tmp_path, monkeypatch)
    before = s.snapshot()
    old_rpc = s.rpc

    def inspect(operation, **fields):
        if operation == "feedback":
            state = s.snapshot()
            assert state["sensory"]["kind"] == "feedback"
            assert state["sensory"]["feedback_kind"] == "reward"
            assert state["sensory"]["neural_input_active"] is True
            assert state["sensory"]["source_step_id"] == 0
            assert state["game"]["step_id"] == 1
            assert state["game"]["food"] is None
            np.testing.assert_array_equal(sensory_rgb(state), fields["rgb"])
            assert state["sensory"]["rgb_sha256"] == before["sensory"]["rgb_sha256"]
            pending = json.loads(s.pending_path.read_text())
            assert pending["sensory_source"] == {"episode_id": 1, "step_id": 0, "direction": 0}
            assert pending["event_id"] == state["sensory"]["event_id"]
        return old_rpc(operation, **fields)

    s.rpc = inspect
    s.step()
    after = s.snapshot()
    assert after["sensory"]["kind"] == "ready"
    assert after["sensory"]["source_step_id"] == 1
    assert after["game"]["food"] is not None
    assert not s.pending_path.exists()


@pytest.mark.parametrize("running", [True, False])
def test_new_episode_shows_actual_neutral_warmup_then_current_game(tmp_path, monkeypatch, running):
    s, _ = harness(tmp_path, monkeypatch)
    s.running = running
    old_rpc = s.rpc
    observed = []

    def inspect(operation, **fields):
        if operation == "reset_episode":
            state = s.snapshot()
            observed.append(state)
            assert state["status"] == "warming"
            assert state["sensory"]["kind"] == "neutral_warmup"
            assert state["sensory"]["view"] == "neutral"
            assert state["sensory"]["neural_input_active"] is True
            assert state["sensory"]["source_episode_id"] is None
            assert state["sensory"]["source_step_id"] is None
            assert state["sensory"]["source_heading"] is None
            assert np.all(sensory_rgb(state) == 128)
        return old_rpc(operation, **fields)

    s.rpc = inspect
    s.new_episode()
    assert len(observed) == 1
    state = s.snapshot()
    assert state["status"] == ("running" if running else "paused")
    assert state["sensory"]["kind"] == ("ready" if running else "snapshot")
    assert state["sensory"]["source_episode_id"] == state["game"]["episode_id"] == 2
    assert state["sensory"]["neural_input_active"] is False
    np.testing.assert_array_equal(sensory_rgb(state), s.game.render_observation())


@pytest.mark.parametrize("committed", [False, True])
def test_recovery_uses_historical_feedback_and_neutral_reset(tmp_path, monkeypatch, committed):
    s, _ = harness(tmp_path, monkeypatch, feedback_failure=True)
    before = s.snapshot()
    with pytest.raises(RuntimeError, match="injected"):
        s.step()
    pending = json.loads(s.pending_path.read_text())
    if committed:
        s.save_memory(completed_event_id=pending["transaction_id"])
    s.game = server.SnakeGame(seed=29)  # A new process starts with an independent board.
    s.game.new_episode()
    s._refresh_observation()
    old_rpc = s.rpc
    kinds = []

    def inspect(operation, **fields):
        if operation == "feedback":
            state = s.snapshot()
            kinds.append("feedback")
            assert state["sensory"]["kind"] == "feedback"
            assert state["sensory"]["source_episode_id"] == 1
            assert state["sensory"]["source_step_id"] == 0
            assert state["game"]["episode_id"] == 2
            assert state["sensory"]["rgb_sha256"] == before["sensory"]["rgb_sha256"]
            np.testing.assert_array_equal(sensory_rgb(state), fields["rgb"])
            return {"stimulus_ms": 200.0, "fixture": True,
                    "observation_sha256": state["sensory"]["rgb_sha256"]}
        if operation == "reset_episode":
            state = s.snapshot()
            kinds.append("reset")
            assert state["sensory"]["kind"] == "neutral_warmup"
            assert state["sensory"]["neural_input_active"] is True
            assert np.all(sensory_rgb(state) == 128)
        return old_rpc(operation, **fields)

    s.rpc = inspect
    s.recover_memory()
    assert kinds == (["reset"] if committed else ["feedback", "reset"])
    assert s.snapshot()["sensory"]["source_episode_id"] == 2
    assert not s.pending_path.exists()


@pytest.mark.parametrize("pause", [False, True])
def test_no_action_retries_same_frame_and_manual_pause_keeps_inactive_frame(tmp_path, monkeypatch, pause):
    s, _ = harness(tmp_path, monkeypatch)
    before = s.snapshot()
    old_rpc = s.rpc

    def inspect(operation, **fields):
        result = old_rpc(operation, **fields)
        if operation == "observe":
            if pause:
                s.command("pause")
            else:
                result["action"] = None
        return result

    s.rpc = inspect
    s.step()
    after = s.snapshot()
    assert after["game"]["step_id"] == 0
    assert after["sensory"]["kind"] == ("snapshot" if pause else "ready")
    assert after["sensory"]["neural_input_active"] is False
    assert after["sensory"]["data_url"] == before["sensory"]["data_url"]
    assert after["status"] == ("paused" if pause else "running")
    assert after["running"] is not pause


def test_isolated_storage_preserves_relative_manifest_and_static_web_root(tmp_path, monkeypatch):
    s, calls = harness(tmp_path, monkeypatch)
    isolated = tmp_path / "runtime" / "isolated"
    s.storage_root = isolated
    s.run_dir = isolated / "runs" / "test"
    s.run_dir.mkdir(parents=True)
    s.journal = s.run_dir / "events.jsonl"
    s.latest_pointer = isolated / "runs" / "latest_memory.json"
    s.pending_path = isolated / "runs" / "pending_feedback.json"
    s.step()
    record = json.loads(s.latest_pointer.read_text())
    assert (isolated / record["path"]).is_file()
    assert not (tmp_path / "runs" / "latest_memory.json").exists()
    assert all(not Path(relative).is_absolute() for relative in record["files"])
    assert s.checked_checkpoint(record, storage_root=isolated) == (isolated / record["path"]).resolve()
    s.restore_record(record)
    assert calls[-1] == ("restore", {"path": str((isolated / record["path"]).resolve())})
    escaped = {**record, "files": {"../outside.npz": "fake"}}
    with pytest.raises(RuntimeError, match="校验失败"):
        s.checked_checkpoint(escaped, storage_root=isolated)
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_bytes(b"TEST-ONLY-STATIC-WEB")
    with serving(s) as client:
        assert request(client, "/")[2] == b"TEST-ONLY-STATIC-WEB"
        assert json.loads(request(client, "/api/state")[2])["run_directory"] == str(s.run_dir)


@pytest.mark.parametrize("isolated", [False, True])
def test_session_constructor_applies_optional_storage_root_without_starting_neurons(tmp_path, monkeypatch, isolated):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    fake_process = SimpleNamespace(start=lambda: None)
    fake_child = SimpleNamespace(close=lambda: None)
    fake_context = SimpleNamespace(Pipe=lambda: (None, fake_child), Process=lambda **kwargs: fake_process)
    monkeypatch.setattr(server.mp, "get_context", lambda method: fake_context)
    monkeypatch.setattr(server.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    storage = tmp_path / "runtime" / "separate" if isolated else tmp_path
    s = server.Session(autorun=False, storage_root=storage if isolated else None)
    assert s.storage_root == storage.resolve()
    assert s.run_dir.is_relative_to(storage / "runs")
    assert s.latest_pointer == storage / "runs" / "latest_memory.json"
    assert s.pending_path == storage / "runs" / "pending_feedback.json"
    assert s.journal.is_file()
