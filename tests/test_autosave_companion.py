from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.autosave_current_session import AutosaveCompanion


SAVED = "a" * 64
DIRTY = "b" * 64


class FakeBackend:
    def __init__(self):
        self.current = {"status": "thinking", "control_source": "MaleCNS v1.0 neural process; no scripted action fallback", "brain": {"loaded": True, "memory": {"sha256": DIRTY}}}
        self.saves = 0
        self.disconnected = False
        self.save_error = False

    def state(self):
        if self.disconnected:
            raise ConnectionError("offline")
        return self.current

    def save(self):
        self.saves += 1
        if self.save_error:
            raise TimeoutError("unknown acceptance")
        return {"ok": True, "action": "save"}


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 1000.0
        self.backend = FakeBackend()
        self.companion = AutosaveCompanion(self.root, backend=self.backend, clock=lambda: self.now)
        self.commit("brain-1", SAVED, 980)

    def commit(self, checkpoint_id, memory, created):
        folder = self.root / "runs" / "checkpoints" / checkpoint_id
        folder.mkdir(parents=True)
        files = {}
        for name in ("brain.npz", "brain.npz.adapter.json"):
            path = folder / name
            content = (checkpoint_id + name + memory).encode()
            path.write_bytes(content)
            files[str(path.relative_to(self.root))] = hashlib.sha256(content).hexdigest()
        record = {"checkpoint_id": checkpoint_id, "path": str((folder / "brain.npz").relative_to(self.root)),
                  "files": files, "created_at": datetime.fromtimestamp(created, timezone.utc).isoformat(),
                  "memory": {"sha256": memory}}
        (folder / "commit.json").write_text(json.dumps(record), encoding="utf-8")
        (self.root / "runs" / "latest_memory.json").write_text(json.dumps(record), encoding="utf-8")

    def test_periodic_uses_checkpoint_time_not_service_start(self):
        self.companion.tick()
        self.assertEqual(self.backend.saves, 0)
        self.now = 1040
        self.companion.tick()
        self.assertEqual(self.backend.saves, 1)
        self.assertTrue(self.companion.snapshot()["saving"])
        self.assertEqual(self.companion.snapshot()["last_saved_at"], datetime.fromtimestamp(980, timezone.utc).isoformat())

    def test_paused_dirty_is_immediate_and_only_queues_once(self):
        self.backend.current["status"] = "paused"
        for _ in range(3):
            self.companion.tick()
        self.assertEqual(self.backend.saves, 1)
        self.assertEqual(self.companion.outstanding["reason"], "paused")

    def test_no_action_dirty_is_immediate(self):
        self.backend.current["status"] = "no_action"
        self.companion.tick()
        self.assertEqual(self.backend.saves, 1)

    def test_unchanged_memory_is_not_saved_even_when_old_and_paused(self):
        self.backend.current["status"] = "paused"
        self.backend.current["brain"]["memory"]["sha256"] = SAVED
        self.now = 5000
        self.companion.tick()
        self.assertEqual(self.backend.saves, 0)

    def test_pending_feedback_blocks_save(self):
        (self.root / "runs" / "pending_feedback.json").write_text("{}")
        self.backend.current["status"] = "paused"
        self.companion.tick()
        self.assertEqual(self.backend.saves, 0)
        self.assertTrue(self.companion.snapshot()["pending_feedback"])

    def test_new_verified_pointer_confirms_save_without_api_checkpoint_id(self):
        self.backend.current["status"] = "paused"
        self.companion.tick()
        self.assertTrue(self.companion.snapshot()["saving"])
        self.commit("brain-2", DIRTY, 1000)
        self.companion.tick()
        status = self.companion.snapshot()
        self.assertFalse(status["saving"])
        self.assertEqual(status["checkpoint_id"], "brain-2")
        self.assertEqual(status["last_reason"], "paused")
        self.assertEqual(self.backend.saves, 1)

    def test_corrupt_checkpoint_is_never_reported_as_saved(self):
        self.backend.current["status"] = "paused"
        self.companion.tick()
        self.commit("brain-2", DIRTY, 1000)
        (self.root / "runs" / "checkpoints" / "brain-2" / "brain.npz").write_bytes(b"corrupt")
        self.companion.tick()
        self.assertTrue(self.companion.snapshot()["saving"])
        self.assertEqual(self.companion.snapshot()["checkpoint_id"], "brain-1")
        self.assertIn("verification failed", self.companion.snapshot()["last_error"])

    def test_cached_generation_detects_modified_file(self):
        self.companion.tick()
        (self.root / "runs" / "checkpoints" / "brain-1" / "brain.npz").write_bytes(b"tampered")
        self.companion.tick()
        self.assertIn("verification failed", self.companion.snapshot()["last_error"])
        self.assertEqual(self.backend.saves, 0)

    def test_uncertain_post_retains_outstanding_slot(self):
        self.backend.current["status"] = "paused"
        self.backend.save_error = True
        self.companion.tick()
        self.now += 100
        self.companion.tick()
        self.assertEqual(self.backend.saves, 1)
        self.assertTrue(self.companion.snapshot()["saving"])
        self.assertIn("uncertain", self.companion.snapshot()["last_error"])

    def test_disconnection_is_visible_and_sends_no_commands(self):
        self.backend.disconnected = True
        self.companion.tick()
        self.assertFalse(self.companion.snapshot()["backend_connected"])
        self.assertIn("disconnected", self.companion.snapshot()["last_error"])
        self.assertEqual(self.backend.saves, 0)

    def test_native_enabled_exits_without_commands(self):
        self.backend.current["autosave"] = {"enabled": True}
        self.assertFalse(self.companion.tick())
        self.assertFalse(self.companion.snapshot()["enabled"])
        self.assertTrue(self.companion.snapshot()["handoff"])
        self.assertEqual(self.backend.saves, 0)

    def test_backend_error_does_not_queue_a_save(self):
        self.backend.current["status"] = "error"
        self.now = 5000
        self.companion.tick()
        self.assertEqual(self.backend.saves, 0)
        self.assertIn("Backend stopped", self.companion.snapshot()["last_error"])

    def test_an_unrecognized_backend_never_receives_commands(self):
        self.backend.current["control_source"] = "another backend"
        self.backend.current["status"] = "paused"
        self.companion.tick()
        self.assertEqual(self.backend.saves, 0)
        self.assertIsNotNone(self.companion.snapshot()["last_error"])

    def test_reading_existing_pointer_does_not_claim_a_restore(self):
        self.companion.tick()
        self.assertIsNone(self.companion.snapshot()["last_reason"])


if __name__ == "__main__":
    unittest.main()
