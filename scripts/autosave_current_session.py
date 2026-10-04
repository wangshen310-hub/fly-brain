"""Save the already-running legacy backend; never create or control a brain.

This compatibility service only reads /api/state and queues action=save. Its
read-only /status endpoint closes automatically once native autosave takes over.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BACKEND = "http://127.0.0.1:8765"
ALLOWED_ORIGINS = {BACKEND, "http://localhost:8765"}
REASONS = {"periodic", "paused", "manual", "feedback", "new_episode", "restored"}


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Checkpoint created_at is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Checkpoint created_at must include its time zone")
    return parsed.timestamp()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class VerifiedPointer:
    """Re-hash only when the manifest or checkpoint file metadata changes."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.folder = (self.root / "runs" / "checkpoints").resolve()
        self.cache_key = None
        self.cached = None

    def read(self):
        pointer = self.root / "runs" / "latest_memory.json"
        if not pointer.exists():
            return None
        raw = pointer.read_bytes()
        record = json.loads(raw)
        checkpoint_id = record.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id.startswith("brain-") or not checkpoint_id[6:].isdigit():
            raise ValueError("Invalid checkpoint_id")
        timestamp(record.get("created_at"))
        if not valid_hash(record.get("memory", {}).get("sha256")):
            raise ValueError("Checkpoint memory hash is missing or invalid")
        checkpoint_folder = (self.folder / checkpoint_id).resolve()
        manifest = record.get("files")
        if not isinstance(manifest, dict) or len(manifest) != 2:
            raise ValueError("Checkpoint must contain the native and adapter files")
        paths = []
        for relative, expected in manifest.items():
            path = (self.root / relative).resolve()
            if path.parent != checkpoint_folder or path.name not in ("brain.npz", "brain.npz.adapter.json") or not valid_hash(expected):
                raise ValueError("Invalid checkpoint file manifest")
            paths.append((path, expected))
        checkpoint = (self.root / record["path"]).resolve()
        if checkpoint != checkpoint_folder / "brain.npz" or checkpoint not in [p for p, _ in paths]:
            raise ValueError("Native checkpoint is not in its manifest")
        commit_path = checkpoint_folder / "commit.json"
        commit_raw = commit_path.read_bytes()
        if json.loads(commit_raw) != record:
            raise ValueError("Latest pointer does not match its committed generation")
        file_metadata = []
        for path, expected in paths:
            stat = path.stat()
            file_metadata.append((str(path), expected, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        key = (raw, commit_raw, tuple(file_metadata))
        if key != self.cache_key:
            for path, expected in paths:
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest != expected:
                    raise ValueError(f"Checkpoint file hash mismatch: {path.name}")
            self.cache_key = key
            self.cached = record
        return deepcopy(self.cached)


class Backend:
    def request(self, route, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(BACKEND + route, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def state(self):
        return self.request("/api/state")

    def save(self):
        return self.request("/api/command", {"action": "save"})


class AutosaveCompanion:
    def __init__(self, root=ROOT, interval=60, *, backend=None, clock=time.time):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Save interval must be finite and positive")
        self.root = Path(root)
        self.pointer = VerifiedPointer(root)
        self.backend = backend or Backend()
        self.clock = clock
        self.lock = threading.Lock()
        self.outstanding = None
        self.seen_checkpoint = None
        self.status = {
            "enabled": True, "interval_seconds": interval,
            "last_saved_at": None, "last_reason": None, "saving": False,
            "mode": "compatibility", "backend_connected": False,
            "last_error": None, "pending_feedback": False,
            "last_checked_at": None, "checkpoint_id": None,
            "last_requested_at": None, "handoff": False,
        }

    def update(self, **fields):
        with self.lock:
            self.status.update(fields)

    def snapshot(self):
        with self.lock:
            return deepcopy(self.status)

    def tick(self):
        """Return False only when native autosave has taken responsibility."""
        now = self.clock()
        self.update(last_checked_at=datetime.fromtimestamp(now, timezone.utc).isoformat())
        try:
            state = self.backend.state()
            if not isinstance(state, dict):
                raise ValueError("Backend returned an invalid state")
            if not str(state.get("control_source", "")).startswith("MaleCNS"):
                raise ValueError("The endpoint is not the expected MaleCNS game")
        except Exception as error:
            self.update(backend_connected=False, last_error=f"Backend disconnected: {error}")
            return True
        self.update(backend_connected=True, last_error=None)
        native = state.get("autosave")
        if isinstance(native, dict) and native.get("enabled") is True:
            self.update(enabled=False, saving=False, handoff=True)
            logging.info("Native autosave is enabled; compatibility service is exiting")
            return False

        try:
            record = self.pointer.read()
        except Exception as error:
            self.update(last_error=f"Checkpoint verification failed: {error}")
            return True
        checkpoint_id = record["checkpoint_id"] if record else None
        if record:
            reason = record.get("reason")
            if self.outstanding is not None and checkpoint_id != self.outstanding["checkpoint_id"]:
                reason = reason if reason in REASONS else self.outstanding["reason"]
                self.outstanding = None
                self.update(saving=False)
                logging.info("Verified saved checkpoint %s", checkpoint_id)
            if checkpoint_id != self.seen_checkpoint:
                self.update(last_saved_at=record["created_at"], last_reason=reason if reason in REASONS else None, checkpoint_id=checkpoint_id)
                self.seen_checkpoint = checkpoint_id

        pending = (self.root / "runs" / "pending_feedback.json").exists()
        self.update(pending_feedback=pending)
        if self.outstanding is not None:
            age = now - self.outstanding["requested_at"]
            error = self.outstanding.get("error")
            if error:
                self.update(last_error=error)
            elif age >= 30:
                self.update(last_error="Save request is still awaiting a verified new checkpoint; no duplicate request will be sent")
            return True
        if pending:
            return True
        brain = state.get("brain") or {}
        if state.get("status") == "error":
            self.update(last_error=f"Backend stopped with an error: {state.get('message', 'unknown error')}")
            return True
        if brain.get("loaded") is not True:
            return True
        memory_hash = (brain.get("memory") or {}).get("sha256")
        if not valid_hash(memory_hash):
            self.update(last_error="Backend memory hash is unavailable; save was not requested")
            return True
        if record and memory_hash == record["memory"]["sha256"]:
            return True
        paused = state.get("status") in ("paused", "no_action")
        due = record is None or now - timestamp(record["created_at"]) >= self.status["interval_seconds"]
        if not paused and not due:
            return True
        # Recheck immediately before sending, in addition to the first observation.
        # The backend queues this command on its serial neural coordinator.
        if (self.root / "runs" / "pending_feedback.json").exists():
            self.update(pending_feedback=True)
            return True
        reason = "paused" if paused else "periodic"
        self.outstanding = {"checkpoint_id": checkpoint_id, "requested_at": now, "reason": reason}
        self.update(saving=True, last_requested_at=datetime.fromtimestamp(now, timezone.utc).isoformat())
        try:
            response = self.backend.save()
            if not isinstance(response, dict) or response.get("ok") is not True or response.get("action") != "save":
                raise ValueError(f"Save was not acknowledged: {response}")
            logging.info("Queued %s save; waiting for a verified pointer change", reason)
        except Exception as error:
            # A timeout may happen after the server accepted the command. Retain
            # the outstanding slot until the actual pointer changes, never retry.
            message = f"Save request failed or its outcome is uncertain: {error}; awaiting a verified checkpoint before any further request"
            self.outstanding["error"] = message
            self.update(last_error=message)
        return True


def make_handler(companion):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body, status=200):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            origin = self.headers.get("Origin")
            if origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            origin = self.headers.get("Origin")
            if origin is not None and origin not in ALLOWED_ORIGINS:
                return self.respond({"error": "Origin denied"}, 403)
            if urlparse(self.path).path != "/status":
                return self.respond({"error": "Not found"}, 404)
            return self.respond(companion.snapshot())

        def do_POST(self):
            return self.respond({"error": "Read-only service"}, 405)

        do_PUT = do_PATCH = do_DELETE = do_POST

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--poll", type=float, default=2)
    args = parser.parse_args()
    if not math.isfinite(args.poll) or args.poll <= 0:
        parser.error("--poll must be finite and positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    companion = AutosaveCompanion(interval=args.interval)
    # Bind before polling: a second instance cannot enqueue any commands.
    server = ThreadingHTTPServer(("127.0.0.1", 8766), make_handler(companion))
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info("Compatibility autosave status: http://127.0.0.1:8766/status")
    try:
        while not stop.is_set():
            if not companion.tick():
                break
            stop.wait(args.poll)
    finally:
        companion.update(enabled=False)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
