"""Download, verify and prepare the full real MaleCNS graph (no substitute).

Run with the project venv. Official downloads total about 1.109 GB. Normalizing
and compiling are separate processes to release temporary arrays between steps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
NEURAL = VENDOR / "stonkfly" / "neural"
COMMIT = "78ef3e05ab0fa086032098558d893667068944a0"


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(stage, **fields):
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    data = Path(os.environ.get("FLYSNAKE_DATA", ROOT / "data" / "malecns_v1")).resolve()
    data.mkdir(parents=True, exist_ok=True)
    os.environ["FLYSNAKE_DATA"] = str(data)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(VENDOR), os.environ.get("PYTHONPATH", "")])
    locks = json.loads((NEURAL / "sources.lock.json").read_text())
    verified = {}
    for name, info in locks.items():
        target = data / name
        partial = target.with_suffix(target.suffix + ".partial")
        if not target.exists() and partial.exists() and partial.stat().st_size == info["bytes"]:
            if digest_file(partial) != info["sha256"]:
                raise RuntimeError(f"Complete partial download has wrong hash: {partial}")
            partial.replace(target)
        if not target.exists():
            if args.skip_download:
                raise FileNotFoundError(f"Official data missing: {target}")
            emit("download", file=name, bytes=info["bytes"], url=info["url"])
            subprocess.run([
                "curl.exe" if sys.platform == "win32" else "curl",
                "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
                "--continue-at", "-", "--output", str(partial), info["url"],
            ], check=True)
            if partial.stat().st_size != info["bytes"] or digest_file(partial) != info["sha256"]:
                raise RuntimeError(f"Official data integrity mismatch: {name}")
            partial.replace(target)
        if target.stat().st_size != info["bytes"] or digest_file(target) != info["sha256"]:
            raise RuntimeError(f"Pinned source digest mismatch: {name}; refusing to import")
        verified[name] = {**info, "license": "CC-BY-4.0", "verified_utc": datetime.now(timezone.utc).isoformat()}
        emit("source_verified", file=name, bytes=info["bytes"], sha256=info["sha256"])
    atomic_json(data / "source.lock.json", verified)
    if args.download_only:
        return
    if not (data / "normalized" / "report.json").exists():
        emit("normalize", message="Streaming source record batches; all retained edges preserved")
        subprocess.run([sys.executable, "-m", "stonkfly.neural.connectome"], cwd=ROOT, check=True)
    if not (data / "graph.npz").exists():
        emit("prepare", message="Building CSR and inferred retinal mapping")
        subprocess.run([sys.executable, "-m", "stonkfly.neural.prepare"], cwd=ROOT, check=True)
    import numpy as np
    expected = json.loads((NEURAL / "arrays.lock.json").read_text())
    checked = {}
    with np.load(data / "graph.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise RuntimeError("Derived graph fields differ from the pinned upstream graph")
        for key, reference in expected.items():
            array = np.ascontiguousarray(archive[key])
            actual = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
            if actual != reference:
                raise RuntimeError(f"Derived {key} does not match pinned upstream graph: {actual}")
            checked[key] = actual
            del array
    import pyarrow.feather as feather
    nodes = feather.read_table(data / "normalized" / "neurons.feather").to_pandas()
    transmitter_values = json.dumps(nodes.neurotransmitter.fillna("").astype(str).tolist(), separators=(",", ":")).encode()
    nt_hash = hashlib.sha256(transmitter_values).hexdigest()
    if nt_hash != json.loads((NEURAL / "neurons.lock.json").read_text())["neurotransmitter_values_sha256"]:
        raise RuntimeError("Normalized neurotransmitter annotations differ from the upstream lock")
    del nodes, transmitter_values
    manifest = json.loads((data / "manifest.json").read_text())
    if (manifest["neurons"], manifest["edges"], manifest["synaptic_contacts"]) != (166700, 25582938, 124177617):
        raise RuntimeError("Retained graph counts differ from the pinned model")
    emit("graph_verified", neurons=manifest["neurons"], edges=manifest["edges"], arrays=len(checked))
    sys.path[:0] = [str(ROOT), str(VENDOR)]
    from stonkfly.neural.brain import build
    emit("compile", message="Compiling the platform-adapted upstream C++ neural kernel")
    binary = build()
    receipt = {
        "upstream_commit": COMMIT,
        "dataset": "MaleCNS v1.0",
        "official_source": "https://male-cns.janelia.org/download/",
        "license": "CC-BY-4.0",
        "citation": "Berg et al. Cell (2026), doi:10.1016/j.cell.2026.08.015",
        "verified_arrays": True,
        "array_sha256": checked,
        "source_sha256": {key: value["sha256"] for key, value in verified.items()},
        "neurotransmitter_values_sha256": nt_hash,
        "source_code_sha256": {str(path.relative_to(ROOT)): digest_file(path) for path in [*NEURAL.glob("*.py"), NEURAL / "kernel.cpp", ROOT / "flysnake" / "neural.py", ROOT / "flysnake" / "decoder.py"]},
        "graph_sha256": digest_file(data / "graph.npz"),
        "counts": {key: manifest[key] for key in ["neurons", "edges", "synaptic_contacts", "retina_mapped"]},
        "build": binary,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "biological_validity_or_learning_established": False,
    }
    atomic_json(data / "prepared.json", receipt)
    if args.skip_smoke:
        emit("prepared", receipt=str(data / "prepared.json"))
        return
    from flysnake.neural import RealFlyBrain
    brain = RealFlyBrain(progress=lambda item: emit("brain_progress", progress=item))
    # Fixed diagnostic pixels, not a policy or substituted snake engine.
    rgb = np.full((180, 320, 3), 210, dtype=np.uint8)
    rgb[50:130, 100:120] = 15
    rgb[60:90, 210:240] = [10, 180, 40]
    observation = brain.observe(rgb)
    reward = brain.feedback("reward", rgb)
    aversive = brain.feedback("aversive", rgb)
    if reward["reward_spikes"] <= 0 or aversive["aversive_spikes"] <= 0:
        raise RuntimeError("The requested dopamine stimulation produced no target spikes")
    path = ROOT / "runs" / "neural-smoke" / "checkpoint.npz"
    saved = brain.checkpoint(path)
    before = brain.memory()
    reset = brain.reset_episode()
    brain.restore(path)
    after = brain.memory()
    if before != after:
        raise RuntimeError("Neural checkpoint round-trip failed")
    result = {
        "real_graph_loaded": True, "native_kernel_ran": True,
        "reward_delivered": True, "aversive_delivered": True,
        "checkpoint_roundtrip": True, "reset_preserved_long_term_memory": True,
        "observation": observation, "reward": reward, "aversive": aversive,
        "reset": reset, "metadata": brain.metadata(),
        "meaning": "A real neural runtime and delivery check, not proof of useful vision or learned Snake behavior",
    }
    atomic_json(data / "smoke.json", result)
    emit("smoke_passed", result=str(data / "smoke.json"), action=observation["action"], reward_spikes=reward["reward_spikes"], aversive_spikes=aversive["aversive_spikes"], memory=after)


if __name__ == "__main__":
    main()
