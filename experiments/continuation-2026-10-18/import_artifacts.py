#!/usr/bin/env python3
"""Import only verified experiment data; never overwrite source or native models.

The original conversation ZIP is required. This importer does not download,
execute, or deserialize code from it. Both the archive and source manifest are
pinned, and all selected members are staged and checked before installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "942cc762b84bf1361f078d27a07471b011e3eb28bb5c43a6e48d39aebf00ef4b"
MANIFEST_SHA256 = "d65121217432fc791a00c5320b92542d566ce7cd1260e31748d850a16ec20a74"
CHUNK = 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name!r}")
    if path.as_posix() != name or name == ".":
        raise ValueError(f"Noncanonical archive member: {name!r}")
    return path


def records(root: Path) -> list[dict]:
    manifest = root / "SOURCE-PROVENANCE.json"
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("Source manifest checksum mismatch")
    selected = []
    seen = set()
    for row in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
        name = row["name"]
        path = safe_name(name)
        if name in seen:
            raise ValueError(f"Duplicate manifest member: {name}")
        seen.add(name)
        if path.parts[0] in ("runtime", "results") or name == "reproduction.log":
            selected.append(row)
    if not selected:
        raise ValueError("No experiment artifacts in the pinned source manifest")
    return selected


def destination(root: Path, name: str) -> Path:
    path = root.joinpath(*safe_name(name).parts)
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Refusing symlink destination: {part}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Destination escapes the experiment directory: {name}")
    return path


def matches(path: Path, row: dict) -> bool:
    return path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]


def check(root: Path = HERE) -> dict:
    root = Path(root).resolve(strict=True)
    expected = records(root)
    invalid = [row["name"] for row in expected if not matches(destination(root, row["name"]), row)]
    if invalid:
        raise ValueError("Missing or changed artifacts; import the original ZIP first: " + ", ".join(invalid))
    return {"verified_files": len(expected), "archive_sha256": ARCHIVE_SHA256,
            "complete_brain_learning_confirmed": False}


def install(archive: Path, root: Path = HERE) -> dict:
    archive = Path(archive)
    root = Path(root).resolve(strict=True)
    expected = records(root)
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch; use the original 2026-10-18 continuation kit")
    pending = []
    present = 0
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix=".artifact-import-", dir=root) as tmp:
        infos = z.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names")
        for info in infos:
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Unexpected directory or symlink in source ZIP: {info.filename}")
        for index, row in enumerate(expected):
            name = row["name"]
            target = destination(root, name)
            info = z.getinfo(name)
            if info.file_size != row["bytes"]:
                raise ValueError(f"Archive member size mismatch: {name}")
            staged = Path(tmp) / str(index)
            h = hashlib.sha256()
            with z.open(info) as src, staged.open("xb") as dst:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    h.update(chunk)
                    dst.write(chunk)
            if h.hexdigest() != row["sha256"]:
                raise ValueError(f"Archive member checksum mismatch: {name}")
            if target.exists():
                if not matches(target, row):
                    raise FileExistsError(f"Refusing to overwrite a different existing artifact: {name}")
                present += 1
            else:
                pending.append((staged, target, row))
        # Same filesystem: hard-link installation is atomic and cannot overwrite
        # an existing file. No target data is written before all checks above.
        created = 0
        for staged, target, row in pending:
            destination(root, row["name"])
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(staged, target)
                created += 1
            except FileExistsError:
                if not matches(target, row):
                    raise
                present += 1
    return {"created": created, "already_present": present, **check(root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, help="Original FlyBrain-continuation-kit-2026-10-18.zip")
    parser.add_argument("--check", action="store_true", help="Verify already imported data without extracting")
    args = parser.parse_args()
    if not args.check and args.archive is None:
        parser.error("supply the original ZIP path, or use --check")
    try:
        result = check() if args.check else install(args.archive)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Artifact import failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
