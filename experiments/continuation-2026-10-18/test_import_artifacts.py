"""Importer safety tests use explicit synthetic fixtures, not neural learning data."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import import_artifacts as importer


def fixture_archive(tmp_path, monkeypatch, *, data=b"tiny synthetic data", declared=None, name="runtime/sample.bin"):
    root = tmp_path / "experiment"
    root.mkdir()
    row = {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest() if declared is None else declared}
    manifest = root / "SOURCE-PROVENANCE.json"
    manifest.write_text(json.dumps({"files": [row]}), encoding="utf-8")
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, data)
    monkeypatch.setattr(importer, "ARCHIVE_SHA256", importer.digest(archive))
    monkeypatch.setattr(importer, "MANIFEST_SHA256", importer.digest(manifest))
    return root, archive


def test_verified_import_is_idempotent(tmp_path, monkeypatch):
    root, archive = fixture_archive(tmp_path, monkeypatch)
    assert importer.install(archive, root)["created"] == 1
    assert importer.install(archive, root)["already_present"] == 1
    assert importer.check(root)["verified_files"] == 1


def test_wrong_archive_is_rejected_before_install(tmp_path, monkeypatch):
    root, archive = fixture_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(importer, "ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        importer.install(archive, root)
    assert not (root / "runtime").exists()


def test_bad_member_checksum_is_rejected_before_install(tmp_path, monkeypatch):
    root, archive = fixture_archive(tmp_path, monkeypatch, declared="0" * 64)
    with pytest.raises(ValueError, match="member checksum mismatch"):
        importer.install(archive, root)
    assert not (root / "runtime").exists()


def test_conflicting_existing_artifact_is_not_overwritten(tmp_path, monkeypatch):
    root, archive = fixture_archive(tmp_path, monkeypatch)
    path = root / "runtime/sample.bin"
    path.parent.mkdir()
    path.write_bytes(b"preserve my file")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        importer.install(archive, root)
    assert path.read_bytes() == b"preserve my file"


def test_traversal_member_is_rejected(tmp_path, monkeypatch):
    root, archive = fixture_archive(tmp_path, monkeypatch, name="runtime/../../escape.bin")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        importer.install(archive, root)
    assert not (tmp_path / "escape.bin").exists()


def test_changed_installed_data_fails_check(tmp_path, monkeypatch):
    root, archive = fixture_archive(tmp_path, monkeypatch)
    importer.install(archive, root)
    (root / "runtime/sample.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Missing or changed artifacts"):
        importer.check(root)
