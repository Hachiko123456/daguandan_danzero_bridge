from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import daguandan_bridge.build_manifest as build_manifest_module
from daguandan_bridge.build_manifest import (
    BUILD_MANIFEST_FILENAME,
    BuildManifestError,
    RELEASE_RECORD_SCHEMA,
    SCHEMA,
    compute_build_id,
    load_build_manifest,
    verify_build_manifest,
    write_build_manifest,
    write_release_record,
)


SOURCE = {
    "commit": "1" * 40,
    "tree": "2" * 40,
    "branch": "main",
    "dirty": False,
    "status_sha256": None,
}
PYTHON = {
    "version": "3.12.0",
    "implementation": "CPython",
    "architecture": "AMD64",
}
DEPENDENCIES = {
    "PyInstaller": "6.21.0",
    "PySide6": "6.11.1",
    "opencv-python": "4.13.0.92",
    "numpy": "2.3.5",
    "torch": "2.13.0",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bundle(root: Path) -> Path:
    root.mkdir()
    (root / "DaguandanAssistant.exe").write_bytes(b"MZ-frozen-executable")
    (root / "_internal").mkdir()
    (root / "_internal" / "runtime.dll").write_bytes(b"runtime")
    profile = root / "data" / "profiles" / "tencent_daguandan"
    (profile / "templates" / "rank").mkdir(parents=True)
    (profile / "models" / "danzero").mkdir(parents=True)
    (profile / "profile.json").write_text("{}", encoding="utf-8")
    (profile / "regions_config.json").write_text("{}", encoding="utf-8")
    (profile / "templates_config.json").write_text("{}", encoding="utf-8")
    (profile / "templates" / "rank" / "3.png").write_bytes(b"template")
    (profile / "models" / "best.npz").write_bytes(b"fabledan")
    (profile / "models" / "danzero" / "q_network.ckpt").write_bytes(b"danzero")
    return root


def _write(project: Path, bundle: Path) -> dict[str, object]:
    return write_build_manifest(
        project,
        bundle,
        bundle / BUILD_MANIFEST_FILENAME,
        source_identity=SOURCE,
        python_identity=PYTHON,
        dependency_versions=DEPENDENCIES,
    )


def test_manifest_is_deterministic_portable_and_does_not_hash_itself(tmp_path: Path):
    project = tmp_path / "source-machine-specific-name"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")

    first = _write(project, bundle)
    first_bytes = (bundle / BUILD_MANIFEST_FILENAME).read_bytes()
    second = _write(project, bundle)

    assert first == second
    assert first_bytes == (bundle / BUILD_MANIFEST_FILENAME).read_bytes()
    assert first["schema"] == SCHEMA
    assert first["build_id"] == compute_build_id(first)
    assert first["source"] == SOURCE
    assert first["executable"]["path"] == "DaguandanAssistant.exe"
    assert first["dependencies"] == DEPENDENCIES
    assert first["bundle_tree"]["file_count"] == 8
    assert first["resources"]["profile"]["file_count"] == 3
    assert first["resources"]["templates"]["file_count"] == 1
    assert first["resources"]["models"]["file_count"] == 2
    entries = {entry["path"]: entry for entry in first["bundle_tree"]["files"]}
    assert entries["DaguandanAssistant.exe"]["classification"] == "executable"
    assert entries["DaguandanAssistant.exe"]["mutability"] == "immutable"
    assert entries["_internal/runtime.dll"]["classification"] == "internal"
    assert entries["_internal/runtime.dll"]["mutability"] == "immutable"
    assert entries[
        "data/profiles/tencent_daguandan/profile.json"
    ]["classification"] == "profile_config"
    assert entries[
        "data/profiles/tencent_daguandan/profile.json"
    ]["mutability"] == "mutable"
    assert entries[
        "data/profiles/tencent_daguandan/templates/rank/3.png"
    ]["mutability"] == "mutable"
    assert entries[
        "data/profiles/tencent_daguandan/models/best.npz"
    ]["mutability"] == "mutable"
    paths = {entry["path"] for entry in first["bundle_tree"]["files"]}
    assert BUILD_MANIFEST_FILENAME not in paths
    serialized = json.dumps(first, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "source-machine-specific-name" not in serialized
    assert not list(bundle.glob(f".{BUILD_MANIFEST_FILENAME}.*.tmp"))


def test_integrity_verification_detects_mutation_and_optional_extra_files(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")
    manifest_path = bundle / BUILD_MANIFEST_FILENAME
    manifest = _write(project, bundle)

    valid = verify_build_manifest(bundle, manifest_path, strict=True)
    assert valid.ok is True
    assert valid.checked_files == manifest["bundle_tree"]["file_count"]

    extra = bundle / "logs" / "startup.log"
    extra.parent.mkdir()
    extra.write_text("runtime-only", encoding="utf-8")
    assert verify_build_manifest(bundle, manifest_path).ok is True
    strict = verify_build_manifest(bundle, manifest_path, strict=True)
    assert strict.ok is True
    assert strict.warnings == ()
    assert strict.unexpected_files == ("logs/startup.log",)

    target = bundle / "data" / "profiles" / "tencent_daguandan" / "models" / "best.npz"
    target.write_bytes(b"changed-model")
    changed = verify_build_manifest(bundle, manifest_path)
    assert changed.ok is True
    assert changed.errors == ()
    assert any("mismatch: data/profiles" in warning for warning in changed.warnings)
    assert changed.mutable_differences == changed.warnings


def test_immutable_mutation_is_an_error_but_missing_mutable_config_is_a_warning(
    tmp_path: Path,
):
    project = tmp_path / "project"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")
    manifest_path = bundle / BUILD_MANIFEST_FILENAME
    _write(project, bundle)

    (bundle / "_internal" / "runtime.dll").write_bytes(b"changed")
    immutable = verify_build_manifest(bundle, manifest_path)
    assert immutable.ok is False
    assert any("_internal/runtime.dll" in error for error in immutable.errors)

    (bundle / "_internal" / "runtime.dll").write_bytes(b"runtime")
    (bundle / "data" / "profiles" / "tencent_daguandan" / "profile.json").unlink()
    mutable = verify_build_manifest(bundle, manifest_path)
    assert mutable.ok is True
    assert mutable.errors == ()
    assert any("profile.json" in warning for warning in mutable.warnings)
    assert mutable.mutable_differences == mutable.warnings


def test_strict_unexpected_file_policy_allows_runtime_warns_profile_and_rejects_root(
    tmp_path: Path,
):
    project = tmp_path / "project"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")
    manifest_path = bundle / BUILD_MANIFEST_FILENAME
    _write(project, bundle)

    runtime = bundle / "data" / "profiles" / "tencent_daguandan" / "sessions" / "game" / "timeline.jsonl"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("", encoding="utf-8")
    user_asset = bundle / "data" / "profiles" / "tencent_daguandan" / "pics" / "sample.png"
    user_asset.parent.mkdir()
    user_asset.write_bytes(b"user")
    profile_only = verify_build_manifest(bundle, manifest_path, strict=True)
    assert profile_only.ok is True
    assert profile_only.warnings == (
        "unexpected bundle file: data/profiles/tencent_daguandan/pics/sample.png",
    )

    (bundle / "rogue.dll").write_bytes(b"not shipped")
    root_extra = verify_build_manifest(bundle, manifest_path, strict=True)
    assert root_extra.ok is False
    assert "unexpected bundle file: rogue.dll" in root_extra.errors


def test_create_and_strict_verify_reject_reparse_directories_before_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "project"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")
    unsafe = bundle / "linked"
    unsafe.mkdir()
    original = build_manifest_module._is_link_or_reparse

    monkeypatch.setattr(
        build_manifest_module,
        "_is_link_or_reparse",
        lambda path: Path(path) == unsafe or original(Path(path)),
    )
    with pytest.raises(BuildManifestError, match="reparse point"):
        _write(project, bundle)

    monkeypatch.setattr(build_manifest_module, "_is_link_or_reparse", original)
    unsafe.rmdir()
    manifest_path = bundle / BUILD_MANIFEST_FILENAME
    _write(project, bundle)
    unsafe.mkdir()
    monkeypatch.setattr(
        build_manifest_module,
        "_is_link_or_reparse",
        lambda path: Path(path) == unsafe or original(Path(path)),
    )
    verified = verify_build_manifest(bundle, manifest_path, strict=True)
    assert verified.ok is False
    assert any("reparse point" in error for error in verified.errors)


def test_verification_rejects_manifest_path_escape_without_reading_it(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")
    manifest = _write(project, bundle)
    outside = tmp_path / "outside.dll"
    outside.write_bytes(b"runtime")
    manifest["bundle_tree"]["files"][0]["path"] = "../outside.dll"

    result = verify_build_manifest(bundle, manifest)

    assert result.ok is False
    assert any("not portable" in error for error in result.errors)


def test_release_record_and_checksum_are_atomic_and_path_portable(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")
    manifest_path = bundle / BUILD_MANIFEST_FILENAME
    manifest = _write(project, bundle)
    archive = tmp_path / "DaguandanAssistant.zip"
    archive.write_bytes(b"portable-archive")
    record_path = tmp_path / "DaguandanAssistant.release.json"
    checksum_path = tmp_path / "DaguandanAssistant.zip.sha256"

    record = write_release_record(
        manifest_path,
        archive,
        record_path,
        checksum_path,
    )

    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert record["schema"] == RELEASE_RECORD_SCHEMA
    assert record["build_id"] == manifest["build_id"]
    assert record["archive"] == {
        "path": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": archive_hash,
    }
    assert record["build_manifest"]["path"] == BUILD_MANIFEST_FILENAME
    assert checksum_path.read_text(encoding="ascii") == f"{archive_hash}  {archive.name}\n"
    assert load_build_manifest(record_path) == record
    assert str(tmp_path) not in record_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_diagnostics_launcher_runs_doctor_without_exporting_raw_logs():
    launcher = (
        PROJECT_ROOT / "release_assets" / "Collect_Diagnostics.bat"
    ).read_text(encoding="utf-8")

    assert "--doctor --doctor-output" in launcher
    assert "launcher.log" in launcher
    assert 'set "DAGUANDAN_DIAGNOSTICS_ROOT=%DIAG_ROOT%\\diagnostics"' in launcher
    assert "diagnostics_subdirectory=diagnostics" in launcher
    assert "doctor_report=doctor.json" in launcher
    assert "Compress-Archive" not in launcher
    assert "Copy-Item" not in launcher
    assert "support ZIP" in launcher


def test_manifest_cli_creates_and_strictly_verifies_a_temporary_bundle(tmp_path: Path):
    bundle = _bundle(tmp_path / "bundle")
    script = PROJECT_ROOT / "scripts" / "generate_build_manifest.py"
    manifest = bundle / BUILD_MANIFEST_FILENAME

    created = subprocess.run(
        [
            sys.executable,
            str(script),
            "create",
            "--project-root",
            str(PROJECT_ROOT),
            "--bundle-root",
            str(bundle),
            "--output",
            str(manifest),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(script),
            "verify",
            "--bundle-root",
            str(bundle),
            "--manifest",
            str(manifest),
            "--strict",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert created.returncode == 0, created.stderr
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert json.loads(created.stdout)["ok"] is True
    assert json.loads(verified.stdout)["ok"] is True
    assert str(PROJECT_ROOT) not in manifest.read_text(encoding="utf-8")
