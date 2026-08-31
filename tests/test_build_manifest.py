from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import daguandan_bridge.build_manifest as build_manifest_module
from daguandan_bridge.build_manifest import (
    BUILD_INPUTS_SCHEMA,
    BUILD_MANIFEST_FILENAME,
    BuildManifestError,
    RELEASE_RECORD_SCHEMA,
    SCHEMA,
    compute_build_id,
    collect_release_build_inputs,
    load_build_manifest,
    verify_build_manifest,
    verify_source_identity,
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
    assert first["pyinstaller"]["version"] == "6.21.0"
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
    ]["mutability"] == "immutable"
    assert entries[
        "data/profiles/tencent_daguandan/templates/rank/3.png"
    ]["mutability"] == "immutable"
    assert entries[
        "data/profiles/tencent_daguandan/models/best.npz"
    ]["mutability"] == "immutable"
    assert first["build_inputs"] == {
        "schema": BUILD_INPUTS_SCHEMA,
        "status": "not_provided",
    }
    paths = {entry["path"] for entry in first["bundle_tree"]["files"]}
    assert BUILD_MANIFEST_FILENAME not in paths
    serialized = json.dumps(first, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "source-machine-specific-name" not in serialized
    assert not list(bundle.glob(f".{BUILD_MANIFEST_FILENAME}.*.tmp"))


def test_qualified_manifest_binds_locks_complete_inventory_and_native_audit(
    tmp_path: Path,
):
    project = tmp_path / "project"
    project.mkdir()
    for name in (
        "requirements-release.in",
        "requirements-release.lock",
        "release_toolchain.lock.json",
        "python_runtime.lock.json",
        "wheelhouse.lock.json",
    ):
        (project / name).write_bytes((PROJECT_ROOT / name).read_bytes())
    bundle = _bundle(tmp_path / "bundle")
    input_audit = bundle / "release_input_audit.json"
    input_audit.write_text(
        json.dumps(
            {
                "schema": "guandan.release-input-audit/1",
                "status": "PASS",
                "installed_distributions": DEPENDENCIES,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    native_audit = bundle / "native_dependency_audit.json"
    native_audit.write_text(
        json.dumps(
            {
                "schema": "guandan.native-dependency-audit/1",
                "status": "PASS",
                "summary": {"pe_file_count": 2, "error_count": 0},
                "files": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    build_inputs = collect_release_build_inputs(
        project,
        bundle,
        release_input_audit=input_audit,
        native_audit=native_audit,
    )

    manifest = write_build_manifest(
        project,
        bundle,
        dependency_versions=DEPENDENCIES,
        source_identity=SOURCE,
        python_identity=PYTHON,
        build_inputs=build_inputs,
    )

    assert manifest["build_inputs"]["status"] == "PASS"
    assert set(manifest["build_inputs"]["locks"]) == {
        "requirements_input",
        "requirements_lock",
        "toolchain_lock",
        "python_runtime_lock",
        "wheelhouse_lock",
    }
    assert manifest["dependencies"] == DEPENDENCIES
    assert manifest["build_inputs"]["native_dependency_audit"]["summary"] == {
        "pe_file_count": 2,
        "error_count": 0,
    }
    assert verify_build_manifest(
        bundle,
        bundle / BUILD_MANIFEST_FILENAME,
        strict=True,
    ).ok is True
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert str(tmp_path) not in serialized

    native_audit.write_text("{}", encoding="utf-8")
    tampered = verify_build_manifest(bundle, bundle / BUILD_MANIFEST_FILENAME)
    assert tampered.ok is False
    assert any("native_dependency_audit.json" in error for error in tampered.errors)


def test_release_build_inputs_reject_failed_native_audit(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    for name in (
        "requirements-release.in",
        "requirements-release.lock",
        "release_toolchain.lock.json",
        "python_runtime.lock.json",
        "wheelhouse.lock.json",
    ):
        (project / name).write_bytes((PROJECT_ROOT / name).read_bytes())
    bundle = _bundle(tmp_path / "bundle")
    input_audit = bundle / "release_input_audit.json"
    input_audit.write_text(
        json.dumps(
            {
                "schema": "guandan.release-input-audit/1",
                "status": "PASS",
                "installed_distributions": DEPENDENCIES,
            }
        ),
        encoding="utf-8",
    )
    native = bundle / "native_dependency_audit.json"
    native.write_text(
        json.dumps(
            {
                "schema": "guandan.native-dependency-audit/1",
                "status": "FAIL",
                "summary": {"error_count": 1},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BuildManifestError, match="native dependency audit did not pass"):
        collect_release_build_inputs(
            project,
            bundle,
            release_input_audit=input_audit,
            native_audit=native,
        )


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
    assert strict.ok is False
    assert "unexpected bundle file: logs/startup.log" in strict.errors
    assert strict.unexpected_files == ("logs/startup.log",)

    target = bundle / "data" / "profiles" / "tencent_daguandan" / "models" / "best.npz"
    target.write_bytes(b"changed-model")
    changed = verify_build_manifest(bundle, manifest_path)
    assert changed.ok is False
    assert any("mismatch: data/profiles" in error for error in changed.errors)
    assert changed.warnings == ()
    assert changed.mutable_differences == ()


def test_strict_manifest_rejects_dirty_source_identity(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    bundle = _bundle(tmp_path / "bundle")
    manifest = _write(project, bundle)
    manifest["source"]["dirty"] = True
    manifest["source"]["status_sha256"] = "f" * 64
    manifest["build_id"] = compute_build_id(manifest)

    result = verify_build_manifest(bundle, manifest, strict=True)

    assert result.ok is False
    assert "strict release manifest source identity must be clean" in result.errors


def test_source_identity_gate_detects_tracked_mutation_during_build(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        ["git", "-C", str(project), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "config", "user.name", "Release Test"],
        check=True,
    )
    tracked = project / "tracked.py"
    tracked.write_text("before = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-qm", "initial"], check=True)
    expected = build_manifest_module.collect_source_identity(project)
    assert verify_source_identity(project, expected)["dirty"] is False

    tracked.write_text("after = 2\n", encoding="utf-8")

    with pytest.raises(BuildManifestError, match="changed or became dirty"):
        verify_source_identity(project, expected)


def test_immutable_runtime_and_missing_seed_config_are_both_errors(
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
    missing_seed = verify_build_manifest(bundle, manifest_path)
    assert missing_seed.ok is False
    assert any("profile.json" in error for error in missing_seed.errors)
    assert missing_seed.warnings == ()
    assert missing_seed.mutable_differences == ()


def test_strict_unexpected_file_policy_rejects_all_package_mutation(
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
    assert profile_only.ok is False
    assert profile_only.warnings == ()
    assert "unexpected bundle file: data/profiles/tencent_daguandan/sessions/game/timeline.jsonl" in profile_only.errors
    assert "unexpected bundle file: data/profiles/tencent_daguandan/pics/sample.png" in profile_only.errors

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
    source_identity = tmp_path / "source_identity.json"
    source_identity.write_text(json.dumps(SOURCE), encoding="utf-8")

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
            "--source-identity",
            str(source_identity),
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
