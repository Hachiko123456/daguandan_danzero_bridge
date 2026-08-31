from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

import daguandan_bridge.release_manager as release_manager_module
from daguandan_bridge.build_manifest import write_build_manifest, write_release_record
from daguandan_bridge.release_manager import (
    BASELINE_AUTH_SCHEMA,
    BASELINE_SOURCE_COMMIT,
    BASELINE_SOURCE_TREE,
    BASELINE_TAG,
    ReleaseManagerError,
    activate_release,
    ensure_install_root,
    install_release,
    register_legacy_baseline,
    release_status,
    rollback_release,
    write_baseline_auth,
)
from daguandan_bridge.doctor import DOCTOR_REQUIRED_CHECK_IDS
from daguandan_bridge.runtime_layout import (
    activate_generation,
    copy_seed_resources,
    layout_for_generation,
    resolve_runtime_layout,
    write_generation_marker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _source(commit: str) -> dict[str, object]:
    return {
        "commit": commit,
        "tree": BASELINE_SOURCE_TREE if commit == BASELINE_SOURCE_COMMIT else "b" * 40,
        "branch": "test",
        "dirty": False,
        "status_sha256": None,
    }


def _release(tmp_path: Path, name: str, *, commit: str):
    root = tmp_path / name
    bundle = root / "DaguandanAssistant"
    profile = bundle / "data" / "profiles" / "tencent_daguandan"
    profile.mkdir(parents=True)
    (bundle / "DaguandanAssistant.exe").write_bytes(b"MZ" + name.encode())
    for filename in ("profile.json", "regions_config.json", "templates_config.json"):
        (profile / filename).write_text("{}\n", encoding="utf-8")
    template = profile / "templates" / "rank" / "7_level.png"
    template.parent.mkdir(parents=True)
    template.write_bytes(b"template")
    fabledan = profile / "models" / "best.npz"
    danzero = profile / "models" / "danzero" / "q_network.ckpt"
    fabledan.parent.mkdir(parents=True)
    danzero.parent.mkdir(parents=True)
    fabledan.write_bytes(b"model")
    danzero.write_bytes(b"checkpoint")
    (bundle / "native_dependency_audit.json").write_text(
        json.dumps(
            {
                "schema": "guandan.native-dependency-audit/1",
                "status": "PASS",
                "files": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    manifest = write_build_manifest(
        PROJECT_ROOT,
        bundle,
        bundle / "build_manifest.json",
        source_identity=_source(commit),
        python_identity={"version": "3.12.0", "implementation": "CPython", "architecture": "AMD64"},
        dependency_versions={},
    )
    archive = root / "DaguandanAssistant.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(root).as_posix())
    record = root / "DaguandanAssistant.release.json"
    checksum = root / "DaguandanAssistant.zip.sha256"
    release_document = write_release_record(
        bundle / "build_manifest.json",
        archive,
        record,
        checksum,
    )
    return archive, record, checksum, manifest, release_document


def _doctor(release, _runtime_root, output_path):
    report = {
        "schema": "guandan.doctor/1",
        "overall_status": "PASS",
        "identity": {
            "schema": "guandan.runtime-identity/1",
            "frozen": True,
            "build_status": "identified",
            "build_id": release.build_id,
        },
        "checks": [
            {
                "id": check_id,
                "status": "PASS",
                "summary": "ok",
                "evidence": {},
                "duration_ms": 0.0,
            }
            for check_id in DOCTOR_REQUIRED_CHECK_IDS
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _legacy_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "external" / "baseline"
    bundle.mkdir(parents=True)
    (bundle / "DaguandanAssistant.exe").write_bytes(b"known-stable-exe")
    (bundle / "stable-resource.bin").write_bytes(b"known-stable-resource")
    auth = tmp_path / "external" / "baseline-auth.json"
    write_baseline_auth(bundle, auth, approved=True)
    return bundle, auth


def _modern_baseline_auth(tmp_path: Path, release_files) -> Path:
    bundle = release_files[0].parent / "DaguandanAssistant"
    auth = tmp_path / "external" / f"{release_files[0].parent.name}-baseline-auth.json"
    write_baseline_auth(bundle, auth, approved=True)
    return auth


def test_install_activate_candidate_and_rollback_preserves_bad_version(tmp_path):
    baseline_files = _release(tmp_path, "baseline", commit=BASELINE_SOURCE_COMMIT)
    candidate_files = _release(tmp_path, "candidate", commit="a" * 40)
    runtime = tmp_path / "用户 数据"
    baseline = install_release(
        baseline_files[0],
        release_record_path=baseline_files[1],
        checksum_path=baseline_files[2],
        runtime_root=runtime,
        baseline=True,
        baseline_auth_path=_modern_baseline_auth(tmp_path, baseline_files),
    )
    candidate = install_release(
        candidate_files[0],
        release_record_path=candidate_files[1],
        checksum_path=candidate_files[2],
        runtime_root=runtime,
    )
    activate_release(baseline.release_id, runtime_root=runtime, doctor_runner=_doctor)
    active = activate_release(candidate.release_id, runtime_root=runtime, doctor_runner=_doctor)
    assert active["release_id"] == candidate.release_id

    def support_exporter(release, destination):
        destination.write_bytes(b"PK\x05\x06" + b"\0" * 18)
        return {"status": "PASS", "release_id": release.release_id}

    rolled_back = rollback_release(runtime_root=runtime, support_exporter=support_exporter)

    assert rolled_back["release_id"] == baseline.release_id
    assert candidate.version_root.is_dir()
    status = release_status(runtime)
    assert status["active"]["release_id"] == baseline.release_id
    receipts = list((runtime / "install" / "rollback-receipts").glob("rollback-*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["bad_directory_preserved"] is True
    assert receipt["support"]["status"] == "PASS"


def test_candidate_cannot_overwrite_immutable_baseline_receipt(tmp_path):
    baseline_files = _release(tmp_path, "baseline", commit=BASELINE_SOURCE_COMMIT)
    runtime = tmp_path / "runtime"
    installed = install_release(
        baseline_files[0],
        release_record_path=baseline_files[1],
        checksum_path=baseline_files[2],
        runtime_root=runtime,
        baseline=True,
        baseline_auth_path=_modern_baseline_auth(tmp_path, baseline_files),
    )
    baseline_path = runtime / "install" / "baseline.json"
    before = baseline_path.read_bytes()

    candidate_files = _release(tmp_path, "candidate", commit="c" * 40)
    install_release(
        candidate_files[0],
        release_record_path=candidate_files[1],
        checksum_path=candidate_files[2],
        runtime_root=runtime,
    )

    assert baseline_path.read_bytes() == before
    assert json.loads(before)["release"]["release_id"] == installed.release_id


def test_install_rejects_zip_traversal_before_writing_outside_staging(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", b"escape")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = tmp_path / "bad.release.json"
    record.write_text(
        json.dumps(
            {
                "schema": "guandan.release-record/1",
                "release_id": "bad",
                "archive": {"sha256": digest},
                "build_manifest": {"sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    checksum = tmp_path / "bad.sha256"
    checksum.write_text(f"{digest}  bad.zip\n", encoding="ascii")

    with pytest.raises(ReleaseManagerError, match="unsafe"):
        install_release(
            archive,
            release_record_path=record,
            checksum_path=checksum,
            runtime_root=tmp_path / "runtime",
        )

    assert not (tmp_path / "escape.txt").exists()


def test_install_root_refuses_nonempty_unowned_runtime_root(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "unrelated.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(ReleaseManagerError, match="non-empty unmarked"):
        ensure_install_root(runtime)

    assert (runtime / "unrelated.txt").read_text(encoding="utf-8") == "mine"


def test_baseline_tag_still_resolves_to_frozen_commit():
    resolved = subprocess.run(
        ["git", "rev-parse", f"{BASELINE_TAG}^{{}}"],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    assert resolved == BASELINE_SOURCE_COMMIT


def test_failed_candidate_doctor_never_changes_active_pointer(tmp_path):
    baseline_files = _release(tmp_path, "baseline", commit=BASELINE_SOURCE_COMMIT)
    candidate_files = _release(tmp_path, "candidate", commit="d" * 40)
    runtime = tmp_path / "runtime"
    baseline = install_release(
        baseline_files[0],
        release_record_path=baseline_files[1],
        checksum_path=baseline_files[2],
        runtime_root=runtime,
        baseline=True,
        baseline_auth_path=_modern_baseline_auth(tmp_path, baseline_files),
    )
    candidate = install_release(
        candidate_files[0],
        release_record_path=candidate_files[1],
        checksum_path=candidate_files[2],
        runtime_root=runtime,
    )
    activate_release(baseline.release_id, runtime_root=runtime, doctor_runner=_doctor)
    before = (runtime / "install" / "active.json").read_bytes()
    data_pointer = runtime / "data" / "v1" / "active.json"
    data_before = data_pointer.read_bytes()
    observed_probe_roots = []

    def failed_doctor(_release, probe_runtime, _output):
        observed_probe_roots.append(probe_runtime)
        return {
            "schema": "guandan.doctor/1",
            "overall_status": "FAIL",
            "checks": [{"id": "FAIL", "status": "FAIL"}],
        }

    with pytest.raises(ReleaseManagerError, match="doctor did not pass"):
        activate_release(candidate.release_id, runtime_root=runtime, doctor_runner=failed_doctor)

    assert (runtime / "install" / "active.json").read_bytes() == before
    assert data_pointer.read_bytes() == data_before
    assert observed_probe_roots
    assert observed_probe_roots[0] != runtime
    assert str(observed_probe_roots[0]).startswith(str(runtime / "install" / "activation-probes"))


def test_rollback_refuses_tampered_previous_version_and_keeps_candidate_active(tmp_path):
    baseline_files = _release(tmp_path, "baseline", commit=BASELINE_SOURCE_COMMIT)
    candidate_files = _release(tmp_path, "candidate", commit="e" * 40)
    runtime = tmp_path / "runtime"
    baseline = install_release(
        baseline_files[0],
        release_record_path=baseline_files[1],
        checksum_path=baseline_files[2],
        runtime_root=runtime,
        baseline=True,
        baseline_auth_path=_modern_baseline_auth(tmp_path, baseline_files),
    )
    candidate = install_release(
        candidate_files[0],
        release_record_path=candidate_files[1],
        checksum_path=candidate_files[2],
        runtime_root=runtime,
    )
    activate_release(baseline.release_id, runtime_root=runtime, doctor_runner=_doctor)
    activate_release(candidate.release_id, runtime_root=runtime, doctor_runner=_doctor)
    baseline.executable.write_bytes(b"tampered")

    with pytest.raises(ReleaseManagerError, match="hash mismatch"):
        rollback_release(runtime_root=runtime)

    active = json.loads((runtime / "install" / "active.json").read_text(encoding="utf-8"))
    assert active["release_id"] == candidate.release_id


def test_legacy_baseline_requires_external_full_tree_preapproval(tmp_path):
    bundle, auth = _legacy_bundle(tmp_path)
    runtime = tmp_path / "runtime"

    installed = register_legacy_baseline(
        bundle,
        baseline_auth_path=auth,
        runtime_root=runtime,
    )

    auth_document = json.loads(auth.read_text(encoding="utf-8"))
    assert auth_document["schema"] == BASELINE_AUTH_SCHEMA
    assert auth_document["approved"] is True
    assert installed.baseline is True
    baseline_receipt = json.loads(
        (runtime / "install" / "baseline.json").read_text(encoding="utf-8")
    )
    assert baseline_receipt["baseline_auth_sha256"] == hashlib.sha256(
        auth.read_bytes()
    ).hexdigest()


def test_legacy_baseline_rejects_any_tree_change_after_preapproval(tmp_path):
    bundle, auth = _legacy_bundle(tmp_path)
    (bundle / "stable-resource.bin").write_bytes(b"changed-after-approval")

    with pytest.raises(ReleaseManagerError, match="artifact tree"):
        register_legacy_baseline(
            bundle,
            baseline_auth_path=auth,
            runtime_root=tmp_path / "runtime",
        )


def test_modern_install_cannot_claim_baseline_without_external_authorization(tmp_path):
    release_files = _release(tmp_path, "synthetic-baseline", commit=BASELINE_SOURCE_COMMIT)
    runtime = tmp_path / "runtime"

    with pytest.raises(ReleaseManagerError, match="external baseline authorization"):
        install_release(
            release_files[0],
            release_record_path=release_files[1],
            checksum_path=release_files[2],
            runtime_root=runtime,
            baseline=True,
        )

    assert not (runtime / "install" / "baseline.json").exists()


def test_dirty_modern_install_cannot_claim_baseline_even_with_tree_preapproval(tmp_path):
    release_files = _release(tmp_path, "dirty-baseline", commit=BASELINE_SOURCE_COMMIT)
    archive, record, checksum, _manifest, _release_document = release_files
    bundle = archive.parent / "DaguandanAssistant"
    manifest_path = bundle / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["dirty"] = True
    # The test intentionally presents an internally consistent but dirty build.
    from daguandan_bridge.build_manifest import compute_build_id

    manifest["build_id"] = compute_build_id(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(archive.parent).as_posix())
    write_release_record(manifest_path, archive, record, checksum)
    auth = tmp_path / "external" / "dirty-auth.json"
    write_baseline_auth(bundle, auth, approved=True)

    with pytest.raises(ReleaseManagerError, match="source identity must be clean"):
        install_release(
            archive,
            release_record_path=record,
            checksum_path=checksum,
            runtime_root=tmp_path / "runtime",
            baseline=True,
            baseline_auth_path=auth,
        )


def test_legacy_baseline_uses_mutable_run_copy_without_mutating_approved_artifact(
    tmp_path,
):
    bundle, auth = _legacy_bundle(tmp_path)
    approved_before = _snapshot_tree(bundle)
    runtime = tmp_path / "runtime"
    baseline = register_legacy_baseline(
        bundle,
        baseline_auth_path=auth,
        runtime_root=runtime,
    )

    runtime_file = (
        baseline.executable.parent
        / "data"
        / "profiles"
        / "tencent_daguandan"
        / "sessions"
        / "game-written-after-registration"
        / "timeline.jsonl"
    )
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text('{"event":"runtime-write"}\n', encoding="utf-8")

    status = release_status(runtime)
    reloaded = next(
        item for item in status["versions"] if item["release_id"] == baseline.release_id
    )
    receipt = json.loads(
        (baseline.version_root / "install_receipt.json").read_text(encoding="utf-8")
    )
    approved_root = baseline.version_root / receipt["approved_artifact_relative"]
    assert runtime_file.is_file()
    assert approved_root != baseline.executable.parent
    assert _snapshot_tree(bundle) == approved_before
    assert _snapshot_tree(approved_root) == approved_before
    assert reloaded["baseline"] is True


def _snapshot_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_preauthorized_legacy_activation_uses_nonempty_hash_bound_doctor(tmp_path):
    bundle, auth = _legacy_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    baseline = register_legacy_baseline(
        bundle,
        baseline_auth_path=auth,
        runtime_root=runtime,
    )

    active = activate_release(baseline.release_id, runtime_root=runtime)

    doctor_path = runtime / "install" / "rollback-receipts" / active["doctor_report"]
    report = json.loads(doctor_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "PASS"
    assert report["legacy_preauthorized"] is True
    assert {item["id"] for item in report["checks"]} == {
        "LEGACY-BASELINE-PREAUTH",
        "LEGACY-ARTIFACT-INTEGRITY",
    }


def test_activation_preserves_real_migrated_generation_and_rollback_restores_marker(
    tmp_path,
):
    baseline_files = _release(tmp_path, "baseline", commit=BASELINE_SOURCE_COMMIT)
    candidate_files = _release(tmp_path, "candidate", commit="f" * 40)
    runtime = tmp_path / "runtime"
    baseline = install_release(
        baseline_files[0],
        release_record_path=baseline_files[1],
        checksum_path=baseline_files[2],
        runtime_root=runtime,
        baseline=True,
        baseline_auth_path=_modern_baseline_auth(tmp_path, baseline_files),
    )
    candidate = install_release(
        candidate_files[0],
        release_record_path=candidate_files[1],
        checksum_path=candidate_files[2],
        runtime_root=runtime,
    )
    baseline_active = activate_release(
        baseline.release_id,
        runtime_root=runtime,
        doctor_runner=_doctor,
    )
    baseline_pointer = baseline_active["data_pointer"]
    baseline_marker_hash = baseline_active["data_marker_sha256"]

    candidate_layout = resolve_runtime_layout(
        frozen=True,
        bundle_root=candidate.executable.parent,
        environ={"DAGUANDAN_DATA_ROOT": str(runtime)},
    )
    migrated = layout_for_generation(candidate_layout, "B-m-test-generation")
    seed = copy_seed_resources(migrated, migrated.generation_root)
    write_generation_marker(migrated.generation_root, migrated, seed_summary=seed)
    activate_generation(migrated, migrated.generation_id)

    candidate_active = activate_release(
        candidate.release_id,
        runtime_root=runtime,
        doctor_runner=_doctor,
    )

    assert candidate_active["data_generation"] == "B-m-test-generation"
    assert candidate_active["data_pointer"]["generation_id"] == "B-m-test-generation"
    assert candidate_active["previous"]["data_pointer"] == baseline_pointer
    rolled = rollback_release(
        runtime_root=runtime,
        support_exporter=lambda _release, _destination: {"status": "PASS"},
    )
    restored_pointer = json.loads(
        (runtime / "data" / "v1" / "active.json").read_text(encoding="utf-8")
    )
    assert rolled["release_id"] == baseline.release_id
    assert {
        key: value for key, value in restored_pointer.items() if key != "transaction_id"
    } == {key: value for key, value in baseline_pointer.items() if key != "transaction_id"}
    assert restored_pointer["transaction_id"] == rolled["transaction_id"]
    assert rolled["data_marker_sha256"] == baseline_marker_hash
    receipt_path = next(
        (runtime / "install" / "rollback-receipts").glob("rollback-*/receipt.json")
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["restored_pointer_verified"] is True
    assert receipt["restored_data_snapshot"]["generation_marker_sha256"] == baseline_marker_hash


def test_pointer_transaction_restores_both_files_when_release_publish_fails(
    tmp_path,
    monkeypatch,
):
    baseline_files = _release(tmp_path, "baseline", commit=BASELINE_SOURCE_COMMIT)
    candidate_files = _release(tmp_path, "candidate", commit="9" * 40)
    runtime = tmp_path / "runtime"
    baseline = install_release(
        baseline_files[0],
        release_record_path=baseline_files[1],
        checksum_path=baseline_files[2],
        runtime_root=runtime,
        baseline=True,
        baseline_auth_path=_modern_baseline_auth(tmp_path, baseline_files),
    )
    candidate = install_release(
        candidate_files[0],
        release_record_path=candidate_files[1],
        checksum_path=candidate_files[2],
        runtime_root=runtime,
    )
    activate_release(baseline.release_id, runtime_root=runtime, doctor_runner=_doctor)
    release_path = runtime / "install" / "active.json"
    data_path = runtime / "data" / "v1" / "active.json"
    release_before = release_path.read_bytes()
    data_before = data_path.read_bytes()
    real_write = release_manager_module.atomic_write_json
    failed_once = False

    def fail_release_pointer(path, value):
        nonlocal failed_once
        target = Path(path)
        if target == release_path and not failed_once:
            failed_once = True
            raise OSError("simulated release pointer publication failure")
        return real_write(path, value)

    monkeypatch.setattr(release_manager_module, "atomic_write_json", fail_release_pointer)

    with pytest.raises(OSError, match="simulated"):
        activate_release(candidate.release_id, runtime_root=runtime, doctor_runner=_doctor)

    assert release_path.read_bytes() == release_before
    assert data_path.read_bytes() == data_before
    transactions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (runtime / "install" / "transactions").glob("activate-*.json")
    ]
    assert any(item["phase"] == "ROLLED_BACK" for item in transactions)


def test_two_processes_install_same_release_without_partial_staging(tmp_path):
    release_files = _release(tmp_path, "concurrent-candidate", commit="6" * 40)
    runtime = tmp_path / "concurrent-runtime"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "manage_release.py"),
        "--runtime-root",
        str(runtime),
        "install",
        str(release_files[0]),
        "--release-record",
        str(release_files[1]),
        "--checksum",
        str(release_files[2]),
    ]
    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    completed = [process.communicate(timeout=30) for process in processes]

    for process, (stdout, stderr) in zip(processes, completed, strict=True):
        assert process.returncode == 0, stdout + stderr
    status = release_status(runtime)
    assert len(status["versions"]) == 1
    assert not list((runtime / "install" / "versions").glob(".*.tmp"))


def test_manage_release_cli_rejects_baseline_flag_without_external_auth(tmp_path):
    release_files = _release(tmp_path, "cli-baseline", commit=BASELINE_SOURCE_COMMIT)
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "manage_release.py"),
            "--runtime-root",
            str(tmp_path / "runtime"),
            "install",
            str(release_files[0]),
            "--release-record",
            str(release_files[1]),
            "--checksum",
            str(release_files[2]),
            "--baseline",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "--baseline requires --baseline-auth" in completed.stderr


def _run_launcher(runtime: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "release_assets" / "Launch_DaguandanAssistant.ps1"),
            "-RuntimeRoot",
            str(runtime),
            "-VerifyOnly",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_launcher_verifies_full_modern_manifest_native_audit_and_data_pointer(tmp_path):
    release_files = _release(tmp_path, "candidate", commit="7" * 40)
    runtime = tmp_path / "runtime"
    installed = install_release(
        release_files[0],
        release_record_path=release_files[1],
        checksum_path=release_files[2],
        runtime_root=runtime,
    )
    activate_release(installed.release_id, runtime_root=runtime, doctor_runner=_doctor)

    passed = _run_launcher(runtime)

    assert passed.returncode == 0, passed.stdout + passed.stderr
    resource = (
        installed.executable.parent
        / "data"
        / "profiles"
        / "tencent_daguandan"
        / "regions_config.json"
    )
    resource.write_text('{"tampered":true}\n', encoding="utf-8")
    failed = _run_launcher(runtime)
    assert failed.returncode != 0
    assert "changed" in (failed.stdout + failed.stderr)


def test_launcher_rejects_unsafe_version_segment_before_path_resolution(tmp_path):
    bundle, auth = _legacy_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    baseline = register_legacy_baseline(
        bundle,
        baseline_auth_path=auth,
        runtime_root=runtime,
    )
    activate_release(baseline.release_id, runtime_root=runtime)
    passed = _run_launcher(runtime)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    active_path = runtime / "install" / "active.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["version_directory"] = "../escape"
    active_path.write_text(json.dumps(active), encoding="utf-8")

    failed = _run_launcher(runtime)

    assert failed.returncode != 0
    assert "safe single path segment" in (failed.stdout + failed.stderr)
