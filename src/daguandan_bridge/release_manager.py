from __future__ import annotations

"""Versioned installation, atomic activation, and recoverable rollback."""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from typing import Callable, Mapping
from uuid import uuid4
import zipfile

from .build_manifest import (
    BUILD_MANIFEST_FILENAME,
    RELEASE_RECORD_SCHEMA,
    load_build_manifest,
    sha256_file,
    verify_build_manifest,
)
from .runtime_layout import (
    APP_DIRECTORY_NAME,
    RUNTIME_ROOT_MARKER,
    RUNTIME_ROOT_SCHEMA,
    assert_safe_tree,
    atomic_write_json,
)


INSTALL_ROOT_SCHEMA = "guandan.install-root/1"
ACTIVE_RELEASE_SCHEMA = "guandan.active-release/1"
ROLLBACK_RECEIPT_SCHEMA = "guandan.rollback-receipt/1"
LEGACY_BASELINE_SCHEMA = "guandan.legacy-baseline/1"
BASELINE_SOURCE_COMMIT = "2db427b0937dfa390eaad407a92a084556af1279"
BASELINE_TAG = "baseline/local-stable-20260831"
_INSTALL_MARKER = ".daguandan-install-root.json"
_ACTIVE_FILE = "active.json"
_HISTORY_FILE = "history.jsonl"
_BASELINE_FILE = "baseline.json"
_MAX_ARCHIVE_FILES = 100_000
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024


class ReleaseManagerError(RuntimeError):
    pass


@dataclass(frozen=True)
class InstalledRelease:
    release_id: str
    build_id: str
    version_root: Path
    executable: Path
    source_commit: str | None
    baseline: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "build_id": self.build_id,
            "version_directory": self.version_root.name,
            "executable_relative": self.executable.relative_to(self.version_root).as_posix(),
            "source_commit": self.source_commit,
            "baseline": self.baseline,
        }


def default_runtime_root(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    override = str(values.get("DAGUANDAN_DATA_ROOT") or "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        local = str(values.get("LOCALAPPDATA") or "").strip()
        if not local:
            raise ReleaseManagerError("LOCALAPPDATA is unavailable")
        root = Path(local) / APP_DIRECTORY_NAME
    if not root.is_absolute():
        raise ReleaseManagerError("runtime root must be absolute")
    return Path(os.path.abspath(os.fspath(root)))


def ensure_install_root(runtime_root: Path | str | None = None) -> Path:
    runtime = Path(runtime_root or default_runtime_root())
    _reject_filesystem_root(runtime, "runtime root")
    _assert_no_reparse_chain(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(runtime)
    _ensure_runtime_marker(runtime)
    install = runtime / "install"
    _assert_no_reparse_chain(install)
    install.mkdir(exist_ok=True)
    marker = install / _INSTALL_MARKER
    if marker.exists():
        value = _json_file(marker, "install marker")
        if value.get("schema") != INSTALL_ROOT_SCHEMA or value.get("application") != APP_DIRECTORY_NAME:
            raise ReleaseManagerError("install root marker is invalid")
    else:
        existing = [entry for entry in install.iterdir() if entry.name != _INSTALL_MARKER]
        if existing:
            raise ReleaseManagerError("refusing to adopt a non-empty unmarked install root")
        atomic_write_json(
            marker,
            {
                "schema": INSTALL_ROOT_SCHEMA,
                "application": APP_DIRECTORY_NAME,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    for name in ("versions", "rollback-receipts"):
        path = install / name
        _assert_no_reparse_chain(path)
        path.mkdir(exist_ok=True)
    return install


def install_release(
    archive_path: Path | str,
    *,
    release_record_path: Path | str,
    checksum_path: Path | str,
    runtime_root: Path | str | None = None,
    baseline: bool = False,
) -> InstalledRelease:
    archive = Path(archive_path)
    release_record = _json_file(Path(release_record_path), "release record")
    if release_record.get("schema") != RELEASE_RECORD_SCHEMA:
        raise ReleaseManagerError("release record schema is unsupported")
    archive_info = release_record.get("archive")
    manifest_info = release_record.get("build_manifest")
    if not isinstance(archive_info, Mapping) or not isinstance(manifest_info, Mapping):
        raise ReleaseManagerError("release record is incomplete")
    expected_archive_hash = str(archive_info.get("sha256") or "")
    actual_archive_hash = sha256_file(archive)
    if expected_archive_hash != actual_archive_hash:
        raise ReleaseManagerError("release archive SHA256 does not match its record")
    _verify_checksum(Path(checksum_path), archive.name, actual_archive_hash)
    release_id = _safe_segment(release_record.get("release_id"), "release id")
    install = ensure_install_root(runtime_root)
    versions = install / "versions"
    destination = versions / release_id
    _assert_below(destination, versions, "release destination")
    if destination.exists():
        installed = _verify_installed_release(destination, release_record, baseline=baseline)
        _record_baseline(install, installed, archive_hash=actual_archive_hash)
        return installed
    staging = versions / f".{release_id}.install-{uuid4().hex}.tmp"
    try:
        bundle = _extract_release_archive(archive, staging)
        manifest_path = bundle / BUILD_MANIFEST_FILENAME
        if sha256_file(manifest_path) != str(manifest_info.get("sha256") or ""):
            raise ReleaseManagerError("extracted build manifest hash does not match release record")
        manifest = load_build_manifest(manifest_path)
        integrity = verify_build_manifest(bundle, manifest_path, strict=True)
        if not integrity.ok:
            raise ReleaseManagerError("extracted release failed strict build integrity: " + "; ".join(integrity.errors))
        _verify_native_audit(bundle)
        source = manifest.get("source") if isinstance(manifest.get("source"), Mapping) else {}
        source_commit = str(source.get("commit") or "") or None
        if baseline and source_commit != BASELINE_SOURCE_COMMIT:
            raise ReleaseManagerError("baseline release does not point to the frozen source commit")
        executable_info = manifest.get("executable")
        if not isinstance(executable_info, Mapping):
            raise ReleaseManagerError("build manifest executable entry is missing")
        executable_relative = _safe_relative(str(executable_info.get("path") or ""))
        executable = bundle.joinpath(*PurePosixPath(executable_relative).parts)
        if not executable.is_file() or sha256_file(executable) != executable_info.get("sha256"):
            raise ReleaseManagerError("release executable failed manifest verification")
        install_receipt = {
            "schema": "guandan.installed-release/1",
            "release_id": release_id,
            "build_id": manifest.get("build_id"),
            "source_commit": source_commit,
            "archive_sha256": actual_archive_hash,
            "release_record_sha256": sha256_file(Path(release_record_path)),
            "manifest_sha256": sha256_file(manifest_path),
            "executable_relative": executable.relative_to(staging).as_posix(),
            "executable_sha256": sha256_file(executable),
            "baseline": bool(baseline),
            "installed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "install_receipt.json", install_receipt)
        staging.rename(destination)
    except BaseException:
        _remove_staging(staging, versions)
        raise
    installed = _verify_installed_release(destination, release_record, baseline=baseline)
    _record_baseline(install, installed, archive_hash=actual_archive_hash)
    return installed


def register_legacy_baseline(
    portable_directory: Path | str,
    *,
    executable_sha256: str,
    runtime_root: Path | str | None = None,
) -> InstalledRelease:
    """Copy a known-good old portable directory without claiming reproducibility."""

    source = Path(portable_directory)
    if not source.is_dir():
        raise ReleaseManagerError("legacy baseline directory does not exist")
    assert_safe_tree(source)
    executable = source / "DaguandanAssistant.exe"
    if not executable.is_file() or sha256_file(executable) != executable_sha256:
        raise ReleaseManagerError("legacy baseline executable hash mismatch")
    release_id = f"legacy-baseline-{BASELINE_SOURCE_COMMIT[:12]}-{executable_sha256[:12]}"
    install = ensure_install_root(runtime_root)
    versions = install / "versions"
    destination = versions / release_id
    if not destination.exists():
        staging = versions / f".{release_id}.install-{uuid4().hex}.tmp"
        try:
            staging.mkdir()
            target = staging / "DaguandanAssistant"
            _copy_safe_tree(source, target)
            atomic_write_json(
                staging / "install_receipt.json",
                {
                    "schema": LEGACY_BASELINE_SCHEMA,
                    "release_id": release_id,
                    "build_id": "legacy-unidentified",
                    "source_commit": BASELINE_SOURCE_COMMIT,
                    "executable_relative": "DaguandanAssistant/DaguandanAssistant.exe",
                    "executable_sha256": executable_sha256,
                    "baseline": True,
                    "reproducible": False,
                },
            )
            staging.rename(destination)
        except BaseException:
            _remove_staging(staging, versions)
            raise
    installed = _load_installed_receipt(destination)
    _record_baseline(install, installed, archive_hash=None)
    return installed


def activate_release(
    release_id: str,
    *,
    runtime_root: Path | str | None = None,
    doctor_runner: Callable[[InstalledRelease, Path, Path], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    install = ensure_install_root(runtime_root)
    selected = _installed_by_id(install, release_id)
    runtime = install.parent
    doctor_path = install / "rollback-receipts" / f"doctor-{selected.release_id}-{uuid4().hex[:8]}.json"
    doctor = (
        dict(doctor_runner(selected, runtime, doctor_path))
        if doctor_runner is not None
        else _run_frozen_doctor(selected, runtime, doctor_path)
    )
    checks = doctor.get("checks") if isinstance(doctor.get("checks"), list) else []
    if doctor.get("schema") != "guandan.doctor/1" or any(
        isinstance(item, Mapping) and item.get("status") == "FAIL" for item in checks
    ):
        raise ReleaseManagerError("candidate doctor did not pass; active release was not changed")
    current = _read_active(install)
    document = {
        "schema": ACTIVE_RELEASE_SCHEMA,
        "release_id": selected.release_id,
        "build_id": selected.build_id,
        "version_directory": selected.version_root.name,
        "executable_relative": selected.executable.relative_to(selected.version_root).as_posix(),
        "data_generation": selected.build_id,
        "activated_at": datetime.now(UTC).isoformat(),
        "previous": (
            {
                "release_id": current.get("release_id"),
                "build_id": current.get("build_id"),
                "data_generation": current.get("data_generation"),
            }
            if current is not None
            else None
        ),
        "doctor_report": doctor_path.name,
    }
    atomic_write_json(install / _ACTIVE_FILE, document)
    _append_history(install, {**document, "operation": "activate"})
    return document


def rollback_release(
    *,
    runtime_root: Path | str | None = None,
    support_exporter: Callable[[InstalledRelease, Path], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    install = ensure_install_root(runtime_root)
    current = _read_active(install)
    if current is None:
        raise ReleaseManagerError("there is no active release to roll back")
    previous = current.get("previous")
    if not isinstance(previous, Mapping) or not previous.get("release_id"):
        raise ReleaseManagerError("active release has no verified previous release")
    bad = _installed_by_id(install, str(current.get("release_id")))
    target = _installed_by_id(install, str(previous.get("release_id")))
    receipt_id = f"rollback-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    receipt_root = install / "rollback-receipts" / receipt_id
    receipt_root.mkdir()
    support_path = receipt_root / "support.zip"
    support_result: dict[str, object]
    try:
        if support_exporter is not None:
            support_result = dict(support_exporter(bad, support_path))
        else:
            support_result = _export_bad_release_support(bad, support_path, install.parent)
    except BaseException as exc:
        support_result = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
    data_snapshot = _snapshot_data_pointer(install.parent, receipt_root)
    target_document = {
        "schema": ACTIVE_RELEASE_SCHEMA,
        "release_id": target.release_id,
        "build_id": target.build_id,
        "version_directory": target.version_root.name,
        "executable_relative": target.executable.relative_to(target.version_root).as_posix(),
        "data_generation": previous.get("data_generation") or target.build_id,
        "activated_at": datetime.now(UTC).isoformat(),
        "previous": {
            "release_id": bad.release_id,
            "build_id": bad.build_id,
            "data_generation": current.get("data_generation"),
        },
        "rollback_receipt": receipt_id,
    }
    receipt = {
        "schema": ROLLBACK_RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "created_at": datetime.now(UTC).isoformat(),
        "bad_release": bad.to_dict(),
        "target_release": target.to_dict(),
        "bad_directory_preserved": bad.version_root.is_dir(),
        "support": support_result,
        "data_snapshot": data_snapshot,
    }
    atomic_write_json(receipt_root / "receipt.json", receipt)
    atomic_write_json(install / _ACTIVE_FILE, target_document)
    _append_history(install, {**target_document, "operation": "rollback"})
    return target_document


def release_status(runtime_root: Path | str | None = None) -> dict[str, object]:
    install = ensure_install_root(runtime_root)
    versions = []
    for path in sorted((install / "versions").iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_dir() or path.name.startswith(".") or _is_link_or_reparse(path):
            continue
        try:
            versions.append(_load_installed_receipt(path).to_dict())
        except ReleaseManagerError:
            versions.append({"release_id": path.name, "status": "INVALID"})
    return {
        "schema": "guandan.release-status/1",
        "active": _read_active(install),
        "baseline": _read_optional_json(install / _BASELINE_FILE),
        "versions": versions,
    }


def _verify_installed_release(
    version_root: Path,
    release_record: Mapping[str, object],
    *,
    baseline: bool,
) -> InstalledRelease:
    assert_safe_tree(version_root)
    receipt = _json_file(version_root / "install_receipt.json", "install receipt")
    if receipt.get("release_id") != release_record.get("release_id"):
        raise ReleaseManagerError("existing installed release has a different identity")
    installed = _installed_from_receipt(version_root, receipt)
    bundle = installed.executable.parent
    manifest_path = bundle / BUILD_MANIFEST_FILENAME
    integrity = verify_build_manifest(bundle, manifest_path, strict=True)
    if not integrity.ok or integrity.build_id != installed.build_id:
        raise ReleaseManagerError("installed release failed strict integrity verification")
    _verify_native_audit(bundle)
    if baseline and installed.source_commit != BASELINE_SOURCE_COMMIT:
        raise ReleaseManagerError("installed baseline source identity is invalid")
    return installed


def _load_installed_receipt(version_root: Path) -> InstalledRelease:
    receipt = _json_file(version_root / "install_receipt.json", "install receipt")
    installed = _installed_from_receipt(version_root, receipt)
    if receipt.get("schema") == "guandan.installed-release/1":
        bundle = installed.executable.parent
        manifest_path = bundle / BUILD_MANIFEST_FILENAME
        integrity = verify_build_manifest(bundle, manifest_path, strict=True)
        if not integrity.ok or integrity.build_id != installed.build_id:
            raise ReleaseManagerError("installed release no longer passes strict integrity")
        _verify_native_audit(bundle)
    return installed


def _installed_from_receipt(
    version_root: Path,
    receipt: Mapping[str, object],
) -> InstalledRelease:
    release_id = _safe_segment(receipt.get("release_id"), "release id")
    relative = _safe_relative(str(receipt.get("executable_relative") or ""))
    executable = version_root.joinpath(*PurePosixPath(relative).parts)
    _assert_below(executable, version_root, "installed executable")
    if not executable.is_file() or _is_link_or_reparse(executable):
        raise ReleaseManagerError("installed executable is unavailable")
    expected_hash = str(receipt.get("executable_sha256") or "")
    if expected_hash and sha256_file(executable) != expected_hash:
        raise ReleaseManagerError("installed executable hash mismatch")
    return InstalledRelease(
        release_id=release_id,
        build_id=str(receipt.get("build_id") or "legacy-unidentified"),
        version_root=version_root,
        executable=executable,
        source_commit=str(receipt.get("source_commit") or "") or None,
        baseline=bool(receipt.get("baseline", False)),
    )


def _installed_by_id(install: Path, release_id: str) -> InstalledRelease:
    safe = _safe_segment(release_id, "release id")
    root = install / "versions" / safe
    _assert_below(root, install / "versions", "installed release")
    return _load_installed_receipt(root)


def _extract_release_archive(archive_path: Path, staging: Path) -> Path:
    identities: set[str] = set()
    files = 0
    total = 0
    roots: set[str] = set()
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_ARCHIVE_FILES:
                raise ReleaseManagerError("release archive contains too many entries")
            for info in infos:
                relative = _safe_relative(info.filename.rstrip("/"))
                identity = relative.casefold()
                if identity in identities:
                    raise ReleaseManagerError(f"archive contains a case-colliding entry: {relative}")
                identities.add(identity)
                roots.add(PurePosixPath(relative).parts[0])
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise ReleaseManagerError("archive links are not allowed")
                if info.flag_bits & 0x1:
                    raise ReleaseManagerError("encrypted archive entries are not allowed")
                if info.is_dir():
                    continue
                files += 1
                total += int(info.file_size)
                if files > _MAX_ARCHIVE_FILES or total > _MAX_ARCHIVE_BYTES:
                    raise ReleaseManagerError("release archive exceeds extraction limits")
                if info.file_size / max(1, info.compress_size) > 1000:
                    raise ReleaseManagerError("release archive compression ratio is unsafe")
                destination = staging.joinpath(*PurePosixPath(relative).parts)
                _assert_below(destination, staging, "archive extraction")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if _is_link_or_reparse(destination.parent):
                    raise ReleaseManagerError("archive extraction traversed a reparse point")
                with archive.open(info) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if destination.stat().st_size != info.file_size:
                    raise ReleaseManagerError("archive entry size changed while extracting")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseManagerError("release archive is unreadable") from exc
    if roots != {"DaguandanAssistant"}:
        raise ReleaseManagerError("release archive must contain one DaguandanAssistant root")
    bundle = staging / "DaguandanAssistant"
    assert_safe_tree(bundle)
    return bundle


def _verify_native_audit(bundle: Path) -> None:
    document = _json_file(bundle / "native_dependency_audit.json", "native dependency audit")
    if document.get("schema") != "guandan.native-dependency-audit/1" or document.get("status") != "PASS":
        raise ReleaseManagerError("native dependency audit is missing or did not pass")


def _record_baseline(
    install: Path,
    release: InstalledRelease,
    *,
    archive_hash: str | None,
) -> None:
    if not release.baseline:
        return
    document = {
        "schema": "guandan.baseline-receipt/1",
        "tag": BASELINE_TAG,
        "source_commit": BASELINE_SOURCE_COMMIT,
        "release": release.to_dict(),
        "archive_sha256": archive_hash,
    }
    path = install / _BASELINE_FILE
    if path.exists():
        if _json_file(path, "baseline receipt") != document:
            raise ReleaseManagerError("immutable baseline receipt cannot be overwritten")
        return
    atomic_write_json(path, document)


def _run_frozen_doctor(
    release: InstalledRelease,
    runtime_root: Path,
    output_path: Path,
) -> dict[str, object]:
    environment = dict(os.environ)
    environment["DAGUANDAN_DATA_ROOT"] = str(runtime_root)
    completed = subprocess.run(
        [str(release.executable), "--doctor", "--doctor-output", str(output_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=180,
        check=False,
        text=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise ReleaseManagerError("frozen doctor failed before activation")
    return _json_file(output_path, "doctor report")


def _export_bad_release_support(
    release: InstalledRelease,
    destination: Path,
    runtime_root: Path,
) -> dict[str, object]:
    environment = dict(os.environ)
    environment["DAGUANDAN_DATA_ROOT"] = str(runtime_root)
    completed = subprocess.run(
        [str(release.executable), "--export-support", str(destination)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=180,
        check=False,
        text=False,
    )
    return {
        "status": "PASS" if completed.returncode == 0 and destination.is_file() else "FAILED",
        "exit_code": int(completed.returncode),
        "path": destination.name if destination.is_file() else None,
        "sha256": sha256_file(destination) if destination.is_file() else None,
    }


def _snapshot_data_pointer(runtime_root: Path, receipt_root: Path) -> dict[str, object]:
    active = runtime_root / "data" / "v1" / "active.json"
    document = _read_optional_json(active)
    generation_marker: Path | None = None
    if isinstance(document, Mapping) and document.get("generation_id"):
        try:
            generation = _safe_segment(document.get("generation_id"), "generation id")
            generation_marker = (
                runtime_root
                / "data"
                / "v1"
                / "generations"
                / generation
                / "runtime_layout.json"
            )
        except ReleaseManagerError:
            generation_marker = None
    snapshot = {
        "schema": "guandan.data-pointer-snapshot/1",
        "active_pointer_present": document is not None,
        "active_pointer": document,
        "active_pointer_sha256": sha256_file(active) if active.is_file() else None,
        "generation_marker_present": bool(
            generation_marker is not None and generation_marker.is_file()
        ),
        "generation_marker_sha256": (
            sha256_file(generation_marker)
            if generation_marker is not None and generation_marker.is_file()
            else None
        ),
        "data_preserved_in_place": True,
    }
    atomic_write_json(receipt_root / "data_snapshot.json", snapshot)
    return snapshot


def _read_active(install: Path) -> dict[str, object] | None:
    path = install / _ACTIVE_FILE
    value = _read_optional_json(path)
    if value is None:
        return None
    if value.get("schema") != ACTIVE_RELEASE_SCHEMA:
        raise ReleaseManagerError("active release pointer schema is invalid")
    _installed_by_id(install, str(value.get("release_id")))
    return value


def _append_history(install: Path, value: Mapping[str, object]) -> None:
    path = install / _HISTORY_FILE
    payload = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_safe_tree(source: Path, destination: Path) -> None:
    assert_safe_tree(source)
    destination.mkdir()
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix().casefold()):
        relative = path.relative_to(source)
        target = destination / relative
        _assert_below(target, destination, "legacy copy")
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _remove_staging(path: Path, versions_root: Path) -> None:
    if not path.exists():
        return
    _assert_below(path, versions_root, "staging cleanup")
    if not path.name.startswith(".") or not path.name.endswith(".tmp"):
        raise ReleaseManagerError("refusing to remove an unrecognized staging path")
    assert_safe_tree(path)
    shutil.rmtree(path)


def _ensure_runtime_marker(runtime: Path) -> None:
    marker = runtime / RUNTIME_ROOT_MARKER
    if marker.exists():
        value = _json_file(marker, "runtime marker")
        if value.get("schema") != RUNTIME_ROOT_SCHEMA or value.get("application") != APP_DIRECTORY_NAME:
            raise ReleaseManagerError("runtime root marker is invalid")
        return
    existing = list(runtime.iterdir())
    if existing:
        raise ReleaseManagerError("refusing to adopt a non-empty unmarked runtime root")
    atomic_write_json(
        marker,
        {
            "schema": RUNTIME_ROOT_SCHEMA,
            "application": APP_DIRECTORY_NAME,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _verify_checksum(path: Path, archive_name: str, expected_hash: str) -> None:
    try:
        parts = path.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as exc:
        raise ReleaseManagerError("archive checksum sidecar is unreadable") from exc
    if not parts or parts[0].casefold() != expected_hash.casefold():
        raise ReleaseManagerError("archive checksum sidecar hash mismatch")
    if len(parts) > 1 and Path(parts[-1]).name.casefold() != archive_name.casefold():
        raise ReleaseManagerError("archive checksum sidecar names a different archive")


def _json_file(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseManagerError(f"{label} is unreadable or invalid") from exc
    if not isinstance(value, dict):
        raise ReleaseManagerError(f"{label} must be a JSON object")
    return value


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return _json_file(path, path.name)


def _safe_segment(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 128 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in text):
        raise ReleaseManagerError(f"{field} is unsafe")
    return text


def _safe_relative(raw: str) -> str:
    if not raw or "\\" in raw or ":" in raw or raw.startswith("/"):
        raise ReleaseManagerError("archive path is unsafe")
    path = PurePosixPath(raw)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseManagerError("archive path is unsafe")
    return path.as_posix()


def _assert_below(path: Path, root: Path, field: str) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ReleaseManagerError(f"{field} escaped its managed root") from exc


def _reject_filesystem_root(path: Path, field: str) -> None:
    resolved = Path(os.path.abspath(os.fspath(path)))
    anchor = Path(resolved.anchor) if resolved.anchor else None
    if anchor is None or resolved == anchor:
        raise ReleaseManagerError(f"{field} must not be a filesystem root")


def _assert_no_reparse_chain(path: Path) -> None:
    current = Path(os.path.abspath(os.fspath(path)))
    while True:
        if current.exists() and _is_link_or_reparse(current):
            raise ReleaseManagerError("managed path traverses a reparse point")
        if current.parent == current:
            return
        current = current.parent


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if callable(is_junction) and is_junction(path):
            return True
        return bool(
            getattr(path.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError:
        return False


__all__ = [
    "ACTIVE_RELEASE_SCHEMA",
    "BASELINE_SOURCE_COMMIT",
    "BASELINE_TAG",
    "InstalledRelease",
    "ReleaseManagerError",
    "activate_release",
    "default_runtime_root",
    "ensure_install_root",
    "install_release",
    "register_legacy_baseline",
    "release_status",
    "rollback_release",
]
