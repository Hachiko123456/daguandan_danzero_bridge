"""Process-stable, privacy-safe runtime and build identity."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from threading import RLock
from typing import Mapping
from uuid import uuid4

from .startup_diagnostics import (
    get_process_run_id,
    initialized_startup_diagnostics,
    record_startup_event,
    resolve_diagnostics_root,
)


BUILD_MANIFEST_ENV = "DAGUANDAN_BUILD_MANIFEST"
BUILD_MANIFEST_FILENAME = "build_manifest.json"
BUILD_MANIFEST_SCHEMA = "guandan.build-manifest/1"
RUNTIME_IDENTITY_SCHEMA = "guandan.runtime-identity/1"

_IDENTITY_LOCK = RLock()
_PROCESS_IDENTITY: dict[str, object] | None = None
_SAFE_BUILD_FIELDS = (
    "schema",
    "build_id",
    "version",
    "created_at_utc",
    "git_commit",
    "git_dirty",
    "platform",
    "artifact_sha256",
    "resource_manifest_sha256",
    "implementation_fingerprint",
)


def application_root(
    *,
    executable_path: Path | str | None = None,
    frozen: bool | None = None,
) -> Path:
    """Return the bundle root in frozen mode and repository root in source mode."""

    is_frozen = getattr(sys, "frozen", False) if frozen is None else bool(frozen)
    if is_frozen:
        return Path(executable_path or sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def default_build_manifest_path(
    *,
    executable_path: Path | str | None = None,
    frozen: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    override = str(values.get(BUILD_MANIFEST_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return application_root(executable_path=executable_path, frozen=frozen) / BUILD_MANIFEST_FILENAME


def build_runtime_identity(
    *,
    build_manifest_path: Path | str | None = None,
    executable_path: Path | str | None = None,
    frozen: bool | None = None,
    run_id: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a JSON-safe identity without exposing usernames or absolute paths."""

    values = os.environ if environ is None else environ
    executable = Path(executable_path or sys.executable)
    is_frozen = getattr(sys, "frozen", False) if frozen is None else bool(frozen)
    manifest_path = (
        Path(build_manifest_path)
        if build_manifest_path is not None
        else default_build_manifest_path(
            executable_path=executable,
            frozen=is_frozen,
            environ=values,
        )
    )
    build = _read_build_identity(manifest_path)
    initialized_diagnostics = (
        initialized_startup_diagnostics() if environ is None else None
    )
    diagnostics_source = (
        initialized_diagnostics.root_source
        if initialized_diagnostics is not None
        else resolve_diagnostics_root(environ=values, frozen=is_frozen).source
    )
    fingerprint = str(build.get("implementation_fingerprint") or "").strip()
    if not fingerprint:
        fingerprint = _module_fingerprint()

    return {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "run_id": str(run_id or get_process_run_id()),
        "build_status": build["status"],
        "build_id": build["build_id"],
        "build": build,
        # Compatibility keys consumed by existing session/replay tooling.
        "implementation_fingerprint": fingerprint,
        "executable_path": executable.name or "python",
        "frozen": is_frozen,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "architecture": platform.machine() or "unknown",
        },
        "diagnostics": {
            "root_source": diagnostics_source,
            "run_directory": f"runs/{str(run_id or get_process_run_id())}",
        },
    }


def get_runtime_identity() -> dict[str, object]:
    """Return a defensive copy of the one identity created for this process."""

    global _PROCESS_IDENTITY
    with _IDENTITY_LOCK:
        if _PROCESS_IDENTITY is None:
            _PROCESS_IDENTITY = build_runtime_identity()
        return deepcopy(_PROCESS_IDENTITY)


def write_runtime_identity_snapshot(
    run_directory: Path | str | None = None,
) -> Path | None:
    """Atomically persist the sanitized process identity, without blocking startup."""

    temporary: Path | None = None
    try:
        state = initialized_startup_diagnostics()
        target_directory = (
            Path(run_directory)
            if run_directory is not None
            else Path(state.run_directory)
            if state is not None and state.run_directory is not None
            else None
        )
        if target_directory is None:
            raise OSError("startup diagnostics run directory is unavailable")
        target_directory.mkdir(parents=True, exist_ok=True)
        destination = target_directory / "runtime_identity.json"
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.tmp"
        )
        payload = get_runtime_identity()
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        temporary = None
        record_startup_event(
            "runtime_identity_snapshot_written",
            {
                "run_id": payload.get("run_id"),
                "output_file": destination.name,
            },
        )
        return destination
    except BaseException as exc:
        record_startup_event(
            "runtime_identity_snapshot_failed",
            {"error_type": type(exc).__name__},
        )
        return None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _read_build_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "status": "unidentified",
            "build_id": "unidentified",
            "reason": "build_manifest_missing",
            "manifest_file": BUILD_MANIFEST_FILENAME,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "status": "invalid",
            "build_id": "invalid",
            "reason": "build_manifest_unreadable_or_invalid_json",
            "manifest_file": BUILD_MANIFEST_FILENAME,
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid",
            "build_id": "invalid",
            "reason": "build_manifest_root_not_object",
            "manifest_file": BUILD_MANIFEST_FILENAME,
        }

    if payload.get("schema") != BUILD_MANIFEST_SCHEMA:
        return {
            "status": "invalid",
            "build_id": "invalid",
            "reason": "build_manifest_schema_invalid",
            "manifest_file": BUILD_MANIFEST_FILENAME,
        }

    source = payload.get("source")
    safe_source: dict[str, object] = {}
    if isinstance(source, dict):
        for key in ("commit", "tree", "dirty", "status_sha256"):
            value = source.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_source[key] = value

    executable = payload.get("executable")
    safe_executable: dict[str, object] = {}
    if isinstance(executable, dict):
        for key in ("path", "bytes", "sha256"):
            value = executable.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_executable[key] = (
                    str(value).replace("\\", "/").rsplit("/", 1)[-1]
                    if key == "path" and isinstance(value, str)
                    else value
                )

    safe_manifest: dict[str, object] = {}
    for key in _SAFE_BUILD_FIELDS:
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_manifest[key] = value
    if safe_source:
        safe_manifest["source"] = safe_source
    if safe_executable:
        safe_manifest["executable"] = safe_executable

    raw_build_id = payload.get("build_id")
    build_id = raw_build_id.strip() if isinstance(raw_build_id, str) else ""
    source_commit = safe_source.get("commit")
    if not isinstance(source_commit, str) or not source_commit.strip():
        return {
            "status": "invalid",
            "build_id": "invalid",
            "reason": "build_manifest_missing_source_commit",
            "manifest_file": BUILD_MANIFEST_FILENAME,
            "manifest": safe_manifest,
        }
    if not build_id:
        return {
            "status": "invalid",
            "build_id": "invalid",
            "reason": "build_manifest_missing_build_id",
            "manifest_file": BUILD_MANIFEST_FILENAME,
            "manifest": safe_manifest,
        }
    return {
        "status": "identified",
        "build_id": build_id,
        "source_commit": source_commit.strip(),
        "reason": "build_manifest_loaded",
        "manifest_file": BUILD_MANIFEST_FILENAME,
        "manifest": safe_manifest,
        "implementation_fingerprint": str(
            payload.get("implementation_fingerprint")
            or payload.get("artifact_sha256")
            or safe_executable.get("sha256")
            or build_id
        ),
    }


def _module_fingerprint() -> str:
    for candidate in (
        Path(__file__).with_name("live") / "orchestrator.py",
        Path(__file__),
    ):
        try:
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            continue
    return "unavailable"
