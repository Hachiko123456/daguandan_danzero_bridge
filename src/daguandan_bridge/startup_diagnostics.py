"""Early, standard-library-only process diagnostics.

This module is safe to import before Qt, OpenCV, DPI setup, or any application
composition code.  Startup diagnostics are deliberately fail-open: an
unwritable diagnostics directory must never prevent the assistant from
starting.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import traceback
from types import TracebackType
from typing import IO, Mapping
from uuid import uuid4

from .bounded_log import RotatingTextLog, append_bounded_event
from .runtime_layout import (
    RuntimeLayout,
    atomic_write_json,
    resolve_application_root,
    resolve_log_diagnostics_root,
    resolve_runtime_layout,
    _assert_no_reparse_chain,
)


DIAGNOSTICS_ROOT_ENV = "DAGUANDAN_DIAGNOSTICS_ROOT"
_DIAGNOSTICS_ROOT_ENV_ALIAS = "DAGUANDAN_DIAGNOSTICS_DIR"
DATA_ROOT_ENV = "DAGUANDAN_DATA_ROOT"
_RUN_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_PROCESS_RUN_ID = (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    + "-"
    + uuid4().hex[:12]
)


@dataclass(frozen=True)
class DiagnosticsRoot:
    path: Path
    source: str


@dataclass(frozen=True)
class StartupDiagnosticsState:
    run_id: str
    root: Path | None
    root_source: str
    run_directory: Path | None
    enabled: bool
    error: str | None = None


_INITIALIZE_LOCK = threading.RLock()
_EVENT_WRITE_LOCK = threading.Lock()
_EXCEPTION_WRITE_LOCK = threading.Lock()
_STATE: StartupDiagnosticsState | None = None
_EXCEPTION_HANDLE: IO[str] | None = None
_FAULT_HANDLE: IO[str] | None = None
_CONSOLE_HANDLE: IO[str] | None = None
_PREVIOUS_SYS_EXCEPTHOOK = sys.excepthook
_PREVIOUS_THREAD_EXCEPTHOOK = getattr(threading, "excepthook", None)
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_STDIO_REDIRECTED = False


def get_process_run_id() -> str:
    """Return one identifier that remains stable for the entire process."""

    return _PROCESS_RUN_ID


def resolve_diagnostics_root(
    *,
    environ: Mapping[str, str] | None = None,
    frozen: bool | None = None,
    bundle_root: Path | str | None = None,
    executable_path: Path | str | None = None,
) -> DiagnosticsRoot:
    """Use the same explicit overrides and diagnostics default as runtime layout."""

    app = resolve_application_root(
        frozen=frozen, bundle_root=bundle_root, executable_path=executable_path
    )
    path, source = resolve_log_diagnostics_root(
        environ=environ, frozen=frozen, application_root=app
    )
    return DiagnosticsRoot(path, source)


def initialize_startup_diagnostics(
    *,
    root: Path | str | None = None,
    run_id: str | None = None,
) -> StartupDiagnosticsState:
    """Install process-wide logging and exception hooks once.

    All setup errors are captured in the returned state.  Callers should keep
    launching the application even when ``enabled`` is false.
    """

    global _STATE, _EXCEPTION_HANDLE, _FAULT_HANDLE, _CONSOLE_HANDLE, _STDIO_REDIRECTED
    with _INITIALIZE_LOCK:
        if _STATE is not None:
            return _STATE

        selected: DiagnosticsRoot | None = None
        safe_run_id = _safe_run_id(run_id or get_process_run_id())
        try:
            # Even the explicit argument must not bypass the frozen resource
            # boundary or become a cwd-relative destination.
            selected = resolve_diagnostics_root(
                environ={DIAGNOSTICS_ROOT_ENV: str(root)} if root is not None else None
            )
            if root is not None:
                selected = DiagnosticsRoot(selected.path, "explicit")
            run_directory = selected.path / "runs" / safe_run_id
            _assert_no_reparse_chain(run_directory)
            run_directory.mkdir(parents=True, exist_ok=True)
            _assert_no_reparse_chain(run_directory)
            exception_path = run_directory / "exceptions.log"
            fault_path = run_directory / "faulthandler.log"
            console_path = run_directory / "startup.log"

            _EXCEPTION_HANDLE = RotatingTextLog(exception_path)
            _assert_no_reparse_chain(fault_path)
            _FAULT_HANDLE = fault_path.open(
                "a", encoding="utf-8", buffering=1, newline="\n"
            )
            _CONSOLE_HANDLE = RotatingTextLog(console_path)
            if sys.stdout is None or sys.stderr is None:
                _STDIO_REDIRECTED = True
                if sys.stdout is None:
                    sys.stdout = _CONSOLE_HANDLE
                if sys.stderr is None:
                    sys.stderr = _CONSOLE_HANDLE

            sys.excepthook = _main_exception_hook
            if hasattr(threading, "excepthook"):
                threading.excepthook = _thread_exception_hook
            faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)

            _STATE = StartupDiagnosticsState(
                run_id=safe_run_id,
                root=selected.path,
                root_source=selected.source,
                run_directory=run_directory,
                enabled=True,
            )
            _CONSOLE_HANDLE.write(
                "[startup] diagnostics initialized "
                f"run_id={safe_run_id} root_source={selected.source} root={selected.path}\n"
            )
            _append_event(
                "startup_diagnostics_initialized",
                {
                    "run_id": safe_run_id,
                    "root_source": selected.source,
                    "diagnostics_root": str(selected.path),
                    "stdout_redirected": _STDIO_REDIRECTED,
                },
            )
            write_startup_report()
            atexit.register(_flush_handles)
            return _STATE
        except BaseException as exc:
            _close_failed_handles()
            _STATE = StartupDiagnosticsState(
                run_id=safe_run_id,
                root=selected.path if selected is not None else None,
                root_source=selected.source if selected is not None else "invalid_configuration",
                run_directory=None,
                enabled=False,
                error=(
                    f"diagnostics unavailable at {selected.path if selected else root or 'configured root'}: "
                    f"{type(exc).__name__}: {exc}; no fallback directory was selected"
                ),
            )
            _report_failure(_STATE.error)
            return _STATE


_CAPTURE_FIELDS = (
    "capture_backend", "allow_screen_fallback", "viewport_mode",
    "viewport_aspect_ratio", "base_size", "target_client_size",
    "detect_black_bars", "allow_resize",
)
_CONFIG_FILES = ("profile.json", "regions_config.json", "templates_config.json")
_REPORT_LOCK = threading.RLock()


def _bounded_document(path: Path, *, max_bytes: int) -> tuple[dict, str]:
    _assert_no_reparse_chain(path)
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"diagnostic input exceeds {max_bytes} bytes: {path}")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"diagnostic input is not a JSON object: {path}")
    return document, hashlib.sha256(raw).hexdigest()


def build_startup_report(
    *,
    layout: RuntimeLayout | None = None,
    profiles_root: Path | str | None = None,
    profile_name: str | None = None,
    profile_config: object | None = None,
    resource_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build local (NOT support-sanitized) startup/selected-profile evidence.

    The controller can pass the actual loaded config and a cached recognition
    identity after selection. No GUI/native imports, Git subprocesses, tree
    walks or model/template hashing occur here. Without that context, selection
    remains explicitly unknown rather than claiming the bundled default is active.
    Call at startup/profile changes, not for every frame.
    """

    report: dict[str, object] = {
        "schema": "guandan.startup-report/1",
        "run_id": _STATE.run_id if _STATE else get_process_run_id(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "privacy": "local_paths_not_support_sanitized",
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "version": platform.version(), "machine": platform.machine(),
        },
        "configured_overrides": {
            key: os.environ[key] for key in (
                DATA_ROOT_ENV, DIAGNOSTICS_ROOT_ENV, _DIAGNOSTICS_ROOT_ENV_ALIAS,
            ) if os.environ.get(key)
        },
        "errors": [],
    }
    errors = report["errors"]
    state = _STATE
    report["diagnostics"] = {
        "root": str(state.root) if state and state.root else None,
        "root_source": state.root_source if state else None,
        "run_directory": str(state.run_directory) if state and state.run_directory else None,
        "enabled": state.enabled if state else False,
        "error": state.error if state else None,
    }
    try:
        selected = layout or resolve_runtime_layout()
    except Exception as exc:
        errors.append(f"runtime_layout: {type(exc).__name__}: {exc}")
        return report
    report.update({
        "build_id": selected.build_id,
        "frozen": selected.frozen,
        "manifest_status": selected.manifest_status,
        "generation_id": selected.generation_id,
        "runtime_root_source": selected.runtime_root_source,
        "paths": {key: str(getattr(selected, key)) for key in (
            "bundle_root", "resource_data_dir", "runtime_root", "app_data_root",
            "generation_root", "data_dir", "profiles_root", "logs_root",
            "diagnostics_root", "preferences_root", "cache_root",
        )},
        "active_data_root": str(selected.data_dir),
        "diagnostics_root": str(state.root if state and state.root else selected.diagnostics_root),
    })
    try:
        manifest, digest = _bounded_document(
            selected.bundle_root / "build_manifest.json", max_bytes=8 * 1024 * 1024
        )
        if manifest.get("schema") != "guandan.build-manifest/1":
            raise ValueError("unsupported build manifest schema")
        source = manifest.get("source", {})
        resources = manifest.get("resources", {})
        report["build_manifest"] = {
            "sha256": digest, "build_id": manifest.get("build_id"),
            "verification": "metadata_only_not_integrity_verification",
            "source": {key: source.get(key) for key in ("commit", "tree", "dirty", "status_sha256")},
            "bundled_resource_fingerprints": {
                key: {field: value.get(field) for field in ("sha256", "file_count", "bytes")}
                for key, value in resources.items() if isinstance(value, dict)
            },
        }
    except FileNotFoundError:
        report["build_manifest"] = {"status": "unavailable"}
    except Exception as exc:
        errors.append(f"build_manifest: {type(exc).__name__}: {exc}")
    # Identity of the actually imported diagnostics module, not a build-machine
    # absolute path or a claimed full-worktree fingerprint.
    module = Path(__file__)
    report["source"] = {"module_file": str(module), "fingerprint_scope": "startup_diagnostics_module_only"}
    if not selected.frozen:
        try:
            with module.open("rb") as handle:
                raw = handle.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("module exceeds diagnostic byte budget")
            report["source"]["sha256"] = hashlib.sha256(raw).hexdigest()
        except Exception as exc:
            errors.append(f"source: {type(exc).__name__}: {exc}")
    profile: dict[str, object] = {"name": profile_name, "root": None, "status": "not_selected"}
    report["active_profile"] = profile
    report["capture"] = {key: None for key in _CAPTURE_FIELDS}
    if profile_name is not None:
        if (not profile_name or profile_name in {".", ".."}
                or any(char in profile_name for char in '/\\:')):
            raise ValueError("profile_name must be a single directory name")
        active_root = Path(profiles_root) if profiles_root is not None else selected.profiles_root
        if not active_root.is_absolute():
            raise ValueError("profiles_root must be absolute")
        profile_root = active_root / profile_name
        profile.update({
            "root": str(profile_root), "profiles_root": str(active_root), "status": "selected",
            "root_source": "caller" if profiles_root is not None else "runtime_layout",
        })
        report["paths"]["profiles_root"] = str(active_root)
        digests: dict[str, str] = {}
        config = profile_config
        for filename in _CONFIG_FILES:
            try:
                document, digest = _bounded_document(profile_root / filename, max_bytes=1024 * 1024)
                digests[filename] = digest
                if filename == "profile.json" and config is None:
                    config = document
            except Exception as exc:
                errors.append(f"{filename}: {type(exc).__name__}: {exc}")
        profile["config_fingerprint"] = {
            "scope": "profile_config_only_excludes_templates_and_models",
            "status": "complete" if len(digests) == len(_CONFIG_FILES) else "partial",
            "files": digests,
            "sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
        }
        report["capture"] = {
            key: config.get(key) if isinstance(config, Mapping) else getattr(config, key, None)
            for key in _CAPTURE_FIELDS
        }
        report["capture_config_source"] = "loaded_config" if profile_config is not None else "profile_json"
    if resource_identity is not None:
        report["active_resource_identity"] = {
            "source": "caller_supplied_cached_identity",
            **{key: resource_identity.get(key) for key in ("status", "algorithm", "profile_name", "sha256")},
        }
    return report


def write_startup_report(**context: object) -> Path | None:
    """Atomically refresh local evidence using build_startup_report arguments.

    Requires initialized diagnostics. Returns None with an explicit event and
    stderr reason on failure, without changing the selected destination.
    """

    state = _STATE
    if state is None or not state.enabled or state.run_directory is None:
        return None
    destination = state.run_directory / "startup_report.json"
    try:
        with _REPORT_LOCK:
            atomic_write_json(destination, build_startup_report(**context))
        return destination
    except Exception as exc:
        cause = exc.__cause__ or exc
        reason = f"startup report unavailable at {destination}: {type(cause).__name__}: {cause}"
        _append_event("startup_report_failed", {"reason": reason})
        _report_failure(reason)
        return None


def _report_failure(reason: str) -> None:
    # Windowless callers also receive the explicit reason in state.error. UI
    # presentation remains the composition/controller's responsibility.
    for stream in (sys.stderr, _ORIGINAL_STDERR, sys.__stderr__):
        if stream is not None:
            try:
                stream.write(f"[startup] {reason}\n")
                stream.flush()
                return
            except Exception:
                continue


def current_startup_diagnostics() -> StartupDiagnosticsState:
    """Return initialized state, initializing lazily when necessary."""

    return initialize_startup_diagnostics()


def initialized_startup_diagnostics() -> StartupDiagnosticsState | None:
    """Return current state without creating directories or installing hooks."""

    return _STATE


def record_startup_event(event: str, evidence: Mapping[str, object] | None = None) -> None:
    """Append a best-effort structured startup event."""

    try:
        if _STATE is None:
            initialize_startup_diagnostics()
        _append_event(str(event), evidence or {})
    except BaseException:
        return


def _safe_run_id(value: str) -> str:
    cleaned = _RUN_ID_PATTERN.sub("-", str(value).strip()).strip(".-")
    return cleaned[:128] or get_process_run_id()


def _append_event(event: str, evidence: Mapping[str, object]) -> None:
    state = _STATE
    if state is None or state.run_directory is None:
        return
    payload = {
        "schema": "guandan.startup-event/1",
        "event": event,
        "run_id": state.run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "evidence": dict(evidence),
    }
    try:
        with _EVENT_WRITE_LOCK:
            append_bounded_event(state.run_directory / "startup.jsonl", payload)
    except BaseException:
        return


def _write_exception(
    source: str,
    exception_type: type[BaseException],
    exception: BaseException,
    tb: TracebackType | None,
    *,
    thread_name: str | None = None,
) -> None:
    try:
        _append_event(
            "uncaught_exception",
            {
                "source": source,
                "exception_type": exception_type.__name__,
                "thread_name": thread_name,
            },
        )
        if _EXCEPTION_HANDLE is not None:
            with _EXCEPTION_WRITE_LOCK:
                timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                _EXCEPTION_HANDLE.write(
                    f"[{timestamp}] source={source} thread={thread_name or '-'}\n"
                )
                traceback.print_exception(
                    exception_type,
                    exception,
                    tb,
                    file=_EXCEPTION_HANDLE,
                )
                _EXCEPTION_HANDLE.flush()
    except BaseException:
        return


def _main_exception_hook(
    exception_type: type[BaseException],
    exception: BaseException,
    tb: TracebackType | None,
) -> None:
    _write_exception("main_thread", exception_type, exception, tb)
    previous = _PREVIOUS_SYS_EXCEPTHOOK
    if previous is not _main_exception_hook:
        try:
            previous(exception_type, exception, tb)
        except BaseException:
            return


def _thread_exception_hook(args: object) -> None:
    exception_type = getattr(args, "exc_type", Exception)
    exception = getattr(args, "exc_value", Exception("unknown thread exception"))
    tb = getattr(args, "exc_traceback", None)
    thread = getattr(args, "thread", None)
    _write_exception(
        "worker_thread",
        exception_type,
        exception,
        tb,
        thread_name=getattr(thread, "name", None),
    )
    previous = _PREVIOUS_THREAD_EXCEPTHOOK
    if previous is not None and previous is not _thread_exception_hook:
        try:
            previous(args)  # type: ignore[arg-type]
        except BaseException:
            return


def _flush_handles() -> None:
    for handle in (_EXCEPTION_HANDLE, _FAULT_HANDLE, _CONSOLE_HANDLE):
        try:
            if handle is not None:
                handle.flush()
        except BaseException:
            continue


def _close_failed_handles() -> None:
    global _EXCEPTION_HANDLE, _FAULT_HANDLE, _CONSOLE_HANDLE, _STDIO_REDIRECTED
    if sys.stdout is _CONSOLE_HANDLE:
        sys.stdout = _ORIGINAL_STDOUT
    if sys.stderr is _CONSOLE_HANDLE:
        sys.stderr = _ORIGINAL_STDERR
    sys.excepthook = _PREVIOUS_SYS_EXCEPTHOOK
    if _PREVIOUS_THREAD_EXCEPTHOOK is not None and hasattr(threading, "excepthook"):
        threading.excepthook = _PREVIOUS_THREAD_EXCEPTHOOK
    for handle in (_EXCEPTION_HANDLE, _FAULT_HANDLE, _CONSOLE_HANDLE):
        try:
            if handle is not None:
                handle.close()
        except BaseException:
            continue
    _EXCEPTION_HANDLE = None
    _FAULT_HANDLE = None
    _CONSOLE_HANDLE = None
    _STDIO_REDIRECTED = False
