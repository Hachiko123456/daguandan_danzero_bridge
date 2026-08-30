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
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import traceback
from types import TracebackType
from typing import IO, Mapping
from uuid import uuid4


DIAGNOSTICS_ROOT_ENV = "DAGUANDAN_DIAGNOSTICS_ROOT"
_DIAGNOSTICS_ROOT_ENV_ALIAS = "DAGUANDAN_DIAGNOSTICS_DIR"
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
) -> DiagnosticsRoot:
    """Select a writable-data location without depending on project modules."""

    values = os.environ if environ is None else environ
    override = str(
        values.get(DIAGNOSTICS_ROOT_ENV)
        or values.get(_DIAGNOSTICS_ROOT_ENV_ALIAS)
        or ""
    ).strip()
    if override:
        return DiagnosticsRoot(Path(override).expanduser(), "environment")

    local_app_data = str(values.get("LOCALAPPDATA") or "").strip()
    if local_app_data:
        return DiagnosticsRoot(
            Path(local_app_data) / "DaguandanAssistant" / "diagnostics",
            "local_app_data_frozen"
            if (getattr(sys, "frozen", False) if frozen is None else frozen)
            else "local_app_data_source",
        )

    temp_root = str(values.get("TEMP") or values.get("TMP") or "").strip()
    base = Path(temp_root) if temp_root else Path(tempfile.gettempdir())
    return DiagnosticsRoot(base / "DaguandanAssistant" / "diagnostics", "temporary")


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

        selected = (
            DiagnosticsRoot(Path(root), "explicit")
            if root is not None
            else resolve_diagnostics_root()
        )
        safe_run_id = _safe_run_id(run_id or get_process_run_id())
        run_directory = selected.path / "runs" / safe_run_id
        try:
            try:
                run_directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                if not selected.source.startswith("local_app_data"):
                    raise
                selected = DiagnosticsRoot(
                    Path(tempfile.gettempdir())
                    / "DaguandanAssistant"
                    / "diagnostics",
                    "temporary_fallback",
                )
                run_directory = selected.path / "runs" / safe_run_id
                run_directory.mkdir(parents=True, exist_ok=True)
            exception_path = run_directory / "exceptions.log"
            fault_path = run_directory / "faulthandler.log"
            console_path = run_directory / "startup.log"

            _EXCEPTION_HANDLE = exception_path.open(
                "a", encoding="utf-8", buffering=1, newline="\n"
            )
            _FAULT_HANDLE = fault_path.open(
                "a", encoding="utf-8", buffering=1, newline="\n"
            )
            _CONSOLE_HANDLE = console_path.open(
                "a", encoding="utf-8", buffering=1, newline="\n"
            )
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
                f"run_id={safe_run_id} root_source={selected.source}\n"
            )
            _append_event(
                "startup_diagnostics_initialized",
                {
                    "run_id": safe_run_id,
                    "root_source": selected.source,
                    "stdout_redirected": _STDIO_REDIRECTED,
                },
            )
            atexit.register(_flush_handles)
            return _STATE
        except BaseException as exc:
            _close_failed_handles()
            _STATE = StartupDiagnosticsState(
                run_id=safe_run_id,
                root=selected.path,
                root_source=selected.source,
                run_directory=None,
                enabled=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return _STATE


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
            with (state.run_directory / "startup.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
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
