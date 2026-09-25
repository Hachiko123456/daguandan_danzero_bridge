from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from PySide6.QtGui import QIcon


APPLICATION_ICON_FILENAME = "app.ico"
APPLICATION_APP_USER_MODEL_ID = "DaguandanAssistant.App"


@dataclass(frozen=True)
class ApplicationIconCandidate:
    path: Path
    source: str


@dataclass(frozen=True)
class ApplicationIconAttempt:
    path: Path
    source: str
    exists: bool
    is_null: bool | None
    available_sizes: tuple[tuple[int, int], ...]
    error: str | None = None


@dataclass(frozen=True)
class ApplicationIconResult:
    icon: QIcon | None
    path: Path | None
    source: str | None
    available_sizes: tuple[tuple[int, int], ...]
    attempts: tuple[ApplicationIconAttempt, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.icon is not None and self.path is not None

    def to_diagnostic(self) -> dict[str, object]:
        return {
            "resource": APPLICATION_ICON_FILENAME,
            "path": str(self.path) if self.path is not None else None,
            "source": self.source,
            "available_sizes": [list(size) for size in self.available_sizes],
            "error": self.error,
            "attempts": [
                {
                    "path": str(attempt.path),
                    "source": attempt.source,
                    "exists": attempt.exists,
                    "is_null": attempt.is_null,
                    "available_sizes": [list(size) for size in attempt.available_sizes],
                    "error": attempt.error,
                }
                for attempt in self.attempts
            ],
        }


def _runtime_module() -> ModuleType:
    return sys.modules[__name__]


def _source_icon_path(module_file: str | Path | None) -> Path | None:
    selected = Path(module_file or getattr(_runtime_module(), "__file__", ""))
    if not selected:
        return None
    try:
        return selected.resolve().parents[2] / APPLICATION_ICON_FILENAME
    except (IndexError, OSError, RuntimeError):
        return None


def _append_candidate(
    candidates: list[ApplicationIconCandidate],
    path: Path | None,
    source: str,
) -> None:
    if path is None:
        return
    try:
        selected = path.expanduser()
    except (OSError, RuntimeError):
        return
    if any(existing.path == selected for existing in candidates):
        return
    candidates.append(ApplicationIconCandidate(selected, source))


def application_icon_candidates(
    *,
    module_file: str | Path | None = None,
    runtime: Any = sys,
) -> tuple[ApplicationIconCandidate, ...]:
    """Return the supported source, frozen-bundle, and executable candidates."""

    candidates: list[ApplicationIconCandidate] = []
    frozen = bool(getattr(runtime, "frozen", False))
    meipass = getattr(runtime, "_MEIPASS", None)
    executable = getattr(runtime, "executable", "")

    # A frozen bundle's _MEIPASS is authoritative for resources collected by
    # PyInstaller.  The executable directory remains a deliberate fallback so
    # an externally shipped app.ico continues to work for onedir releases.
    if not frozen:
        _append_candidate(candidates, _source_icon_path(module_file), "source")
    if meipass:
        _append_candidate(
            candidates,
            Path(str(meipass)) / APPLICATION_ICON_FILENAME,
            "frozen_meipass",
        )
    if frozen and module_file is not None:
        _append_candidate(candidates, _source_icon_path(module_file), "source")
    if executable:
        try:
            _append_candidate(
                candidates,
                Path(str(executable)).resolve().parent / APPLICATION_ICON_FILENAME,
                "executable_directory",
            )
        except (OSError, RuntimeError):
            pass
    return tuple(candidates)


def resolve_application_icon_path(
    *,
    module_file: str | Path | None = None,
    runtime: Any = sys,
) -> Path | None:
    """Return the first existing supported app.ico path, without loading Qt."""

    for candidate in application_icon_candidates(module_file=module_file, runtime=runtime):
        try:
            if candidate.path.is_file():
                return candidate.path
        except OSError:
            continue
    return None


def _available_sizes(icon: QIcon) -> tuple[tuple[int, int], ...]:
    return tuple((int(size.width()), int(size.height())) for size in icon.availableSizes())


def _record_resource_failure(result: ApplicationIconResult) -> None:
    try:
        from .startup_diagnostics import record_startup_event

        record_startup_event("application_icon_load_failed", result.to_diagnostic())
    except BaseException:
        # Diagnostics are deliberately best effort and must not block startup.
        return


def load_application_icon(
    *,
    module_file: str | Path | None = None,
    runtime: Any = sys,
) -> ApplicationIconResult:
    """Find and validate app.ico, including Qt's decoded sizes."""

    attempts: list[ApplicationIconAttempt] = []
    for candidate in application_icon_candidates(module_file=module_file, runtime=runtime):
        try:
            exists = candidate.path.is_file()
        except OSError as exc:
            attempts.append(
                ApplicationIconAttempt(
                    candidate.path,
                    candidate.source,
                    False,
                    None,
                    (),
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if not exists:
            attempts.append(
                ApplicationIconAttempt(candidate.path, candidate.source, False, None, ())
            )
            continue

        try:
            icon = QIcon(str(candidate.path))
            is_null = bool(icon.isNull())
            sizes = _available_sizes(icon)
            attempt = ApplicationIconAttempt(
                candidate.path,
                candidate.source,
                True,
                is_null,
                sizes,
                None if not is_null and sizes else "icon_is_null_or_has_no_sizes",
            )
        except BaseException as exc:
            icon = None
            is_null = None
            sizes = ()
            attempt = ApplicationIconAttempt(
                candidate.path,
                candidate.source,
                True,
                is_null,
                sizes,
                f"{type(exc).__name__}: {exc}",
            )
        attempts.append(attempt)
        if icon is not None and not is_null and sizes:
            return ApplicationIconResult(
                icon=icon,
                path=candidate.path,
                source=candidate.source,
                available_sizes=sizes,
                attempts=tuple(attempts),
            )

    result = ApplicationIconResult(
        icon=None,
        path=None,
        source=None,
        available_sizes=(),
        attempts=tuple(attempts),
        error="application icon resource is missing or invalid",
    )
    _record_resource_failure(result)
    return result


def set_windows_app_user_model_id() -> bool:
    """Set the stable Windows identity used by taskbar grouping and icons."""

    if os.name != "nt":
        return False
    try:
        setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        result = int(setter(APPLICATION_APP_USER_MODEL_ID))
        if result != 0:
            raise OSError(f"HRESULT 0x{result & 0xFFFFFFFF:08X}")
        return True
    except BaseException as exc:
        try:
            from .startup_diagnostics import record_startup_event

            record_startup_event(
                "application_identity_setup_failed",
                {
                    "app_user_model_id": APPLICATION_APP_USER_MODEL_ID,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except BaseException:
            pass
        return False


def install_application_icon(
    application: Any | None = None,
    window: Any | None = None,
    *,
    result: ApplicationIconResult | None = None,
) -> ApplicationIconResult:
    """Set QApplication and optional main-window icons from one validated result."""

    selected = result
    if selected is None:
        set_windows_app_user_model_id()
        selected = load_application_icon()
    if selected.icon is not None:
        if application is not None:
            application.setWindowIcon(selected.icon)
        if window is not None:
            window.setWindowIcon(selected.icon)
    return selected


__all__ = [
    "APPLICATION_APP_USER_MODEL_ID",
    "APPLICATION_ICON_FILENAME",
    "ApplicationIconAttempt",
    "ApplicationIconCandidate",
    "ApplicationIconResult",
    "application_icon_candidates",
    "install_application_icon",
    "load_application_icon",
    "resolve_application_icon_path",
    "set_windows_app_user_model_id",
]
