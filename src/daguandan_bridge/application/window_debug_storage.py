"""Private, bounded storage for read-only window diagnostic reports.

The storage boundary deliberately owns only ``<diagnostics>/window_debug``.  It
never accepts an arbitrary output path, never follows links while cleaning, and
keeps screenshots behind an explicit per-run opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4

try:
    from ..config import DIAGNOSTICS_ROOT
except Exception:  # pragma: no cover - defensive import fallback
    DIAGNOSTICS_ROOT = None

SCHEMA = "guandan.window-debug-storage/v1"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JSON_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.json$")
_IMAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:png|jpe?g)$", re.I)
_SECRET_WORDS = ("password", "passwd", "secret", "token", "api_key", "apikey", "authorization", "cookie", "private_key")
_PATH_WORDS = ("path", "root", "directory", "dir", "filename", "file")
_PRIVATE_WORDS = ("email", "username", "user_name", "machine_name", "computer_name")
_SCREENSHOT_WORDS = ("screenshot", "image", "pixels", "frame_bytes", "encoded_image")


class WindowDebugStorageError(RuntimeError):
    """The requested diagnostic storage operation is unsafe or invalid."""


class ScreenshotOptInRequired(WindowDebugStorageError):
    """Raised when a screenshot is requested without explicit opt-in."""


@dataclass(frozen=True)
class WindowDebugRun:
    run_id: str
    path: Path
    screenshots_enabled: bool = False


@dataclass(frozen=True)
class CleanupResult:
    removed_run_ids: tuple[str, ...]
    skipped_run_ids: tuple[str, ...]
    removed_bytes: int
    remaining_runs: int
    remaining_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "removed_run_ids": list(self.removed_run_ids),
            "skipped_run_ids": list(self.skipped_run_ids),
            "removed_bytes": self.removed_bytes,
            "remaining_runs": self.remaining_runs,
            "remaining_bytes": self.remaining_bytes,
        }


def _reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _assert_plain_ancestors(path: Path) -> None:
    for part in (path, *path.parents):
        if part.exists() and _reparse(part):
            raise WindowDebugStorageError(f"diagnostic path contains a link: {part}")


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not _RUN_ID.fullmatch(value) or value in {".", ".."}:
        raise WindowDebugStorageError("run_id must be one safe path segment")
    return value


def _safe_name(value: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or Path(value).name != value or not pattern.fullmatch(value):
        raise WindowDebugStorageError("artifact name must be a safe file name")
    return value


def _json_value(value: Any, *, key: str = "", redact_paths: bool = True) -> Any:
    normalized = key.casefold().replace("-", "_")
    if any(word in normalized for word in _SECRET_WORDS) or any(word in normalized for word in _PRIVATE_WORDS):
        return "[REDACTED]"
    if any(word in normalized for word in _SCREENSHOT_WORDS):
        return "[OMITTED_SCREENSHOT]"
    if redact_paths and (
        normalized in _PATH_WORDS
        or any(
            normalized.startswith(f"{word}_") or normalized.endswith(f"_{word}")
            for word in _PATH_WORDS
        )
    ):
        return "[REDACTED_PATH]"
    if isinstance(value, Path):
        return "[REDACTED_PATH]" if redact_paths else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("binary diagnostic values are not JSON artifacts")
    if value is None or isinstance(value, (str, bool, int, float)):
        if redact_paths and isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\")):
            return "[REDACTED_PATH]"
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_value(v, key=str(k), redact_paths=redact_paths) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item, redact_paths=redact_paths) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict(), key=key, redact_paths=redact_paths)
    if hasattr(value, "item") and callable(value.item):
        return _json_value(value.item(), key=key, redact_paths=redact_paths)
    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")


def _atomic_write(path: Path, payload: bytes) -> None:
    _assert_plain_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_plain_ancestors(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


class WindowDebugStorage:
    """Store reports below one diagnostics-owned namespace."""

    def __init__(
        self,
        diagnostics_root: Path | str | None = None,
        *,
        runtime_root: Path | str | Any | None = None,
        max_runs: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        if diagnostics_root is not None and runtime_root is not None:
            raise WindowDebugStorageError("choose diagnostics_root or runtime_root, not both")
        if runtime_root is not None:
            diagnostics_root = getattr(runtime_root, "diagnostics_root", None) or Path(runtime_root) / "diagnostics"
        if diagnostics_root is None:
            diagnostics_root = DIAGNOSTICS_ROOT
        if diagnostics_root is None:
            raise WindowDebugStorageError("diagnostics root is unavailable")
        root = Path(diagnostics_root).expanduser()
        if not root.is_absolute():
            raise WindowDebugStorageError("diagnostics root must be absolute")
        _assert_plain_ancestors(root)
        executable_dir = Path(sys.executable).resolve(strict=False).parent
        if root == executable_dir or root.is_relative_to(executable_dir):
            raise WindowDebugStorageError("diagnostics root must not be the executable directory")
        self.diagnostics_root = root
        self.root = root / "window_debug"
        self.max_runs = self._limit(max_runs, "max_runs")
        self.max_bytes = self._limit(max_bytes, "max_bytes")
        self._lock = RLock()

    @staticmethod
    def _limit(value: int | None, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or int(value) < 0:
            raise WindowDebugStorageError(f"{name} must be a non-negative integer")
        return int(value)

    def _ensure_root(self) -> None:
        _assert_plain_ancestors(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        _assert_plain_ancestors(self.root)

    def create_run(self, run_id: str | None = None, *, allow_screenshots: bool = False) -> WindowDebugRun:
        run_id = _safe_id(run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:12]))
        with self._lock:
            self._ensure_root()
            path = self.root / run_id
            _assert_plain_ancestors(path)
            try:
                path.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise WindowDebugStorageError(f"run already exists: {run_id}") from exc
            return WindowDebugRun(run_id, path, bool(allow_screenshots))

    def open_run(self, run_id: str, *, allow_screenshots: bool = False) -> WindowDebugRun:
        run_id = _safe_id(run_id)
        path = self.root / run_id
        _assert_plain_ancestors(path)
        if not path.is_dir() or _reparse(path):
            raise WindowDebugStorageError(f"run does not exist: {run_id}")
        return WindowDebugRun(run_id, path, bool(allow_screenshots))

    def _run(self, run: WindowDebugRun | str) -> WindowDebugRun:
        item = self.open_run(run) if isinstance(run, str) else run
        if not isinstance(item, WindowDebugRun) or item.path.parent != self.root or item.path.name != item.run_id:
            raise WindowDebugStorageError("run is outside this storage namespace")
        _assert_plain_ancestors(item.path)
        if not item.path.is_dir() or _reparse(item.path):
            raise WindowDebugStorageError("run directory is unavailable or unsafe")
        return item

    def _load_report(self, item: WindowDebugRun) -> dict[str, Any]:
        path = item.path / "report.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WindowDebugStorageError("report.json is unreadable") from exc
        if not isinstance(value, dict):
            raise WindowDebugStorageError("report.json must contain an object")
        return value

    def _write_report_payload(self, item: WindowDebugRun, payload: Mapping[str, Any]) -> Path:
        path = item.path / "report.json"
        encoded = json.dumps(_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        _atomic_write(path, encoded)
        return path

    def write_report(self, run: WindowDebugRun | str, report: Mapping[str, Any]) -> Path:
        item = self._run(run)
        with self._lock:
            existing = self._load_report(item)
            payload = dict(_json_value(report))
            if existing.get("events") and "events" not in payload:
                payload["events"] = existing["events"]
            if existing.get("media") and "media" not in payload:
                payload["media"] = existing["media"]
            payload.setdefault("events", [])
            payload.setdefault("media", {"media_saved": False, "items": {}})
            return self._write_report_payload(item, payload)

    def write_json(self, run: WindowDebugRun | str, name: str, value: Any) -> Path:
        if name != "report.json":
            raise WindowDebugStorageError("a diagnostic run may contain only report.json")
        return self.write_report(run, value if isinstance(value, Mapping) else {"value": value})

    def append_event(self, run: WindowDebugRun | str, event: Mapping[str, Any]) -> Path:
        item = self._run(run)
        with self._lock:
            payload = self._load_report(item)
            events = payload.setdefault("events", [])
            if not isinstance(events, list):
                events = []
                payload["events"] = events
            events.append(_json_value(event))
            payload.setdefault("media", {"media_saved": False, "items": {}})
            return self._write_report_payload(item, payload)

    def write_screenshot(self, run: WindowDebugRun | str, name: str, content: bytes | bytearray | memoryview) -> Path:
        item = self._run(run)
        if not item.screenshots_enabled:
            raise ScreenshotOptInRequired("screenshots require create_run(..., allow_screenshots=True)")
        name = _safe_name(name, _IMAGE_NAME)
        with self._lock:
            payload = self._load_report(item)
            media = payload.setdefault("media", {"media_saved": False, "items": {}})
            if not isinstance(media, dict):
                media = {"media_saved": False, "items": {}}
                payload["media"] = media
            screenshots = media.setdefault("items", {})
            if not isinstance(screenshots, dict):
                screenshots = {}
                media["items"] = screenshots
            screenshots[name] = {
                "encoding": "base64",
                "content": base64.b64encode(bytes(content)).decode("ascii"),
            }
            media["media_saved"] = True
            return self._write_report_payload(item, payload)

    def cleanup(self) -> CleanupResult:
        with self._lock:
            if not self.root.exists():
                return CleanupResult((), (), 0, 0, 0)
            _assert_plain_ancestors(self.root)
            records: list[tuple[str, Path, int, bool]] = []
            for child in self.root.iterdir():
                if not child.is_dir() or _reparse(child) or not _RUN_ID.fullmatch(child.name):
                    continue
                size = 0
                safe = True
                stack = [child]
                while stack:
                    current = stack.pop()
                    for entry in current.iterdir():
                        if _reparse(entry):
                            safe = False
                            continue
                        if entry.is_dir():
                            stack.append(entry)
                        elif entry.is_file():
                            size += entry.stat().st_size
                records.append((child.name, child, size, safe))
            records.sort(key=lambda item: (item[0].casefold(), item[0]))
            protected_run_id = next(
                (item[0] for item in reversed(records) if item[3]),
                None,
            )
            remove_ids: set[str] = set()
            if self.max_runs is not None:
                retained_count = max(1, self.max_runs) if protected_run_id is not None else self.max_runs
                excess = max(0, len(records) - retained_count)
                for run_id, _path, _size, safe in records[:excess]:
                    if safe and run_id != protected_run_id:
                        remove_ids.add(run_id)

            keep = [item for item in records if item[0] not in remove_ids]
            total = sum(item[2] for item in keep)
            if self.max_bytes is not None and total > self.max_bytes:
                for run_id, _path, size, safe in keep:
                    if total <= self.max_bytes:
                        break
                    if not safe or run_id == protected_run_id:
                        continue
                    remove_ids.add(run_id)
                    total -= size

            remove = [item for item in records if item[0] in remove_ids]
            removed: list[str] = []
            skipped: list[str] = []
            removed_bytes = 0
            for run_id, path, size, safe in remove:
                if not safe:
                    skipped.append(run_id)
                    continue
                try:
                    _assert_plain_ancestors(path)
                    shutil.rmtree(path)
                except OSError:
                    skipped.append(run_id)
                else:
                    removed.append(run_id)
                    removed_bytes += size
            remaining = [(name, path, size, safe) for name, path, size, safe in records if name not in removed]
            return CleanupResult(tuple(removed), tuple(skipped), removed_bytes, len(remaining), sum(item[2] for item in remaining))


__all__ = [
    "SCHEMA", "CleanupResult", "ScreenshotOptInRequired", "WindowDebugRun",
    "WindowDebugStorage", "WindowDebugStorageError",
]



