"""One stable, application-local evidence case, independent of session storage.

No capture, Qt, models or live-state mutation occurs here. Assigning a case is
memory-only; the evidence writer materializes it on the first save/export.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from threading import RLock
from uuid import uuid4

CASE_SCHEMA = "guandan.problem-case/1"


def safe_path(path: Path) -> Path:
    value = Path(os.path.abspath(path.expanduser()))
    for current in (value, *value.parents):
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"诊断路径不能经过链接或重解析点：{current}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        safe_path(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DiagnosticCases:
    def __init__(self, root: Path, *, profile_name: str) -> None:
        self.root = safe_path(root)
        self.profile_name = profile_name
        self._lock = RLock()
        self._write_lock = RLock()
        self._current: Path | None = None
        self._records: dict[Path, dict] = {}
        self._ended = False

    @property
    def current(self) -> Path | None:
        with self._lock:
            return self._current

    def allocate(self) -> Path:
        with self._lock:
            if self._current is None or self._ended:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = self.root / "cases" / f"case_{stamp}_{uuid4().hex[:8]}"
                self._current = path
                self._ended = False
                self._records[path] = {
                    "schema": CASE_SCHEMA, "case_id": path.name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "profile_name": self.profile_name, "session_id": None,
                    "session_directory": None, "session_links": [],
                    "status": "listening", "frames_directory": "frames",
                }
            return self._current

    def bind_session(self, session_id: str, directory: Path | None, *, formal: bool = True) -> Path:
        with self._lock:
            path = self.allocate()
            record = self._records[path]
            if formal and record["session_id"] not in (None, session_id):
                self._ended = True
                path = self.allocate()
                record = self._records[path]
            link = {"session_id": session_id, "directory": str(directory) if directory else None,
                    "kind": "session" if formal else "preopening"}
            if link not in record["session_links"]:
                record["session_links"].append(link)
            if formal:
                record.update(session_id=session_id, session_directory=link["directory"], status="running")
            return path

    def finish_session(self, session_id: str) -> Path | None:
        with self._lock:
            for path, record in reversed(tuple(self._records.items())):
                if record["session_id"] == session_id:
                    record["status"] = "finished"
                    record["finished_at"] = datetime.now(timezone.utc).isoformat()
                    if path == self._current:
                        self._ended = True
                    return path
        return None

    def end_pending_listener(self) -> Path | None:
        """Explicit stop/new-listen boundary; an initial manual case still joins its first listen."""
        with self._lock:
            if self._current is None:
                return None
            record = self._records[self._current]
            if record["session_id"] is not None:
                return None
            record["status"] = "stopped"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._ended = True
            return self._current

    def describe(self, directory: Path | None = None) -> dict:
        with self._lock:
            path = directory or self._current
            return deepcopy(self._records.get(path, {}))

    def materialize(self, directory: Path, *, runtime: dict | None = None,
                    profile_directory: Path | None = None) -> Path:
        # Only background writers contend on disk. Never hold the capture-side
        # metadata lock across mkdir/read/fsync, including export snapshots.
        with self._write_lock:
            path = safe_path(directory)
            with self._lock:
                if path not in self._records or path.parent != self.root / "cases":
                    raise ValueError("unknown diagnostic case")
                record = self._records[path]
                if runtime:
                    record.update({key: runtime[key] for key in ("run_id", "startup_report", "build_id") if runtime.get(key)})
                    startup = runtime.get("startup_report")
                    if startup:
                        record["run_directory"] = str(Path(startup).parent)
                if profile_directory is not None:
                    record["profile_directory"] = str(profile_directory)
                need_config = profile_directory is not None and not record.get("configuration_snapshot_attempted")
            path.mkdir(parents=True, exist_ok=True)
            safe_path(path)
            if need_config:
                omissions = []
                for name in ("profile.json", "regions_config.json", "templates_config.json"):
                    try:
                        source = safe_path(profile_directory / name)
                        with source.open("rb") as stream:
                            raw = stream.read(2 * 1024 * 1024 + 1)
                        if len(raw) > 2 * 1024 * 1024:
                            raise ValueError("configuration grew beyond limit")
                        config = json.loads(raw)
                        if not isinstance(config, dict):
                            raise ValueError("configuration must be an object")
                        atomic_json(path / "config" / name, config)
                    except (OSError, ValueError) as exc:
                        omissions.append({"file": name, "reason": type(exc).__name__})
                with self._lock:
                    self._records[path]["configuration_snapshot_attempted"] = True
                    self._records[path]["configuration_snapshot_missing"] = omissions
                    self._records[path]["configuration_snapshot_timing"] = "first_evidence_write"
            with self._lock:
                document = deepcopy(self._records[path])
            atomic_json(path / "case.json", document)
            return path
