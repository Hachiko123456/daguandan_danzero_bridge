"""Conservative, cooperative disk admission for pre-session diagnostics.

No retention operation deletes an existing incident: a full store refuses new
media/text. This protects active writers, manifests and user evidence. The lock
covers scans and publication across processes and is OS-released after a crash.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import stat
from threading import Lock
from time import monotonic, sleep
from typing import Iterator


_MEDIA = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".avi", ".mp4"})
_LOCK_REGISTRY_GUARD = Lock()
_PROCESS_LOCKS: dict[str, object] = {}


def is_reparse(path: Path) -> bool:
    info = path.lstat()
    return bool(stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400)


def assert_plain_path(path: Path) -> None:
    """Reject junctions/symlinks in every existing ancestor, including roots."""
    for part in (path, *path.parents):
        if part.exists() and is_reparse(part):
            raise OSError(f"diagnostic path must not contain a link: {part}")


def plain_files(root: Path, *, deadline: float | None = None) -> Iterator[Path]:
    """Walk without ever entering a Windows junction or symbolic link."""
    if not root.exists():
        return
    assert_plain_path(root)
    stack = [root]
    while stack:
        parent = stack.pop()
        for entry in parent.iterdir():
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError("diagnostic quota scan exceeded its time budget")
            if is_reparse(entry):
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                yield entry


@dataclass(frozen=True)
class DiagnosticAllowance:
    media_bytes: int
    text_bytes: int
    incidents_remaining: int


class DiagnosticBudget:
    def __init__(
        self,
        run_root: Path,
        *,
        runs_root: Path | None = None,
        run_media_bytes: int = 64 * 1024 * 1024,
        total_media_bytes: int = 512 * 1024 * 1024,
        run_text_bytes: int = 8 * 1024 * 1024,
        total_text_bytes: int = 64 * 1024 * 1024,
        run_incidents: int = 128,
        total_incidents: int = 1024,
    ) -> None:
        self.run_root = Path(os.path.abspath(run_root))
        # Infer only the production layout, never scan an unrelated parent.
        inferred = self.run_root.parent if self.run_root.parent.name == "runs" else self.run_root
        self.runs_root = Path(os.path.abspath(runs_root or inferred))
        with _LOCK_REGISTRY_GUARD:
            self._process_lock = _PROCESS_LOCKS.setdefault(
                os.path.normcase(str(self.runs_root)), Lock(),
            )
        if not self.run_root.is_relative_to(self.runs_root):
            raise ValueError("diagnostic run is outside the configured runs root")
        self.run_media_bytes = max(0, int(run_media_bytes))
        self.total_media_bytes = max(0, int(total_media_bytes))
        self.run_text_bytes = max(0, int(run_text_bytes))
        self.total_text_bytes = max(0, int(total_text_bytes))
        self.run_incidents = max(1, int(run_incidents))
        self.total_incidents = max(1, int(total_incidents))

    @contextmanager
    def transaction(self, *, timeout: float = 0.2) -> Iterator[DiagnosticAllowance]:
        # Called only by the diagnostic writer, never capture/recognition/UI.
        assert_plain_path(self.runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        deadline = monotonic() + max(0.0, float(timeout))
        if not self._process_lock.acquire(timeout=max(0.0, deadline - monotonic())):
            raise TimeoutError("diagnostic quota process lock timed out")
        try:
            lock_path = self.runs_root / ".opening-budget.lock"
            assert_plain_path(lock_path)
            with lock_path.open("a+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                while True:
                    try:
                        if os.name == "nt":
                            import msvcrt

                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if monotonic() >= deadline:
                            raise TimeoutError("diagnostic quota file lock timed out")
                        sleep(min(0.005, max(0.0, deadline - monotonic())))
                try:
                    yield self._allowance(deadline=deadline)
                finally:
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._process_lock.release()

    def _allowance(self, *, deadline: float | None = None) -> DiagnosticAllowance:
        run_media = total_media = run_text = total_text = 0
        run_count = total_count = 0
        roots = (self.run_root,) if self.runs_root == self.run_root else self.runs_root.iterdir()
        for run in roots:
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError("diagnostic quota scan exceeded its time budget")
            if is_reparse(run) or not run.is_dir():
                continue
            opening = run / "opening"
            if not opening.exists() or is_reparse(opening):
                continue
            for path in plain_files(opening, deadline=deadline):
                size = path.stat().st_size
                current = run == self.run_root
                if path.suffix.lower() in _MEDIA:
                    total_media += size
                    if current:
                        run_media += size
                else:
                    total_text += size
                    if current:
                        run_text += size
                if path.name == "incident.json":
                    total_count += 1
                    if current:
                        run_count += 1
        return DiagnosticAllowance(
            media_bytes=max(0, min(self.run_media_bytes - run_media, self.total_media_bytes - total_media)),
            text_bytes=max(0, min(self.run_text_bytes - run_text, self.total_text_bytes - total_text)),
            incidents_remaining=max(0, min(self.run_incidents - run_count, self.total_incidents - total_count)),
        )
