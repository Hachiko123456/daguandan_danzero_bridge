"""Bounded UTF-8 logs with a fail-open, constant-memory discard mode.

After an I/O or path-safety failure an existing stream never retries appends:
it counts discarded characters instead. This is important when the stream is
the windowed application's stdout/stderr. Construction failures still raise so
the startup diagnostics layer can fall back safely.
"""

import io
import json
import stat
from pathlib import Path
from threading import RLock


class RotatingTextLog(io.TextIOBase):
    def __init__(self, path: Path, max_bytes: int = 1024 * 1024) -> None:
        super().__init__()
        # Initialize lifecycle state before any operation that can fail; IOBase
        # may invoke close() while collecting a partially constructed object.
        self._lock = RLock()
        self._file = None
        self._discarding = False
        self.dropped_writes = 0
        self.dropped_characters = 0
        self.io_errors = 0
        self.last_error = ""
        self._bytes = 0
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 128:
            raise ValueError("log budget must be at least 128 bytes")
        self.path = Path(path).absolute()
        self.max_bytes = max_bytes
        self._validate_paths()
        try:
            self._file = self._open_file("a")
            self._bytes = self.path.stat().st_size
        except BaseException:
            self._close_file_safely()
            raise

    def _validate_paths(self) -> None:
        for candidate in (self.path, self.path.with_name(self.path.name + ".1"), *self.path.parents):
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise OSError(f"Log path traverses a link: {candidate}")

    def _open_file(self, mode: str):
        return self.path.open(mode, encoding="utf-8", errors="replace", newline="\n", buffering=1)

    def _note_io_failure(self, error: Exception) -> None:
        self._discarding = True
        self.io_errors = min(2**63 - 1, self.io_errors + 1)
        self.last_error = f"{type(error).__name__}: {error}"[:256]

    def _close_file_safely(self) -> None:
        handle, self._file = self._file, None
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:
                self._note_io_failure(exc)

    def _discard_after_failure(self, error: Exception) -> None:
        self._note_io_failure(error)
        self._close_file_safely()

    def _note_dropped(self, characters: int) -> None:
        self.dropped_writes = min(2**63 - 1, self.dropped_writes + 1)
        self.dropped_characters = min(2**63 - 1, self.dropped_characters + characters)

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return not self.closed

    def fileno(self) -> int:
        with self._lock:
            if self.closed:
                raise ValueError("I/O operation on closed log")
            if self._file is None:
                raise io.UnsupportedOperation("discarding log has no file descriptor")
            return self._file.fileno()

    def write(self, text: str) -> int:
        with self._lock:
            if self.closed:
                raise ValueError("I/O operation on closed log")
            if not isinstance(text, str):
                raise TypeError("write() argument must be str")
            original_length = len(text)
            if self._discarding:
                # No encode, stat, allocation proportional to input, or disk
                # retry after failure. Counters saturate rather than grow.
                self._note_dropped(original_length)
                return original_length
            clipped_characters = original_length > self.max_bytes
            encoded = text[-self.max_bytes:].encode("utf-8", errors="replace")
            if clipped_characters or len(encoded) > self.max_bytes:
                marker = b"[log entry truncated to storage budget]\n"
                tail = encoded[-(self.max_bytes - len(marker)):].decode("utf-8", errors="ignore")
                encoded = marker + tail.encode("utf-8")
            text = encoded.decode("utf-8")
            try:
                if self._bytes + len(encoded) > self.max_bytes:
                    self._validate_paths()
                    self._file.close()
                    self._file = None
                    # This writer owns exactly these two files, not a tree.
                    self.path.replace(self.path.with_name(self.path.name + ".1"))
                    self._validate_paths()
                    self._file = self._open_file("w")
                    self._bytes = 0
                self._file.write(text)
                self._bytes += len(encoded)
            except Exception as exc:
                self._discard_after_failure(exc)
                self._note_dropped(original_length)
            return original_length

    def flush(self) -> None:
        with self._lock:
            if self._discarding or self._file is None:
                return
            try:
                self._file.flush()
            except Exception as exc:
                self._discard_after_failure(exc)

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self._close_file_safely()
            super().close()


def append_bounded_event(path: Path, payload: dict, max_bytes: int = 1024 * 1024) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    if len(text.encode("utf-8")) > max_bytes:
        payload = {key: value for key, value in payload.items() if key != "evidence"}
        payload["evidence"] = {"truncated": True, "reason": "text_budget"}
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > max_bytes:
            text = '{"event":"entry_omitted","evidence":{"truncated":true,"reason":"text_budget"}}\n'
    with RotatingTextLog(path, max_bytes) as handle:
        handle.write(text)
