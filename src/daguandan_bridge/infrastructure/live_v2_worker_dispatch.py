"""One persistent, serialized sender for a worker command pipe."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Callable


@dataclass(frozen=True, slots=True)
class _Dispatch:
    value: object
    on_start: Callable[[], None]
    on_success: Callable[[], None]
    on_failure: Callable[[str], None]


class WorkerSendDispatcher:
    """Keep one already-running sender thread per worker generation."""

    def __init__(self, send: Callable[[object], None], *, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._send = send
        self._queue: Queue[_Dispatch | None] = Queue(maxsize=capacity)
        self._lock = Lock()
        self._closed = False
        self._thread = Thread(
            target=self._run, name="live-v2-worker-send", daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        value: object,
        *,
        on_start: Callable[[], None] = lambda: None,
        on_success: Callable[[], None] = lambda: None,
        on_failure: Callable[[str], None],
    ) -> None:
        with self._lock:
            if self._closed:
                on_failure("RuntimeError: worker sender is closed")
                return
            try:
                self._queue.put_nowait(
                    _Dispatch(value, on_start, on_success, on_failure)
                )
            except BaseException as exc:
                on_failure(f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._queue.put_nowait(None)
            except BaseException:
                return

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                item.on_start()
                self._send(item.value)
            except BaseException as exc:
                item.on_failure(f"{type(exc).__name__}: {exc}")
            else:
                item.on_success()


__all__ = ["WorkerSendDispatcher"]
