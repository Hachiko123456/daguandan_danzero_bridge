from __future__ import annotations

import threading
import math
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from ..live.latest_worker import LatestOnlyWorker


class CaptureWorker(QObject):
    frame_ready = Signal(object)
    # Preserve typed exceptions for diagnostic consumers. UI adapters decide
    # where to stringify them; the worker must not destroy machine codes/types.
    error = Signal(object)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any], interval_sec: float = 0.1):
        super().__init__()
        self.operation = operation
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()

    @Slot()
    def run(self) -> None:
        deadline = time.monotonic()
        previous_interval = float(self.interval_sec)
        try:
            while not self._stop_event.is_set():
                try:
                    value = self.operation()
                except Exception as exc:
                    self.error.emit(exc)
                    break
                if self._stop_event.is_set():
                    break
                self.frame_ready.emit(value)
                interval = float(self.interval_sec)
                if not math.isfinite(interval) or interval <= 0:
                    raise ValueError("capture interval must be finite and positive")
                now = time.monotonic()
                if interval != previous_interval:
                    # Listening may change between the table and the lobby.
                    deadline = now + interval
                    previous_interval = interval
                else:
                    deadline += interval
                    if deadline <= now:
                        # Skip missed ticks; never burst through an overdue FIFO.
                        deadline += (math.floor((now - deadline) / interval) + 1) * interval
                if self._stop_event.wait(max(0.0, deadline - now)):
                    break
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()


class WorkerHandle(QObject):
    frame_ready = Signal(object)
    error = Signal(object)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any], interval_sec: float = 0.1):
        super().__init__()
        self.thread = QThread()
        self.worker = CaptureWorker(operation, interval_sec)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.frame_ready)
        self.worker.error.connect(self.error)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.DirectConnection)
        self.thread.finished.connect(self.finished)
        self._started = False

    @property
    def is_running(self) -> bool:
        return self.thread.isRunning()

    def start(self) -> None:
        if not self._started:
            self._started = True
            self.thread.start()

    def stop(self) -> None:
        self.worker.stop()

    def wait(self, timeout_ms: int = 3000) -> bool:
        return bool(self.thread.wait(timeout_ms))


class OneShotWorker(QObject):
    """Run one potentially blocking operation outside the GUI thread."""

    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any]):
        super().__init__()
        self.operation = operation

    @Slot()
    def run(self) -> None:
        try:
            self.result.emit(self.operation())
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class OneShotThread(QThread):
    """Run one blocking operation in a thread without a movable worker object."""

    result = Signal(object)
    error = Signal(str)

    def __init__(self, operation: Callable[[], Any], parent: QObject | None = None):
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            self.result.emit(self.operation())
        except Exception as exc:
            self.error.emit(str(exc))
