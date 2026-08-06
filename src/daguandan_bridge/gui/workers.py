from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot


class CaptureWorker(QObject):
    frame_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any], interval_sec: float = 0.1):
        super().__init__()
        self.operation = operation
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    value = self.operation()
                except Exception as exc:
                    self.error.emit(str(exc))
                    break
                self.frame_ready.emit(value)
                if self._stop_event.wait(self.interval_sec):
                    break
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()


class WorkerHandle(QObject):
    frame_ready = Signal(object)
    error = Signal(str)
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


class LatestOnlyWorker:
    """Process one item while retaining only the newest pending replacement."""

    _EMPTY = object()

    def __init__(
        self,
        operation: Callable[[Any], Any],
        *,
        on_result: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.operation = operation
        self.on_result = on_result
        self.on_error = on_error
        self._condition = threading.Condition()
        self._pending: Any = self._EMPTY
        self._stop_requested = False
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                if self._thread.is_alive():
                    return
                raise RuntimeError("latest-only worker 不能重复启动")
            self._thread = threading.Thread(
                target=self._run,
                name="latest-only-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(self, value: Any) -> None:
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("latest-only worker 已停止")
            self._pending = value
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop_requested or self._pending is not self._EMPTY
                )
                if self._stop_requested:
                    return
                value = self._pending
                self._pending = self._EMPTY
            try:
                result = self.operation(value)
            except Exception as exc:
                if self.on_error is not None:
                    self.on_error(exc)
            else:
                if self.on_result is not None:
                    self.on_result(result)

    def stop(self, *, timeout: float | None = None) -> bool:
        with self._condition:
            self._stop_requested = True
            self._pending = self._EMPTY
            self._condition.notify_all()
        thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()
