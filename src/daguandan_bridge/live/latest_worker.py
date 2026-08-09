from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any


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
        self._priority_pending: deque[Any] = deque()
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

    def submit(self, value: Any, *, preserve: bool = False, max_preserved: int = 0) -> None:
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("latest-only worker 已停止")
            if preserve:
                if max_preserved <= 0:
                    raise ValueError("保留队列上限必须为正数")
                while len(self._priority_pending) >= max_preserved:
                    self._priority_pending.popleft()
                self._priority_pending.append(value)
            else:
                self._pending = value
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop_requested
                    or self._priority_pending
                    or self._pending is not self._EMPTY
                )
                if self._stop_requested:
                    return
                if self._priority_pending:
                    value = self._priority_pending.popleft()
                else:
                    value = self._pending
                    self._pending = self._EMPTY
            try:
                result = self.operation(value)
            except Exception as exc:
                with self._condition:
                    stopped = self._stop_requested
                if not stopped and self.on_error is not None:
                    self.on_error(exc)
            else:
                with self._condition:
                    stopped = self._stop_requested
                if not stopped and self.on_result is not None:
                    self.on_result(result)

    def stop(self, *, timeout: float | None = None) -> bool:
        with self._condition:
            self._stop_requested = True
            self._pending = self._EMPTY
            self._priority_pending.clear()
            self._condition.notify_all()
        thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()
