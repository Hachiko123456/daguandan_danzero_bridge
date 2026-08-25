from __future__ import annotations

import threading
import time
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
        on_discard: Callable[[Any, str], None] | None = None,
    ) -> None:
        self.operation = operation
        self.on_result = on_result
        self.on_error = on_error
        self.on_discard = on_discard
        self._condition = threading.Condition()
        self._pending: Any = self._EMPTY
        self._priority_pending: deque[Any] = deque()
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._latest_replaced = 0
        self._preserved_evicted = 0
        self._stop_discarded = 0
        self._inflight = 0
        self._inflight_value: Any = self._EMPTY
        self._inflight_discard_notified = False
        self._max_depth = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict[str, int]:
        """Return a thread-safe point-in-time queue and lifecycle snapshot."""

        with self._condition:
            pending_depth = int(self._pending is not self._EMPTY)
            priority_depth = len(self._priority_pending)
            return {
                "submitted": self._submitted,
                "started": self._started,
                "completed": self._completed,
                "failed": self._failed,
                "latest_replaced": self._latest_replaced,
                "preserved_evicted": self._preserved_evicted,
                "stop_discarded": self._stop_discarded,
                "inflight": self._inflight,
                "pending_depth": pending_depth,
                "priority_depth": priority_depth,
                "max_depth": self._max_depth,
            }

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
        discarded: list[tuple[Any, str]] = []
        with self._condition:
            if self._stop_requested:
                raise RuntimeError("latest-only worker 已停止")
            if preserve and max_preserved <= 0:
                raise ValueError("保留队列上限必须为正数")
            self._submitted += 1
            if preserve:
                while len(self._priority_pending) >= max_preserved:
                    discarded.append((self._priority_pending.popleft(), "preserved_evicted"))
                    self._preserved_evicted += 1
                self._priority_pending.append(value)
            else:
                if self._pending is not self._EMPTY:
                    discarded.append((self._pending, "latest_replaced"))
                    self._latest_replaced += 1
                self._pending = value
            self._max_depth = max(
                self._max_depth,
                len(self._priority_pending) + int(self._pending is not self._EMPTY),
            )
            self._condition.notify()
        self._notify_discards(discarded)

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
                self._started += 1
                self._inflight = 1
                self._inflight_value = value
                self._inflight_discard_notified = False
            try:
                result = self.operation(value)
            except Exception as exc:
                with self._condition:
                    stopped = self._stop_requested
                try:
                    if not stopped and self.on_error is not None:
                        self.on_error(exc)
                finally:
                    with self._condition:
                        self._failed += 1
                        self._inflight = 0
                        self._inflight_value = self._EMPTY
                        self._inflight_discard_notified = False
                        self._condition.notify_all()
            else:
                with self._condition:
                    stopped = self._stop_requested
                try:
                    if not stopped and self.on_result is not None:
                        self.on_result(result)
                finally:
                    with self._condition:
                        self._completed += 1
                        self._inflight = 0
                        self._inflight_value = self._EMPTY
                        self._inflight_discard_notified = False
                        self._condition.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until no operation is running and both pending queues are empty."""

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while (
                self._inflight
                or self._priority_pending
                or self._pending is not self._EMPTY
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def stop(self, *, timeout: float | None = None) -> bool:
        discarded: list[tuple[Any, str]] = []
        with self._condition:
            self._stop_requested = True
            if self._pending is not self._EMPTY:
                discarded.append((self._pending, "stop_discarded"))
                self._pending = self._EMPTY
            while self._priority_pending:
                discarded.append((self._priority_pending.popleft(), "stop_discarded"))
            if self._inflight and not self._inflight_discard_notified:
                discarded.append((self._inflight_value, "stop_discarded"))
                self._inflight_discard_notified = True
            self._stop_discarded += len(discarded)
            self._condition.notify_all()
        self._notify_discards(discarded)
        thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def _notify_discards(self, discarded: list[tuple[Any, str]]) -> None:
        callback = self.on_discard
        if callback is None:
            return
        for value, reason in discarded:
            callback(value, reason)
