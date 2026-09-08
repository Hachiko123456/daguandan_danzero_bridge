"""Bounded request scheduling policy for the live-v2 worker host."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable
from ..application.live_v2_worker_protocol import (
    DeliveryMode,
    WorkerFailure,
    WorkerRequest,
    WorkerResult,
    WorkerResultStatus,
)
from .live_v2_worker_deadlines import RequestDeadlineLedger
Version = tuple[str, int, int]
def _processing_ms() -> int:
    return time.monotonic_ns() // 1_000_000
def item_version(item: WorkerRequest | WorkerResult) -> Version:
    return item.session_id, item.capture_generation, item.state_revision
def local_result(
    request: WorkerRequest,
    status: WorkerResultStatus,
    code: str,
    *,
    worker_generation: int,
    message: str = "",
    worker_pid: int | None = None,
) -> WorkerResult:
    """Create a terminal result for a host-side scheduling decision."""
    return WorkerResult.terminal(
        request,
        status=status,
        worker_generation=worker_generation,
        worker_pid=worker_pid,
        finished_processing_ms=_processing_ms(),
        failure=WorkerFailure(code, status.value, message or code),
    )
@dataclass(frozen=True, slots=True)
class QueueSubmission:
    discarded: tuple[WorkerResult, ...] = ()
@dataclass(slots=True)
class _InFlight:
    request: WorkerRequest
    generation: int
    deadline: float | None = None
    timer: threading.Timer | None = None
class BoundedWorkerQueue:
    """One latest-only slot plus a configurable event FIFO."""
    def __init__(self, fifo_capacity: int) -> None:
        if fifo_capacity < 0:
            raise ValueError("fifo_capacity must be non-negative")
        self._fifo_capacity = fifo_capacity
        self._fifo: deque[WorkerRequest] = deque()
        self._latest: WorkerRequest | None = None
        self._last_sequence = -1
    @property
    def pending_count(self) -> int:
        return len(self._fifo) + int(self._latest is not None)
    def submit(
        self,
        request: WorkerRequest,
        *,
        active_version: Version,
        worker_generation: int,
        has_in_flight: bool,
    ) -> QueueSubmission:
        if item_version(request) != active_version:
            raise ValueError("request version does not match the active host version")
        if request.request_sequence <= self._last_sequence:
            raise ValueError("request_sequence must increase within an active version")
        self._last_sequence = request.request_sequence
        discarded: list[WorkerResult] = []
        if request.delivery is DeliveryMode.FIFO:
            direct_slot = not has_in_flight and self.pending_count == 0
            if len(self._fifo) >= self._fifo_capacity and not direct_slot:
                discarded.append(local_result(
                    request,
                    WorkerResultStatus.DROPPED,
                    "fifo_capacity_exceeded",
                    worker_generation=worker_generation,
                ))
            else:
                self._fifo.append(request)
        else:
            if self._latest:
                discarded.append(local_result(
                    self._latest,
                    WorkerResultStatus.DROPPED,
                    "latest_replaced",
                    worker_generation=worker_generation,
                ))
            self._latest = request
        return QueueSubmission(tuple(discarded))
    def pop_next(self) -> WorkerRequest | None:
        if self._fifo:
            return self._fifo.popleft()
        request, self._latest = self._latest, None
        return request
    def rebind(self, *, reason: str, worker_generation: int) -> tuple[WorkerResult, ...]:
        discarded = self.discard_all(
            reason=reason,
            worker_generation=worker_generation,
        )
        self._last_sequence = -1
        return discarded
    def discard_all(
        self,
        *,
        reason: str,
        worker_generation: int,
    ) -> tuple[WorkerResult, ...]:
        discarded = [
            local_result(
                request,
                WorkerResultStatus.DROPPED,
                reason,
                worker_generation=worker_generation,
            )
            for request in self._fifo
        ]
        self._fifo.clear()
        if self._latest:
            discarded.append(local_result(
                self._latest,
                WorkerResultStatus.DROPPED,
                reason,
                worker_generation=worker_generation,
            ))
            self._latest = None
        return tuple(discarded)
class WorkerRequestCoordinator:
    """Own pending/in-flight state while the host owns lifecycle policy."""

    def __init__(self, fifo_capacity: int) -> None:
        self._pending = BoundedWorkerQueue(fifo_capacity)
        self._in_flight: _InFlight | None = None
        self._deadlines = RequestDeadlineLedger()
    @property
    def pending_count(self) -> int:
        return self._pending.pending_count
    @property
    def has_in_flight(self) -> bool:
        return self._in_flight is not None
    def submit(
        self,
        request: WorkerRequest,
        *,
        active_version: Version,
        worker_generation: int,
    ) -> QueueSubmission:
        decision = self._pending.submit(
            request,
            active_version=active_version,
            worker_generation=worker_generation,
            has_in_flight=self.has_in_flight,
        )
        self._deadlines.record(request, decision.discarded)
        return decision
    def begin_next(self, *, worker_generation: int) -> WorkerRequest | None:
        if self._in_flight:
            return None
        request = self._pending.pop_next()
        if request:
            self._in_flight = _InFlight(
                request,
                worker_generation,
                self._deadlines.take(request),
            )
        return request
    def arm_timeout(
        self,
        request: WorkerRequest,
        callback: Callable[[int, int], None],
    ) -> None:
        current = self._in_flight
        if not current or current.request is not request or not request.timeout_ms:
            return
        remaining = max(
            0.0,
            (current.deadline or time.monotonic()) - time.monotonic(),
        )
        current.timer = threading.Timer(
            remaining,
            callback,
            args=(current.generation, request.request_sequence),
        )
        current.timer.daemon = True
        current.timer.start()
    def complete(
        self,
        result: WorkerResult,
        *,
        active_version: Version,
        allow_state_revision_mismatch: bool = False,
    ) -> WorkerResult | None:
        current = self._in_flight
        expected = None if not current else (
            current.generation,
            current.request.request_sequence,
        )
        if expected != (result.worker_generation, result.request_sequence):
            return None
        assert current is not None
        self._clear_in_flight()
        result_version = item_version(result)
        if result_version == active_version or (
            allow_state_revision_mismatch
            and result_version[:2] == active_version[:2]
            and result_version[2] < active_version[2]
        ):
            return result
        return WorkerResult.terminal(
            current.request,
            status=WorkerResultStatus.REJECTED,
            worker_generation=result.worker_generation,
            worker_pid=result.worker_pid,
            started_processing_ms=result.started_processing_ms,
            finished_processing_ms=_processing_ms(),
            failure=WorkerFailure(
                "stale_result",
                "VersionMismatch",
                "worker result does not belong to the active version",
            ),
        )
    def fail_dispatch(
        self,
        *,
        worker_generation: int,
        request_sequence: int,
        message: str,
    ) -> WorkerResult | None:
        current = self._in_flight
        if not current or (
            current.generation,
            current.request.request_sequence,
        ) != (worker_generation, request_sequence):
            return None
        current = self._take_in_flight()
        if not current:
            return None
        return local_result(
            current.request,
            WorkerResultStatus.ERROR,
            "request_serialization_failed",
            worker_generation=worker_generation,
            message=message,
        )
    def expire(
        self,
        *,
        worker_generation: int,
        request_sequence: int,
        worker_pid: int | None,
    ) -> WorkerResult | None:
        current = self._in_flight
        if not current or (
            current.generation,
            current.request.request_sequence,
        ) != (worker_generation, request_sequence):
            return None
        self._clear_in_flight(cancel_timer=False)
        return local_result(
            current.request,
            WorkerResultStatus.TIMEOUT,
            "worker_timeout",
            worker_generation=worker_generation,
            worker_pid=worker_pid,
        )
    def worker_failed(
        self,
        *,
        worker_generation: int,
        worker_pid: int | None,
        code: str,
    ) -> WorkerResult | None:
        current = self._take_in_flight()
        if not current:
            return None
        return local_result(
            current.request,
            WorkerResultStatus.CRASHED,
            code,
            worker_generation=worker_generation,
            worker_pid=worker_pid,
        )

    def rebind(
        self, *, reason: str, worker_generation: int,
        preserve_in_flight: bool = False,
    ) -> tuple[WorkerResult, ...]:
        results = list(self._pending.rebind(
            reason=reason,
            worker_generation=worker_generation,
        ))
        self._deadlines.clear()
        if preserve_in_flight:
            return tuple(results)
        current = self._take_in_flight()
        if current:
            results.append(local_result(
                current.request,
                WorkerResultStatus.REJECTED,
                "stale_result",
                worker_generation=worker_generation,
            ))
        return tuple(results)

    def discard_all(
        self,
        *,
        reason: str,
        worker_generation: int,
    ) -> tuple[WorkerResult, ...]:
        results = list(self._pending.discard_all(
            reason=reason,
            worker_generation=worker_generation,
        ))
        self._deadlines.clear()
        current = self._take_in_flight()
        if current:
            results.append(local_result(
                current.request,
                WorkerResultStatus.DROPPED,
                reason,
                worker_generation=worker_generation,
            ))
        return tuple(results)

    def _take_in_flight(self) -> _InFlight | None:
        current = self._in_flight
        self._clear_in_flight()
        return current

    def _clear_in_flight(self, *, cancel_timer: bool = True) -> None:
        current, self._in_flight = self._in_flight, None
        if cancel_timer and current and current.timer:
            current.timer.cancel()


__all__ = [
    "BoundedWorkerQueue",
    "QueueSubmission",
    "Version",
    "WorkerRequestCoordinator",
    "item_version",
    "local_result",
]
