"""Public, Qt-free supervisor API for a persistent live-v2 worker."""
from __future__ import annotations
from collections import deque
from dataclasses import replace
from enum import Enum
import threading
import time
from typing import Callable
from ..application.live_v2_worker_protocol import (
    WorkerFailure,
    WorkerReady,
    WorkerReference,
    WorkerRequest,
    WorkerRequestTiming,
    WorkerResult,
    _RunCommand,
    _WorkerTimingEvent,
    _WorkerStartupFailure,
    _WorkerStopped,
)
from .live_v2_worker_process import WorkerProcessEndpoint
from .live_v2_worker_queue import Version, WorkerRequestCoordinator
from .live_v2_worker_cancellation import RebindPolicy, WorkerCancellationMixin
from .live_v2_worker_recovery import WorkerAutoRecoveryMixin, WorkerRestartCircuit
class WorkerHostState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    BROKEN = "broken"
    CLOSING = "closing"
    CLOSED = "closed"
class WorkerHostError(RuntimeError):
    pass
class WorkerStartupError(WorkerHostError):
    pass
class LiveV2WorkerHost(WorkerAutoRecoveryMixin, WorkerCancellationMixin):
    """Supervise lifecycle, version gates and terminal result delivery."""
    def __init__(
        self,
        worker: WorkerReference,
        *,
        session_id: str,
        capture_generation: int,
        state_revision: int,
        rebind_policy: RebindPolicy = RebindPolicy.TERMINATE_IN_FLIGHT,
        fifo_capacity: int = 16,
        restart_limit: int = 3,
        restart_window_seconds: float = 30.0,
        restart_backoff_seconds: float = 0.25,
        timing_capacity: int = 64,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        if min(capture_generation, state_revision) < 0:
            raise ValueError("generation and revision must be non-negative")
        if timing_capacity < 1:
            raise ValueError("timing_capacity must be positive")
        self._worker = worker
        self._session_id = session_id
        self._capture_generation = capture_generation
        self._state_revision = state_revision
        self._rebind_policy = RebindPolicy(rebind_policy)
        self._requests = WorkerRequestCoordinator(fifo_capacity)
        self._condition = threading.Condition(threading.RLock())
        self._results: deque[WorkerResult] = deque()
        self._timing_capacity = timing_capacity
        self._timing_order: deque[tuple[int, int]] = deque()
        self._timings: dict[tuple[int, int], WorkerRequestTiming] = {}
        self._state = WorkerHostState.NEW
        self._generation = 0
        self._ready: WorkerReady | None = None
        self._startup_failure: WorkerFailure | None = None
        self._endpoint: WorkerProcessEndpoint | None = None
        self._recovery = WorkerRestartCircuit(
            restart_limit, restart_window_seconds, restart_backoff_seconds
        )
        self._monotonic_clock = monotonic_clock or time.monotonic
        if not callable(self._monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
    @property
    def state(self) -> WorkerHostState:
        with self._condition:
            return self._state
    @property
    def worker_generation(self) -> int:
        with self._condition:
            return self._generation
    @property
    def worker_pid(self) -> int | None:
        with self._condition:
            return self._ready.worker_pid if self._ready else None
    @property
    def is_worker_alive(self) -> bool:
        with self._condition:
            return bool(self._endpoint and self._endpoint.is_alive)
    @property
    def pending_count(self) -> int:
        with self._condition:
            return self._requests.pending_count
    @property
    def recent_request_timings(self) -> tuple[WorkerRequestTiming, ...]:
        with self._condition:
            return tuple(
                self._timings[key]
                for key in self._timing_order
                if key in self._timings
            )
    @property
    def _version(self) -> Version:
        return self._session_id, self._capture_generation, self._state_revision
    def start(self, *, timeout: float = 10.0) -> WorkerReady:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._condition:
            if self._state is WorkerHostState.CLOSED:
                raise WorkerHostError("closed worker host cannot be started")
            if self._state is WorkerHostState.READY and self._ready:
                return self._ready
            if self._state is not WorkerHostState.NEW:
                raise WorkerHostError(f"cannot start worker from {self._state.value}")
            self._spawn_locked()
            deadline = time.monotonic() + timeout
            while self._state is WorkerHostState.STARTING:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._state is WorkerHostState.READY and self._ready:
                return self._ready
            startup_failure = self._startup_failure
        self._stop_current(timeout=2.0, reason="startup_failed")
        with self._condition:
            if self._state is not WorkerHostState.CLOSED:
                self._state = WorkerHostState.BROKEN
                self._condition.notify_all()
        if startup_failure:
            raise WorkerStartupError(
                f"{startup_failure.error_type}: {startup_failure.message}"
            )
        raise WorkerStartupError("worker did not become ready before timeout")
    def bind_version(
        self,
        *,
        session_id: str,
        capture_generation: int,
        state_revision: int,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        if min(capture_generation, state_revision) < 0:
            raise ValueError("generation and revision must be non-negative")
        self._rebind_worker_version(
            (session_id, capture_generation, state_revision)
        )
    def submit(self, request: WorkerRequest) -> tuple[WorkerResult, ...]:
        """Submit work and return immediate bounded-queue drop results."""
        if self.state is WorkerHostState.BROKEN:
            self._recover_for_submit(request)
        with self._condition:
            if self._state is not WorkerHostState.READY:
                raise WorkerHostError("worker is not ready")
            decision = self._requests.submit(
                request,
                active_version=self._version,
                worker_generation=self._generation,
            )
            rejected = any(
                result.request_sequence == request.request_sequence
                and result.session_id == request.session_id
                for result in decision.discarded
            )
            if not rejected:
                self._accept_timing_locked(request)
            self._publish_many_locked(decision.discarded)
            self._dispatch_next_locked()
            return decision.discarded
    def get_result(self, *, timeout: float | None = None) -> WorkerResult:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._condition:
            ready = self._results or self._condition.wait_for(
                lambda: bool(self._results), timeout
            )
            if not ready:
                raise TimeoutError("no worker result became available")
            return self._results.popleft()
    def drain_results(self) -> tuple[WorkerResult, ...]:
        with self._condition:
            values = tuple(self._results)
            self._results.clear()
            return values
    def restart(
        self,
        *,
        timeout: float = 10.0,
        session_id: str | None = None,
        capture_generation: int | None = None,
        state_revision: int | None = None,
    ) -> WorkerReady:
        with self._condition:
            if self._state is WorkerHostState.CLOSED:
                raise WorkerHostError("closed worker host cannot be restarted")
            version = (
                self._session_id if session_id is None else session_id,
                self._capture_generation if capture_generation is None else capture_generation,
                self._state_revision if state_revision is None else state_revision,
            )
        self.bind_version(
            session_id=version[0],
            capture_generation=version[1],
            state_revision=version[2],
        )
        self._stop_current(timeout=2.0, reason="worker_restarted")
        with self._condition:
            self._state = WorkerHostState.NEW
        return self.start(timeout=timeout)
    def close(self, *, timeout: float = 5.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._condition:
            if self._state is WorkerHostState.CLOSED:
                return
            self._state = WorkerHostState.CLOSING
            self._publish_many_locked(self._requests.discard_all(
                reason="host_closed",
                worker_generation=self._generation,
            ))
            endpoint = self._endpoint
        if endpoint:
            endpoint.shutdown(timeout=timeout, graceful=True)
        with self._condition:
            self._endpoint = None
            self._ready = None
            self._state = WorkerHostState.CLOSED
            self._condition.notify_all()
    def _spawn_locked(self) -> None:
        self._generation += 1
        self._state = WorkerHostState.STARTING
        self._startup_failure = self._ready = None
        self._endpoint = WorkerProcessEndpoint(
            self._worker,
            self._generation,
            self._on_message,
            self._on_exit,
        )
    def _on_message(self, generation: int, message: object) -> None:
        with self._condition:
            if generation != self._generation:
                return
            if isinstance(message, WorkerReady):
                self._ready, self._state = message, WorkerHostState.READY
                self._condition.notify_all()
                self._dispatch_next_locked()
            elif isinstance(message, _WorkerStartupFailure):
                self._startup_failure = message.failure
                self._state = WorkerHostState.BROKEN
                self._condition.notify_all()
            elif isinstance(message, WorkerResult):
                self._mark_timing_locked(
                    generation, message.request_sequence,
                    "result_received_ms", _processing_ms(),
                )
                self._handle_result_locked(message)
            elif isinstance(message, _WorkerTimingEvent):
                field = {
                    "child_received": "child_received_ms",
                    "child_started": "child_started_ms",
                    "child_finished": "child_finished_ms",
                }.get(message.stage)
                if field:
                    self._mark_timing_locked(
                        generation, message.request_sequence,
                        field, message.processing_ms,
                    )
            elif isinstance(message, _WorkerStopped):
                self._condition.notify_all()
    def _on_exit(self, generation: int, pid: int, exitcode: int | None) -> None:
        with self._condition:
            if generation != self._generation:
                return
            if self._state in {WorkerHostState.CLOSING, WorkerHostState.CLOSED}:
                self._condition.notify_all()
                return
            self._state = WorkerHostState.BROKEN
            failed = self._requests.worker_failed(
                worker_generation=self._generation,
                worker_pid=pid,
                code=f"worker_exited_{exitcode}",
            )
            if failed:
                self._publish_locked(failed)
            self._publish_many_locked(self._requests.discard_all(
                reason="worker_unavailable",
                worker_generation=self._generation,
            ))
            self._condition.notify_all()
    def _dispatch_next_locked(self) -> None:
        if self._state is not WorkerHostState.READY:
            return
        request = self._requests.begin_next(worker_generation=self._generation)
        if request is None:
            return
        endpoint = self._endpoint
        if endpoint is None:
            failed = self._requests.worker_failed(
                worker_generation=self._generation,
                worker_pid=None,
                code="worker_pipe_unavailable",
            )
            if failed:
                self._publish_locked(failed)
            return
        self._requests.arm_timeout(request, self._request_timed_out)
        generation = self._generation
        endpoint.dispatch(
            _RunCommand(generation, request),
            on_start=lambda: self._mark_timing(
                generation, request.request_sequence, "send_started_ms"
            ),
            on_success=lambda: self._mark_timing(
                generation, request.request_sequence, "send_finished_ms"
            ),
            on_failure=lambda message: self._dispatch_failed(
                generation, request.request_sequence, message
            ),
        )
    def _dispatch_failed(
        self, generation: int, request_sequence: int, message: str
    ) -> None:
        with self._condition:
            failed = self._requests.fail_dispatch(
                worker_generation=generation,
                request_sequence=request_sequence,
                message=message,
            )
            if failed:
                self._publish_locked(failed)
                self._dispatch_next_locked()
    def _handle_result_locked(self, result: WorkerResult) -> None:
        accepted = self._requests.complete(
            result, active_version=self._version,
            allow_state_revision_mismatch=self._rebind_policy
            is RebindPolicy.PRESERVE_IN_FLIGHT_WITHIN_STREAM,
        )
        if accepted is None:
            return
        self._publish_locked(accepted)
        self._dispatch_next_locked()
    def _request_timed_out(self, generation: int, sequence: int) -> None:
        with self._condition:
            endpoint = self._endpoint
            result = self._requests.expire(
                worker_generation=generation,
                request_sequence=sequence,
                worker_pid=endpoint.pid if endpoint else None,
            )
            if result is None:
                return
            self._publish_locked(result)
            self._state = WorkerHostState.BROKEN
            self._publish_many_locked(self._requests.discard_all(
                reason="worker_unavailable_after_timeout",
                worker_generation=self._generation,
            ))
        if endpoint:
            endpoint.terminate()
    def _publish_locked(self, result: WorkerResult) -> None:
        self._results.append(result)
        self._condition.notify_all()
    def _publish_many_locked(self, results: tuple[WorkerResult, ...]) -> None:
        for result in results:
            self._publish_locked(result)
    def _accept_timing_locked(self, request: WorkerRequest) -> None:
        key = self._generation, request.request_sequence
        if key not in self._timings:
            while len(self._timing_order) >= self._timing_capacity:
                self._timings.pop(self._timing_order.popleft(), None)
            self._timing_order.append(key)
        self._timings[key] = WorkerRequestTiming(
            self._generation, request.request_sequence, _processing_ms()
        )
    def _mark_timing(
        self, generation: int, sequence: int, field: str
    ) -> None:
        with self._condition:
            self._mark_timing_locked(
                generation, sequence, field, _processing_ms()
            )
    def _mark_timing_locked(
        self, generation: int, sequence: int, field: str, value: int
    ) -> None:
        key = generation, sequence
        timing = self._timings.get(key)
        if timing is not None and getattr(timing, field) is None:
            self._timings[key] = replace(timing, **{field: value})
    def _stop_current(self, *, timeout: float, reason: str) -> None:
        with self._condition:
            self._state = WorkerHostState.CLOSING
            self._publish_many_locked(self._requests.discard_all(
                reason=reason,
                worker_generation=self._generation,
            ))
            endpoint = self._endpoint
        if endpoint:
            endpoint.shutdown(timeout=timeout, graceful=False)
        with self._condition:
            self._endpoint = None
            self._ready = None
            self._condition.notify_all()
__all__ = ["LiveV2WorkerHost", "RebindPolicy",
    "WorkerHostError",
    "WorkerHostState",
    "WorkerStartupError",
]


def _processing_ms() -> int:
    return time.monotonic_ns() // 1_000_000
