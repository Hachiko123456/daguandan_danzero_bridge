from __future__ import annotations

from dataclasses import FrozenInstanceError
import multiprocessing
import os
from pathlib import Path
import threading
import time

import pytest

from daguandan_bridge.application.live_v2_worker_protocol import (
    DeliveryMode,
    WorkerReference,
    WorkerRequest,
    WorkerResultStatus,
)
from daguandan_bridge.infrastructure.live_v2_worker_host import (
    LiveV2WorkerHost,
    RebindPolicy,
    WorkerHostState,
    WorkerStartupError,
)
from daguandan_bridge.infrastructure.live_v2_worker_recovery import WorkerRestartCircuit


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance_to(self, seconds: float) -> None:
        self.value = seconds


def _spawn_test_worker(request: WorkerRequest) -> object:
    payload = request.payload
    if isinstance(payload, dict):
        wait_ms = int(payload.get("wait_ms", 0))
        if wait_ms:
            threading.Event().wait(wait_ms / 1000)
        operation = payload.get("operation")
        if operation == "crash":
            os._exit(37)
        if operation == "raise":
            raise LookupError("intentional-worker-error")
        if operation == "unpickleable":
            return lambda: None
    return {
        "pid": os.getpid(),
        "value": payload,
        "sequence": request.request_sequence,
    }


def _reference() -> WorkerReference:
    return WorkerReference.from_callable(_spawn_test_worker)


def _request(
    sequence: int,
    payload: object = None,
    *,
    capture_generation: int = 1,
    state_revision: int = 0,
    delivery: DeliveryMode = DeliveryMode.LATEST_ONLY,
    timeout_ms: int | None = None,
) -> WorkerRequest:
    return WorkerRequest(
        session_id="session-a",
        capture_generation=capture_generation,
        request_sequence=sequence,
        state_revision=state_revision,
        submitted_processing_ms=1_000 + sequence,
        payload=payload,
        delivery=delivery,
        timeout_ms=timeout_ms,
    )


def _host(*, fifo_capacity: int = 2, timing_capacity: int = 64) -> LiveV2WorkerHost:
    host = LiveV2WorkerHost(
        _reference(),
        session_id="session-a",
        capture_generation=1,
        state_revision=0,
        fifo_capacity=fifo_capacity,
        timing_capacity=timing_capacity,
    )
    host.start(timeout=10)
    return host


def _results(host: LiveV2WorkerHost, count: int) -> list:
    return [host.get_result(timeout=10) for _ in range(count)]


def test_protocol_is_immutable_and_rejects_closures() -> None:
    request = _request(1)
    with pytest.raises(FrozenInstanceError):
        request.state_revision = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="module-level"):
        WorkerReference.from_callable(lambda request: request)


def test_spawn_worker_reuses_pid_and_latest_only_reports_replacement() -> None:
    host = _host()
    try:
        ready = host.start()
        host.submit(_request(1, {"wait_ms": 250, "value": "running"}))
        host.submit(_request(2, "superseded"))
        immediate = host.submit(_request(3, "latest"))

        assert len(immediate) == 1
        assert immediate[0].request_sequence == 2
        assert immediate[0].status is WorkerResultStatus.DROPPED
        assert immediate[0].failure and immediate[0].failure.code == "latest_replaced"

        results = _results(host, 3)
        successful = [result for result in results if result.is_success]
        assert [result.request_sequence for result in successful] == [1, 3]
        assert {result.worker_pid for result in successful} == {ready.worker_pid}
        assert host.worker_pid == ready.worker_pid
    finally:
        host.close()


def test_persistent_sender_needs_no_per_request_thread_start(monkeypatch) -> None:
    host = _host()
    try:
        def fail_start(_thread):
            raise AssertionError("request dispatch tried to start a new thread")

        monkeypatch.setattr(threading.Thread, "start", fail_start)
        host.submit(_request(1, "persistent-sender"))
        result = host.get_result(timeout=10)
        assert result.is_success
    finally:
        host.close()


def test_request_timing_trace_is_complete_and_bounded() -> None:
    host = _host(timing_capacity=2)
    try:
        for sequence in range(1, 4):
            host.submit(_request(sequence, f"timed-{sequence}"))
            assert host.get_result(timeout=10).is_success

        timings = host.recent_request_timings
        assert [item.request_sequence for item in timings] == [2, 3]
        for item in timings:
            assert all(
                value is not None
                for value in (
                    item.send_started_ms,
                    item.send_finished_ms,
                    item.child_received_ms,
                    item.child_started_ms,
                    item.child_finished_ms,
                    item.result_received_ms,
                )
            )
            assert item.host_accepted_ms <= item.send_started_ms <= item.send_finished_ms
            assert item.child_received_ms <= item.child_started_ms <= item.child_finished_ms
            assert item.child_finished_ms <= item.result_received_ms
    finally:
        host.close()


def test_fifo_is_bounded_preserved_and_prioritized_over_latest() -> None:
    host = _host(fifo_capacity=2)
    try:
        host.submit(_request(1, {"wait_ms": 250}))
        host.submit(_request(2, "event-2", delivery=DeliveryMode.FIFO))
        host.submit(_request(3, "event-3", delivery=DeliveryMode.FIFO))
        rejected = host.submit(_request(4, "overflow", delivery=DeliveryMode.FIFO))
        host.submit(_request(5, "newest-frame"))

        assert rejected[0].failure and rejected[0].failure.code == "fifo_capacity_exceeded"
        results = _results(host, 5)
        successes = [item.request_sequence for item in results if item.is_success]
        assert successes == [1, 2, 3, 5]
        assert host.pending_count == 0
    finally:
        host.close()


def test_worker_exception_and_unpickleable_result_are_structured_and_reusable() -> None:
    host = _host()
    try:
        pid = host.worker_pid
        host.submit(_request(1, {"operation": "raise"}))
        first = host.get_result(timeout=10)
        assert first.status is WorkerResultStatus.ERROR
        assert first.failure and first.failure.error_type == "LookupError"
        assert "intentional-worker-error" in first.failure.message
        assert "_spawn_test_worker" in first.failure.traceback_text

        host.submit(_request(2, {"operation": "unpickleable"}))
        second = host.get_result(timeout=10)
        assert second.status is WorkerResultStatus.ERROR
        assert second.failure and second.failure.code == "result_serialization_failed"

        host.submit(_request(3, "still-alive"))
        third = host.get_result(timeout=10)
        assert third.is_success and third.worker_pid == pid
    finally:
        host.close()


def test_blocked_pipe_send_is_timed_out_from_submit_and_worker_recovers(
    monkeypatch,
) -> None:
    host = _host()
    endpoint = host._endpoint
    assert endpoint is not None
    release = threading.Event()
    real_send = endpoint.send

    def blocked_send(value):
        release.wait(5)
        return real_send(value)

    monkeypatch.setattr(endpoint, "send", blocked_send)
    started = time.monotonic()
    host.submit(_request(20, "blocked-send", timeout_ms=100))
    submit_elapsed = time.monotonic() - started
    timed_out = host.get_result(timeout=2)

    assert submit_elapsed < 0.25
    assert timed_out.status is WorkerResultStatus.TIMEOUT
    assert timed_out.failure and timed_out.failure.code == "worker_timeout"
    assert host.state is WorkerHostState.BROKEN

    release.set()
    host.submit(_request(21, "after-timeout", timeout_ms=2_000))
    assert host.get_result(timeout=10).is_success
    host.close()
    assert not host.is_worker_alive


def test_host_rejects_old_capture_generation_result_but_keeps_observing() -> None:
    host = _host()
    try:
        pid = host.worker_pid
        host.submit(_request(1, {"wait_ms": 200}))
        host.submit(_request(2, "old-pending"))
        host.bind_version(
            session_id="session-a", capture_generation=2, state_revision=0
        )

        first_two = _results(host, 2)
        statuses = {item.request_sequence: item.status for item in first_two}
        assert statuses == {
            1: WorkerResultStatus.REJECTED,
            2: WorkerResultStatus.DROPPED,
        }
        stale = next(item for item in first_two if item.request_sequence == 1)
        assert stale.failure and stale.failure.code == "stale_result"

        host.submit(_request(1, "new-version", capture_generation=2))
        current = host.get_result(timeout=10)
        assert current.is_success and current.worker_pid != pid
    finally:
        host.close()


def test_vision_policy_preserves_inflight_and_pid_across_state_revision() -> None:
    host = LiveV2WorkerHost(
        _reference(), session_id="session-a", capture_generation=1,
        state_revision=0,
        rebind_policy=RebindPolicy.PRESERVE_IN_FLIGHT_WITHIN_STREAM,
    )
    host.start(timeout=10)
    try:
        pid = host.worker_pid
        host.submit(_request(1, {"wait_ms": 200}, state_revision=0))
        host.submit(_request(2, "old-pending", state_revision=0))
        host.bind_version(
            session_id="session-a", capture_generation=1, state_revision=1,
        )
        dropped = host.get_result(timeout=2)
        assert dropped.request_sequence == 2
        assert dropped.status is WorkerResultStatus.DROPPED

        old = host.get_result(timeout=10)
        assert old.request_sequence == 1 and old.is_success
        assert old.worker_pid == pid == host.worker_pid

        host.submit(_request(3, "current", state_revision=1))
        current = host.get_result(timeout=10)
        assert current.is_success and current.worker_pid == pid
    finally:
        host.close()


def test_advice_policy_terminates_inflight_on_state_revision() -> None:
    host = _host()
    try:
        pid = host.worker_pid
        host.submit(_request(1, {"wait_ms": 200}, state_revision=0))
        host.bind_version(
            session_id="session-a", capture_generation=1, state_revision=1,
        )
        stale = host.get_result(timeout=2)
        assert stale.request_sequence == 1
        assert stale.status is WorkerResultStatus.REJECTED
        assert stale.failure and stale.failure.code == "stale_result"

        host.submit(_request(2, "current", state_revision=1))
        current = host.get_result(timeout=10)
        assert current.is_success and current.worker_pid != pid
    finally:
        host.close()


def test_vision_policy_does_not_accept_future_revision_after_backward_bind() -> None:
    host = LiveV2WorkerHost(
        _reference(), session_id="session-a", capture_generation=1,
        state_revision=2,
        rebind_policy=RebindPolicy.PRESERVE_IN_FLIGHT_WITHIN_STREAM,
    )
    host.start(timeout=10)
    try:
        host.submit(_request(1, {"wait_ms": 100}, state_revision=2))
        host.bind_version(
            session_id="session-a", capture_generation=1, state_revision=1,
        )
        result = host.get_result(timeout=10)
        assert result.status is WorkerResultStatus.REJECTED
        assert result.failure and result.failure.code == "stale_result"
    finally:
        host.close()


def test_processing_timeout_breaks_worker_and_publishes_terminal_result() -> None:
    host = _host()
    try:
        host.submit(_request(1, {"wait_ms": 5_000}, timeout_ms=100))
        result = host.get_result(timeout=5)
        assert result.status is WorkerResultStatus.TIMEOUT
        assert result.failure and result.failure.code == "worker_timeout"
        assert host.state is WorkerHostState.BROKEN
        old_generation = host.worker_generation
        host.submit(_request(2, "after-timeout"))
        recovered = host.get_result(timeout=10)
        assert recovered.is_success
        assert recovered.worker_generation == old_generation + 1
    finally:
        host.close(timeout=2)
    assert not host.is_worker_alive


def test_abnormal_exit_recovers_on_next_legal_request() -> None:
    host = _host()
    try:
        old_pid = host.worker_pid
        old_generation = host.worker_generation
        host.submit(_request(1, {"operation": "crash"}))
        crashed = host.get_result(timeout=10)
        assert crashed.status is WorkerResultStatus.CRASHED
        assert crashed.failure and crashed.failure.code == "worker_exited_37"
        assert host.state is WorkerHostState.BROKEN

        host.submit(_request(2, "after-restart"))
        recovered = host.get_result(timeout=10)
        assert recovered.is_success
        assert recovered.worker_generation == old_generation + 1
        assert recovered.worker_pid != old_pid
    finally:
        host.close()
    assert host.state is WorkerHostState.CLOSED
    assert not host.is_worker_alive


def test_restart_circuit_has_backoff_budget_and_eventual_retry() -> None:
    circuit = WorkerRestartCircuit(
        limit=3, window_seconds=30, base_backoff_seconds=0.25
    )
    assert circuit.begin(0.0) == ""
    circuit.failed(0.0)
    assert circuit.begin(0.1) == "restart_cooldown"
    assert circuit.begin(0.25) == ""
    circuit.failed(0.25)
    assert circuit.begin(0.5) == "restart_cooldown"
    assert circuit.begin(0.75) == ""
    circuit.failed(0.75)
    assert circuit.begin(1.75) == "restart_circuit_open"
    assert circuit.begin(31.0) == ""


def test_failed_auto_restart_is_structured_and_retries_after_cooldown(monkeypatch) -> None:
    clock = ManualClock()
    host = LiveV2WorkerHost(
        _reference(), session_id="session-a", capture_generation=1,
        state_revision=0, restart_backoff_seconds=0.05,
        monotonic_clock=clock,
    )
    host.start(timeout=10)
    host.submit(_request(1, {"operation": "crash"}))
    assert host.get_result(timeout=10).status is WorkerResultStatus.CRASHED
    real_restart = host.restart
    attempts = 0

    def flaky_restart(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise WorkerStartupError("injected restart failure")
        return real_restart(**kwargs)

    monkeypatch.setattr(host, "restart", flaky_restart)
    with pytest.raises(RuntimeError, match="worker_auto_restart_failed"):
        host.submit(_request(2, "first-retry"))
    with pytest.raises(RuntimeError, match="restart_cooldown"):
        host.submit(_request(3, "cooldown"))
    clock.advance_to(0.049)
    with pytest.raises(RuntimeError, match="restart_cooldown"):
        host.submit(_request(4, "before-boundary"))
    clock.advance_to(0.050)
    host.submit(_request(5, "at-boundary"))
    assert host.get_result(timeout=10).is_success
    host.close()
    assert not host.is_worker_alive


def test_close_after_crash_never_auto_restarts_or_leaves_orphan() -> None:
    host = _host()
    host.submit(_request(1, {"operation": "crash"}))
    assert host.get_result(timeout=10).status is WorkerResultStatus.CRASHED
    generation = host.worker_generation
    host.close(timeout=2)
    with pytest.raises(RuntimeError, match="not ready"):
        host.submit(_request(2, "must-not-run"))
    assert host.worker_generation == generation
    assert host.worker_pid is None and not host.is_worker_alive


def test_startup_failure_and_close_are_deterministic_and_idempotent() -> None:
    host = LiveV2WorkerHost(
        WorkerReference(__name__, "missing_worker_function"),
        session_id="session-a",
        capture_generation=1,
        state_revision=0,
    )
    with pytest.raises(WorkerStartupError, match="not callable"):
        host.start(timeout=10)
    host.close(timeout=2)
    host.close(timeout=2)
    assert host.state is WorkerHostState.CLOSED
    assert not host.is_worker_alive


def test_worker_host_stays_small_and_has_no_qt_or_game_state_dependency() -> None:
    root = Path(__file__).parents[1]
    paths = (
        root / "src/daguandan_bridge/application/live_v2_worker_protocol.py",
        root / "src/daguandan_bridge/infrastructure/live_v2_worker_process.py",
        root / "src/daguandan_bridge/infrastructure/live_v2_worker_queue.py",
        root / "src/daguandan_bridge/infrastructure/live_v2_worker_recovery.py",
        root / "src/daguandan_bridge/infrastructure/live_v2_worker_host.py",
    )
    assert all(len(path.read_text(encoding="utf-8").splitlines()) < 450 for path in paths)
    production = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "PySide" not in production
    assert "GuanDanState" not in production


def test_close_leaves_no_worker_process_registered() -> None:
    host = _host()
    pid = host.worker_pid
    host.close()
    assert pid not in {child.pid for child in multiprocessing.active_children()}
