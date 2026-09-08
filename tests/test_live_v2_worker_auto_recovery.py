from __future__ import annotations

import os
import threading

from daguandan_bridge.application.live_v2_advice_protocol import (
    AdvicePrewarmPayload, AdviceWorkerConfig, AdviceWorkerPayload,
    AdviceWorkerSuccess, AdvisorReady,
)
from daguandan_bridge.application.live_v2_advice_runtime import LiveV2AdviceRuntime
from daguandan_bridge.application.live_v2_vision_protocol import VisionWorkerConfig
from daguandan_bridge.application.live_v2_vision_runtime import LiveV2VisionRuntime
from daguandan_bridge.application.live_v2_worker_protocol import (
    WorkerReference, WorkerRequest,
)
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.infrastructure.live_v2_worker_host import LiveV2WorkerHost
from daguandan_bridge.live_v2.game_state import SeatCardCount, TrustedGameSnapshot
from daguandan_bridge.live_v2.identity import Seat, VersionIdentity
from daguandan_bridge.live_v2.results import (
    AdviceOpportunity, OpportunityReason, OpportunityStatus,
)


def _recovering_advice_worker(
    request: WorkerRequest,
) -> AdvisorReady | AdviceWorkerSuccess:
    payload = request.payload
    if isinstance(payload, AdvicePrewarmPayload):
        return AdvisorReady(
            payload.config.advisor_backend,
            "test",
            "recovery-test-model",
            payload.worker_generation,
            os.getpid(),
            0.1,
            False,
        )
    assert isinstance(payload, AdviceWorkerPayload)
    name = payload.opportunity.opportunity_id
    if name == "crash":
        os._exit(43)
    if name == "timeout":
        threading.Event().wait(5)
    return AdviceWorkerSuccess(
        name,
        payload.snapshot.version,
        AdviceResult(
            "recovery-test", (payload.snapshot.my_hand[0],), "Single", False,
            payload.snapshot.version.state_revision, 1.0, payload.request_id,
        ),
        len(payload.snapshot.play_history),
        1.0,
    )


def _snapshot() -> TrustedGameSnapshot:
    version = VersionIdentity("recovery-advice", 1, 0, 0, 0)
    hand = tuple(f"{rank}{suit}" for rank in "3456789" for suit in "HDSC")[:27]
    return TrustedGameSnapshot(
        version, "6", "6", 1, Seat.SELF, Seat.SELF, hand, (), (),
        tuple(SeatCardCount(seat, 27) for seat in Seat), (), True, False, 100,
    )


def _opportunity(snapshot: TrustedGameSnapshot, name: str) -> AdviceOpportunity:
    return AdviceOpportunity(
        name, snapshot.version, Seat.SELF, OpportunityStatus.READY,
        OpportunityReason.TRUSTED_STATE, 100, 110,
    )


def _runtime():
    hosts = []

    def factory(**version):
        host = LiveV2WorkerHost(
            WorkerReference.from_callable(_recovering_advice_worker), **version,
            restart_backoff_seconds=0,
        )
        hosts.append(host)
        return host

    snapshot = _snapshot()
    runtime = LiveV2AdviceRuntime(
        AdviceWorkerConfig(
            "profiles", "fabledan", fabledan_runtime_policy="rule_only"
        ),
        session_id=snapshot.version.session_id,
        capture_generation=snapshot.version.capture_generation,
        state_revision=snapshot.version.state_revision,
        host_factory=factory,
    )
    runtime.start()
    return runtime, snapshot, hosts[0]


def test_advice_crash_recovers_on_next_opportunity_and_closes_process() -> None:
    runtime, snapshot, host = _runtime()
    old_pid, old_generation = runtime.worker_pid, runtime.worker_generation
    runtime.submit(snapshot, _opportunity(snapshot, "crash"), request_sequence=1)
    crashed = runtime.get_result(timeout=10)
    assert crashed.status.value == "worker_crashed"
    runtime.submit(snapshot, _opportunity(snapshot, "after-crash"), request_sequence=2)
    recovered = runtime.get_result(timeout=10)
    assert recovered.status.value == "advice"
    assert recovered.worker_generation == old_generation + 1
    assert recovered.worker_pid != old_pid
    runtime.close()
    assert not host.is_worker_alive and host.worker_pid is None


def test_advice_timeout_recovers_and_late_old_result_never_returns() -> None:
    runtime, snapshot, host = _runtime()
    old_pid = runtime.worker_pid
    old_generation = runtime.worker_generation
    runtime.submit(
        snapshot, _opportunity(snapshot, "timeout"),
        request_sequence=1, timeout_ms=50,
    )
    timed_out = runtime.get_result(timeout=10)
    assert timed_out.status.value == "worker_timeout"
    ready = runtime.advisor_ready
    assert ready is not None
    assert ready.worker_generation == old_generation + 1
    assert ready.worker_pid != old_pid
    assert ready.worker_pid == runtime.worker_pid
    runtime.submit(snapshot, _opportunity(snapshot, "after-timeout"), request_sequence=2)
    recovered = runtime.get_result(timeout=10)
    assert recovered.status.value == "advice"
    assert recovered.worker_generation == ready.worker_generation
    assert recovered.worker_pid == ready.worker_pid
    assert not runtime.drain_results()
    runtime.close()
    assert not host.is_worker_alive


def test_vision_start_uses_one_fresh_process_retry(monkeypatch) -> None:
    host = LiveV2WorkerHost(
        WorkerReference.from_callable(_recovering_advice_worker),
        session_id="vision-start", capture_generation=1, state_revision=0,
    )
    real_start, calls = host.start, 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected first startup failure")
        return real_start(**kwargs)

    monkeypatch.setattr(host, "start", fail_once)
    runtime = LiveV2VisionRuntime(
        VisionWorkerConfig("profiles"), host=host,
        session_id="vision-start", capture_generation=1,
    )
    runtime.start(timeout=10)
    assert calls == 2 and runtime.worker_generation == 1 and host.is_worker_alive
    runtime.close()
    assert not host.is_worker_alive


def test_advice_start_uses_one_fresh_process_retry(monkeypatch) -> None:
    host = LiveV2WorkerHost(
        WorkerReference.from_callable(_recovering_advice_worker),
        session_id="advice-start", capture_generation=1, state_revision=0,
    )
    real_start, calls = host.start, 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected first startup failure")
        return real_start(**kwargs)

    monkeypatch.setattr(host, "start", fail_once)
    runtime = LiveV2AdviceRuntime(
        AdviceWorkerConfig(
            "profiles", "fabledan", fabledan_runtime_policy="rule_only"
        ),
        session_id="advice-start", capture_generation=1, state_revision=0,
        host_factory=lambda **_version: host,
    )
    runtime.start(timeout=10)
    assert calls == 2 and runtime.worker_generation == 1 and host.is_worker_alive
    runtime.close()
    assert not host.is_worker_alive
