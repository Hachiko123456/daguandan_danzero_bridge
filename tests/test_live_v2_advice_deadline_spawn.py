from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import threading
import time

from daguandan_bridge.application.live_v2_advice_protocol import (
    AdvicePrewarmPayload,
    AdviceRequestIdentity,
    AdviceRequestTiming,
    AdviceRuntimeResult,
    AdviceRuntimeStatus,
    AdviceWorkerConfig,
    AdviceWorkerPayload,
    AdviceWorkerSuccess,
    AdvisorReady,
)
from daguandan_bridge.application.live_v2_advice_pump import LiveV2AdvicePump
from daguandan_bridge.application.live_v2_advice_runtime import LiveV2AdviceRuntime
from daguandan_bridge.application.live_v2_worker_protocol import (
    WorkerReference,
    WorkerRequest,
    WorkerRequestTiming,
)
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.infrastructure.live_v2_advice_service_factory import (
    create_live_v2_advice_runtime,
)
from daguandan_bridge.infrastructure.live_v2_worker_host import LiveV2WorkerHost
from daguandan_bridge.live_v2.events import ActionKind
from daguandan_bridge.live_v2.game_state import (
    GameAction,
    SeatCardCount,
    TrustedGameSnapshot,
)
from daguandan_bridge.live_v2.identity import (
    FrameIdentity,
    Seat,
    StateVersion,
    VersionIdentity,
)
from daguandan_bridge.live_v2.results import (
    AdviceOpportunity,
    OpportunityReason,
    OpportunityStatus,
)


PROJECT_ROOT = Path(__file__).parents[1]
HAND = (
    "small_joker", "6H", "6C", "AS", "AH", "KS", "KC", "QH", "QC",
    "JH", "JC", "JC", "JD", "10S", "9S", "9H", "9H", "9C", "8H",
    "8C", "8D", "7D", "5C", "5D", "4H", "4H", "3C",
)


class _AuditStore:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def append_advice(self, record): self.records.append(dict(record))


class _CapturingRuntime:
    worker_generation = 1
    worker_pid = 123

    def __init__(self) -> None:
        self.timeouts: list[int] = []

    def start(self, *, timeout=10.0): pass
    def submit(self, snapshot, opportunity, *, request_sequence, timeout_ms=3_000):
        self.timeouts.append(timeout_ms)
        return ()
    def drain_results(self): return ()
    def close(self, *, timeout=5.0): pass


class _ManualClock:
    def __init__(self, value: int = 0) -> None:
        self._value = value
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._value

    def advance(self, milliseconds: int) -> None:
        with self._lock:
            self._value += milliseconds


class _SilentRuntime:
    worker_generation = 7
    worker_pid = 707

    def __init__(self) -> None:
        self.identity = None
        self.cancelled = threading.Event()
        self.results = []

    def start(self, *, timeout=10.0): pass
    def submit(self, snapshot, opportunity, *, request_sequence, timeout_ms=3_000):
        self.identity = AdviceRequestIdentity(
            opportunity.version, request_sequence, opportunity.opportunity_id
        )
        return ()
    def drain_results(self):
        results, self.results = tuple(self.results), []
        return results
    def cancel(self, identity, *, reason):
        assert identity == self.identity
        assert reason == "opportunity_deadline_expired"
        self.cancelled.set()
        return ()
    def close(self, *, timeout=5.0): pass


class _CancelClearsTimingRuntime:
    worker_generation = 1
    worker_pid = 707

    def __init__(self) -> None:
        self.timing = None
        self.cancelled = False

    def start(self, *, timeout=10.0): pass

    def submit(self, snapshot, opportunity, *, request_sequence, timeout_ms=3_000):
        identity = AdviceRequestIdentity(
            opportunity.version, request_sequence, opportunity.opportunity_id
        )
        worker = WorkerRequestTiming(
            1, 7, 100, send_started_ms=101, send_finished_ms=102,
        )
        self.timing = AdviceRequestTiming(
            identity, 1, 7, worker,
            runtime_submit_entered_ms=90,
            runtime_lock_acquired_ms=91,
            version_bound_ms=92,
            worker_request_built_ms=93,
            host_submit_entered_ms=94,
            host_submit_returned_ms=95,
        )
        return ()

    def timing_for_identity(self, identity):
        return self.timing if self.timing and self.timing.identity == identity else None

    def drain_results(self): return ()

    def cancel(self, identity, *, reason):
        assert reason == "opportunity_deadline_expired"
        assert self.timing and self.timing.identity == identity
        self.cancelled = True
        self.timing = None
        return ()

    def close(self, *, timeout=5.0): pass


class _DeadTimer:
    daemon = True

    def __init__(self, *args, **kwargs): pass
    def start(self): pass
    def cancel(self): pass


def _exact_snapshot(session_id: str = "exact-model-required") -> TrustedGameSnapshot:
    before = StateVersion(session_id, 2, 0)
    after = StateVersion(session_id, 3, 1)
    first = FrameIdentity(session_id, 1, 83, 1_000, "roi", "source-video")
    last = FrameIdentity(session_id, 1, 86, 1_100, "roi", "source-video")
    action = GameAction(
        "truth-left-pair",
        before,
        after,
        Seat.LEFT,
        ActionKind.PLAY,
        ("5H", "5C"),
        (("5H",), ("5C",)),
        1,
        ("source-frame-83-left", "source-frame-86-left"),
        first,
        last,
        0.99,
        1_100,
    )
    return TrustedGameSnapshot(
        version=VersionIdentity.from_state(
            after, capture_generation=1, update_sequence=3
        ),
        round_level="6",
        wild_rank="6",
        trick_index=1,
        current_seat=Seat.SELF,
        lead_seat=Seat.LEFT,
        my_hand=HAND,
        play_history=(action,),
        current_trick=(action,),
        remaining=tuple(
            SeatCardCount(seat, 25 if seat is Seat.LEFT else 27)
            for seat in Seat
        ),
        finished=(),
        trusted=True,
        terminal=False,
        captured_ms=1_100,
    )


def _opportunity(snapshot: TrustedGameSnapshot, name: str) -> AdviceOpportunity:
    return AdviceOpportunity(
        name,
        snapshot.version,
        Seat.SELF,
        OpportunityStatus.READY,
        OpportunityReason.TRUSTED_STATE,
        snapshot.captured_ms,
        snapshot.captured_ms,
    )


def _test_advisor_ready(request: WorkerRequest) -> AdvisorReady | None:
    payload = request.payload
    if not isinstance(payload, AdvicePrewarmPayload):
        return None
    return AdvisorReady(
        payload.config.advisor_backend,
        "test",
        "deadline-test-model",
        payload.worker_generation,
        os.getpid(),
        0.1,
        False,
    )


def _deadline_worker(request: WorkerRequest) -> AdvisorReady | AdviceWorkerSuccess:
    ready = _test_advisor_ready(request)
    if ready is not None:
        return ready
    payload = request.payload
    assert isinstance(payload, AdviceWorkerPayload)
    if payload.opportunity.opportunity_id in {"hang", "slow-state-change"}:
        threading.Event().wait(
            10 if payload.opportunity.opportunity_id == "hang" else 0.2
        )
    return AdviceWorkerSuccess(
        payload.opportunity.opportunity_id,
        payload.snapshot.version,
        AdviceResult(
            "deadline-worker",
            (payload.snapshot.my_hand[0],),
            "Single",
            False,
            payload.snapshot.version.state_revision,
            1.0,
            payload.request_id,
        ),
        len(payload.snapshot.play_history),
        1.0,
    )


def _r3_sequence_worker(request: WorkerRequest) -> AdvisorReady | AdviceWorkerSuccess:
    ready = _test_advisor_ready(request)
    if ready is not None:
        return ready
    payload = request.payload
    assert isinstance(payload, AdviceWorkerPayload)
    if payload.opportunity.opportunity_id in {"turn-2", "turn-10", "turn-14"}:
        threading.Event().wait(10)
    return AdviceWorkerSuccess(
        payload.opportunity.opportunity_id,
        payload.snapshot.version,
        AdviceResult(
            "r3-sequence-worker", (payload.snapshot.my_hand[0],), "Single", False,
            payload.snapshot.version.state_revision, 1.0, payload.request_id,
        ),
        len(payload.snapshot.play_history), 1.0,
    )


def _sequence_snapshot(revision: int, public_turn: int) -> TrustedGameSnapshot:
    version = VersionIdentity(
        "r3-multi-opportunity", 1, revision, revision, public_turn - 1
    )
    return TrustedGameSnapshot(
        version, "6", "6", 1, Seat.SELF, Seat.SELF, HAND, (), (),
        tuple(SeatCardCount(seat, 27) for seat in Seat), (), True, False, 1_000,
    )


def test_real_fabledan_model_required_exact_first_turn_finishes_under_deadline() -> None:
    snapshot = _exact_snapshot()
    runtime = create_live_v2_advice_runtime(
        AdviceWorkerConfig(
            str(PROJECT_ROOT / "data" / "profiles"),
            "fabledan",
            profile_name="tencent_daguandan",
            fabledan_runtime_policy="model_required",
        ),
        snapshot.version,
    )
    delivered = []
    store = _AuditStore()
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=lambda result: delivered.append(result) is None,
        store=store,
        local_hint_window_ms=0,
        request_timeout_ms=3_000,
    )
    try:
        pump.start()
        started = time.monotonic()
        pump.publish(_opportunity(snapshot, "exact-first-self-turn"))
        while not delivered and time.monotonic() - started < 3.5:
            pump.poll()
            time.sleep(0.01)
        elapsed = time.monotonic() - started

        assert delivered and delivered[0].status.value == "advice"
        assert elapsed < 3.0
        assert pump.wait_idle(0.2)
        assert [row["status"] for row in store.records] == [
            "requested", "worker_started", "ready"
        ]
        timing = store.records[-1]["worker_timing"]
        assert timing["worker_request_sequence"] == 1
        assert all(
            timing[name] is not None
            for name in (
                "host_accepted", "send_start", "send_end", "child_received",
                "child_start", "child_end", "result_received",
            )
        )
    finally:
        pump.close()


def test_hint_window_update_sequence_advance_keeps_original_identity_and_reaches_child() -> None:
    snapshot = _exact_snapshot("hint-update-sequence")
    current = [snapshot]

    def host_factory(**version):
        return LiveV2WorkerHost(
            WorkerReference.from_callable(_deadline_worker), **version
        )

    runtime = LiveV2AdviceRuntime(
        AdviceWorkerConfig("unused", "fabledan", fabledan_runtime_policy="rule_only"),
        session_id=snapshot.version.session_id,
        capture_generation=snapshot.version.capture_generation,
        state_revision=snapshot.version.state_revision,
        host_factory=host_factory,
    )
    delivered = []
    store = _AuditStore()
    opportunity = _opportunity(snapshot, "hint-update-sequence")
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: current[0],
        on_result=lambda result: delivered.append(result) is None,
        store=store,
        local_hint_window_ms=200,
        request_timeout_ms=3_000,
    )
    try:
        pump.start()
        started = time.monotonic()
        pump.publish(opportunity)
        current[0] = replace(
            snapshot,
            version=replace(snapshot.version, update_sequence=99),
        )
        assert pump.wait_idle(3)

        assert time.monotonic() - started < 3
        assert delivered and delivered[0].status is AdviceRuntimeStatus.ADVICE
        assert delivered[0].identity.version == opportunity.version
        assert [row["status"] for row in store.records] == [
            "requested", "worker_started", "ready"
        ]
        timing = store.records[-1]["worker_timing"]
        assert all(
            timing[name] is not None
            for name in (
                "host_accepted", "send_start", "send_end", "child_received",
                "child_start", "child_end", "result_received",
            )
        )
    finally:
        pump.close()


def test_hint_window_formal_version_change_is_stale_without_worker_submit() -> None:
    snapshot = _exact_snapshot("hint-formal-change")
    current = [snapshot]
    runtime = _CapturingRuntime()
    delivered = []
    store = _AuditStore()
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: current[0],
        on_result=lambda result: delivered.append(result) is None,
        store=store,
        local_hint_window_ms=200,
        request_timeout_ms=3_000,
    )
    try:
        pump.start()
        pump.publish(_opportunity(snapshot, "hint-formal-change"))
        current[0] = replace(
            snapshot,
            version=replace(snapshot.version, capture_generation=2),
        )
        assert pump.wait_idle(2)

        assert runtime.timeouts == []
        assert delivered and delivered[0].status is AdviceRuntimeStatus.SUPERSEDED
        assert [row["status"] for row in store.records] == ["requested", "stale"]
        assert store.records[-1]["failure_code"] == "snapshot_formal_state_changed"
    finally:
        pump.close()


def test_local_hint_delay_is_deducted_from_the_original_opportunity_deadline() -> None:
    snapshot = _exact_snapshot("opportunity-deadline")
    runtime = _CapturingRuntime()
    clock = [0]
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=lambda result: True,
        local_hint_window_ms=200,
        request_timeout_ms=3_000,
        processing_clock_ms=lambda: clock[0],
    )
    try:
        pump.start()
        pump.publish(_opportunity(snapshot, "hint-window"))
        clock[0] = 200
        deadline = time.monotonic() + 1
        while not runtime.timeouts and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.timeouts == [2_800]
    finally:
        pump.close()


def test_exhausted_local_hint_deadline_times_out_without_worker_submit() -> None:
    snapshot = _exact_snapshot("opportunity-deadline-exhausted")
    runtime = _CapturingRuntime()
    clock = [0]
    delivered = []
    store = _AuditStore()
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=lambda result: delivered.append(result) is None,
        store=store,
        local_hint_window_ms=200,
        request_timeout_ms=200,
        processing_clock_ms=lambda: clock[0],
    )
    try:
        pump.start()
        pump.publish(_opportunity(snapshot, "hint-window-expired"))
        clock[0] = 200
        assert pump.wait_idle(1)
        assert runtime.timeouts == []
        assert delivered[0].status is AdviceRuntimeStatus.WORKER_TIMEOUT
        assert [row["status"] for row in store.records] == ["requested", "timeout"]
    finally:
        pump.close()


def test_pump_absolute_deadline_cancels_silent_runtime_and_ignores_late_result() -> None:
    snapshot = _exact_snapshot("pump-owned-deadline")
    runtime = _SilentRuntime()
    clock = _ManualClock(100)
    delivered, metrics, store = [], {}, _AuditStore()
    timeout_seen = threading.Event()
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=lambda result: (
            delivered.append(result), timeout_seen.set(), True
        )[-1],
        store=store,
        metrics=metrics,
        local_hint_window_ms=0,
        request_timeout_ms=3_000,
        processing_clock_ms=clock,
        poll_seconds=0.005,
    )
    try:
        pump.start()
        pump.publish(_opportunity(snapshot, "silent-host"))
        clock.advance(2_999)
        pump.poll()
        assert not timeout_seen.is_set()

        clock.advance(1)
        assert timeout_seen.wait(1)
        assert runtime.cancelled.wait(1)
        assert pump.wait_idle(0)
        assert metrics["opportunity_late"] == 1
        assert [row["status"] for row in store.records] == [
            "requested", "worker_started", "timeout"
        ]
        assert store.records[-1]["deadline_processing_ms"] == 3_100

        runtime.results.append(AdviceRuntimeResult(
            runtime.identity,
            AdviceRuntimeStatus.WORKER_TIMEOUT,
            runtime.worker_generation,
            runtime.worker_pid,
            elapsed_ms=8_000,
        ))
        pump.poll()
        assert len(delivered) == 1
        assert metrics["opportunity_late"] == 1
        assert [row["status"] for row in store.records].count("timeout") == 1
    finally:
        pump.close()


def test_timeout_snapshots_partial_timing_before_cancel_clears_runtime_mapping() -> None:
    snapshot = _exact_snapshot("timeout-cancel-audit-timing")
    runtime = _CancelClearsTimingRuntime()
    store = _AuditStore()
    clock = _ManualClock(1_000)
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=lambda _result: True,
        store=store,
        local_hint_window_ms=0,
        request_timeout_ms=300,
        processing_clock_ms=clock,
    )
    try:
        pump.start()
        pump.publish(_opportunity(snapshot, "timeout-cancel-audit"))
        clock.advance(301)
        pump.poll()

        assert runtime.cancelled and runtime.timing is None
        terminal = store.records[-1]
        assert terminal["status"] == "timeout"
        timing = terminal["worker_timing"]
        assert timing["host_accepted"] == 100
        assert timing["send_start"] == 101
        assert timing["send_end"] == 102
        assert timing["child_received"] is None
        assert timing["child_start"] is None
        assert timing["child_end"] is None
        assert timing["result_received"] is None
        assert timing["submission"]["host_submit_returned"] == 95
    finally:
        pump.close()


def test_pump_deadline_terminalizes_waiting_hint_when_timer_never_fires(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "daguandan_bridge.application.live_v2_advice_pump.Timer", _DeadTimer
    )
    snapshot = _exact_snapshot("dead-hint-timer")
    runtime = _SilentRuntime()
    clock = _ManualClock()
    timeout_seen = threading.Event()
    store = _AuditStore()
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=lambda _result: timeout_seen.set() is None,
        store=store,
        local_hint_window_ms=200,
        request_timeout_ms=3_000,
        processing_clock_ms=clock,
        poll_seconds=0.005,
    )
    try:
        pump.start()
        pump.publish(_opportunity(snapshot, "lost-host-timer"))
        clock.advance(3_000)
        assert timeout_seen.wait(1)
        assert pump.wait_idle(0)
        assert runtime.identity is None
        assert not runtime.cancelled.is_set()
        assert [row["status"] for row in store.records] == ["requested", "timeout"]
    finally:
        pump.close()


def test_hung_advice_worker_times_out_then_recovers_for_next_opportunity() -> None:
    snapshot = _exact_snapshot("hung-worker-recovery")
    hosts: list[LiveV2WorkerHost] = []

    def host_factory(**version):
        host = LiveV2WorkerHost(
            WorkerReference.from_callable(_deadline_worker),
            **version,
            restart_backoff_seconds=0,
        )
        hosts.append(host)
        return host

    runtime = LiveV2AdviceRuntime(
        AdviceWorkerConfig("unused", "fabledan", fabledan_runtime_policy="rule_only"),
        session_id=snapshot.version.session_id,
        capture_generation=snapshot.version.capture_generation,
        state_revision=snapshot.version.state_revision,
        host_factory=host_factory,
    )
    delivered = []
    store = _AuditStore()
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=lambda result: delivered.append(result) is None,
        store=store,
        local_hint_window_ms=0,
        request_timeout_ms=1_000,
    )
    try:
        pump.start()
        first_pid = runtime.worker_pid
        started = time.monotonic()
        pump.publish(_opportunity(snapshot, "hang"))
        assert pump.wait_idle(2)
        assert delivered[0].status.value == "worker_timeout"
        assert time.monotonic() - started < 2.5
        timeout_timing = store.records[-1]["worker_timing"]
        assert timeout_timing["host_accepted"] is not None
        assert timeout_timing["child_received"] is not None
        assert timeout_timing["child_start"] is not None
        assert timeout_timing["child_end"] is None
        assert timeout_timing["result_received"] is None

        pump.publish(_opportunity(snapshot, "after-timeout"))
        assert pump.wait_idle(5)
        assert delivered[-1].status.value == "advice"
        assert runtime.worker_pid != first_pid
    finally:
        pump.close()
    assert not hosts[0].is_worker_alive


def test_state_advance_turns_late_old_success_into_terminal_stale() -> None:
    snapshot = _exact_snapshot("state-advance-stale")
    current_version = [snapshot.version]
    store = _AuditStore()

    def factory(**version):
        return LiveV2WorkerHost(
            WorkerReference.from_callable(_deadline_worker), **version
        )

    runtime = LiveV2AdviceRuntime(
        AdviceWorkerConfig("unused", "fabledan", fabledan_runtime_policy="rule_only"),
        session_id=snapshot.version.session_id,
        capture_generation=snapshot.version.capture_generation,
        state_revision=snapshot.version.state_revision,
        host_factory=factory,
    )
    delivered = []

    def accept(result):
        delivered.append(result)
        return result.identity.version == current_version[0]

    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: snapshot,
        on_result=accept,
        store=store,
        local_hint_window_ms=0,
        request_timeout_ms=1_000,
    )
    try:
        pump.start()
        pump.publish(_opportunity(snapshot, "slow-state-change"))
        current_version[0] = VersionIdentity(
            snapshot.version.session_id,
            snapshot.version.capture_generation,
            snapshot.version.state_revision + 1,
            snapshot.version.update_sequence + 1,
            snapshot.version.turn_index + 1,
        )
        assert pump.wait_idle(2)
        assert delivered and delivered[0].status.value == "advice"
        assert store.records[-1]["status"] == "stale"
        assert store.records[-1]["worker_timing"]["result_received"] is not None
    finally:
        pump.close()


def test_r3_multi_round_rebind_gives_each_request_exactly_one_terminal() -> None:
    snapshots = {
        turn: _sequence_snapshot(revision, turn)
        for turn, revision in ((2, 3), (6, 7), (10, 11), (14, 15))
    }
    current = [snapshots[2]]
    store = _AuditStore()

    def factory(**version):
        return LiveV2WorkerHost(
            WorkerReference.from_callable(_r3_sequence_worker),
            **version,
            restart_backoff_seconds=0,
        )

    runtime = LiveV2AdviceRuntime(
        AdviceWorkerConfig("unused", "fabledan", fabledan_runtime_policy="rule_only"),
        session_id=current[0].version.session_id,
        capture_generation=1,
        state_revision=3,
        host_factory=factory,
    )
    delivered = []
    pump = LiveV2AdvicePump(
        runtime,
        snapshot_provider=lambda: current[0],
        on_result=lambda result: (
            delivered.append(result) is None
            and result.identity.version == current[0].version
        ),
        store=store,
        local_hint_window_ms=0,
        request_timeout_ms=2_500,
    )
    try:
        pump.start()
        pump.publish(_opportunity(current[0], "turn-2"))
        time.sleep(0.05)

        current[0] = snapshots[6]
        pump.publish(_opportunity(current[0], "turn-6"))
        deadline = time.monotonic() + 2
        while not any(item.identity.opportunity_id == "turn-6" for item in delivered):
            assert time.monotonic() < deadline
            pump.poll()
            time.sleep(0.01)

        current[0] = snapshots[10]
        pump.publish(_opportunity(current[0], "turn-10"))
        time.sleep(0.05)
        current[0] = snapshots[14]
        pump.publish(_opportunity(current[0], "turn-14"))
        assert pump.wait_idle(4)

        terminal = {"ready", "timeout", "stale", "cancelled", "failed", "withheld"}
        by_opportunity = {
            name: [
                row["status"] for row in store.records
                if row["opportunity_id"] == name and row["status"] in terminal
            ]
            for name in ("turn-2", "turn-6", "turn-10", "turn-14")
        }
        assert by_opportunity == {
            "turn-2": ["stale"],
            "turn-6": ["ready"],
            "turn-10": ["stale"],
            "turn-14": ["timeout"],
        }
        assert pump.wait_idle(0)
    finally:
        pump.close()
