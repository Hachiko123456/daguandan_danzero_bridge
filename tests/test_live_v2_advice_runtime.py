from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from daguandan_bridge.application.live_v2_advice_protocol import (
    AdvicePrewarmPayload,
    AdviceRequestIdentity,
    AdviceRuntimeStatus,
    AdviceWorkerConfig,
    AdviceWorkerPayload,
    AdviceWorkerSuccess,
    AdvisorReady,
    request_id,
)
from daguandan_bridge.application.live_v2_advice_runtime import LiveV2AdviceRuntime
from daguandan_bridge.application.live_v2_worker_protocol import (
    DeliveryMode, WorkerFailure, WorkerReady, WorkerRequest, WorkerResult,
    WorkerResultStatus,
)
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.infrastructure import live_v2_advice_worker as worker_module
from daguandan_bridge.infrastructure.live_v2_advice_worker import (
    replay_trusted_snapshot, run_fabledan_advice_worker,
)
from daguandan_bridge.infrastructure.live_v2_advice_service_factory import create_live_v2_advice_runtime
from daguandan_bridge.live_v2.events import ActionKind
from daguandan_bridge.live_v2.game_state import GameAction, SeatCardCount, TrustedGameSnapshot
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, StateVersion, VersionIdentity
from daguandan_bridge.live_v2.results import (
    AdviceOpportunity, OpportunityReason, OpportunityStatus,
)


HAND = tuple(f"{rank}{suit}" for rank in "3456789" for suit in "HDSC")[:27]


class _StubHost:
    def __init__(self) -> None:
        self.worker_generation, self.worker_pid = 0, None
        self.results: list[WorkerResult] = []
        self.state = "new"
        self.prewarm_count = 0
        self.advisor_ready: AdvisorReady | None = None
    def start(self, *, timeout: float = 10.0) -> WorkerReady:
        del timeout
        if self.worker_pid is None:
            self.worker_generation += 1
            self.worker_pid = 10_000 + self.worker_generation
        self.state = "ready"
        return WorkerReady(self.worker_generation, self.worker_pid, 0)
    def bind_version(self, **_version: object) -> None: pass
    def submit(self, request: WorkerRequest) -> tuple[WorkerResult, ...]:
        payload = request.payload
        if isinstance(payload, AdvicePrewarmPayload):
            self.prewarm_count += 1
            self.advisor_ready = AdvisorReady(
                payload.config.advisor_backend,
                payload.config.advisor_backend,
                "stub-model",
                self.worker_generation,
                self.worker_pid,
                0.1,
                self.prewarm_count > 1,
            )
            self.results.append(WorkerResult.terminal(
                request,
                status=WorkerResultStatus.SUCCESS,
                worker_generation=self.worker_generation,
                worker_pid=self.worker_pid,
                finished_processing_ms=1,
                payload=self.advisor_ready,
            ))
            return ()
        assert isinstance(payload, AdviceWorkerPayload)
        name = payload.opportunity.opportunity_id
        status = (
            WorkerResultStatus.CRASHED if name == "crash"
            else WorkerResultStatus.TIMEOUT if name == "slow-timeout"
            else WorkerResultStatus.SUCCESS
        )
        result_payload = None
        failure = None
        if status is WorkerResultStatus.SUCCESS:
            advice = AdviceResult(
                "stub", ("3H",), "single", False,
                request.state_revision, 1.0, payload.request_id,
            )
            result_payload = AdviceWorkerSuccess(
                name, payload.snapshot.version, advice,
                len(payload.snapshot.play_history), 1.0, True,
                replace(self.advisor_ready, cache_hit=True)
                if self.advisor_ready is not None else None,
            )
        else:
            failure = WorkerFailure(name, status.value, name)
        self.results.append(WorkerResult.terminal(
            request, status=status, worker_generation=self.worker_generation,
            worker_pid=self.worker_pid, finished_processing_ms=1,
            payload=result_payload, failure=failure,
        ))
        if status in {WorkerResultStatus.CRASHED, WorkerResultStatus.TIMEOUT}:
            self.state = "broken"
        return ()
    def get_result(self, *, timeout: float | None = None) -> WorkerResult:
        del timeout
        if not self.results:
            raise TimeoutError("no stub result")
        return self.results.pop(0)
    def drain_results(self) -> tuple[WorkerResult, ...]:
        values, self.results = tuple(self.results), []
        return values
    def restart(self, **_options: object) -> WorkerReady:
        self.worker_pid = None
        self.state = "new"
        self.advisor_ready = None
        return self.start()

    def close(self, *, timeout: float = 5.0) -> None:
        del timeout
        self.worker_pid = None
        self.state = "closed"


def _version(revision: int = 0, update: int = 0, turn: int = 0) -> VersionIdentity:
    return VersionIdentity("advice-session", 2, revision, update, turn)


def _counts(self_count: int = 27) -> tuple[SeatCardCount, ...]:
    return tuple(SeatCardCount(s, self_count if s is Seat.SELF else 27) for s in Seat)


def _snapshot(version: VersionIdentity | None = None) -> TrustedGameSnapshot:
    current = version or _version()
    return TrustedGameSnapshot(
        current, "6", "6", 1, Seat.SELF, Seat.SELF, HAND, (), (),
        _counts(), (), True, False, 100,
    )


def _historical_snapshot(
    steps: tuple[tuple[Seat, ActionKind, tuple[str, ...]], ...],
    *,
    current: Seat,
    lead: Seat,
    trick_index: int,
    current_trick_count: int,
    evidence_generation: int = 2,
    snapshot_generation: int = 2,
) -> TrustedGameSnapshot:
    actions: list[GameAction] = []
    before = StateVersion("advice-session", 0, 0)
    hand = list(HAND)
    played = {seat: 0 for seat in Seat}
    for index, (seat, kind, cards) in enumerate(steps, start=1):
        after = StateVersion(
            before.session_id,
            before.state_revision + 1,
            before.turn_index + 1,
        )
        frame = FrameIdentity(
            "advice-session", evidence_generation, index, 100 + index, "roi"
        )
        actions.append(GameAction(
            f"a-{index}", before, after, seat, kind, cards,
            tuple((card,) for card in cards), index, (f"e-{index}",),
            frame, frame, 0.99, frame.captured_ms,
        ))
        if kind is ActionKind.PLAY:
            played[seat] += len(cards)
            if seat is Seat.SELF:
                for card in cards:
                    hand.remove(card)
        before = after
    current_trick = tuple(actions[-current_trick_count:]) if current_trick_count else ()
    return TrustedGameSnapshot(
        version=VersionIdentity.from_state(
            before,
            capture_generation=snapshot_generation,
            update_sequence=len(actions),
        ),
        round_level="6", wild_rank="6", trick_index=trick_index,
        current_seat=current, lead_seat=lead, my_hand=tuple(hand),
        play_history=tuple(actions), current_trick=current_trick,
        remaining=tuple(SeatCardCount(seat, 27 - played[seat]) for seat in Seat),
        finished=(), trusted=True, terminal=False, captured_ms=200,
    )


def _opportunity(
    snapshot: TrustedGameSnapshot,
    name: str = "ready-1",
    status: OpportunityStatus = OpportunityStatus.READY,
) -> AdviceOpportunity:
    reason = (
        OpportunityReason.TRUSTED_STATE
        if status is OpportunityStatus.READY
        else OpportunityReason.HISTORY_GAP
        if status is OpportunityStatus.BLOCKED
        else OpportunityReason.SUPERSEDED
    )
    return AdviceOpportunity(
        name, snapshot.version, Seat.SELF, status, reason, 100, 110
    )


def _runtime(snapshot: TrustedGameSnapshot) -> LiveV2AdviceRuntime:
    host = _StubHost()
    runtime = LiveV2AdviceRuntime(
        AdviceWorkerConfig(
            str(Path.cwd()), "fabledan", fabledan_runtime_policy="rule_only"
        ),
        session_id=snapshot.version.session_id,
        capture_generation=snapshot.version.capture_generation,
        state_revision=snapshot.version.state_revision,
        host_factory=lambda **_version: host,
    )
    runtime.start()
    return runtime


def test_spawn_worker_returns_version_bound_advice_and_closes_idempotently() -> None:
    snapshot = _snapshot()
    runtime = _runtime(snapshot)
    try:
        pid = runtime.worker_pid
        runtime.submit(snapshot, _opportunity(snapshot), request_sequence=1)
        result = runtime.get_result(timeout=10)
        assert result.status is AdviceRuntimeStatus.ADVICE
        assert result.identity == AdviceRequestIdentity(
            snapshot.version, 1, "ready-1"
        )
        assert result.worker_pid == pid
        assert result.advice and result.advice.request_id.endswith(":ready-1")
    finally:
        runtime.close()
        runtime.close()
    assert runtime.worker_pid is None


def test_latest_only_supersedes_inflight_result_and_accepts_latest() -> None:
    snapshot = _snapshot()
    runtime = _runtime(snapshot)
    try:
        runtime.submit(snapshot, _opportunity(snapshot, "slow-old"), request_sequence=1)
        runtime.submit(snapshot, _opportunity(snapshot, "ready-new"), request_sequence=2)
        first = runtime.get_result(timeout=10)
        second = runtime.get_result(timeout=10)
        assert first.status is AdviceRuntimeStatus.SUPERSEDED
        assert first.identity.opportunity_id == "slow-old"
        assert second.status is AdviceRuntimeStatus.ADVICE
        assert second.identity.opportunity_id == "ready-new"
    finally:
        runtime.close()


def test_full_version_identity_rejects_same_revision_old_update() -> None:
    old_snapshot = _snapshot()
    new_snapshot = _snapshot(replace(old_snapshot.version, update_sequence=1))
    runtime = _runtime(old_snapshot)
    try:
        runtime.submit(
            old_snapshot, _opportunity(old_snapshot, "slow-version"),
            request_sequence=1,
        )
        runtime.submit(
            new_snapshot, _opportunity(new_snapshot, "new-version"),
            request_sequence=2,
        )
        assert runtime.get_result(timeout=10).status is AdviceRuntimeStatus.SUPERSEDED
        current = runtime.get_result(timeout=10)
        assert current.status is AdviceRuntimeStatus.ADVICE
        assert current.identity.version.update_sequence == 1
    finally:
        runtime.close()


def test_blocked_and_closed_are_local_and_never_restore_old_advice() -> None:
    snapshot = _snapshot()
    runtime = _runtime(snapshot)
    try:
        pid = runtime.worker_pid
        for sequence, status in enumerate(
            (OpportunityStatus.BLOCKED, OpportunityStatus.CLOSED), start=1
        ):
            immediate = runtime.submit(
                snapshot,
                _opportunity(snapshot, status.value, status),
                request_sequence=sequence,
            )
            assert immediate[0].status.value == status.value
            assert immediate[0].advice is None
            assert runtime.get_result(timeout=0).status.value == status.value
        assert runtime.worker_pid == pid
        assert runtime.drain_results() == ()
    finally:
        runtime.close()


def test_crash_is_typed_and_explicit_restart_changes_worker_generation() -> None:
    snapshot = _snapshot()
    runtime = _runtime(snapshot)
    try:
        old_generation = runtime.worker_generation
        runtime.submit(snapshot, _opportunity(snapshot, "crash"), request_sequence=1)
        crashed = runtime.get_result(timeout=10)
        assert crashed.status is AdviceRuntimeStatus.WORKER_CRASHED
        restarted = runtime.restart(snapshot.version, request_sequence=2)
        assert restarted.status is AdviceRuntimeStatus.WORKER_RESTARTED
        # Crash delivery proactively prepares generation + 1; an explicit
        # restart then creates and prewarms one further generation.
        assert restarted.worker_generation == old_generation + 2
        assert restarted.advisor_ready is not None
        assert restarted.advisor_ready.worker_generation == restarted.worker_generation
        runtime.submit(snapshot, _opportunity(snapshot, "after"), request_sequence=3)
        assert runtime.get_result(timeout=10).status is AdviceRuntimeStatus.ADVICE
    finally:
        runtime.close()


def test_timeout_is_typed_and_cannot_publish_late_advice() -> None:
    snapshot = _snapshot()
    runtime = _runtime(snapshot)
    try:
        runtime.submit(
            snapshot, _opportunity(snapshot, "slow-timeout"),
            request_sequence=1, timeout_ms=50,
        )
        timed_out = runtime.get_result(timeout=10)
        assert timed_out.status is AdviceRuntimeStatus.WORKER_TIMEOUT
        assert timed_out.advice is None
        restarted = runtime.restart(snapshot.version, request_sequence=2)
        assert restarted.status is AdviceRuntimeStatus.WORKER_RESTARTED
        runtime.submit(snapshot, _opportunity(snapshot, "fresh"), request_sequence=3)
        fresh = runtime.get_result(timeout=10)
        assert fresh.status is AdviceRuntimeStatus.ADVICE
        assert fresh.identity.opportunity_id == "fresh"
    finally:
        runtime.close()


def test_timeout_recovery_restarts_and_prewarms_before_next_advice() -> None:
    snapshot = _snapshot()
    runtime = _runtime(snapshot)
    first_ready = runtime.advisor_ready
    assert first_ready is not None
    try:
        runtime.submit(
            snapshot, _opportunity(snapshot, "slow-timeout"),
            request_sequence=1, timeout_ms=50,
        )
        timed_out = runtime.get_result(timeout=10)
        assert timed_out.status is AdviceRuntimeStatus.WORKER_TIMEOUT

        runtime.submit(
            snapshot, _opportunity(snapshot, "auto-recovered"),
            request_sequence=2,
        )
        recovered = runtime.get_result(timeout=10)
        assert recovered.status is AdviceRuntimeStatus.ADVICE
        assert recovered.worker_generation == first_ready.worker_generation + 1
        assert recovered.worker_pid != first_ready.worker_pid
        assert recovered.advisor_cache_hit is True
        assert recovered.advisor_ready is not None
        assert recovered.advisor_ready.worker_generation == recovered.worker_generation
        assert recovered.advisor_ready.worker_pid == recovered.worker_pid
    finally:
        runtime.close()


def test_complete_snapshot_is_replayed_before_adviser_and_cache_is_reused() -> None:
    snapshot = _historical_snapshot(
        (
            (Seat.SELF, ActionKind.PLAY, ("3H",)),
            (Seat.RIGHT, ActionKind.PASS, ()),
            (Seat.OPPOSITE, ActionKind.PASS, ()),
            (Seat.LEFT, ActionKind.PASS, ()),
        ),
        current=Seat.SELF, lead=Seat.SELF,
        trick_index=2, current_trick_count=0,
    )
    state = replay_trusted_snapshot(snapshot)
    assert state.current_player == "self"
    assert state.lead_player == "self"
    assert len(state.play_history) == 4 and state.trick_plays == []
    assert state.remaining_cards["self"] == 26

    class StubAdvisor:
        calls = 0

        def recommend(self, state: object, *, request_id: str = "") -> AdviceResult:
            self.calls += 1
            return AdviceResult(
                "rule-stub", ("4H",), "single", False, 999, 1.0, request_id
            )

    config = AdviceWorkerConfig(
        str(Path.cwd()), "fabledan", fabledan_runtime_policy="rule_only"
    )
    key = (
        str(Path(config.profile_root).resolve()), config.profile_name,
        config.advisor_backend, config.fabledan_runtime_policy,
    )
    adviser = StubAdvisor()
    worker_module._ADVISORS[key] = adviser
    try:
        opportunity = _opportunity(snapshot, "replayed")
        identity = AdviceRequestIdentity(snapshot.version, 7, "replayed")
        payload = AdviceWorkerPayload(config, snapshot, opportunity, request_id(identity))
        request = WorkerRequest(
            snapshot.version.session_id,
            snapshot.version.capture_generation,
            7,
            snapshot.version.state_revision,
            120,
            payload,
            DeliveryMode.LATEST_ONLY,
            1000,
        )
        first = run_fabledan_advice_worker(request)
        second = run_fabledan_advice_worker(request)
        assert isinstance(first, AdviceWorkerSuccess)
        assert isinstance(second, AdviceWorkerSuccess)
        assert first.advice.state_revision == snapshot.version.state_revision
        assert adviser.calls == 2
        assert worker_module._ADVISORS[key] is adviser
    finally:
        worker_module._ADVISORS.pop(key, None)


def test_cross_generation_two_trick_snapshot_replays_and_advises(
    tmp_path: Path,
) -> None:
    snapshot = _historical_snapshot(
        (
            (Seat.RIGHT, ActionKind.PLAY, ("3H",)),
            (Seat.OPPOSITE, ActionKind.PLAY, ("4H",)),
            (Seat.LEFT, ActionKind.PASS, ()),
            (Seat.SELF, ActionKind.PASS, ()),
            (Seat.RIGHT, ActionKind.PASS, ()),
            (Seat.OPPOSITE, ActionKind.PLAY, ("5H",)),
            (Seat.LEFT, ActionKind.PASS, ()),
        ),
        current=Seat.SELF, lead=Seat.OPPOSITE,
        trick_index=2, current_trick_count=2,
        evidence_generation=1, snapshot_generation=2,
    )
    replayed = replay_trusted_snapshot(snapshot)
    assert replayed.lead_player == "opposite"
    assert replayed.current_player == "self"
    assert len(replayed.trick_plays) == 2
    assert {action.first_frame.capture_generation for action in snapshot.play_history} == {1}
    opportunity = _opportunity(snapshot, "real-rule")
    identity = AdviceRequestIdentity(snapshot.version, 1, "real-rule")
    payload = AdviceWorkerPayload(
        AdviceWorkerConfig(
            str(tmp_path), "fabledan", fabledan_runtime_policy="rule_only"
        ),
        snapshot,
        opportunity,
        request_id(identity),
    )
    runtime = create_live_v2_advice_runtime(payload.config, snapshot.version)
    try:
        runtime.start()
        runtime.submit(snapshot, opportunity, request_sequence=1)
        result = runtime.get_result(timeout=10)
        assert result.status is AdviceRuntimeStatus.ADVICE
        assert result.advice and result.advice.strategy == "fabledan-rule"
        assert result.advice.request_id == request_id(identity)
        assert result.worker_pid != os.getpid()
    finally:
        runtime.close()
