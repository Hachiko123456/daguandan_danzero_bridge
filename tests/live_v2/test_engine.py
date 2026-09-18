from __future__ import annotations

from dataclasses import replace

import pytest

from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live_v2.engine import (
    EngineInput,
    InputRejectionReason,
    LiveEngine,
    SideEffectFailureKind,
)
from daguandan_bridge.live_v2.game_state import (
    GameAction,
    SeatCardCount,
    TrustedGameSnapshot,
)
from daguandan_bridge.infrastructure.live_v2_rule_backend import create_reducer_backend
from daguandan_bridge.live_v2.rules_adapter import LiveReducerRuleAdapter
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CommitReason,
    CommitResult,
    ConfirmationReason,
    ConfirmedAction,
    FrameIdentity,
    GapPhase,
    GapReason,
    ObservationKind,
    ObservationReason,
    OpportunityStatus,
    ProjectionReason,
    ProjectionResult,
    Seat,
    SeatObservation,
    StateVersion,
    VersionIdentity,
    CandidateReason,
)


def version(**changes: object) -> VersionIdentity:
    values = dict(session_id="s", capture_generation=1, state_revision=0,
                  update_sequence=0, turn_index=0)
    values.update(changes)
    return VersionIdentity(**values)  # type: ignore[arg-type]


def frame(seq: int, captured_ms: int, *, generation: int = 1) -> FrameIdentity:
    return FrameIdentity("s", generation, seq, captured_ms, "r")


def observation(seq: int, captured_ms: int, *, generation: int = 1,
                seat: Seat = Seat.LEFT) -> SeatObservation:
    return SeatObservation(
        f"o-{generation}-{seq}", frame(seq, captured_ms, generation=generation),
        seat, ObservationKind.EMPTY, (), 0.9, ObservationReason.STABLE_EMPTY,
        captured_ms + 1,
    )


def candidate(candidate_id: str, current: VersionIdentity, captured_ms: int,
              *, generation: int | None = None, seat: Seat = Seat.RIGHT,
              action_epoch: int = 0,
              cards: tuple[str, ...] = ("3H",)) -> ActionCandidate:
    generation = current.capture_generation if generation is None else generation
    first = frame(captured_ms // 10, captured_ms, generation=generation)
    last = frame(captured_ms // 10 + 1, captured_ms + 10, generation=generation)
    candidate_version = replace(current, capture_generation=generation)
    return ActionCandidate(
        candidate_id, candidate_version, seat, ActionKind.PLAY, cards,
        tuple((card,) for card in cards),
        (f"{candidate_id}-1", f"{candidate_id}-2"),
        action_epoch, first, last,
        captured_ms + 20, 0.9, CandidateReason.STABLE_PLAY,
    )


def confirmed(item: ActionCandidate, before: StateVersion) -> ConfirmedAction:
    after = replace(
        before, state_revision=before.state_revision + 1,
        turn_index=before.turn_index + 1,
    )
    return ConfirmedAction.from_candidate(
        action_id=f"a-{item.candidate_id}-{before.state_revision}", candidate=item,
        version_before=before, version_after=after,
        processing_ms=item.processing_ms + 1,
        reason=ConfirmationReason.RULE_VALIDATED,
    )


class Clock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def processing_ms(self) -> int:
        return self.value


class Projector:
    def __init__(self, reasons: list[ProjectionReason]) -> None:
        self.reasons = reasons
        self.calls: list[tuple[ActionCandidate, ...]] = []

    def project(self, *, base_version: VersionIdentity,
                candidates: tuple[ActionCandidate, ...], processing_ms: int) -> ProjectionResult:
        assert all(item.version == base_version for item in candidates)
        self.calls.append(candidates)
        reason = self.reasons.pop(0)
        if reason is ProjectionReason.ACCEPTED:
            actions: list[ConfirmedAction] = []
            current = base_version.state_version
            for item in candidates:
                action = confirmed(item, current)
                actions.append(action)
                current = action.version_after
            return ProjectionResult(
                base_version, tuple(actions), (), reason,
            )
        return ProjectionResult(
            base_version, (), tuple(item.candidate_id for item in candidates), reason
        )


class Committer:
    def __init__(self, reasons: list[CommitReason] | None = None) -> None:
        self.reasons = reasons or [CommitReason.COMMITTED]
        self.calls = 0
        self.committed: list[ConfirmedAction] = []

    def commit(self, *, expected_version: VersionIdentity,
               actions: tuple[ConfirmedAction, ...]) -> CommitResult:
        self.calls += 1
        reason = self.reasons.pop(0)
        if reason is CommitReason.COMMITTED:
            self.committed.extend(actions)
            resulting = expected_version.with_state(
                actions[-1].version_after,
                update_sequence=expected_version.update_sequence + len(actions),
            )
            return CommitResult(expected_version, resulting, actions, reason)
        return CommitResult(expected_version, expected_version, (), reason)


class StateProvider:
    def __init__(self, committer: object, *, current_seat: Seat = Seat.SELF) -> None:
        self.committer = committer
        self.current_seat = current_seat
        self.calls: list[VersionIdentity] = []

    def snapshot_for(
        self, *, version: VersionIdentity, captured_ms: int
    ) -> TrustedGameSnapshot:
        self.calls.append(version)
        confirmed_actions = tuple(getattr(self.committer, "committed", ()))
        history = tuple(GameAction.from_confirmed(item) for item in confirmed_actions)
        played = {
            seat: sum(
                len(item.cards)
                for item in history
                if item.seat is seat and item.kind is ActionKind.PLAY
            )
            for seat in Seat
        }
        remaining = tuple(
            SeatCardCount(seat, 27 - played[seat]) for seat in Seat
        )
        self_count = next(item.count for item in remaining if item.seat is Seat.SELF)
        return TrustedGameSnapshot(
            version=version,
            round_level="6",
            wild_rank="6",
            trick_index=1,
            current_seat=self.current_seat,
            lead_seat=history[0].seat if history else self.current_seat,
            my_hand=tuple(f"CARD-{index}" for index in range(self_count)),
            play_history=history,
            current_trick=history,
            remaining=remaining,
            finished=(),
            trusted=True,
            terminal=False,
            captured_ms=captured_ms,
        )


def engine(clock: Clock, projector: Projector, committer: Committer,
           **changes: object) -> LiveEngine:
    state_provider = changes.pop("state_provider", StateProvider(committer))
    return LiveEngine(
        initial_version=version(), projector=projector, committer=committer,
        state_provider=state_provider, clock=clock, **changes,
    )


def test_projection_failure_does_not_stop_later_observation_or_recovery() -> None:
    clock = Clock(200)
    projector = Projector([ProjectionReason.OUT_OF_ORDER, ProjectionReason.ACCEPTED])
    committer = Committer()
    core = engine(clock, projector, committer)
    bad = candidate("bad", core.state.version, 100)
    first = core.process(EngineInput(candidates=(bad,)))
    assert first.update.gap.phase is GapPhase.RECOVERABLE

    clock.value = 300
    seen = observation(30, 290)
    second = core.process(EngineInput(observations=(seen,)))
    assert second.update.observations == (seen,)
    assert core.state.observation_for(Seat.LEFT) == seen

    clock.value = 400
    recovery = candidate("recovery", core.state.version, 350, seat=Seat.LEFT)
    third = core.process(EngineInput(candidates=(recovery,)))
    assert third.update.confirmed_actions
    assert third.update.gap.phase is GapPhase.CLEAR
    assert len(projector.calls[-1]) == 2


def test_rule_rejected_candidate_is_recoverable_and_fresh_state_can_continue():
    clock = Clock(200)
    projector = Projector([ProjectionReason.RULE_REJECTED, ProjectionReason.ACCEPTED])
    committer = Committer()
    core = engine(clock, projector, committer, evidence_max_age_ms=100)

    bad = candidate("bad", core.state.version, 100)
    rejected = core.process(EngineInput(candidates=(bad,)))
    assert rejected.update.gap.phase is GapPhase.RECOVERABLE
    assert rejected.update.gap.reason is GapReason.RULE_REJECTION

    # Once the provisional evidence ages out, a fresh candidate is projected
    # independently and can clear the recoverable gap.
    clock.value = 400
    fresh = candidate("fresh", core.state.version, 350)
    recovered = core.process(EngineInput(
        candidates=(fresh,), captured_watermark_ms=350
    ))
    assert recovered.update.confirmed_actions
    assert recovered.update.gap.phase is GapPhase.CLEAR
    assert [item.candidate_id for item in projector.calls[-1]] == ["fresh"]



def test_engine_retains_a_complete_provider_snapshot_from_initialization() -> None:
    clock = Clock(100)
    committer = Committer()
    provider = StateProvider(committer)
    core = engine(clock, Projector([]), committer, state_provider=provider)
    assert core.state.snapshot.version == core.state.version
    assert core.state.snapshot.round_level == "6"
    assert len(core.state.snapshot.my_hand) == 27
    assert len(core.state.snapshot.remaining) == 4
    assert core.state.snapshot.trusted is True


def test_commit_requires_provider_to_expose_exact_new_action_chain() -> None:
    clock = Clock(200)
    committer = Committer()
    stale_provider = StateProvider(Committer())
    core = engine(
        clock,
        Projector([ProjectionReason.ACCEPTED]),
        committer,
        state_provider=stale_provider,
    )
    with pytest.raises(ValueError, match="committed action chain"):
        core.process(EngineInput(
            candidates=(candidate("not-exposed", core.state.version, 100),)
        ))


def test_provider_must_return_trusted_snapshot() -> None:
    class UntrustedProvider(StateProvider):
        def snapshot_for(
            self, *, version: VersionIdentity, captured_ms: int
        ) -> TrustedGameSnapshot:
            return replace(
                super().snapshot_for(version=version, captured_ms=captured_ms),
                trusted=False,
            )

    clock = Clock(100)
    committer = Committer()
    with pytest.raises(ValueError, match="trusted rule snapshot"):
        engine(
            clock, Projector([]), committer,
            state_provider=UntrustedProvider(committer),
        )


def test_provider_cannot_move_trick_index_backwards() -> None:
    class RewindingProvider(StateProvider):
        def snapshot_for(
            self, *, version: VersionIdentity, captured_ms: int
        ) -> TrustedGameSnapshot:
            snapshot = super().snapshot_for(
                version=version, captured_ms=captured_ms
            )
            return replace(snapshot, trick_index=2 if len(self.calls) == 1 else 1)

    clock = Clock(100)
    committer = Committer()
    core = engine(
        clock, Projector([]), committer,
        state_provider=RewindingProvider(committer),
    )
    with pytest.raises(ValueError, match="trick_index backwards"):
        core.process(EngineInput())


def test_recovery_produces_ready_from_matching_provider_snapshot() -> None:
    clock = Clock(100)
    projector = Projector([ProjectionReason.OUT_OF_ORDER, ProjectionReason.ACCEPTED])
    core = engine(clock, projector, Committer())
    first = core.process(EngineInput(
        candidates=(candidate("bad", core.state.version, 50),)
    ))
    assert first.update.advice_opportunity is not None
    assert first.update.advice_opportunity.status is OpportunityStatus.BLOCKED

    clock.value = 200
    ready = core.process(EngineInput(
        candidates=(candidate("fix", core.state.version, 150),)
    ))
    assert ready.update.advice_opportunity is not None
    assert ready.update.advice_opportunity.status is OpportunityStatus.READY
    assert ready.update.advice_opportunity.version == ready.update.version
    assert core.state.snapshot.version == ready.update.version
    assert tuple(item.action_id for item in core.state.snapshot.play_history) == (
        "a-bad-0", "a-fix-1",
    )


def test_cross_generation_sync_is_rejected_and_requires_a_new_engine() -> None:
    clock = Clock(200)
    projector = Projector([ProjectionReason.OUT_OF_ORDER])
    core = engine(clock, projector, Committer())
    existing = candidate("existing", core.state.version, 100)
    core.process(EngineInput(
        observations=(observation(9, 90),), candidates=(existing,)
    ))
    before = core.state
    calls_before = len(projector.calls)
    foreign = candidate("foreign", core.state.version, 100, generation=2)
    result = core.process(EngineInput(
        rebind_version=version(capture_generation=2),
        observations=(observation(10, 100, generation=2),), candidates=(foreign,),
    ))
    assert result.update is None
    assert result.rejections[0].reason is InputRejectionReason.STREAM_MISMATCH
    assert core.state == before
    assert len(projector.calls) == calls_before

    new_projector = Projector([ProjectionReason.ACCEPTED])
    new_version = version(capture_generation=2)
    replacement_committer = Committer()
    replacement = LiveEngine(
        initial_version=new_version, projector=new_projector,
        committer=replacement_committer,
        state_provider=StateProvider(replacement_committer), clock=clock,
    )
    new_observation = observation(11, 210, generation=2)
    accepted = replacement.process(EngineInput(
        observations=(new_observation,),
        candidates=(candidate("new", new_version, 210, generation=2),),
    ))
    assert accepted.update is not None
    assert accepted.update.confirmed_actions
    assert replacement.state.observation_for(Seat.LEFT) == new_observation
    assert all(item.candidate_id != "existing" for item in new_projector.calls[0])


def test_new_generation_engine_continues_seeded_rule_history_and_reaches_ready() -> None:
    reducer = LiveReducer("s")
    hand = tuple(
        f"{rank}{suit}"
        for suit in ("S", "H", "C")
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    )[:27]
    reducer.confirm_initial_state(
        round_level="2", hand=hand, lead_player=Seat.RIGHT.value, monotonic_ms=1
    )
    first_version = VersionIdentity("s", 1, reducer.snapshot().revision, 0, 0)
    first_adapter = LiveReducerRuleAdapter(
        create_reducer_backend(reducer, version=first_version, in_memory=True),
        first_version,
    )
    clock = Clock(200)
    first_engine = LiveEngine(
        initial_version=first_version,
        projector=first_adapter,
        committer=first_adapter,
        state_provider=first_adapter,
        clock=clock,
    )
    first_result = first_engine.process(EngineInput(candidates=(
        candidate("gen1-right", first_engine.state.version, 100,
                  seat=Seat.RIGHT, cards=("3D",)),
    )))
    assert first_result.update is not None
    seed = first_result.update.confirmed_actions
    committed_state = first_engine.state.committed_state
    last_capture = first_engine.state.capture_watermark_ms
    del first_engine

    second_version = VersionIdentity.from_state(
        committed_state, capture_generation=2, update_sequence=0
    )
    second_adapter = LiveReducerRuleAdapter(
        create_reducer_backend(
            reducer, version=second_version, seed_actions=seed, in_memory=True
        ),
        second_version,
    )
    second_engine = LiveEngine(
        initial_version=second_version,
        projector=second_adapter,
        committer=second_adapter,
        state_provider=second_adapter,
        clock=clock,
        initial_captured_ms=last_capture,
    )
    assert second_engine.state.version.capture_generation == 2
    assert second_engine.state.snapshot.play_history[0].first_frame.capture_generation == 1

    clock.value = 400
    second_engine.process(EngineInput(observations=(
        observation(20, 200, generation=2, seat=Seat.OPPOSITE),
    )))
    base = second_engine.state.version
    continued = second_engine.process(EngineInput(candidates=(
        candidate("gen2-opposite", base, 220, generation=2,
                  seat=Seat.OPPOSITE, cards=("4D",)),
        candidate("gen2-left", base, 250, generation=2,
                  seat=Seat.LEFT, cards=("5D",)),
    )))
    assert continued.update is not None
    assert len(continued.update.confirmed_actions) == 2
    assert continued.update.version.capture_generation == 2
    assert second_engine.state.committed_state.state_revision == (
        committed_state.state_revision + 2
    )
    assert continued.update.advice_opportunity is not None
    assert continued.update.advice_opportunity.status is OpportunityStatus.READY
    assert [item.first_frame.capture_generation
            for item in second_engine.state.snapshot.play_history] == [1, 2, 2]

    old_state = second_engine.state.committed_state
    clock.value = 500
    old_input = candidate(
        "late-gen1", second_engine.state.version, 450, generation=1,
        seat=Seat.SELF, cards=("6D",),
    )
    old_observation = observation(45, 450, generation=1, seat=Seat.SELF)
    rejected = second_engine.process(EngineInput(
        observations=(old_observation,), candidates=(old_input,)
    ))
    assert len(rejected.rejections) == 2
    assert {item.reason for item in rejected.rejections} == {
        InputRejectionReason.STREAM_MISMATCH
    }
    assert second_engine.state.committed_state == old_state
    assert second_engine.state.observation_for(Seat.SELF) is None


def test_atomic_commit_failure_never_advances_core_revision() -> None:
    clock = Clock(200)
    core = engine(
        clock, Projector([ProjectionReason.ACCEPTED]),
        Committer([CommitReason.TRANSACTION_REJECTED]),
    )
    result = core.process(EngineInput(
        candidates=(candidate("c", core.state.version, 100),)
    ))
    assert result.update.confirmed_actions == ()
    assert result.update.version.state_revision == 0
    assert core.state.candidates
    assert result.update.gap.phase is GapPhase.BLOCKING


def test_committed_candidate_is_idempotent_when_redelivered() -> None:
    clock = Clock(200)
    projector = Projector([ProjectionReason.ACCEPTED])
    committer = Committer()
    core = engine(clock, projector, committer)
    item = candidate("same", core.state.version, 100)
    core.process(EngineInput(candidates=(item,)))
    clock.value = 250
    duplicate = core.process(EngineInput(candidates=(item,)))
    assert duplicate.rejections[0].reason is InputRejectionReason.DUPLICATE_CANDIDATE
    assert committer.calls == 1
    assert len(projector.calls) == 1


def test_candidate_idempotency_key_includes_epoch_and_complete_frames() -> None:
    clock = Clock(200)
    projector = Projector([
        ProjectionReason.ACCEPTED,
        ProjectionReason.ACCEPTED,
        ProjectionReason.ACCEPTED,
    ])
    committer = Committer([CommitReason.COMMITTED] * 3)
    core = engine(clock, projector, committer)
    first = candidate("same-label", core.state.version, 100, action_epoch=1)
    core.process(EngineInput(candidates=(first,)))
    clock.value = 250
    next_epoch = replace(
        candidate("same-label", core.state.version, 200, action_epoch=2),
        first_frame=first.first_frame,
        last_frame=first.last_frame,
    )
    by_epoch = core.process(EngineInput(candidates=(next_epoch,)))
    assert by_epoch.update is not None and not by_epoch.rejections

    clock.value = 400
    next_frames = candidate(
        "same-label", core.state.version, 300, action_epoch=2
    )
    by_frames = core.process(EngineInput(candidates=(next_frames,)))
    assert by_frames.update is not None and not by_frames.rejections
    assert len(projector.calls) == 3


class BrokenJournal:
    def append(self, update: object) -> None:
        raise OSError


class BrokenAdvice:
    def publish(self, opportunity: object) -> None:
        raise RuntimeError


def test_side_effect_failures_are_explicit_and_do_not_rollback_commit() -> None:
    clock = Clock(200)
    core = engine(
        clock, Projector([ProjectionReason.ACCEPTED]), Committer(),
        journal=BrokenJournal(), advice_consumer=BrokenAdvice(),
    )
    result = core.process(EngineInput(
        candidates=(candidate("c", core.state.version, 100),)
    ))
    assert result.update.confirmed_actions
    assert core.state.version.state_revision == 1
    assert {item.kind for item in result.side_effect_failures} == {
        SideEffectFailureKind.JOURNAL_APPEND_FAILED,
        SideEffectFailureKind.ADVICE_PUBLISH_FAILED,
    }


def test_advice_consumer_failure_keeps_ready_core_opportunity() -> None:
    clock = Clock(100)
    core = engine(
        clock, Projector([]), Committer(), advice_consumer=BrokenAdvice()
    )
    result = core.process(EngineInput())
    assert core.state.opportunity.current is not None
    assert core.state.opportunity.current.status is OpportunityStatus.READY
    assert result.side_effect_failures[0].kind is SideEffectFailureKind.ADVICE_PUBLISH_FAILED


def test_provider_cannot_return_a_foreign_snapshot() -> None:
    class ForeignProvider(StateProvider):
        def snapshot_for(
            self, *, version: VersionIdentity, captured_ms: int
        ) -> TrustedGameSnapshot:
            state = super().snapshot_for(version=version, captured_ms=captured_ms)
            if len(self.calls) > 1:
                return replace(
                    state,
                    version=replace(state.version, capture_generation=2),
                )
            return state

    clock = Clock(100)
    committer = Committer()
    core = engine(
        clock, Projector([]), committer,
        state_provider=ForeignProvider(committer),
    )
    before = core.state
    with pytest.raises(ValueError, match="another version"):
        core.process(EngineInput())
    assert core.state == before


def test_same_stream_version_sync_is_controlled_and_monotonic() -> None:
    clock = Clock(100)
    core = engine(clock, Projector([]), Committer())
    sequences = [core.process(EngineInput()).update.version.update_sequence]
    clock.value = 200
    sequences.append(core.process(EngineInput()).update.version.update_sequence)
    clock.value = 300
    synchronized = replace(core.state.version, update_sequence=5)
    result = core.process(EngineInput(rebind_version=synchronized))
    assert result.update is not None
    sequences.append(result.update.version.update_sequence)
    assert sequences == sorted(set(sequences))
    assert sequences[-1] == 6


def test_stale_same_stream_sync_rejects_whole_batch_without_state_change() -> None:
    clock = Clock(100)
    core = engine(clock, Projector([]), Committer())
    core.process(EngineInput(observations=(observation(9, 90),)))
    before = core.state
    stale = replace(core.state.version, update_sequence=0)
    result = core.process(EngineInput(
        rebind_version=stale,
        observations=(observation(10, 95),),
    ))
    assert result.update is None
    assert result.rejections[0].reason is InputRejectionReason.STALE_REVISION
    assert core.state == before


def test_cross_session_sync_is_rejected_without_publishing_side_effects() -> None:
    class Journal:
        def __init__(self) -> None:
            self.updates: list[object] = []

        def append(self, update: object) -> None:
            self.updates.append(update)

    clock = Clock(100)
    journal = Journal()
    core = engine(clock, Projector([]), Committer(), journal=journal)
    before = core.state
    result = core.process(EngineInput(
        rebind_version=version(session_id="another-session"),
        observations=(observation(9, 90),),
    ))
    assert result.update is None
    assert result.rejections[0].reason is InputRejectionReason.STREAM_MISMATCH
    assert core.state == before
    assert journal.updates == []


def test_processing_delay_does_not_expire_capture_evidence() -> None:
    clock = Clock(10_000)
    projector = Projector([ProjectionReason.OUT_OF_ORDER])
    core = engine(clock, projector, Committer(), evidence_max_age_ms=100)
    item = candidate("delayed", core.state.version, 100)
    result = core.process(EngineInput(
        candidates=(item,), captured_watermark_ms=110
    ))
    assert result.update is not None
    assert result.update.candidates == (item,)
    assert result.update.processing_ms == 10_000
    assert result.update.captured_ms == 110


def test_non_commit_updates_do_not_desynchronize_strict_rule_version() -> None:
    clock = Clock(100)
    projector = Projector([ProjectionReason.ACCEPTED])
    core = engine(clock, projector, Committer())
    core.process(EngineInput(observations=(observation(9, 90),)))
    assert core.state.version.update_sequence == 1
    assert core.state.commit_version.update_sequence == 0
    clock.value = 200
    result = core.process(EngineInput(
        candidates=(candidate("after-observation", core.state.version, 150),)
    ))
    assert result.update.confirmed_actions
    assert core.state.version.update_sequence == 2
    assert core.state.commit_version.update_sequence == 1


def test_state_aggregates_one_latest_observation_for_each_seat() -> None:
    clock = Clock(200)
    core = engine(clock, Projector([]), Committer())
    observations = tuple(
        observation(index + 1, 100 + index, seat=seat)
        for index, seat in enumerate(Seat)
    )
    core.process(EngineInput(observations=observations))
    assert tuple(core.state.observation_for(seat) for seat in Seat) == observations


def test_real_rule_adapter_stays_synchronized_after_observation_only_update() -> None:
    reducer = LiveReducer("s")
    hand = tuple(
        f"{rank}{suit}"
        for suit in ("S", "H", "C")
        for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    )[:27]
    reducer.confirm_initial_state(
        round_level="2", hand=hand, lead_player=Seat.RIGHT.value, monotonic_ms=1
    )
    initial = VersionIdentity("s", 1, reducer.snapshot().revision, 0, 0)
    adapter = LiveReducerRuleAdapter(
        create_reducer_backend(reducer, version=initial, in_memory=True), initial
    )
    clock = Clock(100)
    core = LiveEngine(
        initial_version=initial, projector=adapter, committer=adapter,
        state_provider=adapter, clock=clock,
    )
    core.process(EngineInput(observations=(observation(9, 90),)))
    clock.value = 200
    result = core.process(EngineInput(
        candidates=(candidate("legal", core.state.version, 150),)
    ))
    assert result.update.confirmed_actions
    assert reducer.snapshot().play_history[-1].cards == ("3H",)
    assert core.state.commit_version == adapter.version
    assert core.state.version.state_revision == adapter.version.state_revision


def test_stale_revision_and_expired_evidence_are_rejected_before_projection() -> None:
    clock = Clock(200)
    projector = Projector([ProjectionReason.ACCEPTED])
    core = engine(clock, projector, Committer(), evidence_max_age_ms=100)
    original_version = core.state.version
    core.process(EngineInput(candidates=(candidate("first", original_version, 100),)))

    clock.value = 250
    stale = candidate("stale", original_version, 200)
    old = candidate("old", core.state.version, 100)
    result = core.process(EngineInput(
        candidates=(stale, old), captured_watermark_ms=211
    ))
    assert {item.reason for item in result.rejections} == {
        InputRejectionReason.STALE_REVISION,
        InputRejectionReason.EXPIRED_EVIDENCE,
    }
    assert len(projector.calls) == 1


def test_candidate_with_wrong_turn_in_same_revision_is_rejected() -> None:
    clock = Clock(200)
    projector = Projector([])
    core = engine(clock, projector, Committer())
    wrong_turn = candidate(
        "wrong-turn",
        replace(core.state.version, turn_index=core.state.version.turn_index + 1),
        100,
    )
    result = core.process(EngineInput(candidates=(wrong_turn,)))
    assert result.update is not None
    assert result.rejections[0].reason is InputRejectionReason.FUTURE_REVISION
    assert projector.calls == []


def test_capture_watermark_cannot_be_substituted_for_processing_clock() -> None:
    clock = Clock(100)
    core = engine(clock, Projector([]), Committer())
    before = core.state
    try:
        core.process(EngineInput(captured_watermark_ms=101))
    except ValueError as exc:
        assert "processing clock" in str(exc)
    else:
        raise AssertionError("capture watermark ahead of processing clock was accepted")
    assert core.state == before


def test_stale_candidate_cannot_advance_capture_watermark() -> None:
    clock = Clock(10_000)
    current = version(state_revision=1, turn_index=1)
    committer = Committer()
    core = LiveEngine(
        initial_version=current,
        projector=Projector([]),
        committer=committer,
        state_provider=StateProvider(committer),
        clock=clock,
        initial_captured_ms=100,
    )
    stale = candidate("stale-clock", version(), 9_000)
    result = core.process(EngineInput(candidates=(stale,)))
    assert result.update is not None
    assert result.rejections[0].reason is InputRejectionReason.STALE_REVISION
    assert core.state.capture_watermark_ms == 100
