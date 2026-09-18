from __future__ import annotations

from dataclasses import replace

from daguandan_bridge.live_v2.game_state import (
    GameAction, SeatCardCount, TrustedGameSnapshot,
)
from daguandan_bridge.live_v2.opportunity import OpportunityLifecycle, OpportunityState
from daguandan_bridge.live_v2.types import (
    GapPhase,
    GapReason,
    GapState,
    OpportunityReason,
    OpportunityStatus,
    ActionKind,
    FrameIdentity,
    Seat,
    StateVersion,
    VersionIdentity,
)


def version(**changes: object) -> VersionIdentity:
    values = dict(session_id="s", capture_generation=1, state_revision=0,
                  update_sequence=0, turn_index=2)
    values.update(changes)
    return VersionIdentity(**values)  # type: ignore[arg-type]


def gap(current: VersionIdentity, phase: GapPhase = GapPhase.CLEAR) -> GapState:
    return GapState(
        current, phase,
        GapReason.NONE if phase is GapPhase.CLEAR else GapReason.MISSING_EXPECTED_ACTION,
        () if phase is GapPhase.CLEAR else (Seat.SELF,), (), 100, 100,
    )


def snapshot(current: VersionIdentity, **changes: object) -> TrustedGameSnapshot:
    values = dict(
        version=current, round_level="6", wild_rank="6",
        trick_index=1,
        current_seat=Seat.SELF, lead_seat=Seat.SELF,
        my_hand=tuple(f"CARD-{index}" for index in range(27)),
        play_history=(), current_trick=(),
        remaining=tuple(SeatCardCount(seat, 27) for seat in Seat),
        finished=(), trusted=True, terminal=False, captured_ms=100,
    )
    values.update(changes)
    return TrustedGameSnapshot(**values)  # type: ignore[arg-type]


def test_blocked_opportunity_becomes_ready_when_gap_recovers_before_deadline() -> None:
    lifecycle = OpportunityLifecycle(response_deadline_ms=2_000)
    current = version()
    blocked = lifecycle.advance(
        OpportunityState(), snapshot=snapshot(current),
        gap=gap(current, GapPhase.OBSERVING), version=current, processing_ms=200,
    )
    assert blocked.state.current is not None
    assert blocked.state.current.status is OpportunityStatus.BLOCKED
    assert blocked.state.current.reason is OpportunityReason.HISTORY_GAP

    updated = replace(current, update_sequence=1)
    published = lifecycle.advance(
        blocked.state, snapshot=snapshot(updated), gap=gap(current),
        version=updated, processing_ms=500,
    )
    assert published.state.current is not None
    assert published.state.current.status is OpportunityStatus.READY
    assert published.state.current.reason is OpportunityReason.TRUSTED_STATE
    assert len(published.publications) == 1


def test_unresolved_history_blocks_model_opportunity_until_repaired() -> None:
    before = StateVersion("s", 0, 0)
    after = StateVersion("s", 1, 1)
    frame = FrameIdentity("s", 1, 1, 100, "roi", "window")
    action = GameAction(
        "unknown-right", before, after, Seat.RIGHT, ActionKind.PLAY,
        ("3?",), (("3?", "3H", "3D"),), 0, ("e1", "e2"),
        frame, frame, 0.9, 100,
    )
    current = version(state_revision=1, turn_index=1)
    uncertain = snapshot(
        current,
        play_history=(action,),
        current_trick=(action,),
        remaining=tuple(
            SeatCardCount(seat, 26 if seat is Seat.RIGHT else 27)
            for seat in Seat
        ),
        lead_seat=Seat.RIGHT,
    )

    transition = OpportunityLifecycle().advance(
        OpportunityState(), snapshot=uncertain, gap=gap(current),
        version=current, processing_ms=200,
    )

    assert transition.state.current is not None
    assert transition.state.current.status is OpportunityStatus.BLOCKED
    assert transition.state.current.reason is OpportunityReason.OBSERVATION_UNCERTAIN


def test_two_second_deadline_closes_only_opportunity_not_recovery() -> None:
    lifecycle = OpportunityLifecycle(response_deadline_ms=2_000)
    current = version()
    blocked = lifecycle.advance(
        OpportunityState(), snapshot=snapshot(current),
        gap=gap(current, GapPhase.OBSERVING), version=current, processing_ms=100,
    )
    updated = replace(current, update_sequence=1)
    closed = lifecycle.advance(
        blocked.state, snapshot=snapshot(updated), gap=gap(current),
        version=updated, processing_ms=2_100,
    )
    assert closed.state.current is not None
    assert closed.state.current.status is OpportunityStatus.CLOSED
    assert closed.state.current.reason is OpportunityReason.SUPERSEDED
    assert closed.state.current.version.update_sequence == 1

    later = replace(current, update_sequence=2)
    same_turn = lifecycle.advance(
        closed.state, snapshot=snapshot(later), gap=gap(current),
        version=later, processing_ms=2_200,
    )
    assert same_turn.state.current is not None
    assert same_turn.state.current.status is OpportunityStatus.CLOSED


def test_ready_response_remains_open_past_response_deadline() -> None:
    lifecycle = OpportunityLifecycle(response_deadline_ms=2_000)
    current = version()
    ready = lifecycle.advance(
        OpportunityState(), snapshot=snapshot(current), gap=gap(current),
        version=current, processing_ms=200,
    )
    updated = replace(current, update_sequence=1)
    still_ready = lifecycle.advance(
        ready.state, snapshot=snapshot(updated), gap=gap(current),
        version=updated, processing_ms=5_000,
    )
    assert still_ready.state.current is not None
    assert still_ready.state.current.status is OpportunityStatus.READY
    assert still_ready.publications == ()


def test_response_deadline_starts_when_opportunity_is_observed_not_captured() -> None:
    lifecycle = OpportunityLifecycle(response_deadline_ms=2_000)
    current = version()
    delayed = lifecycle.advance(
        OpportunityState(), snapshot=snapshot(current, captured_ms=100),
        gap=gap(current, GapPhase.OBSERVING), version=current,
        processing_ms=10_000,
    )
    assert delayed.state.current is not None
    assert delayed.state.current.status is OpportunityStatus.BLOCKED
    assert delayed.state.opened_processing_ms == 10_000

    before_version = replace(current, update_sequence=1)
    before_deadline = lifecycle.advance(
        delayed.state, snapshot=snapshot(before_version, captured_ms=100),
        gap=gap(current, GapPhase.OBSERVING),
        version=before_version, processing_ms=11_999,
    )
    assert before_deadline.state.current is not None
    assert before_deadline.state.current.status is OpportunityStatus.BLOCKED
    deadline_version = replace(current, update_sequence=2)
    at_deadline = lifecycle.advance(
        before_deadline.state, snapshot=snapshot(deadline_version, captured_ms=100),
        gap=gap(current, GapPhase.OBSERVING),
        version=deadline_version, processing_ms=12_000,
    )
    assert at_deadline.state.current is not None
    assert at_deadline.state.current.status is OpportunityStatus.CLOSED


def test_old_opportunity_is_closed_before_new_turn_is_published() -> None:
    lifecycle = OpportunityLifecycle()
    old_version = version()
    ready = lifecycle.advance(
        OpportunityState(), snapshot=snapshot(old_version), gap=gap(old_version),
        version=old_version, processing_ms=200,
    )
    new_version = replace(old_version, state_revision=1, update_sequence=1, turn_index=3)
    moved = lifecycle.advance(
        ready.state,
        snapshot=snapshot(
            new_version, current_seat=Seat.RIGHT, lead_seat=Seat.RIGHT,
            captured_ms=300,
        ),
        gap=gap(new_version), version=new_version, processing_ms=350,
    )
    assert [item.status for item in moved.publications] == [
        OpportunityStatus.CLOSED,
        OpportunityStatus.BLOCKED,
    ]
    assert moved.publications[0].opportunity_id == ready.state.current.opportunity_id
    assert moved.publications[0].reason is OpportunityReason.SUPERSEDED
    assert moved.state.current is not None
    assert moved.state.current.reason is OpportunityReason.NOT_LOCAL_TURN
