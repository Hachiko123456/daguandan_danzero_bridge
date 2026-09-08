from __future__ import annotations

from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live_v2.event_resolver import UnifiedActionResolver
from daguandan_bridge.infrastructure.live_v2_rule_backend import create_reducer_backend
from daguandan_bridge.live_v2.rules_adapter import LiveReducerRuleAdapter
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    CommitReason,
    FrameIdentity,
    GapPhase,
    GapReason,
    ProjectionReason,
    Seat,
    VersionIdentity,
)


def _setup() -> tuple[
    LiveReducerRuleAdapter, UnifiedActionResolver, LiveReducer
]:
    reducer = LiveReducer("s")
    hand = tuple(f"{rank}{suit}" for suit in "SHC" for rank in (
        "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
    ))[:27]
    reducer.confirm_initial_state(
        round_level="2", hand=hand, lead_player="right", monotonic_ms=1
    )
    version = VersionIdentity("s", 1, 1, 0, 0)
    adapter = LiveReducerRuleAdapter(
        create_reducer_backend(reducer, version=version, in_memory=True), version
    )
    return adapter, UnifiedActionResolver(adapter, adapter), reducer


def _candidate(
    name: str,
    seat: Seat,
    cards: tuple[str, ...],
    *,
    first: int,
    version: VersionIdentity,
    generation: int = 1,
) -> ActionCandidate:
    first_frame = FrameIdentity("s", generation, first, first * 10, "roi", "window")
    last_frame = FrameIdentity("s", generation, first + 1, first * 10 + 10, "roi", "window")
    return ActionCandidate(
        candidate_id=name,
        version=version,
        seat=seat,
        kind=ActionKind.PLAY,
        cards=cards,
        suit_options=tuple((card,) for card in cards),
        evidence_ids=(f"{name}-1", f"{name}-2"),
        action_epoch=first,
        first_frame=first_frame,
        last_frame=last_frame,
        processing_ms=last_frame.captured_ms + 1,
        confidence=0.9,
        reason=CandidateReason.STABLE_PLAY,
    )


def test_missing_candidate_stays_observing_instead_of_inventing_pass() -> None:
    adapter, resolver, reducer = _setup()
    result = resolver.resolve(
        base_version=adapter.version,
        candidates=(),
        processing_ms=100,
        expected_seats=(Seat.RIGHT,),
    )
    assert result.projection.reason is ProjectionReason.NEED_MORE_EVIDENCE
    assert result.gap.phase is GapPhase.OBSERVING
    assert result.gap.reason is GapReason.MISSING_EXPECTED_ACTION
    assert not reducer.snapshot().play_history


def test_normal_resolution_uses_same_projection_and_atomic_commit_path() -> None:
    adapter, resolver, reducer = _setup()
    candidate = _candidate(
        "right-3", Seat.RIGHT, ("3D",), first=10, version=adapter.version
    )
    result = resolver.resolve(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
        expected_seats=(Seat.RIGHT,),
        commit=True,
    )
    assert result.accepted
    assert result.gap.phase is GapPhase.CLEAR
    assert result.commit is not None
    assert result.commit.reason is CommitReason.COMMITTED
    assert reducer.snapshot().current_player == "opposite"


def test_non_unique_legal_order_is_an_explicit_gap_not_a_guess() -> None:
    adapter, resolver, reducer = _setup()
    one = _candidate(
        "right-3", Seat.RIGHT, ("3D",), first=10, version=adapter.version
    )
    alternative = _candidate(
        "right-4", Seat.RIGHT, ("4D",), first=10, version=adapter.version
    )
    follower = _candidate(
        "opposite-5", Seat.OPPOSITE, ("5D",), first=10, version=adapter.version
    )
    result = resolver.resolve(
        base_version=adapter.version,
        candidates=(one, alternative, follower),
        processing_ms=500,
        expected_seats=(Seat.RIGHT, Seat.OPPOSITE),
    )
    assert result.projection.reason is ProjectionReason.OUT_OF_ORDER
    assert result.gap.phase is GapPhase.RECOVERABLE
    assert result.gap.reason is GapReason.OUT_OF_ORDER_EVIDENCE
    assert not reducer.snapshot().play_history


def test_capture_generation_mismatch_expires_the_gap() -> None:
    adapter, resolver, _reducer = _setup()
    candidate = _candidate(
        "right-3", Seat.RIGHT, ("3D",), first=10,
        version=VersionIdentity("s", 2, 1, 0, 0),
        generation=2,
    )
    result = resolver.resolve(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=500,
    )
    assert result.gap.phase is GapPhase.EXPIRED
    assert result.gap.reason is GapReason.CAPTURE_DISCONTINUITY


def test_following_action_without_expected_first_link_remains_recoverable() -> None:
    adapter, resolver, reducer = _setup()
    follower = _candidate(
        "opposite-5", Seat.OPPOSITE, ("5D",), first=10, version=adapter.version
    )
    result = resolver.resolve(
        base_version=adapter.version,
        candidates=(follower,),
        processing_ms=500,
        expected_seats=(Seat.RIGHT, Seat.OPPOSITE),
    )
    assert result.gap.phase is GapPhase.RECOVERABLE
    assert result.gap.reason is GapReason.MISSING_EXPECTED_ACTION
    assert not reducer.snapshot().play_history


def test_shared_evidence_between_candidates_is_an_explicit_conflict() -> None:
    adapter, resolver, _reducer = _setup()
    right = _candidate(
        "right-3", Seat.RIGHT, ("3D",), first=10, version=adapter.version
    )
    opposite = _candidate(
        "opposite-4", Seat.OPPOSITE, ("4D",), first=20, version=adapter.version
    )
    opposite = ActionCandidate(
        candidate_id=opposite.candidate_id,
        version=opposite.version,
        seat=opposite.seat,
        kind=opposite.kind,
        cards=opposite.cards,
        suit_options=opposite.suit_options,
        evidence_ids=(right.evidence_ids[-1], opposite.evidence_ids[-1]),
        action_epoch=opposite.action_epoch,
        first_frame=opposite.first_frame,
        last_frame=opposite.last_frame,
        processing_ms=opposite.processing_ms,
        confidence=opposite.confidence,
        reason=opposite.reason,
    )
    result = resolver.resolve(
        base_version=adapter.version,
        candidates=(right, opposite),
        processing_ms=500,
        expected_seats=(Seat.RIGHT, Seat.OPPOSITE),
    )
    assert result.gap.phase is GapPhase.BLOCKING
    assert result.gap.reason is GapReason.CONFLICTING_EVIDENCE
