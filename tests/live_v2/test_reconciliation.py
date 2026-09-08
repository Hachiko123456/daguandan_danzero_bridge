from __future__ import annotations

from dataclasses import replace

import pytest

from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live_v2.event_resolver import UnifiedActionResolver
from daguandan_bridge.live_v2.reconciliation import BoundedGapReconciler
from daguandan_bridge.infrastructure.live_v2_rule_backend import create_reducer_backend
from daguandan_bridge.live_v2.rules_adapter import LiveReducerRuleAdapter
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    FrameIdentity,
    GapPhase,
    GapReason,
    Seat,
    VersionIdentity,
)


def _system() -> tuple[
    LiveReducerRuleAdapter, BoundedGapReconciler, LiveReducer
]:
    reducer = LiveReducer("s")
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    hand = tuple(f"{rank}{suit}" for suit in "SHC" for rank in ranks)[:27]
    reducer.confirm_initial_state(
        round_level="2", hand=hand, lead_player="right", monotonic_ms=1
    )
    adapter = LiveReducerRuleAdapter(
        create_reducer_backend(
            reducer, version=VersionIdentity("s", 1, 1, 0, 0), in_memory=True
        ),
        VersionIdentity("s", 1, 1, 0, 0),
    )
    resolver = UnifiedActionResolver(adapter, adapter)
    return (
        adapter,
        BoundedGapReconciler(
            resolver, max_actions=3, max_age_ms=1_500, max_turn_span=4
        ),
        reducer,
    )


def _candidate(
    identity: int,
    seat: Seat,
    cards: tuple[str, ...],
    version: VersionIdentity,
    *,
    generation: int = 1,
) -> ActionCandidate:
    first_seq = identity * 10
    first = FrameIdentity(
        "s", generation, first_seq, first_seq * 10, "roi", "window"
    )
    last = FrameIdentity(
        "s", generation, first_seq + 1, first_seq * 10 + 10, "roi", "window"
    )
    return ActionCandidate(
        candidate_id=f"c-{generation}-{identity}",
        version=version,
        seat=seat,
        kind=ActionKind.PLAY,
        cards=cards,
        suit_options=tuple((card,) for card in cards),
        evidence_ids=(
            f"e-{generation}-{identity}-1",
            f"e-{generation}-{identity}-2",
        ),
        action_epoch=identity,
        first_frame=first,
        last_frame=last,
        processing_ms=last.captured_ms + 1,
        confidence=0.9,
        reason=CandidateReason.RECONCILIATION_EVIDENCE,
    )


def _open(reconciler: BoundedGapReconciler, version: VersionIdentity) -> None:
    reconciler.open(
        version=version,
        expected_seats=(Seat.RIGHT, Seat.OPPOSITE, Seat.LEFT),
        opened_captured_ms=50,
        processing_ms=50,
    )


def test_three_action_chain_is_validated_then_committed_atomically() -> None:
    adapter, reconciler, reducer = _system()
    _open(reconciler, adapter.version)
    chain = (
        _candidate(1, Seat.RIGHT, ("3D",), adapter.version),
        _candidate(2, Seat.OPPOSITE, ("4D",), adapter.version),
        _candidate(3, Seat.LEFT, ("5D",), adapter.version),
    )
    attempt = reconciler.attempt(
        candidates=tuple(reversed(chain)),
        current_version=adapter.version,
        processing_ms=500,
    )
    assert attempt.recovered
    assert attempt.gap.phase is GapPhase.CLEAR
    assert [event.player for event in reducer.snapshot().play_history] == [
        "right", "opposite", "left"
    ]
    assert reconciler.gap is None


def test_invalid_tail_does_not_block_unique_maximal_valid_prefix() -> None:
    adapter, reconciler, reducer = _system()
    _open(reconciler, adapter.version)
    chain = (
        _candidate(1, Seat.RIGHT, ("8D",), adapter.version),
        _candidate(2, Seat.OPPOSITE, ("9D",), adapter.version),
        _candidate(3, Seat.LEFT, ("7D",), adapter.version),
    )
    attempt = reconciler.attempt(
        candidates=chain,
        current_version=adapter.version,
        processing_ms=500,
    )
    assert attempt.recovered
    assert attempt.gap.phase is GapPhase.CLEAR
    assert [event.player for event in reducer.snapshot().play_history] == [
        "right",
        "opposite",
    ]
    assert [event.cards for event in reducer.snapshot().play_history] == [
        ("8D",),
        ("9D",),
    ]


def test_more_than_three_actions_expires_without_projection() -> None:
    adapter, reconciler, reducer = _system()
    _open(reconciler, adapter.version)
    candidates = tuple(
        _candidate(index, seat, (f"{index + 2}D",), adapter.version)
        for index, seat in enumerate(
            (Seat.RIGHT, Seat.OPPOSITE, Seat.LEFT, Seat.SELF), start=1
        )
    )
    attempt = reconciler.attempt(
        candidates=candidates,
        current_version=adapter.version,
        processing_ms=600,
    )
    assert attempt.gap.phase is GapPhase.EXPIRED
    assert attempt.gap.reason is GapReason.RECOVERY_BUDGET_EXCEEDED
    assert not reducer.snapshot().play_history


@pytest.mark.parametrize(
    ("processing_ms", "turn_delta", "expected_reason"),
    [
        (2_000, 0, GapReason.STALE_EVIDENCE),
        (500, 5, GapReason.RECOVERY_BUDGET_EXCEEDED),
    ],
)
def test_stale_or_multi_round_recovery_window_is_closed(
    processing_ms: int,
    turn_delta: int,
    expected_reason: GapReason,
) -> None:
    adapter, reconciler, reducer = _system()
    _open(reconciler, adapter.version)
    current = replace(adapter.version, turn_index=adapter.version.turn_index + turn_delta)
    candidate = _candidate(1, Seat.RIGHT, ("3D",), adapter.version)
    attempt = reconciler.attempt(
        candidates=(candidate,),
        current_version=current,
        processing_ms=processing_ms,
    )
    assert attempt.gap.phase is GapPhase.EXPIRED
    assert attempt.gap.reason is expected_reason
    assert not reducer.snapshot().play_history


def test_cross_generation_recovery_is_closed() -> None:
    adapter, reconciler, _reducer = _system()
    _open(reconciler, adapter.version)
    next_version = replace(adapter.version, capture_generation=2)
    candidate = _candidate(
        1, Seat.RIGHT, ("3D",), next_version, generation=2
    )
    attempt = reconciler.attempt(
        candidates=(candidate,),
        current_version=next_version,
        processing_ms=500,
    )
    assert attempt.gap.phase is GapPhase.EXPIRED
    assert attempt.gap.reason is GapReason.CAPTURE_DISCONTINUITY


def test_reconciler_rejects_unbounded_configuration() -> None:
    adapter, _reconciler, _reducer = _system()
    resolver = UnifiedActionResolver(adapter, adapter)
    with pytest.raises(ValueError, match="between one and three"):
        BoundedGapReconciler(resolver, max_actions=4)


def test_expired_window_cannot_be_reopened_by_later_candidate() -> None:
    adapter, reconciler, reducer = _system()
    _open(reconciler, adapter.version)
    expired = reconciler.attempt(
        candidates=(_candidate(1, Seat.RIGHT, ("3D",), adapter.version),),
        current_version=adapter.version,
        processing_ms=2_000,
    )
    assert expired.gap.phase is GapPhase.EXPIRED
    retry = reconciler.attempt(
        candidates=(_candidate(2, Seat.RIGHT, ("4D",), adapter.version),),
        current_version=adapter.version,
        processing_ms=500,
    )
    assert retry.gap == expired.gap
    assert not retry.recovered
    assert not reducer.snapshot().play_history
