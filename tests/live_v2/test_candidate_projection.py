from __future__ import annotations

from dataclasses import replace

from daguandan_bridge.live_v2.candidate_projection import candidate_orders
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    FrameIdentity,
    ProjectionReason,
    Seat,
    VersionIdentity,
)


VERSION = VersionIdentity("s", 1, 1, 0, 0)


def _candidate(identity: int, seat: Seat) -> ActionCandidate:
    first = FrameIdentity("s", 1, identity * 10, identity * 100, "roi", "window")
    last = FrameIdentity(
        "s", 1, identity * 10 + 1, identity * 100 + 10, "roi", "window"
    )
    return ActionCandidate(
        candidate_id=f"c-{identity}",
        version=VERSION,
        seat=seat,
        kind=ActionKind.PLAY,
        cards=(f"{identity + 2}D",),
        suit_options=((f"{identity + 2}D",),),
        evidence_ids=(f"e-{identity}-1", f"e-{identity}-2"),
        action_epoch=identity,
        first_frame=first,
        last_frame=last,
        processing_ms=last.captured_ms + 1,
        confidence=0.9,
        reason=CandidateReason.STABLE_PLAY,
    )


def _orders(*candidates: ActionCandidate) -> tuple[tuple[str, ...], ...]:
    orders, rejection = candidate_orders(
        base_version=VERSION,
        candidates=candidates,
        consumed=frozenset(),
        consumed_evidence=frozenset(),
        max_candidates=4,
    )
    assert rejection is None
    return tuple(
        tuple(candidate.candidate_id for candidate in order) for order in orders
    )


def test_every_non_empty_capture_ordered_subset_is_returned() -> None:
    candidates = tuple(
        _candidate(index, seat)
        for index, seat in enumerate(
            (Seat.RIGHT, Seat.OPPOSITE, Seat.LEFT, Seat.SELF), start=1
        )
    )

    orders = _orders(*candidates)

    assert len(orders) == 15
    assert ("c-1",) in orders
    assert ("c-2", "c-4") in orders
    assert ("c-1", "c-2", "c-3", "c-4") in orders
    assert ("c-4", "c-2") not in orders


def test_each_consistent_permutation_of_an_unordered_subset_is_returned() -> None:
    right = _candidate(1, Seat.RIGHT)
    opposite = replace(
        _candidate(2, Seat.OPPOSITE),
        first_frame=right.first_frame,
        last_frame=right.last_frame,
    )

    assert set(_orders(right, opposite)) == {
        ("c-1",),
        ("c-2",),
        ("c-1", "c-2"),
        ("c-2", "c-1"),
    }


def test_capture_inconsistent_candidates_only_form_consistent_subsets() -> None:
    right = _candidate(1, Seat.RIGHT)
    opposite = _candidate(2, Seat.OPPOSITE)
    other_stream = replace(
        opposite,
        first_frame=replace(opposite.first_frame, source_id="other"),
        last_frame=replace(opposite.last_frame, source_id="other"),
    )

    assert set(_orders(right, other_stream)) == {("c-1",), ("c-2",)}


def test_input_larger_than_the_candidate_budget_is_not_enumerated() -> None:
    candidates = tuple(_candidate(index, Seat.RIGHT) for index in range(1, 6))

    orders, rejection = candidate_orders(
        base_version=VERSION,
        candidates=candidates,
        consumed=frozenset(),
        consumed_evidence=frozenset(),
        max_candidates=4,
    )

    assert orders == ()
    assert rejection is ProjectionReason.NEED_MORE_EVIDENCE
