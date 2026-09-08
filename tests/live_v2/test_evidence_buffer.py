from __future__ import annotations

import pytest

from daguandan_bridge.live_v2.evidence_buffer import EvidenceBuffer, EvidenceDrop
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    EvidenceDropReason,
    FrameIdentity,
    ObservationKind,
    ObservationReason,
    Seat,
    SeatObservation,
    VersionIdentity,
)


def frame(seq: int, *, generation: int = 1) -> FrameIdentity:
    return FrameIdentity("s", generation, seq, seq * 100, "roi-v1", "window-1")


def obs(seq: int, *, seat: Seat = Seat.LEFT, generation: int = 1) -> SeatObservation:
    return SeatObservation(
        f"e-{generation}-{seat.value}-{seq}", frame(seq, generation=generation),
        seat, ObservationKind.PLAY, ("3H",), 0.9,
        ObservationReason.CARDS_RECOGNIZED, seq * 100 + 1,
        (("3H",),), ("template_match",),
    )


def candidate(seq: int, *, seat: Seat = Seat.LEFT) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=f"candidate-{seq}",
        version=VersionIdentity("s", 1, seq, seq, seq),
        seat=seat,
        kind=ActionKind.PLAY,
        cards=("3H",),
        suit_options=(("3H",),),
        evidence_ids=(f"c-{seq}-a", f"c-{seq}-b"),
        action_epoch=seq,
        first_frame=frame(seq),
        last_frame=frame(seq + 1),
        processing_ms=(seq + 1) * 100 + 1,
        confidence=0.9,
        reason=CandidateReason.STABLE_PLAY,
    )


def test_enforces_count_and_byte_budgets_with_reasons() -> None:
    buffer = EvidenceBuffer(max_bytes=20, max_age_ms=1000, max_count=2)
    assert buffer.add(obs(1), size_bytes=8)
    assert buffer.add(obs(2), size_bytes=8)
    assert buffer.add(obs(3), size_bytes=8)
    assert [item.frame.frame_sequence for item in buffer.observations()] == [2, 3]
    assert buffer.retained_bytes == 16
    assert buffer.drops[-1].reason is EvidenceDropReason.COUNT_BUDGET
    assert buffer.add(obs(4), size_bytes=16)
    assert [item.frame.frame_sequence for item in buffer.observations()] == [4]
    assert buffer.drops[-1].reason is EvidenceDropReason.BYTE_BUDGET


def test_age_and_oversize_evidence_are_rejected_explicitly() -> None:
    buffer = EvidenceBuffer(max_bytes=10, max_age_ms=200, max_count=10)
    assert not buffer.add(obs(1), size_bytes=1, now_ms=500)
    assert buffer.drops[-1].reason is EvidenceDropReason.EXPIRED_ON_ARRIVAL
    assert not buffer.add(obs(5), size_bytes=11, now_ms=500)
    assert buffer.drops[-1].reason is EvidenceDropReason.OVERSIZE


def test_consumption_boundary_is_generation_aware() -> None:
    buffer = EvidenceBuffer(max_bytes=100, max_age_ms=1000, max_count=10)
    buffer.add(obs(1), size_bytes=4)
    buffer.add(obs(2), size_bytes=4)
    assert buffer.mark_consumed(Seat.LEFT, frame(2)) == 2
    assert not buffer.add(obs(2), size_bytes=4)
    assert buffer.drops[-1].reason is EvidenceDropReason.CONSUMED_BOUNDARY
    assert buffer.drops[-1].frame == frame(2)
    assert buffer.add(obs(3), size_bytes=4)
    assert buffer.add(obs(1, generation=2), size_bytes=4)


def test_observations_and_candidates_are_queryable_separately() -> None:
    buffer = EvidenceBuffer(max_bytes=100, max_age_ms=1000, max_count=10)
    one, action = obs(1), candidate(2)
    assert buffer.add(one, size_bytes=3)
    assert buffer.add(action, size_bytes=1)
    assert buffer.observations() == (one,)
    assert buffer.candidates() == (action,)
    assert buffer.candidates()[0].action_epoch == 2
    assert buffer.candidates()[0].last_frame == frame(3)


def test_prune_records_age_and_releases_bytes() -> None:
    buffer = EvidenceBuffer(max_bytes=100, max_age_ms=100, max_count=10)
    buffer.add(obs(1), size_bytes=9)
    assert buffer.prune(201) == 1
    assert buffer.retained_bytes == 0
    assert buffer.drops[-1].reason is EvidenceDropReason.AGE_BUDGET


def test_consumption_boundary_does_not_cross_roi_or_source_identity() -> None:
    buffer = EvidenceBuffer(max_bytes=100, max_age_ms=1000, max_count=10)
    buffer.mark_consumed(Seat.LEFT, frame(5))
    changed_roi_frame = FrameIdentity(
        "s", 1, 1, 100, "roi-v2", "window-1"
    )
    changed_roi = SeatObservation(
        "changed-roi", changed_roi_frame, Seat.LEFT, ObservationKind.PLAY,
        ("4H",), 0.9, ObservationReason.CARDS_RECOGNIZED, 101,
        (("4H",),), ("new-roi",),
    )
    assert buffer.add(changed_roi, size_bytes=1)
    assert buffer.drops == ()


def test_evidence_drop_requires_contract_reason_enum() -> None:
    with pytest.raises(TypeError, match="EvidenceDropReason"):
        EvidenceDrop(
            "age_budget",  # type: ignore[arg-type]
            Seat.LEFT,
            "obs",
            frame(1),
            0,
            1,
        )
