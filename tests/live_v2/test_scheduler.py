from __future__ import annotations

from dataclasses import replace

import pytest

from daguandan_bridge.live_v2.scheduler import (
    ObservationScheduler,
    ScheduledRead,
    SchedulingDrop,
)
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    FrameIdentity,
    ObservationReason,
    ScheduledItemKind,
    SchedulingDropReason,
    Seat,
    VersionIdentity,
)


def frame(seq: int) -> FrameIdentity:
    return FrameIdentity("s", 1, seq, seq * 100, "roi-v1", "window-1")


def candidate(seq: int, *, seat: Seat = Seat.LEFT) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=f"candidate-{seq}",
        version=VersionIdentity("s", 1, seq, seq, seq),
        seat=seat,
        kind=ActionKind.PLAY,
        cards=(f"{seq}H",),
        suit_options=((f"{seq}H",),),
        evidence_ids=(f"e-{seq}-a", f"e-{seq}-b"),
        action_epoch=seq,
        first_frame=frame(seq),
        last_frame=frame(seq + 1),
        processing_ms=(seq + 1) * 100 + 1,
        confidence=0.9,
        reason=CandidateReason.STABLE_PLAY,
    )


def test_raw_work_is_latest_wins_per_seat_and_records_age() -> None:
    scheduler = ObservationScheduler(raw_max_age_ms=1000)
    scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(1), payload="old",
        reason=ObservationReason.UNREADABLE,
    )
    scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(2), payload="new", enqueued_ms=250,
        reason=ObservationReason.UNREADABLE,
    )
    item = scheduler.pop_next(now_ms=260)
    assert item is not None and item.payload == "new"
    assert item.item_kind is ScheduledItemKind.RAW
    assert item.reason is ObservationReason.UNREADABLE
    assert scheduler.drops[-1].reason is SchedulingDropReason.RAW_REPLACED
    assert scheduler.drops[-1].item_kind is ScheduledItemKind.RAW
    assert scheduler.drops[-1].age_ms == 150


def test_one_shared_frame_schedules_every_changed_seat() -> None:
    scheduler = ObservationScheduler()
    assert scheduler.submit_frame(
        frame=frame(1), changed_seats=(Seat.LEFT, Seat.OPPOSITE, Seat.RIGHT),
        reason=ObservationReason.STABLE_EMPTY,
    ) == 3
    assert set(scheduler.snapshot().pending_seats) == {
        Seat.LEFT, Seat.OPPOSITE, Seat.RIGHT,
    }


def test_expected_seat_only_increases_priority() -> None:
    scheduler = ObservationScheduler(starvation_ms=1000)
    scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(1), enqueued_ms=100,
        reason=ObservationReason.UNREADABLE,
    )
    scheduler.submit_raw(
        seat=Seat.OPPOSITE, frame=frame(2), enqueued_ms=200,
        reason=ObservationReason.UNREADABLE,
    )
    first = scheduler.pop_next(now_ms=250, expected_seat=Seat.OPPOSITE)
    second = scheduler.pop_next(now_ms=250, expected_seat=Seat.OPPOSITE)
    assert first is not None and first.seat is Seat.OPPOSITE
    assert second is not None and second.seat is Seat.LEFT


def test_self_priority_cannot_starve_other_seats() -> None:
    scheduler = ObservationScheduler(
        starvation_ms=1000, preferred_burst_limit=2, raw_max_age_ms=5000
    )
    scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(1), enqueued_ms=100,
        reason=ObservationReason.UNREADABLE,
    )
    scheduler.submit_raw(
        seat=Seat.SELF, frame=frame(2), enqueued_ms=200,
        reason=ObservationReason.UNREADABLE,
    )
    assert scheduler.pop_next(now_ms=210, self_opportunity=True).seat is Seat.SELF  # type: ignore[union-attr]
    scheduler.submit_raw(
        seat=Seat.SELF, frame=frame(3), enqueued_ms=220,
        reason=ObservationReason.UNREADABLE,
    )
    assert scheduler.pop_next(now_ms=230, self_opportunity=True).seat is Seat.SELF  # type: ignore[union-attr]
    scheduler.submit_raw(
        seat=Seat.SELF, frame=frame(4), enqueued_ms=240,
        reason=ObservationReason.UNREADABLE,
    )
    assert scheduler.pop_next(now_ms=250, self_opportunity=True).seat is Seat.LEFT  # type: ignore[union-attr]
    assert Seat.SELF in scheduler.snapshot().pending_seats


def test_strict_preferred_never_reads_a_foreign_seat() -> None:
    scheduler = ObservationScheduler(starvation_ms=1, raw_max_age_ms=1000)
    scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(1), enqueued_ms=100,
        reason=ObservationReason.UNREADABLE,
    )
    scheduler.submit_raw(
        seat=Seat.RIGHT, frame=frame(2), enqueued_ms=200,
        reason=ObservationReason.UNREADABLE,
    )
    selected = scheduler.pop_next(
        now_ms=500, expected_seat=Seat.RIGHT, strict_preferred=True
    )
    assert selected is not None and selected.seat is Seat.RIGHT
    assert scheduler.pop_next(
        now_ms=500, expected_seat=Seat.RIGHT, strict_preferred=True
    ) is None
    assert Seat.LEFT in scheduler.snapshot().pending_seats


def test_oldest_request_wins_at_starvation_threshold() -> None:
    scheduler = ObservationScheduler(starvation_ms=100, raw_max_age_ms=1000)
    scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(1), enqueued_ms=100,
        reason=ObservationReason.UNREADABLE,
    )
    scheduler.submit_raw(
        seat=Seat.SELF, frame=frame(2), enqueued_ms=190,
        reason=ObservationReason.UNREADABLE,
    )
    item = scheduler.pop_next(
        now_ms=210, expected_seat=Seat.SELF, self_opportunity=True
    )
    assert item is not None and item.seat is Seat.LEFT


def test_candidates_are_fifo_retained_not_latest_wins() -> None:
    scheduler = ObservationScheduler(candidate_capacity=3)
    one, two = candidate(1), candidate(2)
    assert scheduler.retain_candidate(one)
    assert scheduler.retain_candidate(two)
    assert scheduler.candidates() == (one, two)
    assert scheduler.pop_candidate(now_ms=250) == one
    assert scheduler.pop_candidate(now_ms=250) == two


def test_candidate_duplicate_key_includes_epoch_and_complete_frames() -> None:
    scheduler = ObservationScheduler(candidate_capacity=4)
    original = candidate(1)
    assert scheduler.retain_candidate(original)
    assert not scheduler.retain_candidate(
        replace(original, candidate_id="duplicate-id")
    )
    assert scheduler.drops[-1].reason is SchedulingDropReason.CANDIDATE_DUPLICATE

    next_epoch = replace(
        original,
        candidate_id="next-epoch",
        evidence_ids=("next-a", "next-b"),
        action_epoch=original.action_epoch + 1,
        first_frame=frame(3),
        last_frame=frame(4),
        processing_ms=401,
    )
    assert scheduler.retain_candidate(next_epoch)
    assert scheduler.candidates() == (original, next_epoch)


def test_candidate_capacity_and_expiry_have_distinct_reasons() -> None:
    scheduler = ObservationScheduler(candidate_capacity=1, candidate_max_age_ms=100)
    scheduler.retain_candidate(candidate(1), now_ms=100)
    scheduler.retain_candidate(candidate(2), now_ms=200)
    assert scheduler.drops[-1].reason is SchedulingDropReason.CANDIDATE_CAPACITY
    assert scheduler.pop_candidate(now_ms=401) is None
    assert scheduler.drops[-1].reason is SchedulingDropReason.CANDIDATE_EXPIRED


def test_stale_raw_does_not_replace_newer_one() -> None:
    scheduler = ObservationScheduler()
    scheduler.submit_raw(
        seat=Seat.RIGHT, frame=frame(5), reason=ObservationReason.UNREADABLE
    )
    assert not scheduler.submit_raw(
        seat=Seat.RIGHT, frame=frame(4), reason=ObservationReason.UNREADABLE
    )
    assert scheduler.drops[-1].reason is SchedulingDropReason.STALE_RAW
    assert scheduler.pop_next(now_ms=550).frame.frame_sequence == 5  # type: ignore[union-attr]


def test_bound_stream_rejects_delayed_generation_and_clears_prior_work() -> None:
    scheduler = ObservationScheduler()
    scheduler.bind_stream(frame(0), now_ms=0)
    scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(1), reason=ObservationReason.UNREADABLE
    )
    scheduler.retain_candidate(candidate(1))
    scheduler.bind_stream(
        FrameIdentity("s", 2, 0, 200, "roi-v2", "window-1"), now_ms=200
    )
    assert scheduler.snapshot().pending_seats == ()
    assert scheduler.candidates() == ()
    assert not scheduler.submit_raw(
        seat=Seat.LEFT, frame=frame(2), reason=ObservationReason.UNREADABLE
    )
    assert scheduler.drops[-1].reason is SchedulingDropReason.STREAM_MISMATCH


def test_bound_stream_rejects_same_generation_from_different_roi() -> None:
    scheduler = ObservationScheduler()
    scheduler.bind_stream(frame(0), now_ms=0)
    changed_roi = FrameIdentity(
        "s", 1, 1, 100, "roi-v2", "window-1"
    )
    assert not scheduler.submit_raw(
        seat=Seat.LEFT,
        frame=changed_roi,
        reason=ObservationReason.STALE_CAPTURE,
    )
    drop = scheduler.drops[-1]
    assert drop.reason is SchedulingDropReason.STREAM_MISMATCH
    assert drop.frame is changed_roi


def test_scheduler_records_require_contract_enums() -> None:
    with pytest.raises(TypeError, match="ObservationReason"):
        ScheduledRead(
            ScheduledItemKind.RAW,
            Seat.LEFT,
            frame(1),
            None,
            100,
            "changed",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="SchedulingDropReason"):
        SchedulingDrop(
            ScheduledItemKind.RAW,
            "raw_replaced",  # type: ignore[arg-type]
            Seat.LEFT,
            frame(1),
            0,
        )
