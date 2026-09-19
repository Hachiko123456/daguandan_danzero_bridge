"""Public fair scheduler for seat-local observation work.

The scheduler owns only stream binding and priority policy. Replaceable raw
reads and durable candidate evidence are managed by separate queue classes so
their retention semantics cannot be accidentally mixed.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .scheduler_queue import CandidateQueue, RawSeatQueue
from .scheduling_records import (
    SEATS,
    ScheduledRead,
    SchedulerSnapshot,
    SchedulingDrop,
    drop,
    frame_stream,
)
from .types import (
    ActionCandidate,
    FrameIdentity,
    ObservationReason,
    ScheduledItemKind,
    SchedulingDropReason,
    Seat,
)


class ObservationScheduler:
    """Prioritize expected/local reads without starving any changed seat."""

    def __init__(
        self,
        *,
        seats: Iterable[Seat] = SEATS,
        raw_max_age_ms: int = 750,
        candidate_max_age_ms: int = 3000,
        candidate_capacity: int = 128,
        starvation_ms: int = 350,
        preferred_burst_limit: int = 2,
        max_drop_records: int = 256,
    ) -> None:
        self.seats = tuple(seats)
        limits = (
            raw_max_age_ms,
            candidate_max_age_ms,
            candidate_capacity,
            starvation_ms,
            preferred_burst_limit,
            max_drop_records,
        )
        if not self.seats or len(set(self.seats)) != len(self.seats):
            raise ValueError("seats must contain unique values")
        if min(limits) <= 0:
            raise ValueError("all scheduler limits must be positive")

        self.raw_max_age_ms = raw_max_age_ms
        self.candidate_max_age_ms = candidate_max_age_ms
        self.candidate_capacity = candidate_capacity
        self.starvation_ms = starvation_ms
        self.preferred_burst_limit = preferred_burst_limit
        self._raw = RawSeatQueue(max_age_ms=raw_max_age_ms)
        self._candidates = CandidateQueue(
            max_age_ms=candidate_max_age_ms,
            capacity=candidate_capacity,
        )
        self._drops: deque[SchedulingDrop] = deque(maxlen=max_drop_records)
        self._bound_stream: tuple[str, int, str, str] | None = None
        self._last_preferred: Seat | None = None
        self._preferred_burst = 0

    @property
    def drops(self) -> tuple[SchedulingDrop, ...]:
        return tuple(self._drops)

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            pending_seats=self._raw.seats,
            retained_candidates=len(self._candidates.items),
            drops=self.drops,
        )

    def bind_stream(self, frame: FrameIdentity, *, now_ms: int) -> None:
        """Bind all queued work to one complete capture/ROI identity."""

        stream = frame_stream(frame)
        if stream == self._bound_stream:
            return
        self._record(self._raw.clear(now_ms=now_ms))
        self._record(self._candidates.clear(now_ms=now_ms))
        self._bound_stream = stream
        self._last_preferred = None
        self._preferred_burst = 0

    def submit_raw(
        self,
        *,
        seat: Seat,
        frame: FrameIdentity,
        reason: ObservationReason,
        payload: object | None = None,
        enqueued_ms: int | None = None,
    ) -> bool:
        """Keep only the newest raw frame for one changed seat."""

        self._validate_seat(seat)
        queued_at = frame.captured_ms if enqueued_ms is None else enqueued_ms
        if not self._accept_or_bind(frame):
            self._record_one(
                drop(
                    ScheduledItemKind.RAW,
                    SchedulingDropReason.STREAM_MISMATCH,
                    seat,
                    frame,
                    queued_at,
                )
            )
            return False
        item = ScheduledRead(
            item_kind=ScheduledItemKind.RAW,
            seat=seat,
            frame=frame,
            payload=payload,
            enqueued_ms=queued_at,
            reason=reason,
        )
        accepted, replaced = self._raw.submit(item, now_ms=queued_at)
        if replaced is not None:
            self._record_one(replaced)
        return accepted

    def submit_frame(
        self,
        *,
        frame: FrameIdentity,
        changed_seats: Iterable[Seat],
        reason: ObservationReason,
        payload: object | None = None,
        enqueued_ms: int | None = None,
    ) -> int:
        """Schedule one shared capture for every changed seat."""

        accepted = 0
        for seat in dict.fromkeys(changed_seats):
            accepted += int(
                self.submit_raw(
                    seat=seat,
                    frame=frame,
                    reason=reason,
                    payload=payload,
                    enqueued_ms=enqueued_ms,
                )
            )
        return accepted

    def pop_next(
        self,
        *,
        now_ms: int,
        expected_seat: Seat | None = None,
        opening_lead_seat: Seat | None = None,
        self_opportunity: bool = False,
        strict_preferred: bool = False,
    ) -> ScheduledRead | None:
        """Pop one read using priority boosts with bounded starvation."""

        self._record(self._raw.expire(now_ms=now_ms))
        if not self._raw.seats:
            return None
        if expected_seat is not None:
            self._validate_seat(expected_seat)

        preferred = self._preferred(
            expected_seat, opening_lead_seat, self_opportunity
        )
        if strict_preferred and preferred is not None:
            chosen = self._raw.get(preferred)
            if chosen is None:
                return None
            item = self._raw.pop(chosen.seat)
            self._update_burst(preferred, item.seat)
            return item

        oldest = self._raw.oldest()
        if max(0, now_ms - oldest.enqueued_ms) >= self.starvation_ms:
            chosen = oldest
        elif self._must_yield(preferred):
            chosen = self._raw.oldest(exclude=preferred)
        else:
            chosen = self._raw.get(preferred) if preferred is not None else None
            chosen = chosen or oldest

        item = self._raw.pop(chosen.seat)
        self._update_burst(preferred, item.seat)
        return item

    def retain_candidate(
        self, candidate: ActionCandidate, *, now_ms: int | None = None
    ) -> bool:
        """Retain a candidate in FIFO order; never latest-wins replace it."""

        self._validate_seat(candidate.seat)
        clock = candidate.last_captured_ms if now_ms is None else now_ms
        if not self._accept_or_bind(candidate.last_frame):
            self._record_one(
                drop(
                    ScheduledItemKind.CANDIDATE,
                    SchedulingDropReason.STREAM_MISMATCH,
                    candidate.seat,
                    candidate.last_frame,
                    clock,
                )
            )
            return False
        accepted, drops = self._candidates.retain(candidate, now_ms=clock)
        self._record(drops)
        return accepted

    def pop_candidate(self, *, now_ms: int) -> ActionCandidate | None:
        candidate, drops = self._candidates.pop(now_ms=now_ms)
        self._record(drops)
        return candidate

    def candidates(self, *, now_ms: int | None = None) -> tuple[ActionCandidate, ...]:
        if now_ms is not None:
            self._record(self._candidates.expire(now_ms=now_ms))
        return self._candidates.items

    def _preferred(
        self,
        expected_seat: Seat | None,
        opening_lead_seat: Seat | None,
        self_opportunity: bool,
    ) -> Seat | None:
        # Opening evidence has its own explicit priority.  It is deliberately
        # separate from expected_seat because the opening barrier may know the
        # lead before the formal turn cursor has been established.
        if opening_lead_seat is not None and self._raw.get(opening_lead_seat) is not None:
            return opening_lead_seat
        if expected_seat is not None:
            return expected_seat
        if self_opportunity and self._raw.get(Seat.SELF) is not None:
            return Seat.SELF
        return None

    def _must_yield(self, preferred: Seat | None) -> bool:
        return bool(
            preferred is not None
            and self._raw.get(preferred) is not None
            and self._last_preferred == preferred
            and self._preferred_burst >= self.preferred_burst_limit
            and any(seat != preferred for seat in self._raw.seats)
        )

    def _update_burst(self, preferred: Seat | None, selected: Seat) -> None:
        if preferred is not None and selected == preferred:
            self._preferred_burst = (
                self._preferred_burst + 1
                if self._last_preferred == preferred
                else 1
            )
            self._last_preferred = preferred
        else:
            self._last_preferred = None
            self._preferred_burst = 0

    def _accept_or_bind(self, frame: FrameIdentity) -> bool:
        stream = frame_stream(frame)
        if self._bound_stream is None:
            self._bound_stream = stream
            return True
        return self._bound_stream == stream

    def _validate_seat(self, seat: Seat) -> None:
        if seat not in self.seats:
            raise ValueError(f"unknown scheduler seat: {seat!r}")

    def _record(self, drops: Iterable[SchedulingDrop]) -> None:
        self._drops.extend(drops)

    def _record_one(self, item: SchedulingDrop) -> None:
        self._drops.append(item)


__all__ = [
    "ObservationScheduler",
    "ScheduledRead",
    "SchedulerSnapshot",
    "SchedulingDrop",
]
