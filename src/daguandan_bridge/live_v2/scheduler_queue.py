"""Bounded queues used by :mod:`live_v2.scheduler`.

Raw reads and confirmed candidates deliberately have different retention
semantics.  Keeping them in separate classes makes that policy impossible to
accidentally collapse into one latest-wins queue.
"""

from __future__ import annotations

from collections import deque

from .scheduling_records import (
    ScheduledRead,
    SchedulingDrop,
    candidate_key,
    drop,
    is_newer,
)
from .types import (
    ActionCandidate,
    ScheduledItemKind,
    SchedulingDropReason,
    Seat,
)


class RawSeatQueue:
    """Keep one newest raw read per seat."""

    def __init__(self, *, max_age_ms: int) -> None:
        self.max_age_ms = max_age_ms
        self._items: dict[Seat, ScheduledRead] = {}

    @property
    def seats(self) -> tuple[Seat, ...]:
        return tuple(self._items)

    @property
    def values(self) -> tuple[ScheduledRead, ...]:
        return tuple(self._items.values())

    def submit(
        self, item: ScheduledRead, *, now_ms: int
    ) -> tuple[bool, SchedulingDrop | None]:
        current = self._items.get(item.seat)
        if current is not None and not is_newer(item.frame, current.frame):
            return False, drop(
                ScheduledItemKind.RAW,
                SchedulingDropReason.STALE_RAW,
                item.seat,
                item.frame,
                now_ms,
            )
        replaced = None
        if current is not None:
            replaced = drop(
                ScheduledItemKind.RAW,
                SchedulingDropReason.RAW_REPLACED,
                current.seat,
                current.frame,
                now_ms,
            )
        self._items[item.seat] = item
        return True, replaced

    def pop(self, seat: Seat) -> ScheduledRead:
        return self._items.pop(seat)

    def oldest(self, *, exclude: Seat | None = None) -> ScheduledRead:
        return min(
            (item for seat, item in self._items.items() if seat != exclude),
            key=lambda item: (item.enqueued_ms, item.frame.frame_sequence),
        )

    def get(self, seat: Seat) -> ScheduledRead | None:
        return self._items.get(seat)

    def expire(self, *, now_ms: int) -> tuple[SchedulingDrop, ...]:
        expired: list[SchedulingDrop] = []
        for seat, item in tuple(self._items.items()):
            if now_ms - item.frame.captured_ms <= self.max_age_ms:
                continue
            del self._items[seat]
            expired.append(
                drop(
                    ScheduledItemKind.RAW,
                    SchedulingDropReason.RAW_EXPIRED,
                    seat,
                    item.frame,
                    now_ms,
                )
            )
        return tuple(expired)

    def clear(self, *, now_ms: int) -> tuple[SchedulingDrop, ...]:
        removed = tuple(
            drop(
                ScheduledItemKind.RAW,
                SchedulingDropReason.STREAM_REBOUND,
                item.seat,
                item.frame,
                now_ms,
            )
            for item in self._items.values()
        )
        self._items.clear()
        return removed


class CandidateQueue:
    """Retain confirmed candidates in FIFO order with bounded age/count."""

    def __init__(self, *, max_age_ms: int, capacity: int) -> None:
        self.max_age_ms = max_age_ms
        self.capacity = capacity
        self._items: deque[ActionCandidate] = deque()
        self._keys: set[tuple[object, ...]] = set()

    @property
    def items(self) -> tuple[ActionCandidate, ...]:
        return tuple(self._items)

    def retain(
        self, candidate: ActionCandidate, *, now_ms: int
    ) -> tuple[bool, tuple[SchedulingDrop, ...]]:
        drops = list(self.expire(now_ms=now_ms))
        age = now_ms - candidate.last_captured_ms
        if age > self.max_age_ms:
            drops.append(self._drop(candidate, SchedulingDropReason.CANDIDATE_EXPIRED, now_ms))
            return False, tuple(drops)
        key = candidate_key(candidate)
        if key in self._keys:
            drops.append(self._drop(candidate, SchedulingDropReason.CANDIDATE_DUPLICATE, now_ms))
            return False, tuple(drops)
        self._items.append(candidate)
        self._keys.add(key)
        while len(self._items) > self.capacity:
            evicted = self._items.popleft()
            self._keys.discard(candidate_key(evicted))
            drops.append(self._drop(evicted, SchedulingDropReason.CANDIDATE_CAPACITY, now_ms))
        return True, tuple(drops)

    def pop(self, *, now_ms: int) -> tuple[ActionCandidate | None, tuple[SchedulingDrop, ...]]:
        drops = self.expire(now_ms=now_ms)
        if not self._items:
            return None, drops
        candidate = self._items.popleft()
        self._keys.discard(candidate_key(candidate))
        return candidate, drops

    def expire(self, *, now_ms: int) -> tuple[SchedulingDrop, ...]:
        kept: deque[ActionCandidate] = deque()
        expired: list[SchedulingDrop] = []
        while self._items:
            candidate = self._items.popleft()
            if now_ms - candidate.last_captured_ms > self.max_age_ms:
                self._keys.discard(candidate_key(candidate))
                expired.append(
                    self._drop(candidate, SchedulingDropReason.CANDIDATE_EXPIRED, now_ms)
                )
            else:
                kept.append(candidate)
        self._items = kept
        return tuple(expired)

    def clear(self, *, now_ms: int) -> tuple[SchedulingDrop, ...]:
        removed = tuple(
            self._drop(candidate, SchedulingDropReason.STREAM_REBOUND, now_ms)
            for candidate in self._items
        )
        self._items.clear()
        self._keys.clear()
        return removed

    @staticmethod
    def _drop(
        candidate: ActionCandidate,
        reason: SchedulingDropReason,
        now_ms: int,
    ) -> SchedulingDrop:
        return drop(
            ScheduledItemKind.CANDIDATE,
            reason,
            candidate.seat,
            candidate.last_frame,
            now_ms,
        )
