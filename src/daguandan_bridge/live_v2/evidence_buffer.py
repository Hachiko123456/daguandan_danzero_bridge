"""Bounded evidence retention independent from raw image scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TypeAlias

from .types import (
    ActionCandidate,
    EvidenceDropReason,
    FrameIdentity,
    Seat,
    SeatObservation,
)


Evidence: TypeAlias = SeatObservation | ActionCandidate


def _captured_ms(item: Evidence) -> int:
    return item.last_captured_ms if isinstance(item, ActionCandidate) else item.frame.captured_ms


def _last_frame(item: Evidence) -> FrameIdentity:
    return item.last_frame if isinstance(item, ActionCandidate) else item.frame


def _stream(item: Evidence) -> tuple[str, int, str, str]:
    if isinstance(item, ActionCandidate):
        frame = item.last_frame
    else:
        frame = item.frame
    return (
        frame.session_id,
        frame.capture_generation,
        frame.roi_version,
        frame.source_id,
    )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence: Evidence
    size_bytes: int


@dataclass(frozen=True)
class EvidenceDrop:
    reason: EvidenceDropReason
    seat: Seat
    evidence_id: str
    frame: FrameIdentity
    age_ms: int
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.reason, EvidenceDropReason):
            raise TypeError("reason must be an EvidenceDropReason")
        if not isinstance(self.seat, Seat):
            raise TypeError("seat must be a Seat")
        if not isinstance(self.frame, FrameIdentity):
            raise TypeError("frame must be a FrameIdentity")


class EvidenceBuffer:
    """Keep observations/candidates within age, count and byte budgets.

    Consumption is seat-local and generation-aware. Once a boundary is
    marked, delayed evidence at or before that frame is rejected rather than
    re-entering a later action cycle.
    """

    def __init__(
        self,
        *,
        max_bytes: int = 16 * 1024 * 1024,
        max_age_ms: int = 1500,
        max_count: int = 256,
        max_drop_records: int = 256,
    ) -> None:
        if min(max_bytes, max_age_ms, max_count, max_drop_records) <= 0:
            raise ValueError("all evidence buffer limits must be positive")
        self.max_bytes = max_bytes
        self.max_age_ms = max_age_ms
        self.max_count = max_count
        self._records: deque[EvidenceRecord] = deque()
        self._retained_bytes = 0
        self._consumed: dict[tuple[Seat, str, int, str, str], FrameIdentity] = {}
        self._drops: deque[EvidenceDrop] = deque(maxlen=max_drop_records)

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def drops(self) -> tuple[EvidenceDrop, ...]:
        return tuple(self._drops)

    def add(
        self,
        evidence: Evidence,
        *,
        size_bytes: int = 0,
        now_ms: int | None = None,
    ) -> bool:
        if size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        captured_ms = _captured_ms(evidence)
        clock = captured_ms if now_ms is None else now_ms
        self.prune(clock)
        if self._is_consumed(evidence):
            self._drop(EvidenceDropReason.CONSUMED_BOUNDARY, evidence, size_bytes, clock)
            return False
        if clock - captured_ms > self.max_age_ms:
            self._drop(EvidenceDropReason.EXPIRED_ON_ARRIVAL, evidence, size_bytes, clock)
            return False
        if size_bytes > self.max_bytes:
            self._drop(EvidenceDropReason.OVERSIZE, evidence, size_bytes, clock)
            return False

        self._records.append(EvidenceRecord(evidence=evidence, size_bytes=size_bytes))
        self._retained_bytes += size_bytes
        while len(self._records) > self.max_count:
            self._evict_left(EvidenceDropReason.COUNT_BUDGET, clock)
        while self._retained_bytes > self.max_bytes:
            self._evict_left(EvidenceDropReason.BYTE_BUDGET, clock)
        return any(record.evidence is evidence for record in self._records)

    def available(self, *, seat: Seat | None = None) -> tuple[Evidence, ...]:
        return tuple(
            record.evidence
            for record in self._records
            if seat is None or record.evidence.seat == seat
        )

    def observations(self, *, seat: Seat | None = None) -> tuple[SeatObservation, ...]:
        return tuple(
            item
            for item in self.available(seat=seat)
            if isinstance(item, SeatObservation)
        )

    def candidates(self, *, seat: Seat | None = None) -> tuple[ActionCandidate, ...]:
        return tuple(
            item
            for item in self.available(seat=seat)
            if isinstance(item, ActionCandidate)
        )

    def mark_consumed(self, seat: Seat, through: FrameIdentity) -> int:
        key = self._boundary_key(seat, through)
        previous = self._consumed.get(key)
        if previous is None or self._is_at_or_before(previous, through):
            self._consumed[key] = through
        removed = 0
        kept: deque[EvidenceRecord] = deque()
        for record in self._records:
            if record.evidence.seat == seat and self._evidence_at_or_before(
                record.evidence, through
            ):
                self._retained_bytes -= record.size_bytes
                removed += 1
            else:
                kept.append(record)
        self._records = kept
        return removed

    def prune(self, now_ms: int) -> int:
        removed = 0
        kept: deque[EvidenceRecord] = deque()
        for record in self._records:
            captured_ms = _captured_ms(record.evidence)
            if now_ms - captured_ms > self.max_age_ms:
                self._retained_bytes -= record.size_bytes
                self._drop(
                    EvidenceDropReason.AGE_BUDGET,
                    record.evidence,
                    record.size_bytes,
                    now_ms,
                )
                removed += 1
            else:
                kept.append(record)
        self._records = kept
        return removed

    def clear(self) -> None:
        self._records.clear()
        self._retained_bytes = 0
        self._consumed.clear()

    def _evict_left(self, reason: EvidenceDropReason, now_ms: int) -> None:
        record = self._records.popleft()
        self._retained_bytes -= record.size_bytes
        self._drop(reason, record.evidence, record.size_bytes, now_ms)

    def _drop(
        self,
        reason: EvidenceDropReason,
        evidence: Evidence,
        size_bytes: int,
        now_ms: int,
    ) -> None:
        captured_ms = _captured_ms(evidence)
        evidence_id = (
            evidence.observation_id
            if isinstance(evidence, SeatObservation)
            else evidence.candidate_id
        )
        self._drops.append(
            EvidenceDrop(
                reason=reason,
                seat=evidence.seat,
                evidence_id=evidence_id,
                frame=_last_frame(evidence),
                age_ms=max(0, now_ms - captured_ms),
                size_bytes=size_bytes,
            )
        )

    def _is_consumed(self, evidence: Evidence) -> bool:
        session_id, generation, roi_version, source_id = _stream(evidence)
        key = (evidence.seat, session_id, generation, roi_version, source_id)
        boundary = self._consumed.get(key)
        return boundary is not None and self._evidence_at_or_before(evidence, boundary)

    @staticmethod
    def _boundary_key(
        seat: Seat, frame: FrameIdentity
    ) -> tuple[Seat, str, int, str, str]:
        return (
            seat,
            frame.session_id,
            frame.capture_generation,
            frame.roi_version,
            frame.source_id,
        )

    @staticmethod
    def _evidence_at_or_before(evidence: Evidence, boundary: FrameIdentity) -> bool:
        if _stream(evidence) != (
            boundary.session_id,
            boundary.capture_generation,
            boundary.roi_version,
            boundary.source_id,
        ):
            return False
        return EvidenceBuffer._is_at_or_before(_last_frame(evidence), boundary)

    @staticmethod
    def _is_at_or_before(frame: FrameIdentity, boundary: FrameIdentity) -> bool:
        if (
            frame.session_id != boundary.session_id
            or frame.capture_generation != boundary.capture_generation
            or frame.roi_version != boundary.roi_version
            or frame.source_id != boundary.source_id
        ):
            return False
        return (frame.frame_sequence, frame.captured_ms) <= (
            boundary.frame_sequence,
            boundary.captured_ms,
        )
