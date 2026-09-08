"""Immutable scheduling records and identity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .types import (
    ActionCandidate,
    FrameIdentity,
    ObservationReason,
    ScheduledItemKind,
    SchedulingDropReason,
    Seat,
)


StreamIdentity: TypeAlias = tuple[str, int, str, str]
SEATS: tuple[Seat, ...] = tuple(Seat)


@dataclass(frozen=True)
class ScheduledRead:
    item_kind: ScheduledItemKind
    seat: Seat
    frame: FrameIdentity
    payload: object | None
    enqueued_ms: int
    reason: ObservationReason

    def __post_init__(self) -> None:
        if self.item_kind is not ScheduledItemKind.RAW:
            raise ValueError("ScheduledRead item_kind must be RAW")
        if not isinstance(self.reason, ObservationReason):
            raise TypeError("reason must be an ObservationReason")
        if not isinstance(self.frame, FrameIdentity):
            raise TypeError("frame must be a FrameIdentity")


@dataclass(frozen=True)
class SchedulingDrop:
    item_kind: ScheduledItemKind
    reason: SchedulingDropReason
    seat: Seat
    frame: FrameIdentity
    age_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.item_kind, ScheduledItemKind):
            raise TypeError("item_kind must be a ScheduledItemKind")
        if not isinstance(self.reason, SchedulingDropReason):
            raise TypeError("reason must be a SchedulingDropReason")
        if not isinstance(self.frame, FrameIdentity):
            raise TypeError("frame must be a FrameIdentity")


@dataclass(frozen=True)
class SchedulerSnapshot:
    pending_seats: tuple[Seat, ...]
    retained_candidates: int
    drops: tuple[SchedulingDrop, ...]


def frame_stream(frame: FrameIdentity) -> StreamIdentity:
    return (
        frame.session_id,
        frame.capture_generation,
        frame.roi_version,
        frame.source_id,
    )


def frame_key(frame: FrameIdentity) -> tuple[object, ...]:
    return (
        *frame_stream(frame),
        frame.frame_sequence,
        frame.captured_ms,
    )


def is_newer(frame: FrameIdentity, current: FrameIdentity) -> bool:
    if frame_stream(frame) != frame_stream(current):
        return True
    return (frame.frame_sequence, frame.captured_ms) > (
        current.frame_sequence,
        current.captured_ms,
    )


def candidate_key(candidate: ActionCandidate) -> tuple[object, ...]:
    return (
        candidate.seat,
        candidate.kind,
        candidate.cards,
        candidate.suit_options,
        candidate.action_epoch,
        candidate.evidence_ids,
        frame_key(candidate.first_frame),
        frame_key(candidate.last_frame),
    )


def drop(
    item_kind: ScheduledItemKind,
    reason: SchedulingDropReason,
    seat: Seat,
    frame: FrameIdentity,
    now_ms: int,
) -> SchedulingDrop:
    return SchedulingDrop(
        item_kind=item_kind,
        reason=reason,
        seat=seat,
        frame=frame,
        age_ms=max(0, now_ms - frame.captured_ms),
    )
