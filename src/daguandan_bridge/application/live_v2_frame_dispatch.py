"""Budgeted deep-read dispatch built from the existing live-v2 primitives."""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter_ns
from typing import Any, Callable

from ..live_v2.evidence_buffer import EvidenceBuffer, EvidenceDrop
from ..live_v2.scheduler import ObservationScheduler, SchedulingDrop
from ..live_v2.seat_tracker import SeatTracker, SeatTrackerSnapshot
from ..live_v2.types import (
    ActionCandidate,
    FrameIdentity,
    ObservationKind,
    ObservationReason,
    Seat,
    SeatObservation,
    VersionIdentity,
)
from ..live_v2.vision_adapter import VisionAdapter
from .live_v2_frame_types import (
    FramePipelineConfig,
    FramePipelineDrop,
    SeatSurfaceMetrics,
)
from .ports import RecognitionPort


SEATS: tuple[Seat, ...] = tuple(Seat)


@dataclass(frozen=True, slots=True)
class DispatchBatch:
    observations: tuple[SeatObservation, ...]
    candidates: tuple[ActionCandidate, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DropCursor:
    scheduling: tuple[SchedulingDrop, ...]
    evidence: tuple[EvidenceDrop, ...]


@dataclass(frozen=True, slots=True)
class _ReadPayload:
    image: Any
    metrics: SeatSurfaceMetrics


class FrameReadDispatcher:
    """Schedule expensive reads and feed existing trackers/evidence stores."""

    def __init__(
        self,
        recognition: RecognitionPort,
        *,
        config: FramePipelineConfig,
        vision_adapter: VisionAdapter | None = None,
        scheduler: ObservationScheduler | None = None,
        evidence_buffer: EvidenceBuffer | None = None,
        trackers: dict[Seat, SeatTracker] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.recognition = recognition
        self.config = config
        self.vision_adapter = vision_adapter or VisionAdapter()
        self.scheduler = scheduler or ObservationScheduler(
            raw_max_age_ms=config.raw_max_age_ms,
            candidate_max_age_ms=config.candidate_max_age_ms,
            candidate_capacity=config.candidate_capacity,
            starvation_ms=config.probe_starvation_ms,
        )
        self.evidence_buffer = evidence_buffer or EvidenceBuffer(
            max_bytes=config.evidence_max_bytes,
            max_age_ms=config.evidence_max_age_ms,
            max_count=config.evidence_max_count,
        )
        self.trackers = trackers or {seat: SeatTracker(seat) for seat in SEATS}
        if set(self.trackers) != set(SEATS):
            raise ValueError("trackers must contain exactly the four seats")
        self.clock_ms = clock_ms or (lambda: perf_counter_ns() // 1_000_000)
        self._last_deep_read_ms: dict[Seat, int | None] = {
            seat: None for seat in SEATS
        }

    def bind_stream(self, frame: FrameIdentity, *, now_ms: int) -> None:
        self.scheduler.bind_stream(frame, now_ms=now_ms)
        self.evidence_buffer.clear()
        for tracker in self.trackers.values():
            tracker.reset()
        self._last_deep_read_ms = {seat: None for seat in SEATS}

    def arm_persistent_passes(
        self, version: VersionIdentity, seats: tuple[Seat, ...]
    ) -> tuple[Seat, ...]:
        return tuple(
            seat for seat in seats
            if self.trackers[seat].arm_persistent_pass(
                version, turnover_confirmed=True
            )
        )

    def drop_cursor(self) -> DropCursor:
        return DropCursor(self.scheduler.drops, self.evidence_buffer.drops)

    def drops_since(self, cursor: DropCursor) -> tuple[FramePipelineDrop, ...]:
        scheduling = (
            _scheduler_drop(item)
            for item in _new_suffix(cursor.scheduling, self.scheduler.drops)
        )
        evidence = (
            _evidence_drop(item)
            for item in _new_suffix(cursor.evidence, self.evidence_buffer.drops)
        )
        return tuple(scheduling) + tuple(evidence)

    def schedule(
        self,
        image: Any,
        *,
        frame: FrameIdentity,
        metrics: tuple[SeatSurfaceMetrics, ...],
        self_opportunity: bool,
        expected_seat: Seat | None = None,
    ) -> None:
        for item in metrics:
            if expected_seat is not None and item.seat is not expected_seat:
                continue
            tracker = self.trackers[item.seat].snapshot()
            should_read = bool(
                item.content_changed
                or item.pass_visible
                or item.animating
                or item.followup_due
                or tracker.pending_count
                or self._is_starved(item.seat, frame.captured_ms)
                or (item.seat is Seat.SELF and self_opportunity)
                or (tracker.emitted_signature is not None and item.stable_empty)
            )
            if should_read:
                self.scheduler.submit_raw(
                    seat=item.seat,
                    frame=frame,
                    payload=_ReadPayload(image, item),
                    enqueued_ms=frame.captured_ms,
                    reason=_schedule_reason(item),
                )

    def drain(
        self,
        *,
        version: VersionIdentity,
        wild_rank: str,
        expected_seat: Seat | None,
        self_opportunity: bool,
        pass_eligible_seats: tuple[Seat, ...],
        formal_action_boundary: FrameIdentity | None,
        processing_base: int,
        started_clock: int,
    ) -> DispatchBatch:
        observations: list[SeatObservation] = []
        candidates: list[ActionCandidate] = []
        diagnostics: list[str] = []
        for _ in range(self.config.max_deep_reads_per_frame):
            item = self.scheduler.pop_next(
                now_ms=self.processing_ms(processing_base, started_clock),
                expected_seat=expected_seat,
                self_opportunity=self_opportunity,
            )
            if item is None:
                break
            if not isinstance(item.payload, _ReadPayload):
                diagnostics.append(f"invalid_read_payload:{item.seat.value}")
                continue
            observation = self._deep_read(
                item.payload,
                seat=item.seat,
                frame=item.frame,
                wild_rank=wild_rank,
                processing_base=processing_base,
                started_clock=started_clock,
            )
            pass_problem = (
                "pass_not_eligible_for_current_turn"
                if item.seat not in pass_eligible_seats
                else "pass_not_after_formal_action_boundary"
                if not _after_formal_boundary(
                    observation.frame, formal_action_boundary
                )
                else ""
            ) if observation.kind is ObservationKind.PASS else ""
            if pass_problem:
                observation = replace(
                    observation,
                    kind=ObservationKind.UNKNOWN,
                    confidence=0.0,
                    reason=ObservationReason.UNREADABLE,
                    diagnostics=observation.diagnostics + (pass_problem,),
                )
            self._last_deep_read_ms[item.seat] = item.frame.captured_ms
            observations.append(observation)
            processed_ms = max(
                observation.processing_ms,
                self.processing_ms(processing_base, started_clock),
            )
            self.evidence_buffer.add(
                observation,
                size_bytes=_observation_size(observation),
                now_ms=processed_ms,
            )
            candidate = self.trackers[item.seat].ingest(observation, version=version)
            if candidate is None:
                continue
            retained = self.scheduler.retain_candidate(candidate, now_ms=processed_ms)
            buffered = self.evidence_buffer.add(
                candidate,
                size_bytes=_candidate_size(candidate),
                now_ms=processed_ms,
            )
            if retained and buffered:
                candidates.append(candidate)
                diagnostics.append(
                    "candidate_confirmed:"
                    f"{candidate.candidate_id}:confidence={candidate.confidence:.3f}:"
                    f"diagnostics={'|'.join(candidate.diagnostics) or 'none'}"
                )
        return DispatchBatch(tuple(observations), tuple(candidates), tuple(diagnostics))

    def processing_ms(self, base: int, started_clock: int) -> int:
        return max(base, base + max(0, self.clock_ms() - started_clock))

    def pending_candidates(self, *, now_ms: int | None = None) -> tuple[ActionCandidate, ...]:
        return self.scheduler.candidates(now_ms=now_ms)

    def pop_candidate(self, *, now_ms: int) -> ActionCandidate | None:
        return self.scheduler.pop_candidate(now_ms=now_ms)

    def tracker_snapshots(self) -> tuple[SeatTrackerSnapshot, ...]:
        return tuple(self.trackers[seat].snapshot() for seat in SEATS)

    def _deep_read(
        self,
        payload: _ReadPayload,
        *,
        seat: Seat,
        frame: FrameIdentity,
        wild_rank: str,
        processing_base: int,
        started_clock: int,
    ) -> SeatObservation:
        try:
            result = self.recognition.recognize_play_region(
                payload.image,
                seat.value,
                wild_rank=wild_rank,
                allow_pass=True,
                allow_unknown_suit=True,
            )
            processing_ms = self.processing_ms(processing_base, started_clock)
            if result is None:
                return _failure(seat, frame, processing_ms, "play_region_returned_none")
            if Seat(result.player) is not seat:
                return _failure(seat, frame, processing_ms, "play_region_seat_mismatch")
            return self.vision_adapter.normalize_play_region(
                result,
                frame=frame,
                processing_ms=processing_ms,
                animating=payload.metrics.animating,
                empty_confirmed=payload.metrics.stable_empty,
            )
        except Exception as exc:
            return _failure(
                seat,
                frame,
                self.processing_ms(processing_base, started_clock),
                f"play_region_failed:{type(exc).__name__}",
            )

    def _is_starved(self, seat: Seat, captured_ms: int) -> bool:
        previous = self._last_deep_read_ms[seat]
        return previous is None or captured_ms - previous >= self.config.probe_starvation_ms


def _schedule_reason(item: SeatSurfaceMetrics) -> ObservationReason:
    if item.effect_visible or item.animating:
        return ObservationReason.ANIMATION_DETECTED
    if item.pass_visible:
        return ObservationReason.PASS_MARKER
    if item.stable_empty:
        return ObservationReason.STABLE_EMPTY
    return ObservationReason.UNREADABLE


def _failure(seat: Seat, frame: FrameIdentity, processing_ms: int, detail: str) -> SeatObservation:
    return SeatObservation(
        f"{frame.session_id}:{frame.capture_generation}:{frame.frame_sequence}:"
        f"{frame.roi_version}:{frame.source_id}:{seat.value}:failure",
        frame,
        seat,
        ObservationKind.UNKNOWN,
        (),
        0.0,
        ObservationReason.DETECTOR_FAILURE,
        max(frame.captured_ms, processing_ms),
        diagnostics=(detail,),
    )


def _observation_size(item: SeatObservation) -> int:
    return 192 + sum(len(value) for value in item.cards + item.diagnostics)


def _candidate_size(item: ActionCandidate) -> int:
    return 256 + sum(len(value) for value in item.cards + item.evidence_ids)


def _after_formal_boundary(
    frame: FrameIdentity, boundary: FrameIdentity | None
) -> bool:
    if boundary is None:
        return True
    if frame.captured_ms <= boundary.captured_ms:
        return False
    same_capture_stream = (
        frame.session_id, frame.capture_generation,
        frame.roi_version, frame.source_id,
    ) == (
        boundary.session_id, boundary.capture_generation,
        boundary.roi_version, boundary.source_id,
    )
    return not same_capture_stream or frame.frame_sequence > boundary.frame_sequence

def _new_suffix(before: tuple[Any, ...], after: tuple[Any, ...]) -> tuple[Any, ...]:
    for size in range(min(len(before), len(after)), -1, -1):
        if size == 0 or before[len(before) - size :] == after[:size]:
            return after[size:]
    return after


def _scheduler_drop(item: SchedulingDrop) -> FramePipelineDrop:
    return FramePipelineDrop(
        "scheduler", item.reason.value, item.seat, item.frame, item.age_ms
    )


def _evidence_drop(item: EvidenceDrop) -> FramePipelineDrop:
    return FramePipelineDrop(
        "evidence", item.reason.value, item.seat, item.frame, item.age_ms
    )

__all__ = ["DispatchBatch", "DropCursor", "FrameReadDispatcher"]
