"""Frame-level orchestration for the replacement four-seat observer."""

from __future__ import annotations

from typing import Any, Callable

from ..domain.recognition import FastSignalResult
from ..live_v2.evidence_buffer import EvidenceBuffer
from ..live_v2.scheduler import ObservationScheduler
from ..live_v2.seat_tracker import SeatTracker, SeatTrackerSnapshot
from ..live_v2.types import ActionCandidate, FrameIdentity, Seat, VersionIdentity
from ..live_v2.vision_adapter import VisionAdapter
from .live_v2_frame_dispatch import FrameReadDispatcher
from .live_v2_frame_types import (
    FramePipelineConfig,
    FramePipelineDrop,
    FramePipelineResult,
    SeatSurfaceMetrics,
    SurfaceProbeConfig,
)
from .live_v2_surface_probe import FourSeatSurfaceProbe
from .ports import RecognitionPort


SEATS: tuple[Seat, ...] = tuple(Seat)


class LiveV2FramePipeline:
    """Probe one full frame, schedule bounded reads, and return evidence."""

    def __init__(
        self,
        recognition: RecognitionPort,
        *,
        config: FramePipelineConfig | None = None,
        surface_config: SurfaceProbeConfig | None = None,
        vision_adapter: VisionAdapter | None = None,
        scheduler: ObservationScheduler | None = None,
        evidence_buffer: EvidenceBuffer | None = None,
        trackers: dict[Seat, SeatTracker] | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.recognition = recognition
        self.config = config or FramePipelineConfig()
        self.dispatcher = FrameReadDispatcher(
            recognition,
            config=self.config,
            vision_adapter=vision_adapter,
            scheduler=scheduler,
            evidence_buffer=evidence_buffer,
            trackers=trackers,
            clock_ms=clock_ms,
        )
        self.surface_probe = FourSeatSurfaceProbe(
            recognition.play_roi,
            config=surface_config,
            seats=SEATS,
        )
        # Expose the composed primitives for diagnostics and migration code.
        self.scheduler = self.dispatcher.scheduler
        self.evidence_buffer = self.dispatcher.evidence_buffer
        self.trackers = self.dispatcher.trackers
        self.vision_adapter = self.dispatcher.vision_adapter
        self._stream: tuple[str, int, str, str] | None = None
        self._last_frame: FrameIdentity | None = None

    def process_frame(
        self,
        image: Any,
        *,
        frame: FrameIdentity,
        version: VersionIdentity,
        wild_rank: str,
        expected_seat: Seat | str | None = None,
        now_ms: int | None = None,
        formal_action_boundary: FrameIdentity | None = None,
    ) -> FramePipelineResult:
        if not version.belongs_to(frame):
            raise ValueError("version and frame must belong to the same capture stream")
        if not isinstance(wild_rank, str) or not wild_rank.strip():
            raise ValueError("wild_rank must be a non-empty string")
        expected = Seat(expected_seat) if expected_seat is not None else None
        processing_base = max(
            frame.captured_ms,
            frame.captured_ms if now_ms is None else now_ms,
        )
        started_clock = self.dispatcher.clock_ms()
        cursor = self.dispatcher.drop_cursor()

        stale_reason = self._stale_reason(frame)
        if stale_reason is not None:
            return self._stale_result(
                frame,
                expected=expected,
                processing_base=processing_base,
                reason=stale_reason,
            )
        self._bind_stream(frame, now_ms=processing_base)

        diagnostics: list[str] = []
        fast_seat = expected or Seat.SELF
        try:
            fast = self.recognition.recognize_fast_signals(
                image,
                fast_seat.value,
                allow_pass=True,
            )
        except Exception as exc:
            fast = _empty_fast(fast_seat)
            diagnostics.append(f"fast_signal_failed:{type(exc).__name__}")

        metrics = self.surface_probe.probe(
            image,
            frame=frame,
            pass_seats=_pass_seats(fast),
            effect_seats=(fast_seat,) if fast.effect_visible else (),
        )
        turnovers = _persistent_pass_turnovers(expected, fast)
        rearmed = self.dispatcher.arm_persistent_passes(version, turnovers)
        diagnostics.extend(
            f"persistent_pass_rearmed:{seat.value}" for seat in rearmed
        )
        self_opportunity = bool(
            fast.active_player == Seat.SELF.value
            or fast.self_action_buttons_visible
        )
        self.dispatcher.schedule(
            image,
            frame=frame,
            metrics=metrics,
            self_opportunity=self_opportunity,
        )
        batch = self.dispatcher.drain(
            version=version,
            wild_rank=wild_rank,
            expected_seat=expected,
            self_opportunity=self_opportunity,
            pass_eligible_seats=_pass_eligible_seats(expected, turnovers),
            formal_action_boundary=formal_action_boundary,
            processing_base=processing_base,
            started_clock=started_clock,
        )
        self._last_frame = frame
        final_ms = self.dispatcher.processing_ms(processing_base, started_clock)
        snapshot = self.scheduler.snapshot()
        return FramePipelineResult(
            frame=frame,
            fast_signals=fast,
            surface_metrics=metrics,
            observations=batch.observations,
            candidates=batch.candidates,
            drops=self.dispatcher.drops_since(cursor),
            pending_seats=snapshot.pending_seats,
            candidate_backlog=len(self.scheduler.candidates(now_ms=final_ms)),
            diagnostics=tuple(diagnostics) + batch.diagnostics,
        )

    def pending_candidates(self, *, now_ms: int | None = None) -> tuple[ActionCandidate, ...]:
        return self.dispatcher.pending_candidates(now_ms=now_ms)

    def pop_candidate(self, *, now_ms: int) -> ActionCandidate | None:
        return self.dispatcher.pop_candidate(now_ms=now_ms)

    def tracker_snapshots(self) -> tuple[SeatTrackerSnapshot, ...]:
        return self.dispatcher.tracker_snapshots()

    def _bind_stream(self, frame: FrameIdentity, *, now_ms: int) -> None:
        stream = _stream(frame)
        if stream == self._stream:
            return
        self.dispatcher.bind_stream(frame, now_ms=now_ms)
        self.surface_probe.reset()
        self._stream = stream

    def _stale_reason(self, frame: FrameIdentity) -> str | None:
        if self._stream is not None:
            session_id, generation, _, _ = self._stream
            if frame.session_id == session_id and frame.capture_generation < generation:
                return "older_capture_generation"
        previous = self._last_frame
        if (
            previous is not None
            and frame.session_id == previous.session_id
            and frame.capture_generation == previous.capture_generation
            and (
                frame.frame_sequence <= previous.frame_sequence
                or frame.captured_ms <= previous.captured_ms
            )
        ):
            return "stale_frame_identity"
        return None

    def _stale_result(
        self,
        frame: FrameIdentity,
        *,
        expected: Seat | None,
        processing_base: int,
        reason: str,
    ) -> FramePipelineResult:
        drops = tuple(
            FramePipelineDrop(
                "pipeline",
                reason,
                seat,
                frame,
                max(0, processing_base - frame.captured_ms),
            )
            for seat in SEATS
        )
        snapshot = self.scheduler.snapshot()
        return FramePipelineResult(
            frame,
            _empty_fast(expected or Seat.SELF),
            (),
            (),
            (),
            drops,
            snapshot.pending_seats,
            len(self.scheduler.candidates(now_ms=processing_base)),
            (reason,),
        )


def _empty_fast(expected: Seat) -> FastSignalResult:
    return FastSignalResult(expected.value, None, False, False, False)


def _pass_seats(fast: FastSignalResult) -> tuple[Seat, ...]:
    values: list[Seat] = []
    raw_values = list(fast.pass_marker_players)
    if fast.pass_marker_player is not None:
        raw_values.append(fast.pass_marker_player)
    if fast.pass_visible:
        raw_values.append(fast.expected_player)
    for raw in raw_values:
        try:
            seat = Seat(raw)
        except ValueError:
            continue
        if seat not in values:
            values.append(seat)
    return tuple(values)


def _persistent_pass_turnovers(
    expected: Seat | None, fast: FastSignalResult
) -> tuple[Seat, ...]:
    if (
        expected is None
        or fast.active_player is None
        or fast.effect_visible
        or fast.game_end_control is not None
    ):
        return ()
    try:
        active = Seat(fast.active_player)
    except ValueError:
        return ()
    if active is expected:
        return ()
    marked = set(_pass_seats(fast))
    order = tuple(Seat)
    crossed: list[Seat] = []
    cursor = expected
    for _ in range(len(order) - 1):
        if cursor is active:
            break
        crossed.append(cursor)
        cursor = order[(order.index(cursor) + 1) % len(order)]
        if cursor is active:
            # The active-seat jump may cross players that have already
            # finished and therefore never display a PASS marker.  Rearm only
            # crossed seats whose own marker is visible; never manufacture a
            # pass for an unmarked intermediate seat.  SeatTracker still
            # requires a newer formal turn, and the dispatcher still requires
            # two observations strictly after the formal action boundary.
            return tuple(seat for seat in crossed if seat in marked)
    return ()


def _pass_eligible_seats(
    expected: Seat | None, turnovers: tuple[Seat, ...]
) -> tuple[Seat, ...]:
    return tuple(dict.fromkeys((() if expected is None else (expected,)) + turnovers))


def _stream(frame: FrameIdentity) -> tuple[str, int, str, str]:
    return (
        frame.session_id,
        frame.capture_generation,
        frame.roi_version,
        frame.source_id,
    )


__all__ = [
    "FramePipelineConfig",
    "FramePipelineDrop",
    "FramePipelineResult",
    "LiveV2FramePipeline",
    "SeatSurfaceMetrics",
    "SurfaceProbeConfig",
]
