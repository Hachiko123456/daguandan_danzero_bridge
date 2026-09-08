"""Immutable application contracts for the real-frame live-v2 adapter."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.recognition import FastSignalResult
from ..live_v2.types import (
    ActionCandidate,
    FrameIdentity,
    Seat,
    SeatObservation,
)


@dataclass(frozen=True, slots=True)
class SurfaceProbeConfig:
    fingerprint_width: int = 32
    minimum_fingerprint_height: int = 8
    change_threshold: float = 0.004
    occupied_delta_threshold: float = 0.035
    motion_threshold: float = 0.060
    quiet_threshold: float = 0.010
    bright_neutral_fraction: float = 0.008
    edge_fraction: float = 0.030
    contrast_threshold: float = 0.055
    empty_confirmations: int = 2
    followup_frames: int = 3

    def __post_init__(self) -> None:
        if self.fingerprint_width < 8 or self.minimum_fingerprint_height < 4:
            raise ValueError("fingerprint dimensions are too small")
        thresholds = (
            self.change_threshold,
            self.occupied_delta_threshold,
            self.motion_threshold,
            self.quiet_threshold,
            self.bright_neutral_fraction,
            self.edge_fraction,
            self.contrast_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("surface thresholds must be between zero and one")
        if self.empty_confirmations < 2:
            raise ValueError("empty_confirmations must require at least two frames")
        if self.followup_frames < 1:
            raise ValueError("followup_frames must be positive")


@dataclass(frozen=True, slots=True)
class SeatSurfaceMetrics:
    frame: FrameIdentity
    seat: Seat
    roi_shape: tuple[int, ...]
    fingerprint: str
    change_score: float
    motion_score: float
    baseline_delta: float
    card_like_fraction: float
    edge_fraction: float
    content_changed: bool
    visible_surface: bool
    animating: bool
    stable_empty: bool
    empty_streak: int
    followup_due: bool
    pass_visible: bool
    effect_visible: bool
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FramePipelineConfig:
    max_deep_reads_per_frame: int = 2
    probe_starvation_ms: int = 350
    raw_max_age_ms: int = 750
    candidate_max_age_ms: int = 3_000
    candidate_capacity: int = 128
    evidence_max_age_ms: int = 3_000
    evidence_max_count: int = 512
    evidence_max_bytes: int = 2 * 1024 * 1024
    strict_current_seat_only: bool = False

    def __post_init__(self) -> None:
        if min(
            self.max_deep_reads_per_frame,
            self.probe_starvation_ms,
            self.raw_max_age_ms,
            self.candidate_max_age_ms,
            self.candidate_capacity,
            self.evidence_max_age_ms,
            self.evidence_max_count,
            self.evidence_max_bytes,
        ) <= 0:
            raise ValueError("frame pipeline limits must be positive")
        if not isinstance(self.strict_current_seat_only, bool):
            raise TypeError("strict_current_seat_only must be bool")


@dataclass(frozen=True, slots=True)
class FramePipelineDrop:
    source: str
    reason: str
    seat: Seat
    frame: FrameIdentity
    age_ms: int


@dataclass(frozen=True, slots=True)
class FramePipelineResult:
    frame: FrameIdentity
    fast_signals: FastSignalResult
    surface_metrics: tuple[SeatSurfaceMetrics, ...]
    observations: tuple[SeatObservation, ...]
    candidates: tuple[ActionCandidate, ...]
    drops: tuple[FramePipelineDrop, ...]
    pending_seats: tuple[Seat, ...]
    candidate_backlog: int
    diagnostics: tuple[str, ...] = ()

    @property
    def seat_metrics(self) -> tuple[SeatSurfaceMetrics, ...]:
        return self.surface_metrics


__all__ = [
    "FramePipelineConfig",
    "FramePipelineDrop",
    "FramePipelineResult",
    "SeatSurfaceMetrics",
    "SurfaceProbeConfig",
]
