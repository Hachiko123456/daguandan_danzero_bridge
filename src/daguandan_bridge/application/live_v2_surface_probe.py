"""Cheap, seat-local surface measurements for the live-v2 frame pipeline.

The probe deliberately performs no card classification and knows nothing about
turn order. Its only job is to decide which seat regions deserve an expensive
read and whether an empty result has independent visual support. A first frame
is never learned as an empty baseline: blank evidence needs a quiet streak and
card-like pixels are checked before baseline calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Callable, Iterable

import numpy as np

from ..live_v2.types import FrameIdentity, Seat
from .live_v2_frame_types import SeatSurfaceMetrics, SurfaceProbeConfig


SEATS: tuple[Seat, ...] = tuple(Seat)


@dataclass(slots=True)
class _SeatProbeState:
    previous: np.ndarray | None = None
    empty_baseline: np.ndarray | None = None
    empty_streak: int = 0
    followup_remaining: int = 0


class FourSeatSurfaceProbe:
    """Measure all four play ROIs once per supplied capture frame."""

    def __init__(
        self,
        roi_provider: Callable[[Any, Any], Any],
        *,
        config: SurfaceProbeConfig | None = None,
        seats: Iterable[Seat] = SEATS,
    ) -> None:
        self.roi_provider = roi_provider
        self.config = config or SurfaceProbeConfig()
        self.seats = tuple(Seat(seat) for seat in seats)
        if not self.seats or len(set(self.seats)) != len(self.seats):
            raise ValueError("seats must contain unique values")
        self._states = {seat: _SeatProbeState() for seat in self.seats}
        self._stream: tuple[str, int, str, str] | None = None

    def reset(self) -> None:
        self._states = {seat: _SeatProbeState() for seat in self.seats}
        self._stream = None

    def probe(
        self,
        image: Any,
        *,
        frame: FrameIdentity,
        pass_seats: Iterable[Seat] = (),
        effect_seats: Iterable[Seat] = (),
    ) -> tuple[SeatSurfaceMetrics, ...]:
        stream = _stream(frame)
        if self._stream != stream:
            self._states = {seat: _SeatProbeState() for seat in self.seats}
            self._stream = stream
        passes = {Seat(seat) for seat in pass_seats}
        effects = {Seat(seat) for seat in effect_seats}
        return tuple(
            self._probe_seat(
                image,
                frame=frame,
                seat=seat,
                pass_visible=seat in passes,
                effect_visible=seat in effects,
            )
            for seat in self.seats
        )

    def _probe_seat(
        self,
        image: Any,
        *,
        frame: FrameIdentity,
        seat: Seat,
        pass_visible: bool,
        effect_visible: bool,
    ) -> SeatSurfaceMetrics:
        state = self._states[seat]
        diagnostics: list[str] = []
        try:
            roi = np.asarray(self.roi_provider(image, seat.value))
            gray, card_like = _gray_and_card_fraction(roi)
            fingerprint = _fingerprint(gray, self.config)
        except Exception as exc:
            diagnostics.append(f"surface_probe_failed:{type(exc).__name__}")
            return SeatSurfaceMetrics(
                frame, seat, (), "", 0.0, 0.0, 0.0, 0.0, 0.0,
                False, False, bool(effect_visible), False, 0, False,
                bool(pass_visible), bool(effect_visible), tuple(diagnostics),
            )

        motion = _difference(fingerprint, state.previous)
        baseline_delta = _difference(fingerprint, state.empty_baseline)
        edge_fraction, contrast = _edge_and_contrast(fingerprint)
        independent_surface = bool(
            card_like >= self.config.bright_neutral_fraction
            or (
                edge_fraction >= self.config.edge_fraction
                and contrast >= self.config.contrast_threshold
            )
        )
        visible = bool(
            pass_visible
            or independent_surface
            or (
                state.empty_baseline is not None
                and baseline_delta >= self.config.occupied_delta_threshold
            )
        )
        changed = bool(
            state.previous is not None and motion >= self.config.change_threshold
        )
        animating = bool(effect_visible or motion >= self.config.motion_threshold)

        quiet_blank = bool(
            not visible
            and not pass_visible
            and not effect_visible
            and motion <= self.config.quiet_threshold
        )
        state.empty_streak = state.empty_streak + 1 if quiet_blank else 0
        stable_empty = state.empty_streak >= self.config.empty_confirmations
        if stable_empty:
            if state.empty_baseline is None:
                state.empty_baseline = fingerprint.copy()
            elif baseline_delta < self.config.occupied_delta_threshold:
                state.empty_baseline = (
                    state.empty_baseline * 0.9 + fingerprint * 0.1
                ).astype(np.float32)

        followup_due = state.followup_remaining > 0
        if state.followup_remaining > 0:
            state.followup_remaining -= 1
        if changed or animating:
            state.followup_remaining = self.config.followup_frames
            followup_due = True
        state.previous = fingerprint

        return SeatSurfaceMetrics(
            frame=frame,
            seat=seat,
            roi_shape=tuple(int(value) for value in roi.shape),
            fingerprint=blake2b(fingerprint.tobytes(), digest_size=8).hexdigest(),
            change_score=motion,
            motion_score=motion,
            baseline_delta=baseline_delta,
            card_like_fraction=card_like,
            edge_fraction=edge_fraction,
            content_changed=changed,
            visible_surface=visible,
            animating=animating,
            stable_empty=stable_empty,
            empty_streak=state.empty_streak,
            followup_due=followup_due,
            pass_visible=bool(pass_visible),
            effect_visible=bool(effect_visible),
            diagnostics=tuple(diagnostics),
        )


def _stream(frame: FrameIdentity) -> tuple[str, int, str, str]:
    return (
        frame.session_id,
        frame.capture_generation,
        frame.roi_version,
        frame.source_id,
    )


def _gray_and_card_fraction(roi: np.ndarray) -> tuple[np.ndarray, float]:
    if roi.ndim not in {2, 3} or roi.size == 0:
        raise ValueError("play ROI must be a non-empty image")
    values = roi.astype(np.float32, copy=False)
    if values.max(initial=0.0) <= 1.0:
        values = values * 255.0
    if values.ndim == 2:
        gray = values
        neutral = np.ones(values.shape, dtype=bool)
    else:
        if values.shape[2] < 3:
            raise ValueError("colour play ROI requires at least three channels")
        bgr = values[..., :3]
        gray = bgr[..., 0] * 0.114 + bgr[..., 1] * 0.587 + bgr[..., 2] * 0.299
        neutral = bgr.max(axis=2) - bgr.min(axis=2) <= 60.0
    card_like = float(np.mean(neutral & (gray >= 140.0)))
    return np.clip(gray / 255.0, 0.0, 1.0), card_like


def _fingerprint(gray: np.ndarray, config: SurfaceProbeConfig) -> np.ndarray:
    height, width = gray.shape
    target_width = min(width, config.fingerprint_width)
    target_height = min(
        height,
        max(
            config.minimum_fingerprint_height,
            int(round(config.fingerprint_width * height / max(1, width))),
        ),
    )
    rows = np.linspace(0, height - 1, target_height).round().astype(int)
    cols = np.linspace(0, width - 1, target_width).round().astype(int)
    return gray[np.ix_(rows, cols)].astype(np.float32, copy=False)


def _difference(current: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None or previous.shape != current.shape:
        return 0.0
    return float(np.mean(np.abs(current - previous)))


def _edge_and_contrast(fingerprint: np.ndarray) -> tuple[float, float]:
    horizontal = np.abs(np.diff(fingerprint, axis=1)).ravel()
    vertical = np.abs(np.diff(fingerprint, axis=0)).ravel()
    edges = np.concatenate((horizontal, vertical))
    edge_fraction = float(np.mean(edges >= (24.0 / 255.0))) if edges.size else 0.0
    return edge_fraction, float(np.std(fingerprint))


__all__ = [
    "FourSeatSurfaceProbe",
    "SeatSurfaceMetrics",
    "SurfaceProbeConfig",
]
