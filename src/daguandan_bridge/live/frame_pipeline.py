"""Shared entry from a canonical frame envelope into the live core."""

from __future__ import annotations

from typing import Mapping

from ..application.ports import LiveRuntimePort
from ..domain.frame import FrameEnvelope
from ..domain.live_runtime import LiveUpdate


def analyze_frame_envelope(
    runtime: LiveRuntimePort,
    envelope: FrameEnvelope,
    *,
    trace_context: Mapping[str, object] | None = None,
) -> LiveUpdate:
    """Analyze one frame without exposing its screen/video source to the core."""

    return runtime.analyze_frame(
        envelope.image,
        monotonic_ms=int(envelope.captured_monotonic_ms),
        trace_context=envelope.trace_context(trace_context),
    )


__all__ = ["analyze_frame_envelope"]