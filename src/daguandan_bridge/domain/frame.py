"""Canonical frame input shared by live capture and recorded replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FrameEnvelope:
    """One standardized image plus its capture identity and original time.

    Production capture and recorded-session replay create the same object.  The
    listener core therefore does not need to know whether an image came from a
    Windows window or an immutable AVI file.
    """

    image: Any
    captured_monotonic_ms: int
    wall_time: str
    frame_index: int | None = None
    capture_seq: int | None = None
    capture_generation: int = 0
    evidence_frame_id: str = ""
    roi_version: str = "capture-v1"
    source_id: str = ""

    def trace_context(
        self,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "frame_source": "canonical_envelope",
            "source_wall_time": self.wall_time,
            "captured_ms": int(self.captured_monotonic_ms),
            "capture_generation": int(self.capture_generation),
            "roi_version": self.roi_version,
            "source_id": self.source_id or self.roi_version,
        }
        if self.frame_index is not None:
            result["frame_index"] = int(self.frame_index)
        if self.capture_seq is not None:
            result["capture_seq"] = int(self.capture_seq)
        if self.evidence_frame_id:
            result["evidence_frame_id"] = self.evidence_frame_id
        if extra:
            result.update(dict(extra))
        return result


__all__ = ["FrameEnvelope"]