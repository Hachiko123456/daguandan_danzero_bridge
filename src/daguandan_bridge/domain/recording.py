from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RecorderWarning:
    reason: str
    monotonic_ms: int
    details: str


@dataclass(frozen=True)
class IncidentMediaFailure:
    reason: str
    incident_directory: Path
    trigger_ms: int
    details: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "incident_directory": str(self.incident_directory),
            "trigger_ms": self.trigger_ms,
            "details": self.details,
        }


@dataclass(frozen=True)
class RecordingResult:
    video_path: Path
    index_path: Path
    frame_count: int
    dropped_frames: int
    incident_media_failures: tuple[IncidentMediaFailure, ...] = ()
    integrity: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IncidentMedia:
    clip_path: Path
    contact_sheet_path: Path
    trigger_frame_path: Path
    frame_count: int
