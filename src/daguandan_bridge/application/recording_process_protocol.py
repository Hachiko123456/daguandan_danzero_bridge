"""Spawn-safe DTOs for the per-session recording process."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class RecordingOperation(str, Enum):
    WRITE = "write"
    SAVE_EVIDENCE = "save_evidence"
    SCHEDULE_INCIDENT = "schedule_incident"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class RecordingProcessConfig:
    session_directory: str
    size: tuple[int, int]
    fps: float
    codec: str = "MJPG"
    max_video_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RecordingCommand:
    request_id: int
    operation: RecordingOperation
    payload: Any = None


@dataclass(frozen=True, slots=True)
class RecordingResponse:
    request_id: int
    status: str
    payload: Any = None
    error_type: str = ""
    message: str = ""


@dataclass(frozen=True, slots=True)
class RecordingReady:
    pid: int


class RecordingConnection(Protocol):
    def send(self, value: object) -> None: ...
    def recv(self) -> object: ...
    def poll(self, timeout: float = 0.0) -> bool: ...
    def close(self) -> None: ...


__all__ = [
    "RecordingCommand", "RecordingConnection", "RecordingOperation",
    "RecordingProcessConfig", "RecordingReady", "RecordingResponse",
]
