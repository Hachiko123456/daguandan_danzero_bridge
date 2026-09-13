"""Typed cross-process contracts for the live-v2 vision service.

Exactly one full screenshot is carried by each request. Seat ROI extraction,
surface probing and budgeted deep reads happen inside the worker process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from ..live_v2.identity import FrameIdentity, Seat, VersionIdentity
from .live_v2_frame_types import (
    FramePipelineConfig,
    FramePipelineResult,
    SurfaceProbeConfig,
)
from .live_v2_worker_protocol import WorkerReady, WorkerRequest, WorkerResult


class VisionRuntimeStatus(str, Enum):
    FRAME = "frame"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    WORKER_ERROR = "worker_error"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_CRASHED = "worker_crashed"
    WORKER_RESTARTED = "worker_restarted"
    SERVICE_CLOSED = "service_closed"


@dataclass(frozen=True, slots=True)
class VisionWorkerConfig:
    profile_root: str
    profile_name: str = "tencent_daguandan"
    pipeline: FramePipelineConfig = FramePipelineConfig()
    surface: SurfaceProbeConfig = SurfaceProbeConfig()
    diagnostic_tracing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile_root, str) or not self.profile_root.strip():
            raise ValueError("profile_root is required")
        if not isinstance(self.profile_name, str) or not self.profile_name.strip():
            raise ValueError("profile_name is required")
        if not isinstance(self.pipeline, FramePipelineConfig):
            raise TypeError("pipeline must be FramePipelineConfig")
        if not isinstance(self.surface, SurfaceProbeConfig):
            raise TypeError("surface must be SurfaceProbeConfig")
        if not isinstance(self.diagnostic_tracing, bool):
            raise TypeError("diagnostic_tracing must be bool")


@dataclass(frozen=True, slots=True)
class VisionRequestIdentity:
    frame: FrameIdentity
    version: VersionIdentity
    request_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.frame, FrameIdentity):
            raise TypeError("frame must be FrameIdentity")
        if not isinstance(self.version, VersionIdentity):
            raise TypeError("version must be VersionIdentity")
        if not self.version.belongs_to(self.frame):
            raise ValueError("version and frame belong to different capture streams")
        if isinstance(self.request_sequence, bool) or self.request_sequence < 0:
            raise ValueError("request_sequence must be non-negative")


@dataclass(frozen=True, slots=True)
class VisionWorkerPayload:
    config: VisionWorkerConfig
    identity: VisionRequestIdentity
    expected_seat: Seat | None
    visual_self_opportunity: bool
    wild_rank: str
    image: np.ndarray
    formal_action_boundary: FrameIdentity | None = None
    repair_seats: tuple[Seat, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config, VisionWorkerConfig):
            raise TypeError("config must be VisionWorkerConfig")
        if not isinstance(self.identity, VisionRequestIdentity):
            raise TypeError("identity must be VisionRequestIdentity")
        if self.expected_seat is not None and not isinstance(self.expected_seat, Seat):
            raise TypeError("expected_seat must be Seat or None")
        if not isinstance(self.repair_seats, tuple) or any(
            not isinstance(item, Seat) for item in self.repair_seats
        ):
            raise TypeError("repair_seats must be a tuple of Seat")
        if len(set(self.repair_seats)) != len(self.repair_seats):
            raise ValueError("repair_seats must be unique")
        if not isinstance(self.visual_self_opportunity, bool):
            raise TypeError("visual_self_opportunity must be bool")
        if not isinstance(self.wild_rank, str) or not self.wild_rank.strip():
            raise ValueError("wild_rank is required")
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if self.image.ndim not in {2, 3} or not self.image.size:
            raise ValueError("image must be one non-empty full screenshot")
        if self.formal_action_boundary is not None:
            if not isinstance(self.formal_action_boundary, FrameIdentity):
                raise TypeError("formal_action_boundary must be FrameIdentity or None")
            if (
                self.formal_action_boundary.session_id != self.identity.frame.session_id
                or self.formal_action_boundary.capture_generation
                != self.identity.frame.capture_generation
            ):
                raise ValueError("formal action boundary belongs to another stream")


@dataclass(frozen=True, slots=True)
class VisionWorkerSuccess:
    identity: VisionRequestIdentity
    pipeline_result: FramePipelineResult
    worker_pid: int
    elapsed_ms: float
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class VisionRuntimeResult:
    identity: VisionRequestIdentity
    status: VisionRuntimeStatus
    worker_generation: int
    worker_pid: int | None
    elapsed_ms: float = 0.0
    end_to_end_ms: float = 0.0
    pipeline_result: FramePipelineResult | None = None
    cache_hit: bool = False
    failure_code: str = ""
    failure_type: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        if self.status is VisionRuntimeStatus.FRAME:
            if self.pipeline_result is None:
                raise ValueError("FRAME requires a pipeline result")
        elif self.pipeline_result is not None:
            raise ValueError("non-frame result cannot carry observations")
        if min(self.worker_generation, self.elapsed_ms, self.end_to_end_ms) < 0:
            raise ValueError("generation and durations must be non-negative")


@runtime_checkable
class VisionWorkerHostPort(Protocol):
    """Narrow supervisor surface consumed by the application runtime."""

    @property
    def state(self) -> object: ...

    @property
    def worker_pid(self) -> int | None: ...

    @property
    def worker_generation(self) -> int: ...

    def start(self, *, timeout: float = 10.0) -> WorkerReady: ...

    def bind_version(
        self,
        *,
        session_id: str,
        capture_generation: int,
        state_revision: int,
    ) -> None: ...

    def submit(self, request: WorkerRequest) -> tuple[WorkerResult, ...]: ...

    def get_result(self, *, timeout: float | None = None) -> WorkerResult: ...

    def drain_results(self) -> tuple[WorkerResult, ...]: ...

    def restart(
        self,
        *,
        timeout: float = 10.0,
        session_id: str | None = None,
        capture_generation: int | None = None,
        state_revision: int | None = None,
    ) -> WorkerReady: ...

    def close(self, *, timeout: float = 5.0) -> None: ...


__all__ = [
    "VisionRequestIdentity",
    "VisionRuntimeResult",
    "VisionRuntimeStatus",
    "VisionWorkerConfig",
    "VisionWorkerHostPort",
    "VisionWorkerPayload",
    "VisionWorkerSuccess",
]
