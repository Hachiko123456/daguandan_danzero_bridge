"""Typed messages for the live-v2 long-lived advice worker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..domain.advice import AdviceResult
from ..live_v2.game_state import TrustedGameSnapshot
from ..live_v2.identity import VersionIdentity
from ..live_v2.results import AdviceOpportunity
from .live_v2_worker_protocol import (
    WorkerReady, WorkerRequest, WorkerRequestTiming, WorkerResult,
)


class AdviceRuntimeStatus(str, Enum):
    ADVICE = "advice"
    BLOCKED = "blocked"
    CLOSED = "closed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    WORKER_ERROR = "worker_error"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_CRASHED = "worker_crashed"
    WORKER_RESTARTED = "worker_restarted"
    SERVICE_CLOSED = "service_closed"


class AdviceFailureKind(str, Enum):
    INVALID_REQUEST = "invalid_request"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    MODEL_ERROR = "model_error"


class WorkerHostPort(Protocol):
    """Application-facing contract for one supervised worker host."""

    @property
    def worker_generation(self) -> int: ...

    @property
    def worker_pid(self) -> int | None: ...

    @property
    def recent_request_timings(self) -> tuple[WorkerRequestTiming, ...]: ...

    def start(self, *, timeout: float = 10.0) -> WorkerReady: ...
    def bind_version(
        self, *, session_id: str, capture_generation: int, state_revision: int
    ) -> None: ...
    def submit(self, request: WorkerRequest) -> tuple[WorkerResult, ...]: ...
    def cancel_all(self, *, reason: str) -> tuple[WorkerResult, ...]: ...
    def get_result(self, *, timeout: float | None = None) -> WorkerResult: ...
    def drain_results(self) -> tuple[WorkerResult, ...]: ...
    def restart(
        self, *, timeout: float = 10.0, session_id: str | None = None,
        capture_generation: int | None = None, state_revision: int | None = None,
    ) -> WorkerReady: ...
    def close(self, *, timeout: float = 5.0) -> None: ...


class WorkerHostFactory(Protocol):
    def __call__(
        self, *, session_id: str, capture_generation: int, state_revision: int
    ) -> WorkerHostPort: ...


@dataclass(frozen=True, slots=True)
class AdviceWorkerConfig:
    profile_root: str
    advisor_backend: str
    profile_name: str = "tencent_daguandan"
    fabledan_runtime_policy: str = "model_required"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_root, str) or not isinstance(
            self.profile_name, str
        ) or not self.profile_root.strip() or not self.profile_name.strip():
            raise ValueError("profile_root and profile_name are required")
        if self.advisor_backend not in {"fabledan", "danzero"}:
            raise ValueError("advisor_backend must be fabledan or danzero")
        if self.fabledan_runtime_policy not in {
            "auto", "model_required", "rule_only"
        }:
            raise ValueError("unsupported FableDan runtime policy")


@dataclass(frozen=True, slots=True)
class AdvicePrewarmPayload:
    """Construct and initialize the configured adviser inside the child."""

    config: AdviceWorkerConfig
    worker_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.config, AdviceWorkerConfig):
            raise TypeError("config must be AdviceWorkerConfig")
        if self.worker_generation <= 0:
            raise ValueError("worker_generation must be positive")


@dataclass(frozen=True, slots=True)
class AdvisorReady:
    """Proof that one concrete adviser/model is initialized in the worker."""

    advisor_backend: str
    runtime_backend: str
    model_identity: str
    worker_generation: int
    worker_pid: int
    elapsed_ms: float
    cache_hit: bool
    model_path: str = ""
    model_digest: str = ""
    model_schema: str = ""
    model_status: str = ""

    def __post_init__(self) -> None:
        if self.advisor_backend not in {"fabledan", "danzero"}:
            raise ValueError("unsupported advisor backend")
        if not self.runtime_backend or not self.model_identity:
            raise ValueError("runtime backend and model identity are required")
        if self.worker_generation <= 0 or self.worker_pid <= 0:
            raise ValueError("worker generation and pid must be positive")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")

    @property
    def backend(self) -> str:
        return self.advisor_backend

    @property
    def pid(self) -> int:
        return self.worker_pid

    @property
    def generation(self) -> int:
        return self.worker_generation


@dataclass(frozen=True, slots=True)
class AdviceRequestIdentity:
    version: VersionIdentity
    request_sequence: int
    opportunity_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.version, VersionIdentity):
            raise TypeError("version must be VersionIdentity")
        if self.request_sequence < 0:
            raise ValueError("request_sequence must be non-negative")
        if not self.opportunity_id.strip():
            raise ValueError("opportunity_id is required")


@dataclass(frozen=True, slots=True)
class AdviceRequestTiming:
    """Bind a public advice request to its internal worker timing trace."""

    identity: AdviceRequestIdentity
    worker_generation: int
    worker_request_sequence: int
    worker: WorkerRequestTiming | None = None
    runtime_submit_entered_ms: int | None = None
    runtime_lock_acquired_ms: int | None = None
    version_bound_ms: int | None = None
    worker_request_built_ms: int | None = None
    host_submit_entered_ms: int | None = None
    host_submit_returned_ms: int | None = None


@dataclass(frozen=True, slots=True)
class AdviceWorkerPayload:
    config: AdviceWorkerConfig
    snapshot: TrustedGameSnapshot
    opportunity: AdviceOpportunity
    request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, AdviceWorkerConfig):
            raise TypeError("config must be AdviceWorkerConfig")
        if not isinstance(self.snapshot, TrustedGameSnapshot):
            raise TypeError("snapshot must be TrustedGameSnapshot")
        if not isinstance(self.opportunity, AdviceOpportunity):
            raise TypeError("opportunity must be AdviceOpportunity")
        if not self.request_id.strip():
            raise ValueError("request_id is required")


@dataclass(frozen=True, slots=True)
class AdviceWorkerSuccess:
    opportunity_id: str
    version: VersionIdentity
    advice: AdviceResult
    replayed_actions: int
    elapsed_ms: float
    advisor_cache_hit: bool = False
    advisor_ready: AdvisorReady | None = None


@dataclass(frozen=True, slots=True)
class AdviceWorkerFailure:
    opportunity_id: str
    version: VersionIdentity
    kind: AdviceFailureKind
    code: str
    error_type: str
    message: str
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class AdviceRuntimeResult:
    identity: AdviceRequestIdentity
    status: AdviceRuntimeStatus
    worker_generation: int
    worker_pid: int | None
    elapsed_ms: float = 0.0
    advice: AdviceResult | None = None
    failure_code: str = ""
    failure_type: str = ""
    message: str = ""
    advisor_cache_hit: bool = False
    advisor_ready: AdvisorReady | None = None

    @property
    def is_advice(self) -> bool:
        return self.status is AdviceRuntimeStatus.ADVICE

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AdviceRequestIdentity):
            raise TypeError("identity must be AdviceRequestIdentity")
        if not isinstance(self.status, AdviceRuntimeStatus):
            raise TypeError("status must be AdviceRuntimeStatus")
        if self.worker_generation < 0 or self.elapsed_ms < 0:
            raise ValueError("worker_generation and elapsed_ms must be non-negative")
        if self.status is AdviceRuntimeStatus.ADVICE:
            if self.advice is None:
                raise ValueError("ADVICE result requires advice")
        elif self.advice is not None:
            raise ValueError("non-advice result cannot retain advice")
        if self.advisor_ready is not None and not isinstance(
            self.advisor_ready, AdvisorReady
        ):
            raise TypeError("advisor_ready must be AdvisorReady")


def request_id(identity: AdviceRequestIdentity) -> str:
    version = identity.version
    return (
        f"{version.session_id}:{version.capture_generation}:"
        f"{identity.request_sequence}:{version.state_revision}:"
        f"{identity.opportunity_id}"
    )


def same_formal_advice_version(
    left: VersionIdentity, right: VersionIdentity
) -> bool:
    """Ignore only publication/update sequence for advice identity."""

    return (
        left.session_id,
        left.capture_generation,
        left.state_revision,
        left.turn_index,
    ) == (
        right.session_id,
        right.capture_generation,
        right.state_revision,
        right.turn_index,
    )


__all__ = [
    "AdvicePrewarmPayload",
    "AdviceFailureKind",
    "AdviceRequestIdentity",
    "AdviceRequestTiming",
    "AdviceRuntimeResult",
    "AdviceRuntimeStatus",
    "AdviceWorkerConfig",
    "AdviceWorkerFailure",
    "AdviceWorkerPayload",
    "AdviceWorkerSuccess",
    "AdvisorReady",
    "WorkerHostFactory",
    "WorkerHostPort",
    "request_id",
    "same_formal_advice_version",
]
