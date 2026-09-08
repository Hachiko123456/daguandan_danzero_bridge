"""Serializable protocol for supervised ``live_v2`` worker processes.

Only standard-library, immutable value objects live here. A worker callable
receives a :class:`WorkerRequest` and returns one pickleable value. The
callable itself is identified by module and function name so that Windows'
``spawn`` start method and frozen applications never have to pickle a closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib
from typing import Any, Callable


class DeliveryMode(str, Enum):
    """How a request waits while the worker is busy."""

    LATEST_ONLY = "latest_only"
    FIFO = "fifo"


class WorkerResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    DROPPED = "dropped"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    CRASHED = "crashed"


@dataclass(frozen=True, slots=True)
class WorkerReference:
    """Import address of one module-level worker function."""

    module_path: str
    function_name: str

    def __post_init__(self) -> None:
        if not self.module_path or not self.function_name:
            raise ValueError("worker module_path and function_name are required")
        if not all(part.isidentifier() for part in self.module_path.split(".")):
            raise ValueError("worker module_path must be an importable dotted path")
        if not self.function_name.isidentifier():
            raise ValueError("worker function_name must name a module-level function")

    @classmethod
    def from_callable(cls, worker: Callable[["WorkerRequest"], Any]) -> "WorkerReference":
        module = getattr(worker, "__module__", "")
        name = getattr(worker, "__name__", "")
        qualname = getattr(worker, "__qualname__", "")
        if name == "<lambda>" or "<locals>" in qualname or qualname != name:
            raise ValueError("worker must be an importable module-level function")
        reference = cls(module_path=module, function_name=name)
        if reference.resolve() is not worker:
            raise ValueError("worker is not exported by its declared module")
        return reference

    def resolve(self) -> Callable[["WorkerRequest"], Any]:
        module = importlib.import_module(self.module_path)
        worker = getattr(module, self.function_name, None)
        if not callable(worker):
            raise TypeError(
                f"{self.module_path}.{self.function_name} is not callable"
            )
        return worker


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    """One version-bound unit of work sent to the long-lived process."""

    session_id: str
    capture_generation: int
    request_sequence: int
    state_revision: int
    submitted_processing_ms: int
    payload: Any = None
    delivery: DeliveryMode = DeliveryMode.LATEST_ONLY
    timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        for name in (
            "capture_generation",
            "request_sequence",
            "state_revision",
            "submitted_processing_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive when supplied")
        if not isinstance(self.delivery, DeliveryMode):
            raise TypeError("delivery must be a DeliveryMode")


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Structured failure information safe to log or cross a process pipe."""

    code: str
    error_type: str
    message: str
    traceback_text: str = ""


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Terminal result for exactly one request."""

    session_id: str
    capture_generation: int
    request_sequence: int
    state_revision: int
    submitted_processing_ms: int
    status: WorkerResultStatus
    worker_generation: int
    worker_pid: int | None
    started_processing_ms: int | None
    finished_processing_ms: int
    payload: Any = None
    failure: WorkerFailure | None = None

    @property
    def is_success(self) -> bool:
        return self.status is WorkerResultStatus.SUCCESS

    @classmethod
    def terminal(
        cls,
        request: WorkerRequest,
        *,
        status: WorkerResultStatus,
        worker_generation: int,
        finished_processing_ms: int,
        worker_pid: int | None = None,
        started_processing_ms: int | None = None,
        payload: Any = None,
        failure: WorkerFailure | None = None,
    ) -> "WorkerResult":
        return cls(
            session_id=request.session_id,
            capture_generation=request.capture_generation,
            request_sequence=request.request_sequence,
            state_revision=request.state_revision,
            submitted_processing_ms=request.submitted_processing_ms,
            status=status,
            worker_generation=worker_generation,
            worker_pid=worker_pid,
            started_processing_ms=started_processing_ms,
            finished_processing_ms=finished_processing_ms,
            payload=payload,
            failure=failure,
        )


@dataclass(frozen=True, slots=True)
class WorkerReady:
    worker_generation: int
    worker_pid: int
    ready_processing_ms: int


@dataclass(frozen=True, slots=True)
class WorkerRequestTiming:
    """Bounded host-visible timing trace for one worker request."""

    worker_generation: int
    request_sequence: int
    host_accepted_ms: int
    send_started_ms: int | None = None
    send_finished_ms: int | None = None
    child_received_ms: int | None = None
    child_started_ms: int | None = None
    child_finished_ms: int | None = None
    result_received_ms: int | None = None


@dataclass(frozen=True, slots=True)
class _RunCommand:
    worker_generation: int
    request: WorkerRequest


@dataclass(frozen=True, slots=True)
class _WorkerTimingEvent:
    worker_generation: int
    request_sequence: int
    stage: str
    processing_ms: int


@dataclass(frozen=True, slots=True)
class _StopCommand:
    worker_generation: int


@dataclass(frozen=True, slots=True)
class _WorkerStopped:
    worker_generation: int
    worker_pid: int


@dataclass(frozen=True, slots=True)
class _WorkerStartupFailure:
    worker_generation: int
    worker_pid: int
    failure: WorkerFailure


__all__ = [
    "DeliveryMode",
    "WorkerFailure",
    "WorkerReady",
    "WorkerReference",
    "WorkerRequest",
    "WorkerRequestTiming",
    "WorkerResult",
    "WorkerResultStatus",
]
