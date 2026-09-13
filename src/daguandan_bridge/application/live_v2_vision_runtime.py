"""Latest-only supervisor for the long-lived live-v2 vision process."""
from __future__ import annotations
from threading import RLock
import time
import numpy as np
from ..live_v2.identity import FrameIdentity, Seat, VersionIdentity
from .live_v2_vision_protocol import (
    VisionRequestIdentity,
    VisionRuntimeResult,
    VisionRuntimeStatus,
    VisionWorkerConfig,
    VisionWorkerHostPort,
    VisionWorkerPayload,
    VisionWorkerSuccess,
)
from .live_v2_worker_protocol import (
    DeliveryMode,
    WorkerRequest,
    WorkerResult,
    WorkerResultStatus,
)
from .live_v2_vision_sync import VisionSyncMixin


class LiveV2VisionRuntime(VisionSyncMixin):
    """Own one worker, one latest frame slot and strict result version gates."""
    def __init__(
        self,
        config: VisionWorkerConfig,
        *,
        host: VisionWorkerHostPort,
        session_id: str,
        capture_generation: int,
    ) -> None:
        self._config = config
        if not isinstance(host, VisionWorkerHostPort):
            raise TypeError("host must implement VisionWorkerHostPort")
        self._host = host
        self._lock = RLock()
        self._identities: dict[tuple[str, int, int, int], VisionRequestIdentity] = {}
        self._latest: VisionRequestIdentity | None = None
        self._last_submitted: VisionRequestIdentity | None = None
        self._active_stream = (session_id, capture_generation)
        self._retired_streams: set[tuple[str, int]] = set()
        self._closed = False
    @property
    def state(self) -> object:
        return self._host.state
    @property
    def worker_pid(self) -> int | None:
        return self._host.worker_pid
    @property
    def worker_generation(self) -> int:
        return self._host.worker_generation
    def start(self, *, timeout: float = 10.0) -> None:
        try:
            self._host.start(timeout=timeout)
        except RuntimeError as first:
            if self._closed:
                raise
            try:
                self._host.restart(timeout=timeout)
            except RuntimeError as second:
                raise RuntimeError(
                    f"vision worker failed two bounded starts: {first}; {second}"
                ) from second
    def submit(
        self,
        image: np.ndarray,
        *,
        frame: FrameIdentity,
        version: VersionIdentity,
        expected_seat: Seat | str | None,
        visual_self_opportunity: bool,
        wild_rank: str,
        request_sequence: int,
        formal_action_boundary: FrameIdentity | None = None,
        repair_seats: tuple[Seat | str, ...] = (),
        timeout_ms: int = 2_000,
    ) -> tuple[VisionRuntimeResult, ...]:
        identity = VisionRequestIdentity(frame, version, request_sequence)
        expected = Seat(expected_seat) if expected_seat is not None else None
        with self._lock:
            if self._closed:
                return self._publish_local(identity, VisionRuntimeStatus.SERVICE_CLOSED)
            problem = self._submission_problem(identity)
            if problem:
                return self._publish_local(
                    identity,
                    VisionRuntimeStatus.REJECTED,
                    problem,
                    "StaleSubmission",
                    "vision request is older than the active request",
                )
            owned = _owned_frame(image)
            try:
                self._prepare_worker(identity.version)
            except Exception as exc:
                return self._publish_local(
                    identity,
                    VisionRuntimeStatus.WORKER_ERROR,
                    "worker_unavailable",
                    type(exc).__name__,
                    str(exc),
                )
            self._latest = self._last_submitted = identity
            request = WorkerRequest(
                session_id=version.session_id,
                capture_generation=version.capture_generation,
                request_sequence=request_sequence,
                state_revision=version.state_revision,
                submitted_processing_ms=_processing_ms(),
                payload=VisionWorkerPayload(
                    self._config,
                    identity,
                    expected,
                    visual_self_opportunity,
                    wild_rank,
                    owned,
                    formal_action_boundary,
                    tuple(dict.fromkeys(Seat(item) for item in repair_seats)),
                ),
                delivery=DeliveryMode.LATEST_ONLY,
                timeout_ms=timeout_ms,
            )
            self._identities[_worker_key(request)] = identity
            try:
                self._host.submit(request)
            except RuntimeError as exc:
                return self._publish_local(
                    identity,
                    VisionRuntimeStatus.WORKER_ERROR,
                    "worker_unavailable",
                    type(exc).__name__,
                    str(exc),
                )
            except Exception as exc:
                return self._publish_local(
                    identity,
                    VisionRuntimeStatus.REJECTED,
                    "request_rejected",
                    type(exc).__name__,
                    str(exc),
                )
            return tuple(self._convert(item, remove=True) for item in self._host.drain_results())
    def restart(self, *, version: VersionIdentity, timeout: float = 10.0) -> int:
        """Explicitly replace the worker; its process-local pipeline cache dies."""

        with self._lock:
            if self._closed:
                raise RuntimeError("closed vision runtime cannot be restarted")
            ready = self._host.restart(
                timeout=timeout,
                session_id=version.session_id,
                capture_generation=version.capture_generation,
                state_revision=version.state_revision,
            )
            if self._active_stream != (version.session_id, version.capture_generation):
                self._retired_streams.add(self._active_stream)
            self._active_stream = (version.session_id, version.capture_generation)
            self._latest = self._last_submitted = None
            return ready.worker_pid

    def get_result(self, *, timeout: float | None = None) -> VisionRuntimeResult:
        return self._convert(self._host.get_result(timeout=timeout), remove=True)

    def drain_results(self) -> tuple[VisionRuntimeResult, ...]:
        return tuple(
            self._convert(item, remove=True) for item in self._host.drain_results()
        )

    def close(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._host.close(timeout=timeout)

    def _prepare_worker(self, version: VersionIdentity) -> None:
        stream = (version.session_id, version.capture_generation)
        if stream != self._active_stream:
            self._retired_streams.add(self._active_stream)
            self._host.restart(
                session_id=version.session_id,
                capture_generation=version.capture_generation,
                state_revision=version.state_revision,
            )
            self._active_stream = stream
            return
        if _host_state(self._host) == "new":
            self.start()
        state = _host_state(self._host)
        if state not in {"ready", "broken"}:
            raise RuntimeError(f"vision worker is {state}")
        self._host.bind_version(
            session_id=version.session_id,
            capture_generation=version.capture_generation,
            state_revision=version.state_revision,
        )

    def _submission_problem(self, identity: VisionRequestIdentity) -> str:
        new_stream = (identity.version.session_id, identity.version.capture_generation)
        if new_stream != self._active_stream and new_stream in self._retired_streams:
            return "stale_session"
        previous = self._last_submitted
        if previous is None:
            return ""
        old_stream = (previous.version.session_id, previous.version.capture_generation)
        if old_stream != new_stream:
            if (
                previous.version.session_id == identity.version.session_id
                and identity.version.capture_generation < previous.version.capture_generation
            ):
                return "stale_capture_generation"
            return ""
        checks = (
            (identity.version.state_revision < previous.version.state_revision, "stale_state_revision"),
            (identity.version.update_sequence <= previous.version.update_sequence, "stale_update_sequence"),
            (identity.request_sequence <= previous.request_sequence, "stale_request_sequence"),
            (identity.frame.frame_sequence <= previous.frame.frame_sequence, "stale_frame_sequence"),
            (identity.version.turn_index < previous.version.turn_index, "stale_turn_index"),
        )
        return next((code for failed, code in checks if failed), "")

    def _publish_local(
        self,
        identity: VisionRequestIdentity,
        status: VisionRuntimeStatus,
        code: str = "",
        failure_type: str = "",
        message: str = "",
    ) -> tuple[VisionRuntimeResult, ...]:
        result = VisionRuntimeResult(
            identity,
            status,
            self._host.worker_generation,
            self._host.worker_pid,
            failure_code=code,
            failure_type=failure_type,
            message=message,
        )
        return (result,)

    def _convert(self, value: WorkerResult, *, remove: bool) -> VisionRuntimeResult:
        key = _worker_key(value)
        with self._lock:
            identity = self._identities.get(key) or _fallback_identity(value)
            if remove:
                self._identities.pop(key, None)
            latest = self._latest
        special = _worker_status(value)
        if special is not None:
            return self._result(identity, special, value)
        stale = _stale_stream_reason(identity, latest)
        if stale:
            return self._result(
                identity, VisionRuntimeStatus.REJECTED, value, code=stale
            )
        payload = value.payload
        if not isinstance(payload, VisionWorkerSuccess):
            return self._result(
                identity, VisionRuntimeStatus.WORKER_ERROR, value,
                code="invalid_worker_payload", failure_type="TypeError",
            )
        if (
            payload.identity != identity
            or payload.worker_pid != value.worker_pid
            or payload.pipeline_result.frame != identity.frame
        ):
            return self._result(
                identity, VisionRuntimeStatus.REJECTED, value,
                code="worker_payload_identity_mismatch", failure_type="VersionMismatch",
            )
        return self._result(
            identity,
            VisionRuntimeStatus.FRAME,
            value,
            elapsed_ms=payload.elapsed_ms,
            pipeline_result=payload.pipeline_result,
            cache_hit=payload.cache_hit,
        )

    def _result(
        self,
        identity: VisionRequestIdentity,
        status: VisionRuntimeStatus,
        value: WorkerResult,
        *,
        code: str = "",
        failure_type: str = "",
        elapsed_ms: float | None = None,
        pipeline_result: object = None,
        cache_hit: bool = False,
    ) -> VisionRuntimeResult:
        failure = value.failure
        return VisionRuntimeResult(
            identity=identity,
            status=status,
            worker_generation=value.worker_generation,
            worker_pid=value.worker_pid,
            elapsed_ms=float(elapsed_ms or 0),
            end_to_end_ms=float(max(0, value.finished_processing_ms - value.submitted_processing_ms)),
            pipeline_result=pipeline_result,  # type: ignore[arg-type]
            cache_hit=cache_hit,
            failure_code=code or (failure.code if failure else ""),
            failure_type=failure_type or (failure.error_type if failure else ""),
            message="" if failure is None else failure.message,
        )


def _worker_status(value: WorkerResult) -> VisionRuntimeStatus | None:
    if value.status is WorkerResultStatus.SUCCESS:
        return None
    code = "" if value.failure is None else value.failure.code
    if code == "latest_replaced":
        return VisionRuntimeStatus.SUPERSEDED
    if code == "worker_restarted":
        return VisionRuntimeStatus.WORKER_RESTARTED
    return {
        WorkerResultStatus.TIMEOUT: VisionRuntimeStatus.WORKER_TIMEOUT,
        WorkerResultStatus.CRASHED: VisionRuntimeStatus.WORKER_CRASHED,
        WorkerResultStatus.REJECTED: VisionRuntimeStatus.REJECTED,
        WorkerResultStatus.DROPPED: VisionRuntimeStatus.REJECTED,
        WorkerResultStatus.ERROR: VisionRuntimeStatus.WORKER_ERROR,
    }[value.status]


def _stale_stream_reason(
    identity: VisionRequestIdentity,
    latest: VisionRequestIdentity | None,
) -> str:
    if latest is None:
        return "no_active_request"
    pairs = (
        (identity.version.session_id, latest.version.session_id, "stale_session"),
        (identity.version.capture_generation, latest.version.capture_generation, "stale_capture_generation"),
    )
    return next((code for actual, expected, code in pairs if actual != expected), "")


def _owned_frame(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or not image.size:
        raise ValueError("image must be a non-empty numpy screenshot")
    owned = np.array(image, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _worker_key(value: WorkerRequest | WorkerResult) -> tuple[str, int, int, int]:
    return (
        value.session_id,
        value.capture_generation,
        value.state_revision,
        value.request_sequence,
    )


def _fallback_identity(value: WorkerResult) -> VisionRequestIdentity:
    version = VersionIdentity(
        value.session_id, value.capture_generation, value.state_revision, 0, 0
    )
    frame = FrameIdentity(
        value.session_id, value.capture_generation, 0, 0, "unknown", "unknown"
    )
    return VisionRequestIdentity(frame, version, value.request_sequence)


def _processing_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _host_state(host: VisionWorkerHostPort) -> str:
    value = host.state
    return str(getattr(value, "value", value)).lower()


VisionRuntime = LiveV2VisionRuntime

__all__ = ["LiveV2VisionRuntime", "VisionRuntime"]
