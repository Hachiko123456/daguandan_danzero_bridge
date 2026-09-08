"""Latest-only runtime for version-bound live-v2 advice requests."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import replace
from threading import Lock, RLock
import time

from .live_v2_advice_protocol import (
    AdvicePrewarmPayload,
    AdviceFailureKind,
    AdviceRequestIdentity,
    AdviceRequestTiming,
    AdviceRuntimeResult,
    AdviceRuntimeStatus,
    AdviceWorkerConfig,
    AdviceWorkerFailure,
    AdviceWorkerPayload,
    AdviceWorkerSuccess,
    AdvisorReady,
    WorkerHostFactory,
    request_id,
    same_formal_advice_version,
)
from .live_v2_worker_protocol import (
    DeliveryMode,
    WorkerReady,
    WorkerRequest,
    WorkerRequestTiming,
    WorkerResult,
    WorkerResultStatus,
)
from ..live_v2.game_state import TrustedGameSnapshot
from ..live_v2.identity import VersionIdentity
from ..live_v2.results import AdviceOpportunity, OpportunityStatus


class LiveV2AdviceRuntime:
    """Own one persistent adviser process and never surface stale advice."""

    def __init__(
        self,
        config: AdviceWorkerConfig,
        *,
        session_id: str,
        capture_generation: int,
        state_revision: int,
        host_factory: WorkerHostFactory,
    ) -> None:
        self._config = config
        self._host = host_factory(
            session_id=session_id,
            capture_generation=capture_generation,
            state_revision=state_revision,
        )
        self._lock = RLock()
        self._timing_lock = Lock()
        self._local: deque[AdviceRuntimeResult] = deque()
        self._identities: dict[tuple[str, int, int, int], AdviceRequestIdentity] = {}
        self._timing_keys: OrderedDict[
            AdviceRequestIdentity, AdviceRequestTiming
        ] = OrderedDict()
        self._latest: AdviceRequestIdentity | None = None
        self._version = (session_id, capture_generation, state_revision)
        self._worker_sequence = -1
        self._advisor_ready: AdvisorReady | None = None
        self._requires_restart = False
        self._started = False
        self._closed = False

    @property
    def worker_pid(self) -> int | None:
        return self._host.worker_pid

    @property
    def worker_generation(self) -> int:
        return self._host.worker_generation

    @property
    def advisor_ready(self) -> AdvisorReady | None:
        return self._advisor_ready

    @property
    def recent_request_timings(self) -> tuple[WorkerRequestTiming, ...]:
        return tuple(getattr(self._host, "recent_request_timings", ()))

    def timing_for_identity(
        self, identity: AdviceRequestIdentity
    ) -> AdviceRequestTiming | None:
        with self._timing_lock:
            mapped = self._timing_keys.get(identity)
        if mapped is None:
            return None
        worker = next(
            (
                item for item in self.recent_request_timings
                if (item.worker_generation, item.request_sequence)
                == (mapped.worker_generation, mapped.worker_request_sequence)
            ),
            None,
        )
        return replace(mapped, worker=worker)

    def start(self, *, timeout: float = 10.0) -> AdvisorReady:
        with self._lock:
            return self._start_locked(timeout)

    def _start_locked(self, timeout: float) -> AdvisorReady:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if self._closed:
            raise RuntimeError("closed advice runtime cannot be started")
        if self._is_prewarmed():
            assert self._advisor_ready is not None
            return self._advisor_ready
        deadline = time.monotonic() + timeout
        try:
            ready = self._start_host(deadline)
            advisor_ready = self._prewarm(ready, deadline)
        except RuntimeError as first:
            self._started = False
            self._advisor_ready = None
            self._requires_restart = self._host.worker_generation > 0
            raise RuntimeError(f"advice advisor prewarm failed: {first}") from first
        self._advisor_ready = advisor_ready
        self._requires_restart = False
        self._started = True
        return advisor_ready

    def submit(
        self,
        snapshot: TrustedGameSnapshot,
        opportunity: AdviceOpportunity,
        *,
        request_sequence: int,
        timeout_ms: int = 3_000,
    ) -> tuple[AdviceRuntimeResult, ...]:
        identity = AdviceRequestIdentity(
            opportunity.version, request_sequence, opportunity.opportunity_id
        )
        if same_formal_advice_version(snapshot.version, opportunity.version):
            if snapshot.version != opportunity.version:
                snapshot = replace(snapshot, version=opportunity.version)
        self._remember_timing_identity(identity, self._host.worker_generation, -1)
        self._mark_submission(identity, "runtime_submit_entered_ms")
        with self._lock:
            self._mark_submission(identity, "runtime_lock_acquired_ms")
            self._latest = identity
            if self._closed:
                return self._publish_local(identity, AdviceRuntimeStatus.SERVICE_CLOSED)
            local = self._preflight(identity, snapshot, opportunity)
            if local is not None:
                return self._publish_local(identity, *local)
            try:
                self._version = (
                    snapshot.version.session_id,
                    snapshot.version.capture_generation,
                    snapshot.version.state_revision,
                )
                self._host.bind_version(
                    session_id=snapshot.version.session_id,
                    capture_generation=snapshot.version.capture_generation,
                    state_revision=snapshot.version.state_revision,
                )
                self._mark_submission(identity, "version_bound_ms")
                if not self._is_prewarmed():
                    self.start()
            except RuntimeError as exc:
                return self._publish_local(
                    identity,
                    AdviceRuntimeStatus.WORKER_ERROR,
                    "worker_start_failed",
                    type(exc).__name__,
                    str(exc),
                )
            worker_request = WorkerRequest(
                session_id=snapshot.version.session_id,
                capture_generation=snapshot.version.capture_generation,
                request_sequence=self._next_worker_sequence(),
                state_revision=snapshot.version.state_revision,
                submitted_processing_ms=_processing_ms(),
                payload=AdviceWorkerPayload(
                    self._config, snapshot, opportunity, request_id(identity)
                ),
                delivery=DeliveryMode.LATEST_ONLY,
                timeout_ms=timeout_ms,
            )
            self._mark_submission(identity, "worker_request_built_ms")
            self._identities[_worker_key(worker_request)] = identity
            self._remember_timing_identity(
                identity, self._host.worker_generation,
                worker_request.request_sequence,
            )
            try:
                self._mark_submission(identity, "host_submit_entered_ms")
                immediate = self._host.submit(worker_request)
                self._mark_submission(identity, "host_submit_returned_ms")
            except (RuntimeError, ValueError) as exc:
                return self._publish_local(
                    identity,
                    (
                        AdviceRuntimeStatus.REJECTED
                        if isinstance(exc, ValueError)
                        else AdviceRuntimeStatus.WORKER_ERROR
                    ),
                    (
                        "request_rejected"
                        if isinstance(exc, ValueError)
                        else "worker_unavailable"
                    ),
                    type(exc).__name__,
                    str(exc),
                )
            return tuple(self._convert(item, remove=False) for item in immediate)

    def _remember_timing_identity(
        self, identity: AdviceRequestIdentity,
        worker_generation: int, worker_sequence: int,
    ) -> None:
        with self._timing_lock:
            previous = self._timing_keys.get(identity)
            if previous is None:
                self._timing_keys[identity] = AdviceRequestTiming(
                    identity, worker_generation, worker_sequence,
                )
            else:
                self._timing_keys[identity] = replace(
                    previous,
                    worker_generation=worker_generation,
                    worker_request_sequence=worker_sequence,
                )
            self._timing_keys.move_to_end(identity)
            while len(self._timing_keys) > 128:
                self._timing_keys.popitem(last=False)

    def _mark_submission(self, identity: AdviceRequestIdentity, field: str) -> None:
        with self._timing_lock:
            current = self._timing_keys.get(identity)
            if current is not None and getattr(current, field) is None:
                self._timing_keys[identity] = replace(
                    current, **{field: _processing_ms()}
                )

    def get_result(self, *, timeout: float | None = None) -> AdviceRuntimeResult:
        with self._lock:
            if self._local:
                return self._local.popleft()
        return self._convert(self._host.get_result(timeout=timeout), remove=True)

    def drain_results(self) -> tuple[AdviceRuntimeResult, ...]:
        with self._lock:
            local = tuple(self._local)
            self._local.clear()
        worker = tuple(
            self._convert(item, remove=True) for item in self._host.drain_results()
        )
        return local + worker

    def cancel(
        self,
        identity: AdviceRequestIdentity,
        *,
        reason: str = "opportunity_superseded",
    ) -> tuple[AdviceRuntimeResult, ...]:
        with self._lock:
            known = identity in self._identities.values()
        if not known:
            return ()
        worker = self._host.cancel_all(reason=reason)
        converted = tuple(self._convert(item, remove=True) for item in worker)
        with self._lock:
            if self._latest == identity:
                self._latest = None
        return converted

    def restart(
        self,
        version: VersionIdentity,
        *,
        request_sequence: int,
        timeout: float = 10.0,
    ) -> AdviceRuntimeResult:
        identity = AdviceRequestIdentity(version, request_sequence, "worker-restart")
        with self._lock:
            if self._closed:
                return self._result(identity, AdviceRuntimeStatus.SERVICE_CLOSED)
            self._latest = identity
            self._version = (
                version.session_id, version.capture_generation,
                version.state_revision,
            )
            deadline = time.monotonic() + timeout
            try:
                ready = self._host.restart(
                    timeout=timeout,
                    session_id=version.session_id,
                    capture_generation=version.capture_generation,
                    state_revision=version.state_revision,
                )
                advisor_ready = self._prewarm(ready, deadline)
            except RuntimeError:
                self._started = False
                self._advisor_ready = None
                self._requires_restart = True
                raise
            self._advisor_ready = advisor_ready
            self._requires_restart = False
            self._started = True
            return self._result(
                identity,
                AdviceRuntimeStatus.WORKER_RESTARTED,
                worker_generation=ready.worker_generation,
                worker_pid=ready.worker_pid,
                advisor_ready=advisor_ready,
            )

    def close(self, *, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._latest = None
            self._started = False
            self._advisor_ready = None
            self._requires_restart = False
        self._host.close(timeout=timeout)

    def _is_prewarmed(self) -> bool:
        ready = self._advisor_ready
        state = getattr(self._host, "state", None)
        state_value = getattr(state, "value", state)
        return bool(
            self._started
            and ready is not None
            and ready.worker_generation == self._host.worker_generation
            and ready.worker_pid == self._host.worker_pid
            and state_value not in {"broken", "closing", "closed"}
        )

    def _start_host(self, deadline: float) -> WorkerReady:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("advice startup deadline expired before worker start")
        state = getattr(self._host, "state", None)
        state_value = getattr(state, "value", state)
        if self._requires_restart or state_value == "broken":
            return self._host.restart(
                timeout=remaining,
                session_id=self._version[0],
                capture_generation=self._version[1],
                state_revision=self._version[2],
            )
        try:
            return self._host.start(timeout=remaining)
        except RuntimeError as first:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"worker start failed before deadline: {first}"
                ) from first
            try:
                return self._host.restart(
                    timeout=remaining,
                    session_id=self._version[0],
                    capture_generation=self._version[1],
                    state_revision=self._version[2],
                )
            except RuntimeError as second:
                raise RuntimeError(
                    f"advice worker failed two bounded starts: {first}; {second}"
                ) from second

    def _prewarm(self, ready: WorkerReady, deadline: float) -> AdvisorReady:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("advice startup deadline expired before advisor prewarm")
        timeout_ms = max(1, int(remaining * 1000) - 25)
        request = WorkerRequest(
            session_id=self._version[0],
            capture_generation=self._version[1],
            request_sequence=self._next_worker_sequence(),
            state_revision=self._version[2],
            submitted_processing_ms=_processing_ms(),
            payload=AdvicePrewarmPayload(self._config, ready.worker_generation),
            delivery=DeliveryMode.FIFO,
            timeout_ms=timeout_ms,
        )
        immediate = self._host.submit(request)
        if immediate:
            raise RuntimeError("advisor prewarm was rejected by the worker queue")
        key = _worker_key(request)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("advisor did not become ready before timeout")
            try:
                result = self._host.get_result(timeout=remaining)
            except TimeoutError as exc:
                raise TimeoutError(
                    "advisor did not become ready before timeout"
                ) from exc
            if _worker_key(result) != key:
                self._local.append(self._convert(result, remove=True))
                continue
            if result.status is not WorkerResultStatus.SUCCESS:
                failure = result.failure
                detail = (
                    result.status.value
                    if failure is None
                    else f"{failure.code}: {failure.error_type}: {failure.message}"
                )
                raise RuntimeError(f"advisor prewarm worker failure: {detail}")
            payload = result.payload
            if not isinstance(payload, AdvisorReady):
                raise RuntimeError("advisor prewarm returned no AdvisorReady proof")
            if (
                payload.worker_generation != result.worker_generation
                or payload.worker_pid != result.worker_pid
                or payload.worker_generation != ready.worker_generation
                or payload.worker_pid != ready.worker_pid
                or payload.advisor_backend != self._config.advisor_backend
            ):
                raise RuntimeError("AdvisorReady identity does not match worker/config")
            return payload

    def _next_worker_sequence(self) -> int:
        self._worker_sequence += 1
        return self._worker_sequence

    def _preflight(
        self,
        identity: AdviceRequestIdentity,
        snapshot: TrustedGameSnapshot,
        opportunity: AdviceOpportunity,
    ) -> tuple[AdviceRuntimeStatus, str, str, str] | None:
        if not same_formal_advice_version(
            opportunity.version, snapshot.version
        ):
            return (
                AdviceRuntimeStatus.REJECTED,
                "opportunity_formal_version_mismatch",
                "VersionMismatch",
                "opportunity and snapshot formal versions differ",
            )
        if opportunity.version != snapshot.version:
            return (
                AdviceRuntimeStatus.REJECTED,
                "opportunity_update_sequence_not_canonical",
                "VersionMismatch",
                "snapshot update sequence was not canonicalized to opportunity",
            )
        if opportunity.status is OpportunityStatus.BLOCKED:
            return AdviceRuntimeStatus.BLOCKED, opportunity.reason.value, "", ""
        if opportunity.status is OpportunityStatus.CLOSED:
            return AdviceRuntimeStatus.CLOSED, opportunity.reason.value, "", ""
        if not snapshot.trusted or snapshot.terminal:
            return (
                AdviceRuntimeStatus.REJECTED,
                "snapshot_not_advisable",
                "SnapshotNotTrusted",
                "READY requires a trusted active snapshot",
            )
        if snapshot.current_seat is not opportunity.seat:
            return (
                AdviceRuntimeStatus.REJECTED,
                "opportunity_seat_mismatch",
                "SnapshotMismatch",
                "READY opportunity does not match snapshot current seat",
            )
        return None

    def _publish_local(
        self,
        identity: AdviceRequestIdentity,
        status: AdviceRuntimeStatus,
        code: str = "",
        failure_type: str = "",
        message: str = "",
    ) -> tuple[AdviceRuntimeResult, ...]:
        result = self._result(identity, status, code, failure_type, message)
        self._local.append(result)
        return (result,)

    def _convert(self, value: WorkerResult, *, remove: bool) -> AdviceRuntimeResult:
        terminal_worker_failure = (
            value.status in {WorkerResultStatus.TIMEOUT, WorkerResultStatus.CRASHED}
            and value.worker_generation == self._host.worker_generation
        )
        if terminal_worker_failure:
            self._started = False
            self._advisor_ready = None
            self._requires_restart = True
            if not self._closed:
                try:
                    self.start(timeout=10.0)
                except RuntimeError:
                    # Preserve the original typed terminal result. A later
                    # submit/start will retry the same bounded prewarm gate.
                    pass
        key = _worker_key(value)
        with self._lock:
            identity = self._identities.get(key) or AdviceRequestIdentity(
                VersionIdentity(
                    value.session_id, value.capture_generation,
                    value.state_revision, 0, 0,
                ),
                value.request_sequence,
                "unknown-opportunity",
            )
            if remove:
                self._identities.pop(key, None)
            latest = self._latest
        if not _same_opportunity(latest, identity):
            return self._result(
                identity, AdviceRuntimeStatus.SUPERSEDED,
                "latest_request_changed", "Superseded",
                "result no longer belongs to the latest opportunity",
                value=value,
            )
        if value.status is WorkerResultStatus.SUCCESS:
            payload = value.payload
            if isinstance(payload, AdviceWorkerSuccess):
                if (payload.version, payload.opportunity_id) != (
                    identity.version, identity.opportunity_id,
                ):
                    return self._result(
                        identity, AdviceRuntimeStatus.REJECTED,
                        "worker_payload_version_mismatch", "VersionMismatch",
                        "worker payload does not match request identity", value=value,
                    )
                return self._result(
                    identity, AdviceRuntimeStatus.ADVICE,
                    value=value, elapsed_ms=payload.elapsed_ms, advice=payload.advice,
                    advisor_cache_hit=payload.advisor_cache_hit,
                    advisor_ready=payload.advisor_ready,
                )
            if isinstance(payload, AdviceWorkerFailure):
                status = (
                    AdviceRuntimeStatus.REJECTED
                    if payload.kind is not AdviceFailureKind.MODEL_ERROR
                    else AdviceRuntimeStatus.WORKER_ERROR
                )
                return self._result(
                    identity, status, payload.code, payload.error_type,
                    payload.message, value=value, elapsed_ms=payload.elapsed_ms,
                )
            return self._result(
                identity, AdviceRuntimeStatus.WORKER_ERROR,
                "invalid_worker_payload", "TypeError",
                "worker returned an unsupported payload", value=value,
            )
        statuses = {
            WorkerResultStatus.TIMEOUT: AdviceRuntimeStatus.WORKER_TIMEOUT,
            WorkerResultStatus.CRASHED: AdviceRuntimeStatus.WORKER_CRASHED,
            WorkerResultStatus.DROPPED: AdviceRuntimeStatus.SUPERSEDED,
            WorkerResultStatus.REJECTED: AdviceRuntimeStatus.REJECTED,
            WorkerResultStatus.ERROR: AdviceRuntimeStatus.WORKER_ERROR,
        }
        failure = value.failure
        return self._result(
            identity, statuses[value.status],
            "" if failure is None else failure.code,
            "" if failure is None else failure.error_type,
            "" if failure is None else failure.message,
            value=value,
        )
    def _result(
        self,
        identity: AdviceRequestIdentity,
        status: AdviceRuntimeStatus,
        code: str = "",
        failure_type: str = "",
        message: str = "",
        *,
        value: WorkerResult | None = None,
        worker_generation: int | None = None,
        worker_pid: int | None = None,
        elapsed_ms: float | None = None,
        advice: object = None,
        advisor_cache_hit: bool = False,
        advisor_ready: AdvisorReady | None = None,
    ) -> AdviceRuntimeResult:
        duration = elapsed_ms
        if duration is None and value is not None:
            duration = max(0, value.finished_processing_ms - value.submitted_processing_ms)
        generation = (
            value.worker_generation
            if value is not None
            else self._host.worker_generation
            if worker_generation is None
            else worker_generation
        )
        return AdviceRuntimeResult(
            identity=identity,
            status=status,
            worker_generation=generation,
            worker_pid=(value.worker_pid if value is not None else worker_pid),
            elapsed_ms=float(duration or 0),
            advice=advice,  # type: ignore[arg-type]
            failure_code=code,
            failure_type=failure_type,
            message=message,
            advisor_cache_hit=advisor_cache_hit,
            advisor_ready=advisor_ready,
        )


def _same_opportunity(
    latest: AdviceRequestIdentity | None,
    identity: AdviceRequestIdentity,
) -> bool:
    if latest is None or latest.opportunity_id != identity.opportunity_id:
        return False
    return same_formal_advice_version(latest.version, identity.version)


def _worker_key(value: WorkerRequest | WorkerResult) -> tuple[str, int, int, int]:
    return (
        value.session_id,
        value.capture_generation,
        value.state_revision,
        value.request_sequence,
    )


def _processing_ms() -> int:
    return time.monotonic_ns() // 1_000_000


AdviceRuntime = LiveV2AdviceRuntime

__all__ = ["AdviceRuntime", "LiveV2AdviceRuntime"]
