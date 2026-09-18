"""Threaded bridge from engine opportunities to latest-only advice results."""

from __future__ import annotations

from dataclasses import replace
from threading import Event, Lock, Thread, Timer
from time import monotonic_ns
from typing import Callable, Protocol

from ..live_v2.game_state import TrustedGameSnapshot
from ..live_v2.identity import VersionIdentity
from ..live_v2.results import AdviceOpportunity, OpportunityStatus
from .live_v2_advice_protocol import (
    AdviceRequestIdentity, AdviceRuntimeResult, AdviceRuntimeStatus,
    same_formal_advice_version,
)
from .live_v2_advice_audit import (
    AdviceAuditJournal, metric_for, terminal_status, timing_record,
)
from .live_v2_advice_ledger import AdviceOpportunityLedger
from .live_v2_local_pass import LocalPassOpportunityPhase
from .ports import SessionPersistencePort


class AdviceRuntimeLike(Protocol):
    def start(self, *, timeout: float = 10.0) -> None: ...
    def submit(
        self, snapshot: TrustedGameSnapshot, opportunity: AdviceOpportunity,
        *, request_sequence: int, timeout_ms: int = 3000,
    ) -> tuple[AdviceRuntimeResult, ...]: ...
    def drain_results(self) -> tuple[AdviceRuntimeResult, ...]: ...
    def cancel(
        self, identity: AdviceRequestIdentity, *, reason: str
    ) -> tuple[AdviceRuntimeResult, ...]: ...
    def close(self, *, timeout: float = 5.0) -> None: ...


def result_matches_opportunity(
    result: AdviceRuntimeResult,
    current: VersionIdentity,
    opportunity: AdviceOpportunity | None,
) -> bool:
    return opportunity_is_current(
        result.identity.version, result.identity.opportunity_id,
        current, opportunity,
    )


def opportunity_is_current(
    incoming: VersionIdentity,
    opportunity_id: str,
    current: VersionIdentity,
    opportunity: AdviceOpportunity | None,
) -> bool:
    return bool(
        opportunity
        and opportunity.status is OpportunityStatus.READY
        and opportunity.opportunity_id == opportunity_id
        and (incoming.session_id, incoming.capture_generation,
             incoming.state_revision, incoming.turn_index)
        == (current.session_id, current.capture_generation,
            current.state_revision, current.turn_index)
    )


class LiveV2AdvicePump:
    """Submit READY exactly once and deliver results without GUI polling."""

    def __init__(
        self, runtime: AdviceRuntimeLike, *,
        snapshot_provider: Callable[[], TrustedGameSnapshot],
        on_result: Callable[[AdviceRuntimeResult], bool | str],
        on_local_pass: Callable[[AdviceOpportunity, object], bool] | None = None,
        on_failure: Callable[[str], None] | None = None,
        store: SessionPersistencePort | None = None,
        metrics: dict[str, int] | None = None,
        local_hint_window_ms: int = 200,
        request_timeout_ms: int = 3_000,
        processing_clock_ms: Callable[[], int] | None = None,
        poll_seconds: float = 0.02,
    ) -> None:
        self._runtime = runtime
        self._snapshot_provider = snapshot_provider
        self._on_result = on_result
        self._on_local_pass = on_local_pass or (lambda _opportunity, _hint: False)
        self._on_failure = on_failure or (lambda _message: None)
        self._poll_seconds = max(0.005, float(poll_seconds))
        if (
            isinstance(local_hint_window_ms, bool)
            or local_hint_window_ms not in range(150, 251)
            and local_hint_window_ms != 0
        ):
            raise ValueError("local hint window must be 150-250ms, or 0 to disable")
        self._local_hint_window_ms = int(local_hint_window_ms)
        if isinstance(request_timeout_ms, bool) or request_timeout_ms <= 0:
            raise ValueError("request timeout must be a positive integer")
        self._request_timeout_ms = int(request_timeout_ms)
        self._clock_ms = processing_clock_ms or (lambda: monotonic_ns() // 1_000_000)
        self._stop = Event()
        self._thread: Thread | None = None
        self._buffer_lock = Lock()
        self._buffered_results: dict[AdviceRequestIdentity, AdviceRuntimeResult] = {}
        self._hint_open: set[AdviceRequestIdentity] = set()
        self._ledger = AdviceOpportunityLedger()
        self._audit = AdviceAuditJournal(
            store, metrics, self._on_failure, self._ledger.requested_ms
        )

    def start(self) -> None:
        self._runtime.start()
        self._thread = Thread(
            target=self._run, name="live-v2-advice-pump", daemon=True
        )
        self._thread.start()

    def publish(self, opportunity: AdviceOpportunity) -> None:
        if self._stop.is_set():
            return
        retired = self._ledger.retire_superseded(opportunity)
        for identity, phase in retired:
            with self._buffer_lock:
                self._hint_open.discard(identity)
                self._buffered_results.pop(identity, None)
            timing_extra = self._timing_extra(identity)
            if phase is LocalPassOpportunityPhase.MODEL_SUBMITTED:
                cancel = getattr(self._runtime, "cancel", None)
                if callable(cancel):
                    try:
                        cancel(identity, reason="opportunity_superseded")
                    except Exception as exc:
                        self._on_failure(f"cancel:{type(exc).__name__}: {exc}")
            self._record(
                identity,
                "cancelled",
                failure_code=f"opportunity_{opportunity.status.value}",
                discard_reason="opportunity_superseded",
                **timing_extra,
            )
            self._increment("opportunity_no_result")
        if opportunity.status is not OpportunityStatus.READY:
            return
        identity = self._ledger.register(
            opportunity,
            requested_ms=self._clock_ms(),
            request_timeout_ms=self._request_timeout_ms,
        )
        if identity is None:
            return
        self._record(
            identity, "requested",
            local_hint_window_ms=self._local_hint_window_ms,
            submission_policy="parallel_local_hint_race",
        )
        self._increment("opportunity_total")
        # Start the prewarmed model immediately.  A concurrent timer closes the
        # local-hint race and releases any early model result.
        if self._local_hint_window_ms:
            timer = Timer(
                self._local_hint_window_ms / 1000,
                self._finish_hint_window,
                args=(identity, opportunity),
            )
            timer.daemon = True
            self._ledger.set_timer(identity, timer)
            with self._buffer_lock:
                self._hint_open.add(identity)
            timer.start()
        self._submit(identity, opportunity)

    def _submit(
        self, identity: AdviceRequestIdentity, opportunity: AdviceOpportunity
    ) -> None:
        self._expire_due(self._clock_ms())
        if self._stop.is_set() or not self._ledger.begin_model(identity):
            return
        try:
            snapshot = self._snapshot_provider()
            now_ms = self._clock_ms()
            self._expire_due(now_ms)
            deadline_ms = self._ledger.deadline_ms(identity)
            if deadline_ms is not None and now_ms >= deadline_ms:
                return
            if not same_formal_advice_version(
                snapshot.version, opportunity.version
            ):
                self._deliver(AdviceRuntimeResult(
                    identity,
                    AdviceRuntimeStatus.SUPERSEDED,
                    int(getattr(self._runtime, "worker_generation", 0)),
                    getattr(self._runtime, "worker_pid", None),
                    elapsed_ms=float(max(0, now_ms - (
                        self._ledger.requested_ms(identity) or now_ms
                    ))),
                    failure_code="snapshot_formal_state_changed",
                    failure_type="VersionMismatch",
                    message="snapshot formal state changed during hint window",
                ))
                return
            if snapshot.version != opportunity.version:
                snapshot = replace(snapshot, version=opportunity.version)
            worker_generation = int(getattr(self._runtime, "worker_generation", 0))
            worker_pid = getattr(self._runtime, "worker_pid", None)
            self._record(
                identity, "worker_started",
                worker_generation=worker_generation,
                worker_pid=worker_pid,
                worker_start_semantics="request_accepted_by_runtime",
            )
            immediate = self._runtime.submit(
                snapshot,
                opportunity,
                request_sequence=identity.request_sequence,
                timeout_ms=max(1, (deadline_ms or now_ms + 1) - now_ms),
            )
            for result in immediate:
                self._deliver(result)
        except Exception as exc:
            if self._complete(identity):
                self._record(
                    identity, "failed", message=str(exc),
                    failure_code="submit_failed", failure_type=type(exc).__name__,
                    worker_generation=int(getattr(self._runtime, "worker_generation", 0)),
                    worker_pid=getattr(self._runtime, "worker_pid", None),
                )
                self._increment("opportunity_no_result")
            self._on_failure(f"{type(exc).__name__}: {exc}")

    def confirm_local_pass(self, opportunity: AdviceOpportunity, hint: object) -> bool:
        with self._buffer_lock:
            open_identity = next((
                identity for identity in self._hint_open
                if identity.opportunity_id == opportunity.opportunity_id
                and same_formal_advice_version(
                    identity.version, opportunity.version
                )
            ), None)
            if open_identity is None:
                return False
            selected = self._ledger.take_local_pass(
                opportunity,
                now_ms=self._clock_ms(),
                window_ms=self._local_hint_window_ms,
            )
            if selected is None:
                return False
            identity, phase = selected
            self._hint_open.discard(identity)
            self._buffered_results.pop(identity, None)
        if phase is LocalPassOpportunityPhase.MODEL_SUBMITTED:
            cancel = getattr(self._runtime, "cancel", None)
            if callable(cancel):
                try:
                    cancel(identity, reason="local_pass_won_race")
                except Exception as exc:
                    self._on_failure(f"cancel:{type(exc).__name__}: {exc}")
        try:
            accepted = bool(self._on_local_pass(opportunity, hint))
        except Exception as exc:
            accepted = False
            self._on_failure(f"local_pass_callback:{type(exc).__name__}: {exc}")
        if not accepted:
            self._record(
                identity, "stale", failure_code="session_rejected_local_pass"
            )
            self._increment("opportunity_no_result")
            return False
        self._record(identity, "local_pass", response_source="button_cannot_beat")
        self._increment("opportunity_valid")
        return True

    def poll(self) -> None:
        # Drain completed worker results before expiring deadlines. A fast
        # result that is already in the pipe must win over the timeout check,
        # especially when replay/capture time and worker wall time advance at
        # different rates.
        try:
            for result in self._runtime.drain_results():
                self._deliver(result)
        except Exception as exc:
            self._on_failure(f"{type(exc).__name__}: {exc}")
        self._expire_due(self._clock_ms())

    def wait_idle(self, timeout: float) -> bool:
        # Results may already be available in the worker pipe while the replay
        # clock has advanced beyond the logical opportunity deadline. Drain
        # them before expiring the ledger, otherwise a completed fast model
        # response is falsely reported as a timeout.
        try:
            for result in self._runtime.drain_results():
                self._deliver(result)
        except Exception as exc:
            self._on_failure(f"{type(exc).__name__}: {exc}")
        self._expire_due(self._clock_ms())
        return self._ledger.wait_idle(timeout)

    def cancel_pending(
        self, *, reason: str, preserve_worker: bool = False,
    ) -> None:
        """Cancel logical requests before a formal state rewrite.

        A visual correction supersedes the old opportunity. Terminalizing the
        old identities prevents their late results from being reported as
        ordinary ``stale`` advice. ``preserve_worker`` skips process-level
        cancellation, so an in-flight same-stream result is merely ignored and
        the prewarmed worker can bind the corrected revision on its next job.
        """

        pending = self._ledger.pending()
        with self._buffer_lock:
            for identity in pending:
                self._hint_open.discard(identity)
                self._buffered_results.pop(identity, None)
        cancel = (
            None if preserve_worker else getattr(self._runtime, "cancel", None)
        )
        timing_by_identity = {
            identity: self._timing_extra(identity) for identity in pending
        }
        for identity in pending:
            if callable(cancel):
                try:
                    cancel(identity, reason=reason)
                except Exception as exc:
                    self._on_failure(f"cancel:{type(exc).__name__}: {exc}")
            if self._complete(identity):
                self._record(
                    identity, "cancelled",
                    failure_code=reason,
                    discard_reason="opportunity_superseded_by_correction",
                    **timing_by_identity[identity],
                )
                self._increment("opportunity_no_result")
        self._ledger.notify_idle()

    def close(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        for timer in self._ledger.cancel_timers():
            timer.cancel()
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, timeout))
        pending = self._ledger.pending()
        with self._buffer_lock:
            self._hint_open.clear()
            self._buffered_results.clear()
        timing_by_identity = {
            identity: self._timing_extra(identity) for identity in pending
        }
        try:
            self._runtime.close(timeout=max(0.0, timeout))
        finally:
            for identity in pending:
                if self._complete(identity):
                    self._record(
                        identity, "cancelled", failure_code="runtime_closed",
                        discard_reason="stop_discarded",
                        worker_generation=int(getattr(self._runtime, "worker_generation", 0)),
                        worker_pid=getattr(self._runtime, "worker_pid", None),
                        **timing_by_identity[identity],
                    )
                    self._increment("opportunity_no_result")
        self._ledger.notify_idle()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            self.poll()

    def _expire_due(self, now_ms: int) -> None:
        expired = self._ledger.expire_due(now_ms)
        for identity, phase, deadline_ms in expired:
            with self._buffer_lock:
                self._hint_open.discard(identity)
                self._buffered_results.pop(identity, None)
            timing_extra = self._timing_extra(identity)
            requested_ms = self._ledger.requested_ms(identity)
            elapsed_ms = max(0, now_ms - requested_ms) if requested_ms is not None else 0
            result = AdviceRuntimeResult(
                identity, AdviceRuntimeStatus.WORKER_TIMEOUT,
                int(getattr(self._runtime, "worker_generation", 0)),
                getattr(self._runtime, "worker_pid", None),
                elapsed_ms=float(elapsed_ms),
                failure_code="opportunity_deadline_expired",
                failure_type="TimeoutError",
                message="advice opportunity exceeded its absolute deadline",
            )
            if phase is LocalPassOpportunityPhase.MODEL_SUBMITTED:
                cancel = getattr(self._runtime, "cancel", None)
                if callable(cancel):
                    try:
                        cancel(identity, reason="opportunity_deadline_expired")
                    except Exception as exc:
                        self._on_failure(f"cancel:{type(exc).__name__}: {exc}")
            self._record(
                identity, "timeout", elapsed_ms=float(elapsed_ms),
                deadline_processing_ms=deadline_ms,
                worker_generation=result.worker_generation,
                worker_pid=result.worker_pid,
                failure_code=result.failure_code,
                failure_type=result.failure_type,
                message=result.message,
                **timing_extra,
            )
            self._increment("opportunity_late")
            try:
                self._on_result(result)
            except Exception as exc:
                self._on_failure(f"session_callback:{type(exc).__name__}: {exc}")
        if expired:
            self._ledger.notify_idle()

    def _deliver(
        self, result: AdviceRuntimeResult, *, allow_hint_buffer: bool = True
    ) -> None:
        if (
            allow_hint_buffer
            and self._local_hint_window_ms
            and result.status is AdviceRuntimeStatus.ADVICE
        ):
            with self._buffer_lock:
                if (
                    result.identity in self._hint_open
                    and self._ledger.result_pending(result.identity)
                ):
                    self._buffered_results[result.identity] = result
                    return
        timing_extra = self._timing_extra(result.identity)
        if not self._ledger.accept_result(result.identity):
            return
        if self._stop.is_set():
            self._record(
                result.identity, "cancelled", failure_code="runtime_closed",
                discard_reason="stop_discarded",
                worker_generation=result.worker_generation, worker_pid=result.worker_pid,
                **timing_extra,
            )
            self._increment("opportunity_no_result")
            return
        try:
            acceptance = self._on_result(result)
            accepted = acceptance is True
            rejection_code = (
                str(acceptance)
                if isinstance(acceptance, str) and acceptance
                else "session_rejected_result"
            )
        except Exception as exc:
            self._record(
                result.identity, "failed", worker_generation=result.worker_generation,
                worker_pid=result.worker_pid, elapsed_ms=result.elapsed_ms,
                failure_code="session_callback_failed",
                failure_type=type(exc).__name__, message=str(exc),
                **timing_extra,
            )
            self._increment("opportunity_no_result")
            self._on_failure(f"session_callback:{type(exc).__name__}: {exc}")
            return
        known_supersession = {
            "opportunity_closed",
            "superseded_by_state_change",
            "superseded_by_self_action",
            "capture_generation_changed",
            "session_terminal",
        }
        status = (
            terminal_status(result.status)
            if accepted
            else "cancelled"
            if rejection_code in known_supersession
            else "stale"
        )
        self._record(
            result.identity, status,
            worker_generation=result.worker_generation,
            worker_pid=result.worker_pid, elapsed_ms=result.elapsed_ms,
            failure_code=(result.failure_code or ("" if accepted else rejection_code)),
            failure_type=result.failure_type,
            message=result.message,
            **timing_extra,
        )
        self._increment(metric_for(status))

    def _finish_hint_window(
        self, identity: AdviceRequestIdentity, opportunity: AdviceOpportunity
    ) -> None:
        self._ledger.clear_timer(identity)
        with self._buffer_lock:
            self._hint_open.discard(identity)
            buffered = self._buffered_results.pop(identity, None)
        if not self._ledger.result_pending(identity):
            return
        try:
            snapshot = self._snapshot_provider()
        except Exception as exc:
            self._on_failure(f"snapshot_provider:{type(exc).__name__}: {exc}")
            snapshot = None
        if snapshot is None or not same_formal_advice_version(
            snapshot.version, opportunity.version
        ):
            with self._buffer_lock:
                self._hint_open.discard(identity)
                self._buffered_results.pop(identity, None)
            cancel = getattr(self._runtime, "cancel", None)
            if callable(cancel):
                try:
                    cancel(identity, reason="snapshot_formal_state_changed")
                except Exception as exc:
                    self._on_failure(f"cancel:{type(exc).__name__}: {exc}")
            if self._complete(identity):
                self._record(
                    identity, "stale",
                    failure_code="snapshot_formal_state_changed",
                    discard_reason="local_hint_window_state_changed",
                )
                self._increment("opportunity_no_result")
            return
        if buffered is not None:
            self._deliver(buffered, allow_hint_buffer=False)

    def _complete(self, identity: AdviceRequestIdentity) -> bool:
        return self._ledger.complete(identity)

    def _record(self, identity: AdviceRequestIdentity, status: str, **extra: object) -> None:
        if "worker_timing" not in extra:
            extra.update(self._timing_extra(identity))
        self._audit.record(identity, status, **extra)

    def _timing_extra(
        self, identity: AdviceRequestIdentity
    ) -> dict[str, object]:
        timing_for_identity = getattr(self._runtime, "timing_for_identity", None)
        if not callable(timing_for_identity):
            return {}
        timing = timing_for_identity(identity)
        return {} if timing is None else {"worker_timing": timing_record(timing)}

    def _increment(self, name: str) -> None:
        self._audit.increment(name)
__all__ = [
    "AdviceRuntimeLike", "LiveV2AdvicePump", "opportunity_is_current",
    "result_matches_opportunity",
]
