"""Persistence and metric projection for advice opportunity terminals."""

from __future__ import annotations

from time import monotonic_ns
from typing import Callable

from .live_v2_advice_protocol import (
    AdviceRequestIdentity,
    AdviceRequestTiming,
    AdviceRuntimeStatus,
    request_id,
)
from .ports import SessionPersistencePort


class AdviceAuditJournal:
    def __init__(
        self,
        store: SessionPersistencePort | None,
        metrics: dict[str, int] | None,
        on_failure: Callable[[str], None],
        requested_ms: Callable[[AdviceRequestIdentity], int | None],
    ) -> None:
        self._store = store
        self._metrics = metrics
        self._on_failure = on_failure
        self._requested_ms = requested_ms

    def record(
        self, identity: AdviceRequestIdentity, status: str, **extra: object
    ) -> None:
        if self._store is None:
            return
        version = identity.version
        record = {
            "schema": "guandan.live-v2.advice/1",
            "request_id": request_id(identity),
            "opportunity_id": identity.opportunity_id,
            "status": status,
            "session_id": version.session_id,
            "capture_generation": version.capture_generation,
            "request_generation": version.capture_generation,
            "state_revision": version.state_revision,
            "turn_id": version.turn_index + 1,
            "request_sequence": identity.request_sequence,
            "requested_processing_ms": self._requested_ms(identity),
            "finished_processing_ms": monotonic_ns() // 1_000_000,
            **extra,
        }
        try:
            self._store.append_advice(record)
        except Exception as exc:
            self._on_failure(f"advice_audit:{type(exc).__name__}: {exc}")

    def increment(self, name: str) -> None:
        if self._metrics is not None and name:
            self._metrics[name] = int(self._metrics.get(name, 0)) + 1


def terminal_status(status: AdviceRuntimeStatus) -> str:
    if status is AdviceRuntimeStatus.ADVICE:
        return "ready"
    if status is AdviceRuntimeStatus.BLOCKED:
        return "withheld"
    if status is AdviceRuntimeStatus.WORKER_TIMEOUT:
        return "timeout"
    if status in {AdviceRuntimeStatus.CLOSED, AdviceRuntimeStatus.SUPERSEDED}:
        return "stale"
    if status is AdviceRuntimeStatus.SERVICE_CLOSED:
        return "cancelled"
    return "failed"


def metric_for(status: str) -> str:
    return {
        "ready": "opportunity_valid",
        "timeout": "opportunity_late",
        "withheld": "opportunity_unrecoverable",
    }.get(status, "opportunity_no_result")


def timing_record(value: AdviceRequestTiming) -> dict[str, object]:
    worker = value.worker
    fields = {
        "host_accepted": None if worker is None else worker.host_accepted_ms,
        "send_start": None if worker is None else worker.send_started_ms,
        "send_end": None if worker is None else worker.send_finished_ms,
        "child_received": None if worker is None else worker.child_received_ms,
        "child_start": None if worker is None else worker.child_started_ms,
        "child_end": None if worker is None else worker.child_finished_ms,
        "result_received": None if worker is None else worker.result_received_ms,
    }
    submission = {
        "runtime_submit_entered": value.runtime_submit_entered_ms,
        "runtime_lock_acquired": value.runtime_lock_acquired_ms,
        "version_bound": value.version_bound_ms,
        "worker_request_built": value.worker_request_built_ms,
        "host_submit_entered": value.host_submit_entered_ms,
        "host_submit_returned": value.host_submit_returned_ms,
    }

    def delta(start: str, end: str) -> int | None:
        left, right = fields[start], fields[end]
        if not isinstance(left, int) or not isinstance(right, int):
            return None
        return max(0, right - left)

    return {
        "worker_generation": value.worker_generation,
        "worker_request_sequence": value.worker_request_sequence,
        "submission": submission,
        **fields,
        "delta_ms": {
            "accepted_to_send_start": delta("host_accepted", "send_start"),
            "send": delta("send_start", "send_end"),
            "send_to_child_received": delta("send_end", "child_received"),
            "child_queue": delta("child_received", "child_start"),
            "child_execution": delta("child_start", "child_end"),
            "child_to_result_received": delta("child_end", "result_received"),
            "total": delta("host_accepted", "result_received"),
        },
    }


__all__ = ["AdviceAuditJournal", "metric_for", "terminal_status", "timing_record"]
