"""Bounded, UI-independent listener evidence and a single best-effort writer.

Callers copy capture buffers *before* probing/analysis and pass those copies here.
No window capture, image dependency, or Qt object belongs in this module. Disk
work is injected so the session diagnostic store remains the PNG/JSON authority.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
from threading import Condition, Event, Lock, Thread, current_thread
from time import monotonic
from typing import Callable, Any
from uuid import uuid4


@dataclass(frozen=True)
class ListenerFrame:
    session_directory: Path | None
    session_id: str
    snapshot: Any
    capture_generation: int
    capture_seq: int
    source_phase: str
    source: str = "live_listener_frame"
    details: dict[str, object] = field(default_factory=dict, compare=False)

    # Capture scope is independent of a later fallback storage-directory bind.
    # The controller supplies a run/session key; standalone callers use session_id.
    scope_id: str = ""

    @property
    def capture_scope(self) -> tuple[str, int, str]:
        return (str(self.details.get("listener_phase", self.source_phase)),
                self.capture_generation, self.scope_id or self.session_id)

    @property
    def identity(self) -> tuple[str, int, str, int, str]:
        return (*self.capture_scope, self.capture_seq,
                str(getattr(self.snapshot, "evidence_frame_id", "")))


class EvidenceWriteHandle:
    """Small completion handle, not a Qt thread; waits are explicit and bounded."""

    def __init__(self) -> None:
        self.done = Event()
        self.result: object = None
        self.error: Exception | None = None

    def isRunning(self) -> bool:
        return not self.done.is_set()

    def wait(self, timeout_ms: int = 5000) -> bool:
        return self.done.wait(max(0, timeout_ms) / 1000)


class BoundedEvidenceWriter:
    """One lazy daemon worker; a bounded FIFO never blocks a capture/Qt producer.

    Idle workers exit. close() rejects new work, drains already accepted jobs,
    and joins only up to the supplied deadline. A stuck filesystem operation is
    never force-killed mid-write and never owns a QThread/QObject lifetime.
    """

    def __init__(self, *, queue_limit: int = 4) -> None:
        self.queue_limit = max(1, queue_limit)
        self._condition = Condition()
        self._pending: deque = deque()
        self._thread: Thread | None = None
        self._active = False
        self._closed = False

    def submit(self, operation: Callable[[], object],
               completed: Callable[[object, Exception | None], None] | None = None,
               ) -> EvidenceWriteHandle | None:
        with self._condition:
            if self._closed or len(self._pending) >= self.queue_limit:
                return None
            handle = EvidenceWriteHandle()
            self._pending.append((handle, operation, completed))
            if self._thread is None:
                self._thread = Thread(target=self._run, name="listener-evidence", daemon=True)
                self._thread.start()
            self._condition.notify_all()
            return handle

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._pending:
                    self._active = False
                    self._thread = None
                    self._condition.notify_all()
                    return
                handle, operation, completed = self._pending.popleft()
                self._active = True
            try:
                handle.result = operation()
            except Exception as exc:
                handle.error = exc
            finally:
                try:
                    if completed is not None:
                        completed(handle.result, handle.error)
                except Exception:
                    # Reporting failures cannot recursively schedule evidence.
                    pass
                handle.done.set()
                with self._condition:
                    self._active = False
                    self._condition.notify_all()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        deadline = monotonic() + max(0, timeout)
        with self._condition:
            while self._active or self._pending:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout: float = 2.0) -> bool:
        with self._condition:
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(max(0, timeout))
        return thread is None or not thread.is_alive()


def _bounded_json(value: object, depth: int = 0) -> object:
    """Keep untrusted diagnostic details small, JSON-safe and non-recursive."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:1000]
    if depth >= 4:
        return str(value)[:200]
    if isinstance(value, dict):
        return {str(k)[:100]: _bounded_json(v, depth + 1)
                for k, v in list(value.items())[:24]}
    if isinstance(value, (list, tuple)):
        return [_bounded_json(v, depth + 1) for v in value[:24]]
    return str(value)[:200]


@dataclass
class _Incident:
    incident_id: str
    reason: str
    failure: ListenerFrame
    frames: tuple[ListenerFrame, ...]
    details: object
    failure_role: str = "failure"
    records: list[dict[str, object]] = field(default_factory=list)
    manifest: Path | None = None


class ListenerEvidence:
    """Four-frame ring, reason/episode dedup, and conservative per-run quotas.

    Eight incidents and 64 MiB of reservations per run are the default ceiling.
    Reservations for accepted jobs (including failed writes) are not refunded, so
    broken storage cannot create an unbounded retry loop. PNG worst-case space,
    sidecars and a <=64 KiB manifest are reserved *before* enqueueing. Recovery
    adds at most one frame to each incident and updates that same manifest.
    """

    def __init__(self, persist: Callable[[ListenerFrame], dict[str, object]],
                 completed: Callable[[object, Exception | None], None], *,
                 prior_frames: int = 3, max_incidents: int = 8,
                 max_bytes: int = 64 * 1024 * 1024, queue_limit: int = 4) -> None:
        self.persist = persist
        self.completed = completed
        self.writer = BoundedEvidenceWriter(queue_limit=queue_limit)
        self._lock = Lock()
        self._ring: deque[ListenerFrame] = deque(maxlen=max(1, prior_frames + 1))
        self.max_incidents = max(0, max_incidents)
        self.max_bytes = max(0, max_bytes)
        self._seen: dict[tuple[tuple[str, int, str], str], int] = {}
        self._recovered: dict[tuple[tuple[str, int, str], str], int] = {}
        # Origins stay pinned across runs: manifests refer to these exact paths.
        self._pinned_directories: set[str] = set()
        self._pending_recovery: list[_Incident] = []
        self.incident_count = 0
        self.reserved_bytes = 0

    def begin_run(self) -> bool:
        with self._lock:
            if not self.writer.wait_idle(0):
                return False
            self._ring.clear()
            self._seen.clear()
            self._recovered.clear()
            self._pending_recovery.clear()
            self.incident_count = 0
            self.reserved_bytes = 0
            return True

    def pin_directory(self, directory: Path) -> None:
        with self._lock:
            self._pinned_directories.add(os.path.normcase(os.path.abspath(directory)))

    def directory_is_pinned(self, directory: Path) -> bool:
        with self._lock:
            return os.path.normcase(os.path.abspath(directory)) in self._pinned_directories

    def rebind_directory(self, source: Path, target: Path, session_id: str) -> None:
        """Rebind migratable cached frames without changing capture identity."""
        origin = os.path.normcase(os.path.abspath(source))
        with self._lock:
            if origin in self._pinned_directories:
                raise RuntimeError("incident directory cannot be migrated")
            self._ring = deque((
                replace(frame, session_directory=target, session_id=session_id,
                        scope_id=frame.scope_id or frame.session_id)
                if frame.session_directory is not None
                and os.path.normcase(os.path.abspath(frame.session_directory)) == origin
                else frame for frame in self._ring
            ), maxlen=self._ring.maxlen)

    def remember(self, frame: ListenerFrame) -> ListenerFrame:
        with self._lock:
            for current in self._ring:
                if current.identity == frame.identity:
                    return current
            if self._ring and self._ring[-1].capture_scope != frame.capture_scope:
                self._ring.clear()
            if not self._ring or frame.capture_seq > self._ring[-1].capture_seq:
                self._ring.append(frame)
            return frame

    def find(self, snapshot: object, generation: int, phase: str, *,
             scope_id: str | None = None, capture_seq: int | None = None) -> ListenerFrame | None:
        identity = str(getattr(snapshot, "evidence_frame_id", ""))
        with self._lock:
            return next((frame for frame in reversed(self._ring)
                         if frame.capture_generation == generation
                         and frame.capture_scope[0] == phase
                         and (scope_id is None or frame.capture_scope[2] == scope_id)
                         and (capture_seq is None or frame.capture_seq == capture_seq)
                         and (frame.snapshot is snapshot or identity and
                              str(getattr(frame.snapshot, "evidence_frame_id", "")) == identity)), None)

    def find_processed(self, *, generation: int, capture_seq: int, captured_ms: int,
                       phase: str, scope_id: str) -> ListenerFrame | None:
        with self._lock:
            return next((frame for frame in reversed(self._ring)
                         if frame.capture_scope == (phase, generation, scope_id)
                         and frame.capture_seq == capture_seq
                         and getattr(frame.snapshot, "captured_monotonic_ms", None) == captured_ms), None)

    def annotate(self, frame: ListenerFrame, **details: object) -> ListenerFrame:
        with self._lock:
            for index, current in enumerate(self._ring):
                if current.identity == frame.identity:
                    # A worker can hold the pre-probe context. Merge with the
                    # newest annotation rather than erasing page/recognition data.
                    annotated = replace(frame, details=deepcopy({
                        **current.details, **frame.details, **details,
                    }))
                    self._ring[index] = annotated
                    return annotated
            return replace(frame, details=deepcopy({**frame.details, **details}))

    @staticmethod
    def _frame_budget(frame: ListenerFrame) -> int:
        # PNG overhead for tiny frames and sidecar metadata is included. The
        # standardized uint8 arrays are stored, not overlays or raw desktop crops.
        return int(getattr(frame.snapshot.image, "nbytes", 0)) * 2 + 64 * 1024

    def incident(self, reason: str, failure: ListenerFrame | None, *,
                 details: dict[str, object] | None = None, failure_role: str = "failure") -> bool:
        if failure_role not in {"failure", "context"}:
            raise ValueError("invalid failure_role")
        if failure is None or failure.session_directory is None or failure.capture_seq < 0:
            return False
        with self._lock:
            key = (failure.capture_scope, reason)
            if failure.capture_seq <= self._recovered.get(key, -1):
                return False
            if key in self._seen:
                self._seen[key] = max(self._seen[key], failure.capture_seq)
                return False
            if self.incident_count >= self.max_incidents:
                return False
            prior = tuple(frame for frame in self._ring
                          if frame.capture_scope == failure.capture_scope
                          and frame.capture_seq < failure.capture_seq)[-3:]
            frames = (*prior, failure)
            budget = sum(self._frame_budget(frame) for frame in frames) + 64 * 1024
            if self.reserved_bytes + budget > self.max_bytes:
                return False
            incident = _Incident(uuid4().hex, reason, failure, frames,
                                 _bounded_json(details or {}), failure_role=failure_role)
            handle = self.writer.submit(lambda: self._write_incident(incident), self.completed)
            if handle is None:
                return False
            self._seen[key] = failure.capture_seq
            self._pinned_directories.add(os.path.normcase(os.path.abspath(failure.session_directory)))
            self.incident_count += 1
            self.reserved_bytes += budget
            self._pending_recovery.append(incident)
            return True

    def recover(self, frame: ListenerFrame | None) -> None:
        if frame is None:
            return
        with self._lock:
            keys = [key for key in self._seen if key[0] == frame.capture_scope]
            if not keys or frame.capture_seq <= max(self._seen[key] for key in keys):
                return
            incidents = [item for item in self._pending_recovery
                         if item.failure.capture_scope == frame.capture_scope]
            for incident in incidents:
                budget = self._frame_budget(frame) + 64 * 1024
                if self.reserved_bytes + budget > self.max_bytes:
                    continue
                handle = self.writer.submit(lambda item=incident: self._write_recovery(item, frame), self.completed)
                if handle is None:
                    continue  # Leave pending state/accounting unchanged; a later frame may retry.
                self.reserved_bytes += budget
                self._pending_recovery.remove(incident)
                accepted_key = (incident.failure.capture_scope, incident.reason)
                self._recovered[accepted_key] = frame.capture_seq
                self._seen.pop(accepted_key, None)
            if not any(item.failure.capture_scope == frame.capture_scope for item in self._pending_recovery):
                for key in keys:
                    self._recovered[key] = frame.capture_seq
                    self._seen.pop(key, None)

    def _save_record(self, incident: _Incident, frame: ListenerFrame, role: str) -> dict[str, object]:
        context = replace(
            frame, session_directory=incident.failure.session_directory,
            session_id=incident.failure.session_id,
            source_phase=("failed_listener_frame" if role == "failure" else
                          "last_listener_frame" if role == "context" else frame.source_phase),
            details={**frame.details, "incident": {
                "id": incident.incident_id, "reason": incident.reason,
                "role": role, "failure": incident.details,
            }},
        )
        result = self.persist(context)
        record = {
            "role": role, "capture_generation": frame.capture_generation,
            "capture_scope": list(frame.capture_scope),
            "capture_seq": frame.capture_seq,
            "evidence_frame_id": str(getattr(frame.snapshot, "evidence_frame_id", "")),
            "source_phase": context.source_phase,
            "image_path": str(result["image_path"]),
            "metadata_path": str(result["metadata_path"]),
            "details": _bounded_json(frame.details),
        }
        incident.records.append(record)
        return result

    def _write_incident(self, incident: _Incident) -> dict[str, object]:
        # Always save the failure first, even if an earlier-frame write fails.
        result = self._save_record(incident, incident.failure, incident.failure_role)
        incident.manifest = Path(result["metadata_path"]).parent / f"incident_{incident.incident_id}.json"
        try:
            for frame in incident.frames[:-1]:
                self._save_record(incident, frame, "prior")
        finally:
            self._write_manifest(incident)
            incident.frames = ()
        return {**result, "automatic": True, "reason": incident.reason,
                "incident_manifest": str(incident.manifest)}

    def _write_recovery(self, incident: _Incident, frame: ListenerFrame) -> dict[str, object]:
        if incident.manifest is None:
            return {"status": "FAILURE", "automatic": True, "reason": incident.reason,
                    "message": "事故帧未保存，跳过恢复截图"}
        result = self._save_record(incident, frame, "recovery")
        self._write_manifest(incident)
        return {**result, "automatic": True, "reason": incident.reason,
                "incident_manifest": str(incident.manifest), "recovery": True}

    @staticmethod
    def _write_manifest(incident: _Incident) -> None:
        if incident.manifest is None:
            return
        content = json.dumps({
            "schema": "guandan.listener-incident/v1", "incident_id": incident.incident_id,
            "reason": incident.reason, "details": incident.details,
            "has_failure_frame": incident.failure_role == "failure",
            "failure_role": incident.failure_role,
            "frames": sorted(incident.records, key=lambda item: (
                item["role"] == "recovery", int(item["capture_seq"]))),
        }, ensure_ascii=False, indent=2).encode("utf-8")
        if len(content) > 64 * 1024:
            raise ValueError("listener incident manifest exceeds 64 KiB")
        temporary = incident.manifest.with_suffix(".tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
            os.replace(temporary, incident.manifest)
        finally:
            temporary.unlink(missing_ok=True)
