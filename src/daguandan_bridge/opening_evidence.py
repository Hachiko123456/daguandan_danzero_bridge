from __future__ import annotations

"""Fail-open, pre-session evidence collection for opening-state failures.

The live controller intentionally owns this sidecar instead of the reducer.  A
diagnostic failure must never change a recognition decision, delay a capture,
or manufacture a game event.  Frames are retained by reference in a bounded
memory ring and are serialized only after an incident has been queued.
"""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from threading import BoundedSemaphore, RLock
from threading import Thread, current_thread
from time import monotonic_ns
from time import monotonic as monotonic_seconds
from time import sleep
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4
from weakref import ref as weakref_ref
from queue import Empty, Full, Queue

import cv2
import numpy as np

from .danzero.state import RANKS
from .runtime_identity import get_runtime_identity
from .resource_fingerprint import recognition_resource_identity
from .startup_diagnostics import current_startup_diagnostics
from .storage import atomic_write_json
from .diagnostic_budget import DiagnosticBudget, assert_plain_path, is_reparse


OPENING_EVIDENCE_SCHEMA = "guandan.opening-evidence/1"
OPENING_INCIDENT_SCHEMA = "guandan.opening-incident/1"
RECOGNITION_TRACE_SCHEMA = "guandan.recognition-trace/1"

OPENING_WINDOW_NOT_FOUND = "OPENING-WINDOW-NOT-FOUND"
OPENING_WINDOW_MINIMIZED = "OPENING-WINDOW-MINIMIZED"
OPENING_GEOMETRY_CHANGED = "OPENING-GEOMETRY-CHANGED"
OPENING_GEOMETRY_RECOVERY = "OPENING-GEOMETRY-RECOVERY"
OPENING_CAPTURE_BLACK_FRAME = "OPENING-CAPTURE-BLACK-FRAME"
OPENING_CAPTURE_ERROR = "OPENING-CAPTURE-ERROR"
OPENING_ANCHOR_TIMEOUT = "OPENING-ANCHOR-TIMEOUT"
OPENING_LEVEL_MISSING = "OPENING-LEVEL-MISSING"
OPENING_LEVEL_CONFLICT = "OPENING-LEVEL-CONFLICT"
OPENING_HAND_COUNT_MISMATCH = "OPENING-HAND-COUNT-MISMATCH"
OPENING_HAND_UNSTABLE = "OPENING-HAND-UNSTABLE"
OPENING_LEAD_UNSTABLE = "OPENING-LEAD-UNSTABLE"
OPENING_RESOURCE_ERROR = "OPENING-RESOURCE-ERROR"
OPENING_TIMEOUT = "OPENING-TIMEOUT"


@dataclass
class _OpeningFrame:
    frame_id: str
    seq: int
    monotonic_ms: int
    wall_time: str
    snapshot: object
    byte_size: int
    frame_metadata: dict[str, object]
    recognition: dict[str, object] | None = None
    recognition_trace: dict[str, object] | None = None
    anchor_score: float | None = None
    analysis: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OpeningEvidenceMetrics:
    retained_frames: int
    retained_bytes: int
    latest_seq: int
    dropped_age: int
    dropped_budget: int
    dropped_writer_queue: int
    incidents_queued: int
    incidents_written: int
    writer_failures: int
    orphan_recognitions: int
    pending_snapshot_bytes: int
    incident_occurrences: int
    suppressed_incidents: int = 0
    media_budget_exhausted: int = 0
    text_budget_exhausted: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "retained_frames": self.retained_frames,
            "retained_bytes": self.retained_bytes,
            "latest_seq": self.latest_seq,
            "dropped_age": self.dropped_age,
            "dropped_budget": self.dropped_budget,
            "dropped_writer_queue": self.dropped_writer_queue,
            "incidents_queued": self.incidents_queued,
            "incidents_written": self.incidents_written,
            "writer_failures": self.writer_failures,
            "orphan_recognitions": self.orphan_recognitions,
            "pending_snapshot_bytes": self.pending_snapshot_bytes,
            "incident_occurrences": self.incident_occurrences,
            "suppressed_incidents": self.suppressed_incidents,
            "media_budget_exhausted": self.media_budget_exhausted,
            "text_budget_exhausted": self.text_budget_exhausted,
        }


class _DaemonSingleWorker:
    """Minimal Future executor whose worker cannot block interpreter shutdown."""

    def __init__(self, *, name: str) -> None:
        self._queue: Queue[tuple[Future[None], Callable[..., None], tuple[object, ...]] | None] = Queue()
        self._lock = RLock()
        self._closed = False
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, operation: Callable[..., None], *args: object) -> Future[None]:
        with self._lock:
            if self._closed:
                raise RuntimeError("opening evidence writer is closed")
            future: Future[None] = Future()
            self._queue.put_nowait((future, operation, args))
            return future

    def shutdown(
        self,
        *,
        wait: bool,
        cancel_futures: bool,
        timeout: float | None = None,
    ) -> bool:
        with self._lock:
            if not self._closed:
                self._closed = True
                if cancel_futures:
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except Empty:
                            break
                        if item is not None:
                            item[0].cancel()
                        self._queue.task_done()
                self._queue.put_nowait(None)
        if wait and self._thread is not current_thread():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                future, operation, args = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    operation(*args)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(None)
            finally:
                self._queue.task_done()


class OpeningEvidenceMonitor:
    """Maintain a bounded opening ring and persist deduplicated incidents.

    All public observation methods swallow their own failures.  This is a
    deliberate safety property: the sidecar is evidence-only and cannot make
    capture, recognition, session creation, or game-state progression fail.
    """

    def __init__(
        self,
        *,
        diagnostics_root: Path | None = None,
        profiles_root: Path | None = None,
        profile_name: str = "tencent_daguandan",
        max_age_seconds: float = 8.0,
        max_bytes: int = 64 * 1024 * 1024,
        field_timeout_seconds: float = 8.0,
        writer_queue_size: int = 4,
        max_persisted_frames: int = 3,
        max_persisted_image_bytes: int = 16 * 1024 * 1024,
        max_run_image_bytes: int = 64 * 1024 * 1024,
        max_total_image_bytes: int = 512 * 1024 * 1024,
        max_run_text_bytes: int = 8 * 1024 * 1024,
        max_total_text_bytes: int = 64 * 1024 * 1024,
        max_incident_text_bytes: int = 256 * 1024,
        max_run_incidents: int = 128,
        max_total_incidents: int = 1024,
        diagnostics_runs_root: Path | None = None,
        delivery_settle_seconds: float = 0.5,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if max_age_seconds <= 0 or field_timeout_seconds <= 0:
            raise ValueError("opening evidence timeouts must be positive")
        if max_bytes <= 0 or writer_queue_size <= 0:
            raise ValueError("opening evidence budgets must be positive")
        startup = current_startup_diagnostics()
        default_root = getattr(startup, "run_directory", None)
        self.root = Path(diagnostics_root or default_root or Path.cwd()) / "opening"
        self.profiles_root = Path(profiles_root) if profiles_root is not None else None
        self.profile_name = str(profile_name)
        self.max_age_ms = max(1, int(max_age_seconds * 1000))
        self.max_bytes = int(max_bytes)
        self.field_timeout_ms = max(1, int(field_timeout_seconds * 1000))
        self.max_persisted_frames = max(1, int(max_persisted_frames))
        self.max_persisted_image_bytes = max(1, int(max_persisted_image_bytes))
        self.max_incident_text_bytes = max(1, int(max_incident_text_bytes))
        self.max_run_incidents = max(1, int(max_run_incidents))
        self._disk_budget = DiagnosticBudget(
            self.root.parent,
            runs_root=diagnostics_runs_root,
            run_media_bytes=max_run_image_bytes,
            total_media_bytes=max_total_image_bytes,
            run_text_bytes=max_run_text_bytes,
            total_text_bytes=max_total_text_bytes,
            run_incidents=max_run_incidents,
            total_incidents=max_total_incidents,
        )
        self._media_exhausted = False
        self._text_exhausted = False
        self._suppressed_incidents = 0
        self._shared_media: dict[tuple[str, str], dict[str, object]] = {}
        self._page = "unknown"
        self._page_observed = False
        self._table_phase_active = False
        self._last_recognition_document: dict[str, object] | None = None
        self._last_capture_metadata: dict[str, object] | None = None
        self.delivery_settle_seconds = max(0.0, float(delivery_settle_seconds))
        self._clock_ms = clock_ms or (lambda: monotonic_ns() // 1_000_000)
        self._ring: deque[_OpeningFrame] = deque()
        self._frame_by_identity: dict[int, _OpeningFrame] = {}
        self._frame_by_id: dict[str, _OpeningFrame] = {}
        self._ambiguous_frame_ids: set[str] = set()
        self._known_frame_inputs: dict[
            int, tuple[object, str, str, dict[str, object]]
        ] = {}
        self._ring_record_ids: set[int] = set()
        self._retained_bytes = 0
        self._latest_seq = 0
        self._dropped_age = 0
        self._dropped_budget = 0
        self._dropped_writer_queue = 0
        self._incidents_queued = 0
        self._incidents_written = 0
        self._writer_failures = 0
        self._orphan_recognitions = 0
        self._incident_occurrences = 0
        self._dedup: set[tuple[str, str]] = set()
        self._active_incident_ids: dict[tuple[str, str], str] = {}
        self._occurrences_by_incident: dict[str, int] = {}
        self._future_keys: dict[Future[None], tuple[str, str]] = {}
        self._levels: deque[str] = deque(maxlen=3)
        self._hands: deque[tuple[str, ...]] = deque(maxlen=3)
        self._anchor_ready = False
        self._listener_started_ms: int | None = None
        self._armed_ms: int | None = None
        self._field_blocked_since: dict[str, int] = {}
        self._lock = RLock()
        self._writer_slots = BoundedSemaphore(int(writer_queue_size))
        self._executor: _DaemonSingleWorker | None = None
        self._futures: set[Future[None]] = set()
        self._future_record_ids: dict[Future[None], tuple[int, ...]] = {}
        self._pending_record_refs: dict[int, tuple[int, int]] = {}
        self._pending_record_bytes = 0
        self._closed = False

    def begin(self, *, monotonic_ms: int | None = None) -> None:
        """Start a fresh listener interval without touching captured evidence."""

        try:
            with self._lock:
                if self._closed:
                    return
                self._listener_started_ms = int(monotonic_ms if monotonic_ms is not None else self._clock_ms())
                self._armed_ms = None
                self._field_blocked_since = {
                    "table_anchor": self._listener_started_ms,
                }
                self._dedup.clear()
                self._active_incident_ids.clear()
                self._levels.clear()
                self._hands.clear()
                self._anchor_ready = False
                self._page = "unknown"
                self._page_observed = False
                self._table_phase_active = False
                self._last_recognition_document = None
                self._last_capture_metadata = None
                self._clear_ring_locked()
                self._frame_by_identity.clear()
                self._frame_by_id.clear()
                self._ambiguous_frame_ids.clear()
                self._known_frame_inputs.clear()
                self._ring_record_ids.clear()
        except BaseException:
            return

    def observe_page(self, page: str, *, monotonic_ms: int | None = None) -> None:
        """Disarm screenshot/timeouts off-table; rearm only a new table phase.

        Page classification is supplied by the live controller. Explicit unknown
        disarms media without inventing a new-game boundary. Legacy callers that
        do not supply page signals retain their previous diagnostic behavior.
        """
        try:
            normalized = str(getattr(page, "value", page)).casefold()
            if normalized not in {"table", "lobby", "settlement", "unknown"}:
                normalized = "unknown"
            now = int(monotonic_ms if monotonic_ms is not None else self._clock_ms())
            with self._lock:
                self._page = normalized
                self._page_observed = True
                if normalized != "table":
                    if normalized in {"lobby", "settlement"}:
                        self._table_phase_active = False
                    self._listener_started_ms = None
                    self._armed_ms = None
                    self._field_blocked_since.clear()
                    self._hands.clear()
                    self._levels.clear()
                    self._anchor_ready = False
                    self._clear_ring_locked()
                    self._frame_by_identity.clear()
                    self._frame_by_id.clear()
                    self._known_frame_inputs.clear()
                elif not self._table_phase_active or self._listener_started_ms is None:
                    new_phase = not self._table_phase_active
                    self._table_phase_active = True
                    self._listener_started_ms = now
                    self._armed_ms = None
                    self._field_blocked_since = {"table_anchor": now}
                    if new_phase:
                        self._dedup.clear()
                        self._active_incident_ids.clear()
        except BaseException:
            return

    def observe_frame(self, snapshot: object, *, monotonic_ms: int | None = None) -> None:
        """Append one immutable capture reference in O(1), never encode it here."""

        try:
            with self._lock:
                if self._closed or (self._page_observed and self._page != "table"):
                    return
                metadata_only = self._media_exhausted or self._text_exhausted
                if metadata_only:
                    self._latest_seq += 1
                    self._last_capture_metadata = _frame_metadata(snapshot)
            if metadata_only:
                self._emit_timeouts_if_due()
                return
            now = int(monotonic_ms if monotonic_ms is not None else self._clock_ms())
            byte_size, metadata, black = _frame_capture_details(snapshot)
            with self._lock:
                if self._closed:
                    return
                self._append_frame_locked(
                    snapshot,
                    now=now,
                    byte_size=byte_size,
                    metadata=metadata,
                    analysis=_initial_analysis(now),
                )
                if not black:
                    self._clear_error_stage_locked("capture")
            if black:
                standard = getattr(snapshot, "image", None)
                self.emit_incident(
                    OPENING_CAPTURE_BLACK_FRAME,
                    field="capture",
                    reason="captured standardized client frame is effectively black",
                    monotonic_ms=now,
                    evidence={"pixel_max": int(np.max(standard)) if isinstance(standard, np.ndarray) and standard.size else 0},
                )
            self._emit_timeouts_if_due()
        except BaseException:
            return

    def _append_frame_locked(
        self,
        snapshot: object,
        *,
        now: int,
        byte_size: int,
        metadata: Mapping[str, object],
        analysis: Mapping[str, object],
        recognition: Mapping[str, object] | None = None,
        trace: Mapping[str, object] | None = None,
    ) -> _OpeningFrame:
        """Publish a fully initialized frame while the evidence lock is held."""

        self._latest_seq += 1
        frame_metadata = dict(metadata)
        requested_frame_id = str(
            getattr(snapshot, "evidence_frame_id", "") or uuid4().hex
        )
        frame_id = requested_frame_id
        if frame_id in self._frame_by_id or frame_id in self._ambiguous_frame_ids:
            self._ambiguous_frame_ids.add(requested_frame_id)
            self._frame_by_id.pop(requested_frame_id, None)
            frame_id = uuid4().hex
            frame_metadata["duplicate_source_frame_id"] = requested_frame_id
        frame_metadata["frame_id"] = frame_id
        pixel_hash = str(frame_metadata.get("standardized_pixel_sha256") or "")
        try:
            snapshot_ref: object = weakref_ref(snapshot)
        except TypeError:
            snapshot_ref = lambda: None
        analysis_document = dict(analysis)
        self._known_frame_inputs[id(snapshot)] = (
            snapshot_ref,
            requested_frame_id,
            pixel_hash,
            analysis_document,
        )
        for identity, known in tuple(self._known_frame_inputs.items()):
            dereference = known[0]
            if callable(dereference) and dereference() is None:
                self._known_frame_inputs.pop(identity, None)
        item = _OpeningFrame(
            frame_id=frame_id,
            seq=self._latest_seq,
            monotonic_ms=now,
            wall_time=_wall_time(snapshot),
            snapshot=snapshot,
            byte_size=byte_size,
            frame_metadata=frame_metadata,
            recognition=dict(recognition) if isinstance(recognition, Mapping) else None,
            recognition_trace=dict(trace) if isinstance(trace, Mapping) else None,
            analysis=analysis_document,
        )
        self._ring.append(item)
        self._frame_by_identity[id(snapshot)] = item
        if requested_frame_id not in self._ambiguous_frame_ids:
            self._frame_by_id[requested_frame_id] = item
        self._frame_by_id[frame_id] = item
        self._ring_record_ids.add(id(item))
        self._retained_bytes += byte_size
        self._trim_locked(now)
        return item

    def _restore_and_bind_recognition_locked(
        self,
        snapshot: object,
        document: Mapping[str, object],
        trace: Mapping[str, object] | None,
        *,
        now: int,
    ) -> tuple[_OpeningFrame | None, bool]:
        """Atomically restore an exact evicted input already bound to its result."""

        item = self._correlated_frame_locked(snapshot, trace=trace)
        recovered = False
        if item is None:
            if not self._can_recover_exact_locked(snapshot, trace=trace):
                return None, False
            known = self._known_frame_inputs.get(id(snapshot))
            if known is None:
                return None, False
            analysis = dict(known[3])
            analysis["completed_ms"] = now
            analysis["recovered_exact_analysis_frame"] = True
            if analysis.get("status") != "dropped":
                analysis["status"] = "completed"
            byte_size, metadata, _black = _frame_capture_details(snapshot)
            item = self._append_frame_locked(
                snapshot,
                now=now,
                byte_size=byte_size,
                metadata=metadata,
                analysis=analysis,
                recognition=document,
                trace=trace,
            )
            recovered = id(item) in self._ring_record_ids
            if not recovered:
                return None, False
        else:
            item.recognition = dict(document)
            item.recognition_trace = (
                dict(trace) if isinstance(trace, Mapping) else None
            )
            item.analysis["completed_ms"] = now
            item.analysis["recovered_exact_analysis_frame"] = False
            if item.analysis.get("status") != "dropped":
                item.analysis["status"] = "completed"
        return item, recovered

    def observe_analysis_submitted(self, snapshot: object) -> None:
        self._mark_analysis(snapshot, "submitted")

    def observe_analysis_started(self, snapshot: object) -> None:
        self._mark_analysis(snapshot, "started")

    def observe_analysis_dropped(self, snapshot: object, *, reason: str) -> None:
        try:
            now = self._clock_ms()
            with self._lock:
                item = self._correlated_frame_locked(snapshot)
                if item is None:
                    return
                item.analysis.update(
                    {
                        "status": "dropped",
                        "dropped_ms": now,
                        "drop_reason": str(reason)[:128],
                        "gate_delivered": False,
                    }
                )
        except BaseException:
            return

    def observe_delivery(
        self,
        snapshot: object,
        *,
        gate_eligible: bool,
    ) -> None:
        try:
            now = self._clock_ms()
            with self._lock:
                item = self._correlated_frame_locked(snapshot)
                if item is None:
                    return
                item.analysis.update(
                    {
                        "status": "delivered" if gate_eligible else "delivered_ineligible",
                        "delivered_ms": now,
                        "gate_delivered": bool(gate_eligible),
                    }
                )
        except BaseException:
            return

    def _mark_analysis(self, snapshot: object, state: str) -> None:
        try:
            now = self._clock_ms()
            with self._lock:
                item = self._correlated_frame_locked(snapshot)
                if item is None:
                    return
                order = {"captured": 0, "submitted": 1, "started": 2}
                current = str(item.analysis.get("status") or "captured")
                if current in {"completed", "dropped", "delivered", "delivered_ineligible"}:
                    return
                if order.get(state, 0) < order.get(current, 0):
                    return
                item.analysis["status"] = state
                item.analysis[f"{state}_ms"] = now
        except BaseException:
            return

    def observe_anchor(
        self,
        snapshot: object,
        score: float,
        *,
        required_score: float,
    ) -> None:
        try:
            now = self._clock_ms()
            with self._lock:
                if self._page_observed and self._page != "table":
                    return
                item = self._correlated_frame_locked(snapshot)
                if item is not None:
                    item.anchor_score = float(score)
                if float(score) >= float(required_score):
                    self._anchor_ready = True
                    if self._armed_ms is None:
                        self._armed_ms = now
                    self._field_blocked_since.pop("table_anchor", None)
                    self._clear_field_dedup_locked("table_anchor")
                elif self._armed_ms is not None:
                    self._field_blocked_since.setdefault("table_anchor", now)
            self._emit_timeouts_if_due()
        except BaseException:
            return

    def observe_recognition(
        self,
        snapshot: object,
        result: object,
        trace: Mapping[str, object] | None = None,
        *,
        opening_seed_valid: bool | None = None,
    ) -> None:
        try:
            document = _recognition_document(result)
            level = str(document.get("round_level") or "")
            hand = tuple(sorted(str(card) for card in document.get("my_hand", [])))
            now = self._clock_ms()
            with self._lock:
                if self._page_observed and self._page != "table":
                    return
                self._last_recognition_document = {
                    key: document.get(key) for key in (
                        "round_level", "hand_count", "current_player", "lead_player", "elapsed_ms",
                    )
                }
                metadata_only = self._media_exhausted or self._text_exhausted
                if metadata_only:
                    item = None
                else:
                    item, _recovered = self._restore_and_bind_recognition_locked(
                        snapshot, document, trace, now=now,
                    )
                if item is None and not metadata_only:
                    self._orphan_recognitions += 1
                self._clear_error_stage_locked("recognition")
                if level in RANKS:
                    self._levels.append(level)
                if len(hand) == 27:
                    self._hands.append(hand)
                else:
                    # An incomplete hand is neither stability nor recovery.
                    # Do not combine a pre-gap complete hand with a later one.
                    self._hands.clear()
                if self._armed_ms is None and (level in RANKS or bool(hand)):
                    self._armed_ms = now
                    if not self._anchor_ready:
                        self._field_blocked_since.setdefault("table_anchor", now)
                level_conflict = len(set(self._levels)) > 1
                if level in RANKS and not level_conflict:
                    self._set_field_state_locked("round_level", True, now)
                elif level not in RANKS:
                    self._set_field_state_locked("round_level", False, now)
                else:
                    self._field_blocked_since.pop("round_level", None)
                self._set_field_state_locked("hand_count", len(hand) == 27, now)
                hand_stable = len(self._hands) >= 2 and len(set(self._hands)) == 1
                self._set_field_state_locked(
                    "hand_stability",
                    len(hand) == 27 and hand_stable,
                    now,
                )
                if len(hand) == 27 and hand_stable:
                    self._clear_field_dedup_locked("my_hand")
                hand_unstable = len(self._hands) >= 2 and len(set(self._hands)) > 1
                if level in RANKS and len(hand) == 27 and opening_seed_valid is not None:
                    self._set_field_state_locked("lead_player", bool(opening_seed_valid), now)
                if not level_conflict:
                    self._resolve_episode_locked(
                        (OPENING_LEVEL_CONFLICT, "round_level")
                    )
                if len(hand) == 27 and hand_stable:
                    self._resolve_episode_locked((OPENING_HAND_UNSTABLE, "my_hand"))
            trace_candidates = (
                trace.get("candidates", ())
                if isinstance(trace, Mapping)
                else ()
            )
            level_candidates = [
                item
                for item in trace_candidates
                if isinstance(item, Mapping)
                and str(item.get("field", "")).casefold() in {"level_rank", "round_level"}
            ]
            if level not in RANKS and level_candidates:
                best = max(level_candidates, key=lambda item: _number(item.get("score")))
                threshold = _number(best.get("threshold"), default=1.0)
                if _number(best.get("score")) < threshold:
                    self.emit_incident(
                        "OPENING-LEVEL-BELOW-THRESHOLD",
                        field="round_level",
                        reason="level candidates were present but none reached the production threshold",
                        monotonic_ms=now,
                        evidence={
                            "best_candidate": _json_safe(dict(best)),
                            "candidate_count": len(level_candidates),
                        },
                    )
                labels = {
                    str(item.get("label", ""))
                    for item in level_candidates
                    if _number(item.get("score")) >= threshold
                }
                if len(labels) > 1:
                    self.emit_incident(
                        "OPENING-LEVEL-CANDIDATE-CONFLICT",
                        field="round_level",
                        reason="multiple level candidates cleared the threshold",
                        monotonic_ms=now,
                        evidence={"labels": sorted(labels)},
                    )
            if hand:
                try:
                    from .danzero.state import GuanDanState

                    GuanDanState().confirm_hand(hand)
                except Exception as exc:
                    self.emit_incident(
                        "OPENING-HAND-INVALID",
                        field="my_hand",
                        reason="recognized hand failed canonical deck validation",
                        monotonic_ms=now,
                        evidence={"error_type": type(exc).__name__},
                    )
            if level_conflict:
                self.emit_incident(
                    OPENING_LEVEL_CONFLICT,
                    field="round_level",
                    reason="valid level observations conflict within the opening stability window",
                    monotonic_ms=now,
                    evidence={"observed_levels": list(self._levels)},
                )
            if hand_unstable:
                self.emit_incident(
                    OPENING_HAND_UNSTABLE,
                    field="my_hand",
                    reason="27-card observations are not stable across opening frames",
                    monotonic_ms=now,
                    evidence={"hand_hashes": [_cards_hash(cards) for cards in self._hands]},
                )
            self._emit_timeouts_if_due(result=result)
        except BaseException:
            return

    def observe_failure(
        self,
        error: object,
        *,
        stage: str = "capture",
        monotonic_ms: int | None = None,
        snapshot: object | None = None,
    ) -> str:
        """Classify a window/capture/resource failure and queue it once."""

        normalized_stage = str(stage).strip().lower() or "unknown"
        code = classify_opening_failure(error)
        if normalized_stage in {"recognition", "anchor"} and code == OPENING_CAPTURE_ERROR:
            code = "OPENING-RECOGNITION-ERROR"
        try:
            source_code = str(getattr(error, "code", "") or "") or None
            evidence: dict[str, object] = {
                "error_type": type(error).__name__,
                "source_error_code": source_code,
            }
            details = getattr(error, "details", None)
            if isinstance(details, Mapping):
                evidence.update(_json_safe(dict(details)))
            with self._lock:
                correlated = (
                    self._correlated_frame_locked(snapshot)
                    if snapshot is not None
                    else None
                )
                recoverable = (
                    self._can_recover_exact_locked(snapshot)
                    if snapshot is not None
                    else False
                )
            recovered = False
            if correlated is None and snapshot is not None and recoverable:
                # A slow recognition can outlive the ordinary 8-second ring.
                # Reinsert that exact id+pixel input as failure evidence; do
                # not substitute the newest unrelated frame.
                self.observe_frame(
                    snapshot,
                    monotonic_ms=self._clock_ms(),
                )
                with self._lock:
                    correlated = self._correlated_frame_locked(snapshot)
                recovered = correlated is not None
            with self._lock:
                if correlated is not None:
                    evidence.update(
                        {
                            "frame_id": correlated.frame_id,
                            "frame_seq": correlated.seq,
                            "input_pixel_sha256": correlated.frame_metadata.get(
                                "standardized_pixel_sha256"
                            ),
                            "recovered_exact_failure_frame": recovered,
                        }
                    )
            self.emit_incident(
                code,
                field=normalized_stage,
                reason=str(error),
                monotonic_ms=monotonic_ms,
                evidence=evidence,
            )
        except BaseException:
            pass
        return code

    def observe_geometry_recovery(
        self,
        *,
        result: str,
        generation: int,
        attempt_count: int,
        details: Mapping[str, object] | None = None,
        reason: str = "",
        monotonic_ms: int | None = None,
    ) -> None:
        """Persist one auditable recovery outcome through the incident chain."""

        normalized_result = str(result).strip().lower() or "unknown"
        evidence = dict(details or {})
        evidence.update(
            {
                "result": normalized_result,
                "generation": int(generation),
                "attempt_count": int(attempt_count),
            }
        )
        self.emit_incident(
            OPENING_GEOMETRY_RECOVERY,
            field=f"geometry_recovery_{normalized_result}",
            reason=str(reason or f"geometry recovery {normalized_result}"),
            monotonic_ms=monotonic_ms,
            evidence=evidence,
        )

    def mark_session_started(self) -> None:
        """Stop opening timeouts; retained evidence remains exportable."""

        try:
            with self._lock:
                self._listener_started_ms = None
                self._armed_ms = None
                self._field_blocked_since.clear()
        except BaseException:
            return

    def emit_incident(
        self,
        code: str,
        *,
        field: str,
        reason: str,
        monotonic_ms: int | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> bool:
        """Queue a deduplicated snapshot; return immediately if the queue is full."""

        slot_acquired = False
        record_ids: tuple[int, ...] = ()
        incident_id: str | None = None
        key: tuple[str, str] | None = None
        try:
            now = int(monotonic_ms if monotonic_ms is not None else self._clock_ms())
            normalized_code = str(code).strip().upper()
            normalized_field = str(field).strip().lower() or "unknown"
            key = (normalized_code, normalized_field)
            with self._lock:
                if self._closed:
                    return False
                if self._page_observed and self._page != "table":
                    return False
                if key in self._dedup:
                    incident_id = self._active_incident_ids.get(key)
                    if incident_id is not None:
                        self._occurrences_by_incident[incident_id] = (
                            min(2**63 - 1, self._occurrences_by_incident.get(incident_id, 1) + 1)
                        )
                        self._incident_occurrences += 1
                    return False
                if self._text_exhausted or self._incidents_queued >= self.max_run_incidents:
                    self._text_exhausted = True
                    self._clear_ring_locked()
                    self._frame_by_identity.clear()
                    self._frame_by_id.clear()
                    self._known_frame_inputs.clear()
                    self._suppressed_incidents = min(2**63 - 1, self._suppressed_incidents + 1)
                    return False
            if not self._writer_slots.acquire(blocking=False):
                with self._lock:
                    self._dropped_writer_queue += 1
                return False
            slot_acquired = True
            incident_id = f"OPEN-{now}-{uuid4().hex[:8]}"
            with self._lock:
                if self._closed:
                    self._writer_slots.release()
                    return False
                self._dedup.add(key)
                self._active_incident_ids[key] = incident_id
                self._occurrences_by_incident[incident_id] = 1
                self._incident_occurrences += 1
                records = self._select_incident_records_locked()
                record_ids = self._retain_pending_records_locked(records)
                metrics = self._metrics_locked().to_dict()
                last_observation = {
                    "recognition": dict(self._last_recognition_document or {}),
                    "capture": dict(self._last_capture_metadata or {}),
                    "latest_seq": self._latest_seq,
                }
            payload = {
                "schema": OPENING_INCIDENT_SCHEMA,
                "incident_id": incident_id,
                "code": normalized_code,
                "blocked_stage": "opening",
                "field": normalized_field,
                "reason": str(reason)[:2048],
                "monotonic_ms": now,
                "wall_time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "evidence": _json_safe(dict(evidence or {})),
                "episode": {"occurrence_count": 1},
                "last_observation": last_observation,
                "runtime_identity": get_runtime_identity(),
            }
            with self._lock:
                if self._executor is None:
                    self._executor = _DaemonSingleWorker(name="opening-evidence")
                self._incidents_queued += 1
                future = self._executor.submit(
                    self._write_incident,
                    incident_id,
                    payload,
                    records,
                    metrics,
                )
                self._futures.add(future)
                self._future_keys[future] = key
                self._future_record_ids[future] = record_ids
                future.add_done_callback(self._writer_finished)
            return True
        except BaseException:
            with self._lock:
                if record_ids:
                    self._release_pending_records_locked(record_ids)
                if key is not None:
                    self._resolve_episode_locked(key)
                if incident_id is not None:
                    self._occurrences_by_incident.pop(incident_id, None)
            if slot_acquired:
                try:
                    self._writer_slots.release()
                except (ValueError, RuntimeError):
                    pass
            return False

    def _select_incident_records_locked(self) -> tuple[_OpeningFrame, ...]:
        if self._media_exhausted:
            return ()
        # Preserve the exact completed/started recognition input ahead of the
        # newest capture-only frames. Reserving writer memory leaves room for a
        # subsequent recognition frame instead of pinning the entire ring.
        priority = sorted(
            self._ring,
            key=lambda item: (
                item.recognition is not None,
                item.analysis.get("status") in {"started", "submitted"},
                item.seq,
            ),
            reverse=True,
        )
        pending = sum(size for _count, size in self._pending_record_refs.values())
        reserve = self.max_bytes // 2
        selected: list[_OpeningFrame] = []
        for record in priority:
            if len(selected) >= self.max_persisted_frames:
                break
            extra = 0 if id(record) in self._pending_record_refs else record.byte_size
            if pending + extra > reserve:
                # A deliberately tiny test/custom ring can fit only one frame.
                if pending or selected or record.byte_size > self.max_bytes:
                    continue
            selected.append(record)
            pending += extra
        return tuple(sorted(selected, key=lambda item: item.seq))

    def flush(self, timeout: float = 10.0) -> bool:
        try:
            deadline = monotonic_seconds() + max(0.0, float(timeout))
            with self._lock:
                futures = tuple(self._futures)
            for future in futures:
                remaining = deadline - monotonic_seconds()
                if remaining <= 0:
                    return False
                future.result(timeout=remaining)
            while True:
                with self._lock:
                    callbacks_pending = bool(self._futures)
                if not callbacks_pending:
                    break
                if monotonic_seconds() >= deadline:
                    return False
                sleep(0.001)
            self._sync_episode_occurrences(deadline=deadline)
            with self._lock:
                return self._writer_failures == 0
        except BaseException:
            return False

    def close(self, timeout: float = 5.0) -> None:
        # Diagnostics are a fail-open sidecar, never a multi-second GUI shutdown
        # dependency. Explicit flush() remains available for support exports.
        deadline = monotonic_seconds() + min(0.25, max(0.0, float(timeout)))
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.flush(max(0.0, deadline - monotonic_seconds()))
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(
                wait=True,
                cancel_futures=True,
                timeout=max(0.0, deadline - monotonic_seconds()),
            )

    def metrics(self) -> OpeningEvidenceMetrics:
        with self._lock:
            return self._metrics_locked()

    def _metrics_locked(self) -> OpeningEvidenceMetrics:
        return OpeningEvidenceMetrics(
            retained_frames=len(self._ring),
            retained_bytes=self._retained_bytes,
            latest_seq=self._latest_seq,
            dropped_age=self._dropped_age,
            dropped_budget=self._dropped_budget,
            dropped_writer_queue=self._dropped_writer_queue,
            incidents_queued=self._incidents_queued,
            incidents_written=self._incidents_written,
            writer_failures=self._writer_failures,
            orphan_recognitions=self._orphan_recognitions,
            pending_snapshot_bytes=self._pending_record_bytes,
            incident_occurrences=self._incident_occurrences,
            suppressed_incidents=self._suppressed_incidents,
            media_budget_exhausted=int(self._media_exhausted),
            text_budget_exhausted=int(self._text_exhausted),
        )

    def _trim_locked(self, now_ms: int) -> None:
        while self._ring and now_ms - self._ring[0].monotonic_ms > self.max_age_ms:
            self._drop_left_locked(age=True)
        while self._ring and self._retained_bytes > self.max_bytes:
            self._drop_left_locked(age=False)

    def _drop_left_locked(self, *, age: bool) -> None:
        item = self._ring.popleft()
        if self._frame_by_identity.get(id(item.snapshot)) is item:
            self._frame_by_identity.pop(id(item.snapshot), None)
        for frame_id in (item.frame_id, str(getattr(item.snapshot, "evidence_frame_id", ""))):
            if self._frame_by_id.get(frame_id) is item:
                self._frame_by_id.pop(frame_id, None)
        self._ring_record_ids.discard(id(item))
        pending = self._pending_record_refs.get(id(item))
        if pending is None:
            self._retained_bytes = max(0, self._retained_bytes - item.byte_size)
        else:
            self._pending_record_bytes += item.byte_size
        if age:
            self._dropped_age += 1
        else:
            self._dropped_budget += 1

    def _clear_ring_locked(self) -> None:
        """Release a listener ring while preserving queued-writer accounting."""

        while self._ring:
            item = self._ring.popleft()
            identity = id(item)
            self._ring_record_ids.discard(identity)
            if identity in self._pending_record_refs:
                self._pending_record_bytes += item.byte_size
            else:
                self._retained_bytes = max(
                    0,
                    self._retained_bytes - item.byte_size,
                )

    def _emit_timeouts_if_due(self, *, result: object | None = None) -> None:
        now = self._clock_ms()
        with self._lock:
            if self._page_observed and self._page != "table":
                return
            armed = self._armed_ms
            listener_started = self._listener_started_ms
            blocked = dict(self._field_blocked_since)
        if listener_started is None:
            return
        if now - listener_started >= self.field_timeout_ms:
            self.emit_incident(
                OPENING_TIMEOUT,
                field="opening",
                reason="opening listener did not form a session before timeout",
                monotonic_ms=now,
            )
        if now - blocked.get("table_anchor", now) >= self.field_timeout_ms:
            self.emit_incident(
                OPENING_ANCHOR_TIMEOUT,
                field="table_anchor",
                reason="table anchor did not meet the readiness score before timeout",
                monotonic_ms=now,
            )
        if armed is None:
            return
        if now - blocked.get("round_level", now) >= self.field_timeout_ms:
            self.emit_incident(
                OPENING_LEVEL_MISSING,
                field="round_level",
                reason="round level remained unresolved until opening timeout",
                monotonic_ms=now,
            )
        if now - blocked.get("hand_count", now) >= self.field_timeout_ms:
            hand = tuple(getattr(result, "my_hand", ()) or ()) if result is not None else None
            self.emit_incident(
                OPENING_HAND_COUNT_MISMATCH,
                field="my_hand",
                reason="opening hand count did not equal 27 before timeout",
                monotonic_ms=now,
                evidence={
                    "actual_count": len(hand) if hand is not None else None,
                    "observation_status": "observed" if hand is not None else "unknown",
                    "required_count": 27,
                },
            )
        if now - blocked.get("hand_stability", now) >= self.field_timeout_ms:
            self.emit_incident(
                OPENING_HAND_UNSTABLE,
                field="my_hand",
                reason="27-card observations did not reach a stable two-frame consensus",
                monotonic_ms=now,
            )
        if now - blocked.get("lead_player", now) >= self.field_timeout_ms:
            self.emit_incident(
                OPENING_LEAD_UNSTABLE,
                field="lead_player",
                reason="complete opening fields could not form a valid lead/opening anchor",
                monotonic_ms=now,
            )

    def _set_field_state_locked(self, field: str, valid: bool, now: int) -> None:
        if valid:
            self._field_blocked_since.pop(field, None)
            self._clear_field_dedup_locked(field)
        elif self._armed_ms is not None:
            self._field_blocked_since.setdefault(field, now)

    def _clear_field_dedup_locked(self, field: str) -> None:
        normalized = field.casefold()
        for key in tuple(self._dedup):
            if key[1].casefold() == normalized:
                self._resolve_episode_locked(key)

    def _resolve_episode_locked(self, key: tuple[str, str]) -> None:
        self._dedup.discard(key)
        self._active_incident_ids.pop(key, None)

    def _clear_error_stage_locked(self, stage: str) -> None:
        normalized = str(stage).casefold()
        for key in tuple(self._dedup):
            if key[1].casefold() == normalized:
                self._resolve_episode_locked(key)

    def _correlated_frame_locked(
        self,
        snapshot: object,
        *,
        trace: Mapping[str, object] | None = None,
    ) -> _OpeningFrame | None:
        """Correlate only by a unique frame id plus exact standardized pixels."""

        if snapshot is None:
            return None
        requested_id = str(getattr(snapshot, "evidence_frame_id", "") or "")
        direct = self._frame_by_identity.get(id(snapshot))
        if direct is not None:
            candidate = direct
        elif requested_id and requested_id not in self._ambiguous_frame_ids:
            candidate = self._frame_by_id.get(requested_id)
        else:
            candidate = None
        if candidate is None:
            return None
        if direct is None and requested_id not in {
            candidate.frame_id,
            str(getattr(candidate.snapshot, "evidence_frame_id", "") or ""),
        }:
            return None
        expected_hash = str(
            candidate.frame_metadata.get("standardized_pixel_sha256") or ""
        )
        image = getattr(snapshot, "image", None)
        if not isinstance(image, np.ndarray) or not image.size:
            return None
        actual_hash = _array_sha256(image)
        trace_hash = (
            str(trace.get("input_sha256") or "")
            if isinstance(trace, Mapping)
            else ""
        )
        if expected_hash and actual_hash != expected_hash:
            return None
        if trace_hash and trace_hash != expected_hash:
            return None
        return candidate

    def _can_recover_exact_locked(
        self,
        snapshot: object,
        *,
        trace: Mapping[str, object] | None = None,
    ) -> bool:
        known = self._known_frame_inputs.get(id(snapshot))
        if known is None:
            return False
        dereference, frame_id, pixel_hash, _analysis = known
        if not callable(dereference) or dereference() is not snapshot:
            return False
        if str(getattr(snapshot, "evidence_frame_id", "") or "") != frame_id:
            return False
        image = getattr(snapshot, "image", None)
        if not isinstance(image, np.ndarray) or not image.size:
            return False
        if pixel_hash and _array_sha256(image) != pixel_hash:
            return False
        trace_hash = (
            str(trace.get("input_sha256") or "")
            if isinstance(trace, Mapping)
            else ""
        )
        return not trace_hash or trace_hash == pixel_hash

    def _retain_pending_records_locked(
        self,
        records: tuple[_OpeningFrame, ...],
    ) -> tuple[int, ...]:
        ids: list[int] = []
        for record in records:
            identity = id(record)
            count, byte_size = self._pending_record_refs.get(
                identity,
                (0, record.byte_size),
            )
            self._pending_record_refs[identity] = (count + 1, byte_size)
            ids.append(identity)
        return tuple(ids)

    def _release_pending_records_locked(self, record_ids: tuple[int, ...]) -> None:
        for identity in record_ids:
            value = self._pending_record_refs.get(identity)
            if value is None:
                continue
            count, byte_size = value
            if count > 1:
                self._pending_record_refs[identity] = (count - 1, byte_size)
                continue
            self._pending_record_refs.pop(identity, None)
            if identity not in self._ring_record_ids:
                self._pending_record_bytes = max(
                    0,
                    self._pending_record_bytes - byte_size,
                )
                self._retained_bytes = max(0, self._retained_bytes - byte_size)

    def _sync_episode_occurrences(self, *, deadline: float | None = None) -> None:
        with self._lock:
            occurrences = dict(self._occurrences_by_incident)
        incidents_root = self.root / "incidents"
        for incident_id, count in occurrences.items():
            if deadline is not None and monotonic_seconds() >= deadline:
                return
            path = incidents_root / incident_id / "incident.json"
            if not path.is_file():
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(document, dict):
                    continue
                episode = document.get("episode")
                episode = dict(episode) if isinstance(episode, Mapping) else {}
                if int(episode.get("occurrence_count", 0) or 0) == int(count):
                    continue
                episode["occurrence_count"] = int(count)
                document["episode"] = episode
                # Account for the atomic replacement's temporary copy too.
                remaining = 0.2 if deadline is None else max(0.0, deadline - monotonic_seconds())
                with self._disk_budget.transaction(timeout=min(0.2, remaining)) as allowance:
                    if len(_document_bytes(document)) <= allowance.text_bytes:
                        assert_plain_path(path)
                        atomic_write_json(path, document)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                with self._lock:
                    self._writer_failures += 1

    def _writer_finished(self, future: Future[None]) -> None:
        failed = False
        try:
            future.result()
        except BaseException:
            failed = True
            with self._lock:
                self._writer_failures += 1
        finally:
            with self._lock:
                self._futures.discard(future)
                key = self._future_keys.pop(future, None)
                record_ids = self._future_record_ids.pop(future, ())
                self._release_pending_records_locked(record_ids)
                if failed and key is not None:
                    self._resolve_episode_locked(key)
            try:
                self._writer_slots.release()
            except ValueError:
                pass

    def _write_incident(
        self,
        incident_id: str,
        incident: dict[str, object],
        records: tuple[_OpeningFrame, ...],
        metrics: dict[str, int],
    ) -> None:
        try:
            self._write_incident_with_budget(incident_id, incident, records, metrics)
        except TimeoutError:
            # An incomplete quota scan grants no allowance. Do not repeatedly
            # rescan a huge legacy tree on every bad frame; keep this run's
            # sidecar metadata-only until the next launch/explicit cleanup.
            with self._lock:
                self._media_exhausted = self._text_exhausted = True
                self._suppressed_incidents = min(2**63 - 1, self._suppressed_incidents + 1)
                self._clear_ring_locked()
                self._frame_by_identity.clear()
                self._frame_by_id.clear()
                self._known_frame_inputs.clear()

    def _write_incident_with_budget(
        self,
        incident_id: str,
        incident: dict[str, object],
        records: tuple[_OpeningFrame, ...],
        metrics: dict[str, int],
    ) -> None:
        self._settle_analysis_delivery(records)
        with self._disk_budget.transaction() as allowance:
            with self._lock:
                self._media_exhausted = allowance.media_bytes <= 0
                if self._media_exhausted:
                    self._clear_ring_locked()
                    self._frame_by_identity.clear()
                    self._frame_by_id.clear()
                    self._known_frame_inputs.clear()
                if allowance.incidents_remaining <= 0 or allowance.text_bytes < 2048:
                    self._text_exhausted = True
                    self._clear_ring_locked()
                    self._frame_by_identity.clear()
                    self._frame_by_id.clear()
                    self._known_frame_inputs.clear()
                    self._suppressed_incidents = min(2**63 - 1, self._suppressed_incidents + 1)
                    return
            self._publish_incident(
                incident_id, incident, records, metrics,
                image_limit=min(self.max_persisted_image_bytes, allowance.media_bytes),
                text_limit=min(self.max_incident_text_bytes, allowance.text_bytes),
            )

    def _publish_incident(
        self,
        incident_id: str,
        incident: dict[str, object],
        records: tuple[_OpeningFrame, ...],
        metrics: dict[str, int],
        *,
        image_limit: int,
        text_limit: int,
    ) -> None:
        assert_plain_path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        incidents_root = self.root / "incidents"
        assert_plain_path(incidents_root)
        incidents_root.mkdir(exist_ok=True)
        target = incidents_root / incident_id
        staging = incidents_root / f".{incident_id}.{uuid4().hex}.tmp"
        staging.mkdir()
        try:
            frame_documents: list[dict[str, object]] = []
            trace_lines: list[str] = []
            selected = records[-self.max_persisted_frames :]
            image_bytes = 0
            shared_updates: dict[tuple[str, str], dict[str, object]] = {}
            for record in selected:
                document = {
                    "frame_id": record.frame_id,
                    "seq": record.seq,
                    "monotonic_ms": record.monotonic_ms,
                    "wall_time": record.wall_time,
                    "capture": record.frame_metadata,
                    "analysis": _json_safe(record.analysis),
                    "anchor_score": record.anchor_score,
                    "recognition": record.recognition,
                }
                if record.recognition_trace is not None:
                    correlated_trace = dict(record.recognition_trace)
                    correlated_trace.update(
                        {
                            "frame_seq": record.seq,
                            "monotonic_ms": record.monotonic_ms,
                            "wall_time": record.wall_time,
                        }
                    )
                    trace_lines.append(
                        json.dumps(
                            correlated_trace,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                media_key = (
                    str(record.frame_metadata.get("standardized_pixel_sha256", "")),
                    str(record.frame_metadata.get("raw_pixel_sha256", "")),
                )
                shared = shared_updates.get(media_key) or self._shared_media.get(media_key)
                if shared is not None:
                    bytes_written, artifacts = self._link_shared_images(
                        staging, record, shared, incident_id=incident_id,
                        remaining=max(0, image_limit - image_bytes),
                    )
                    document["media_status"] = "hardlinked" if artifacts else "omitted_budget"
                    document["shared_pixel_owner"] = {
                        "incident_id": shared["incident_id"],
                        "frame_id": shared["frame_id"],
                    }
                else:
                    bytes_written, artifacts = self._write_record_images(
                        staging, record, remaining=max(0, image_limit - image_bytes),
                    )
                    document["media_status"] = "stored" if artifacts else "omitted_budget"
                    if artifacts:
                        shared_updates[media_key] = {
                            "incident_id": incident_id,
                            "frame_id": record.frame_id,
                            "reason": "identical_capture_pixels",
                            "artifacts": artifacts,
                        }
                image_bytes += bytes_written
                document["artifacts"] = artifacts
                frame_documents.append(document)
            evidence = {
                "schema": OPENING_EVIDENCE_SCHEMA,
                "incident_id": incident_id,
                "profile": self.profile_name,
                "buffer": {
                    "max_age_ms": self.max_age_ms,
                    "max_bytes": self.max_bytes,
                    "metrics_at_trigger": metrics,
                    "record_count": len(frame_documents),
                },
                "privacy": {
                    "contains_sensitive_images": image_bytes > 0,
                    "support_export_requires_explicit_image_opt_in": True,
                },
                "resource_identity": self._resource_identity(),
                "storage": {
                    "media_bytes": image_bytes,
                    "media_limit": image_limit,
                    "frame_limit": self.max_persisted_frames,
                    "existing_evidence_deleted": False,
                    "policy": "refuse_new_when_full",
                },
                "frames": frame_documents,
            }
            repro = {
                    "schema": "guandan.repro-manifest/1",
                    "incident_id": incident_id,
                    "field": incident.get("field"),
                    "symptom_code": incident.get("code"),
                    "expected_truth": None,
                    "truth_status": "not_provided",
                    "resource_identity": evidence["resource_identity"],
                    "frame_sequence": [
                        {
                            "frame_id": item.get("frame_id"),
                            "seq": item["seq"],
                            "monotonic_ms": item["monotonic_ms"],
                            "artifacts": item.get("artifacts", []),
                            "media_status": item.get("media_status"),
                            "media_reference": item.get("media_reference"),
                        }
                        for item in frame_documents
                    ],
                }
            latest = {
                    "schema": "guandan.opening-latest/1",
                    "incident_id": incident_id,
                    "relative_directory": f"incidents/{incident_id}",
                    "code": incident.get("code"),
                }
            documents = {
                "incident.json": incident,
                "opening_evidence.json": evidence,
                "repro.json": repro,
            }
            encoded = {name: _document_bytes(value) for name, value in documents.items()}
            traces = ("\n".join(trace_lines) + ("\n" if trace_lines else "")).encode("utf-8")
            latest_bytes = _document_bytes(latest)
            if sum(map(len, encoded.values())) + len(traces) + len(latest_bytes) > text_limit:
                # Retain frame correlation and explicit truncation rather than
                # letting a large template trace bypass the text quota.
                traces = b""
                for document in frame_documents:
                    document["recognition"] = {
                        key: value for key, value in dict(document.get("recognition") or {}).items()
                        if key in {"hand_count", "round_level", "lead_player", "current_player"}
                    }
                evidence["resource_identity"] = {"status": "omitted_text_budget"}
                repro["resource_identity"] = evidence["resource_identity"]
                evidence["storage"]["text_status"] = "trace_and_details_omitted_budget"
                incident["evidence"] = {"status": "omitted_text_budget"}
                encoded = {name: _document_bytes(value) for name, value in documents.items()}
            if sum(map(len, encoded.values())) + len(traces) + len(latest_bytes) > text_limit:
                _remove_tree_best_effort(staging)
                with self._lock:
                    self._text_exhausted = True
                    self._clear_ring_locked()
                    self._frame_by_identity.clear()
                    self._frame_by_id.clear()
                    self._known_frame_inputs.clear()
                    self._suppressed_incidents = min(2**63 - 1, self._suppressed_incidents + 1)
                return
            for name, content in encoded.items():
                (staging / name).write_bytes(content)
            (staging / "recognition_trace.jsonl").write_bytes(traces)
            staging.replace(target)
            # This file is a bounded pointer, not an append-only journal.
            atomic_write_json(self.root / "latest.json", latest)
            self._shared_media.update(shared_updates)
            with self._lock:
                self._incidents_written += 1
        except BaseException:
            _remove_tree_best_effort(staging)
            raise

    def _settle_analysis_delivery(
        self,
        records: tuple[_OpeningFrame, ...],
    ) -> None:
        deadline = monotonic_seconds() + self.delivery_settle_seconds
        while monotonic_seconds() < deadline:
            with self._lock:
                pending = any(
                    record.analysis.get("status")
                    in {"submitted", "started", "completed"}
                    for record in records
                )
            if not pending:
                return
            sleep(0.005)

    def _write_record_images(
        self,
        staging: Path,
        record: _OpeningFrame,
        *,
        remaining: int,
    ) -> tuple[int, list[dict[str, object]]]:
        if remaining <= 0:
            return 0, []
        standard = getattr(record.snapshot, "image", None)
        captured = getattr(record.snapshot, "frame", None)
        raw = getattr(captured, "raw_image", None)
        written = 0
        artifacts: list[dict[str, object]] = []
        images: tuple[tuple[str, str, object], ...] = (
            ("raw_client", f"frames/raw_client_{record.seq:06d}.png", raw),
            ("standardized", f"frames/standardized_{record.seq:06d}.png", standard),
        )
        for kind, relative, image in images:
            if not isinstance(image, np.ndarray) or image.size == 0:
                continue
            saved = _write_png_bounded(
                staging / relative,
                image,
                remaining=max(0, remaining - written),
            )
            if saved is None:
                continue
            size, file_sha256 = saved
            written += size
            artifacts.append(
                {
                    "frame_id": record.frame_id,
                    "frame_seq": record.seq,
                    "kind": kind,
                    "field": None,
                    "path": relative.replace("\\", "/"),
                    "bytes": size,
                    "sha256": file_sha256,
                    "pixel_sha256": _array_sha256(image),
                }
            )
        if isinstance(standard, np.ndarray) and standard.size:
            for name, box, crop in self._opening_rois(standard):
                relative = Path("roi") / f"{name}_{record.seq:06d}.png"
                saved = _write_png_bounded(
                    staging / relative,
                    crop,
                    remaining=max(0, remaining - written),
                )
                if saved is None:
                    continue
                size, file_sha256 = saved
                written += size
                artifacts.append(
                    {
                        "frame_id": record.frame_id,
                        "frame_seq": record.seq,
                        "kind": "roi",
                        "field": name,
                        "path": relative.as_posix(),
                        "bytes": size,
                        "sha256": file_sha256,
                        "pixel_sha256": _array_sha256(crop),
                        "box": list(box),
                    }
                )
        return written, artifacts

    def _link_shared_images(
        self,
        staging: Path,
        record: _OpeningFrame,
        shared: Mapping[str, object],
        *,
        incident_id: str,
        remaining: int,
    ) -> tuple[int, list[dict[str, object]]]:
        """Share physical PNG storage while each incident stays self-contained.

        Budgets deliberately count hardlink sizes conservatively as logical
        bytes. They can stop admission early, never allow more disk usage than
        the configured limit. Unsupported filesystems get fresh bounded PNGs.
        """
        source_root = staging if shared.get("incident_id") == incident_id else (
            self.root / "incidents" / str(shared["incident_id"])
        )
        written = 0
        artifacts: list[dict[str, object]] = []
        for original in shared.get("artifacts", []):
            original = dict(original)
            size = int(original.get("bytes", 0))
            if size <= 0 or size > remaining - written:
                continue
            relative = Path(str(original["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            source = source_root / relative
            name = str(original.get("field") or original["kind"])
            output = Path("roi" if original["kind"] == "roi" else "frames") / f"{name}_{record.seq:06d}.png"
            target = staging / output
            try:
                assert_plain_path(source)
                if source.stat().st_size != size or hashlib.sha256(source.read_bytes()).hexdigest() != original["sha256"]:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
            except OSError:
                continue
            linked = dict(original)
            linked.update(frame_id=record.frame_id, frame_seq=record.seq, path=output.as_posix())
            artifacts.append(linked)
            written += size
        if not artifacts and remaining > 0:
            return self._write_record_images(staging, record, remaining=remaining)
        return written, artifacts

    def _opening_rois(
        self,
        image: np.ndarray,
    ) -> Iterable[tuple[str, tuple[int, int, int, int], np.ndarray]]:
        if self.profiles_root is None:
            return ()
        try:
            from .annotation_service import AnnotationService

            service = AnnotationService(self.profiles_root, self.profile_name)
            allowed = {
                "level_rank",
                "my_hand",
                "first_play_self",
                "first_play_left",
                "first_play_opposite",
                "first_play_right",
            }
            crops: list[tuple[str, tuple[int, int, int, int], np.ndarray]] = []
            for region in service.list_regions():
                if region.name not in allowed:
                    continue
                box = service._box_for_image(region, image)
                if not box.fits_within((image.shape[1], image.shape[0])):
                    continue
                crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
                if crop.size:
                    crops.append(
                        (
                            region.name,
                            (box.x, box.y, box.w, box.h),
                            crop,
                        )
                    )
            return tuple(crops)
        except BaseException:
            return ()

    def _resource_identity(self) -> dict[str, object]:
        if self.profiles_root is None:
            return {"status": "unavailable", "files": []}
        return recognition_resource_identity(self.profiles_root, self.profile_name)


class NonBlockingOpeningEvidenceSink:
    """Serialize observer calls on a bounded daemon queue.

    This protects the capture and recognition workers even when a test or a
    future plugin injects a slow/throwing evidence sink.  Queue saturation is
    observable but always fail-open.
    """

    def __init__(self, target: object, *, queue_size: int = 8) -> None:
        if queue_size <= 0:
            raise ValueError("evidence dispatch queue size must be positive")
        self.target = target
        self._queue: Queue[tuple[str, tuple[object, ...], dict[str, object]] | None] = Queue(
            maxsize=int(queue_size)
        )
        self._lock = RLock()
        self._worker: Thread | None = None
        self._closed = False
        self.dropped_calls = 0
        self.failed_calls = 0
        self.completed_calls = 0

    def begin(self, **kwargs: object) -> None:
        self._submit("begin", (), kwargs)

    def observe_frame(self, snapshot: object, **kwargs: object) -> None:
        self._submit("observe_frame", (snapshot,), kwargs)

    def observe_analysis_submitted(self, snapshot: object) -> None:
        self._submit("observe_analysis_submitted", (snapshot,), {})

    def observe_analysis_started(self, snapshot: object) -> None:
        self._submit("observe_analysis_started", (snapshot,), {})

    def observe_analysis_dropped(self, snapshot: object, *, reason: str) -> None:
        self._submit(
            "observe_analysis_dropped",
            (snapshot,),
            {"reason": str(reason)},
        )

    def observe_delivery(self, snapshot: object, *, gate_eligible: bool) -> None:
        self._submit(
            "observe_delivery",
            (snapshot,),
            {"gate_eligible": bool(gate_eligible)},
        )

    def observe_anchor(self, snapshot: object, score: float, **kwargs: object) -> None:
        self._submit("observe_anchor", (snapshot, score), kwargs)

    def observe_page(self, page: str, **kwargs: object) -> None:
        self._submit("observe_page", (page,), kwargs)

    def observe_recognition(
        self,
        snapshot: object,
        result: object,
        trace: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        self._submit("observe_recognition", (snapshot, result, trace), kwargs)

    def observe_failure(self, error: object, **kwargs: object) -> None:
        self._submit("observe_failure", (error,), kwargs)

    def observe_geometry_recovery(self, **kwargs: object) -> None:
        self._submit("observe_geometry_recovery", (), kwargs)

    def mark_session_started(self) -> None:
        self._submit("mark_session_started", (), {})

    def flush(self, timeout: float = 10.0) -> bool:
        deadline = monotonic_seconds() + max(0.0, float(timeout))
        while self._queue.unfinished_tasks and monotonic_seconds() < deadline:
            sleep(0.005)
        if self._queue.unfinished_tasks:
            return False
        flush = getattr(self.target, "flush", None)
        try:
            return bool(flush(max(0.0, deadline - monotonic_seconds()))) if callable(flush) else True
        except BaseException:
            self.failed_calls += 1
            return False

    def close(self, timeout: float = 5.0) -> None:
        deadline = monotonic_seconds() + min(0.25, max(0.0, float(timeout)))
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.flush(max(0.0, deadline - monotonic_seconds()))
        worker = self._worker
        if worker is not None and worker.is_alive():
            try:
                self._queue.put_nowait(None)
            except Full:
                pass
            if worker is not current_thread():
                worker.join(max(0.0, deadline - monotonic_seconds()))
        close = getattr(self.target, "close", None)
        try:
            if callable(close):
                close(timeout=max(0.0, deadline - monotonic_seconds()))
        except TypeError:
            try:
                close()
            except BaseException:
                self.failed_calls += 1
        except BaseException:
            self.failed_calls += 1

    def _submit(
        self,
        method: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        try:
            with self._lock:
                if self._closed:
                    self.dropped_calls += 1
                    return
                self._ensure_worker()
                self._queue.put_nowait((method, args, dict(kwargs)))
        except Full:
            with self._lock:
                self.dropped_calls += 1
        except BaseException:
            with self._lock:
                self.failed_calls += 1

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = Thread(
                target=self._run,
                name="opening-evidence-dispatch",
                daemon=True,
            )
            self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=2.0)
            except Empty:
                with self._lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            try:
                if item is None:
                    return
                method, args, kwargs = item
                operation = getattr(self.target, method, None)
                if callable(operation):
                    operation(*args, **kwargs)
                with self._lock:
                    self.completed_calls += 1
            except BaseException:
                with self._lock:
                    self.failed_calls += 1
            finally:
                self._queue.task_done()


class NullOpeningEvidenceSink:
    """No-op fallback used only when diagnostics cannot be constructed."""

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_args, **_kwargs: None


def build_opening_evidence_monitor(**kwargs: object) -> object:
    """Construct the concrete monitor without making application startup depend on it."""

    try:
        return OpeningEvidenceMonitor(**kwargs)
    except BaseException:
        return NullOpeningEvidenceSink()


def classify_opening_failure(error: object) -> str:
    machine_code = str(getattr(error, "code", "")).strip().upper()
    typed = {
        "WINDOW-NOT-FOUND": OPENING_WINDOW_NOT_FOUND,
        "WINDOW-AMBIGUOUS": OPENING_WINDOW_NOT_FOUND,
        "WINDOW-MINIMIZED": OPENING_WINDOW_MINIMIZED,
        "GEOMETRY-CHANGED": OPENING_GEOMETRY_CHANGED,
        "CAPTURE-BLACK-FRAME": OPENING_CAPTURE_BLACK_FRAME,
        "CAPTURE-OCCLUDED": OPENING_CAPTURE_ERROR,
        "CAPTURE-BACKEND-FAILED": OPENING_CAPTURE_ERROR,
        "RESOURCE-MISMATCH": OPENING_RESOURCE_ERROR,
    }
    if machine_code in typed:
        return typed[machine_code]
    text = str(error).casefold()
    if any(token in text for token in ("没有找到", "not found", "no matching window")):
        return OPENING_WINDOW_NOT_FOUND
    if any(token in text for token in ("最小化", "minimized", "iconic")):
        return OPENING_WINDOW_MINIMIZED
    if any(token in text for token in ("geometry changed", "尺寸", "client rect")):
        return OPENING_GEOMETRY_CHANGED
    if any(token in text for token in ("全黑", "black frame", "empty image")):
        return OPENING_CAPTURE_BLACK_FRAME
    if any(token in text for token in ("template", "模板", "profile", "resource", "模型")):
        return OPENING_RESOURCE_ERROR
    if any(token in text for token in ("timeout", "超时")):
        return OPENING_TIMEOUT
    return OPENING_CAPTURE_ERROR


def _initial_analysis(now: int) -> dict[str, object]:
    return {
        "status": "captured",
        "captured_observer_ms": now,
        "submitted_ms": None,
        "started_ms": None,
        "completed_ms": None,
        "delivered_ms": None,
        "dropped_ms": None,
        "drop_reason": None,
        "gate_delivered": False,
    }


def _frame_capture_details(
    snapshot: object,
) -> tuple[int, dict[str, object], bool]:
    standard = getattr(snapshot, "image", None)
    captured = getattr(snapshot, "frame", None)
    raw = getattr(captured, "raw_image", None)
    byte_size = _unique_array_bytes(raw, standard)
    metadata = _frame_metadata(snapshot)
    if isinstance(standard, np.ndarray) and standard.size:
        metadata.update(
            {
                "pixel_max": int(np.max(standard)),
                "pixel_mean": float(np.mean(standard)),
                "standardized_pixel_sha256": _array_sha256(standard),
            }
        )
    if isinstance(raw, np.ndarray) and raw.size:
        metadata["raw_pixel_sha256"] = _array_sha256(raw)
    return byte_size, metadata, _is_black_frame(standard)


def _frame_metadata(snapshot: object) -> dict[str, object]:
    captured = getattr(snapshot, "frame", None)
    standardization = getattr(captured, "standardization", None)
    rect = getattr(captured, "rect", None)
    source_viewport = getattr(standardization, "source_viewport", None)
    content_box = getattr(standardization, "content_box", None)
    standard_image = getattr(snapshot, "image", None)
    source_viewport_values = _box_like(source_viewport)
    content_box_values = _box_like(content_box)
    interpolation = None
    if source_viewport_values is not None and content_box_values is not None:
        interpolation = (
            "INTER_AREA"
            if source_viewport_values[2] > content_box_values[2]
            or source_viewport_values[3] > content_box_values[3]
            else "INTER_LINEAR"
        )
    return {
        "backend": str(getattr(captured, "backend", "unknown")),
        "captured_monotonic_ms": int(
            getattr(snapshot, "captured_monotonic_ms", 0) or 0
        ),
        "dpi": int(getattr(captured, "dpi", 0) or 0),
        "window_title": str(getattr(captured, "window_title", ""))[:256],
        "client_rect": _box_like(rect, client=True),
        "standardization": {
            "source_size": list(getattr(standardization, "source_size", ()) or ()),
            "source_viewport": source_viewport_values,
            "content_box": content_box_values,
            "scale": float(getattr(standardization, "scale", 0.0) or 0.0),
            "padding": list(getattr(standardization, "padding", ()) or ()),
            "standardized_size": (
                [int(standard_image.shape[1]), int(standard_image.shape[0])]
                if isinstance(standard_image, np.ndarray) and standard_image.ndim >= 2
                else []
            ),
            "interpolation": interpolation,
            "border_value": 0,
            "aspect_error": float(getattr(standardization, "aspect_error", 0.0) or 0.0),
            "aspect_compatible": bool(getattr(standardization, "aspect_compatible", False)),
        },
    }


def _box_like(value: object, *, client: bool = False) -> list[int] | None:
    if value is None:
        return None
    fields = ("left", "top", "width", "height") if client else ("x", "y", "w", "h")
    try:
        return [int(getattr(value, name)) for name in fields]
    except (AttributeError, TypeError, ValueError):
        return None


def _recognition_document(result: object) -> dict[str, object]:
    return {
        "round_level": getattr(result, "round_level", None),
        "wild_rank": getattr(result, "wild_rank", None),
        "my_hand": [str(card) for card in tuple(getattr(result, "my_hand", ()) or ())],
        "hand_count": len(tuple(getattr(result, "my_hand", ()) or ())),
        "lead_player": getattr(result, "lead_player", None),
        "current_player": getattr(result, "current_player", None),
        "field_confidences": _json_safe(dict(getattr(result, "field_confidences", {}) or {})),
        "sources": _json_safe(dict(getattr(result, "sources", {}) or {})),
        "unresolved_fields": list(getattr(result, "unresolved_fields", ()) or ()),
        "diagnostics": list(getattr(result, "diagnostics", ()) or ()),
        "elapsed_ms": float(getattr(result, "elapsed_ms", 0.0) or 0.0),
    }


def _unique_array_bytes(*values: object) -> int:
    seen: set[int] = set()
    total = 0
    for value in values:
        if isinstance(value, np.ndarray) and id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
    return total


def _is_black_frame(image: object) -> bool:
    return bool(
        isinstance(image, np.ndarray)
        and image.size > 0
        and float(np.max(image)) <= 1.0
    )


def _cards_hash(cards: Iterable[str]) -> str:
    payload = "\0".join(str(card) for card in cards).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest()


def _write_png_bounded(
    path: Path,
    image: np.ndarray,
    *,
    remaining: int,
) -> tuple[int, str] | None:
    """Encode first, then enforce the hard on-disk byte budget before publish."""

    if remaining <= 0:
        return None
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"diagnostic PNG encoding failed: {path.name}")
    content = bytes(encoded)
    if len(content) > int(remaining):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return len(content), hashlib.sha256(content).hexdigest()


def _wall_time(snapshot: object) -> str:
    value = getattr(snapshot, "captured_at", None)
    if isinstance(value, datetime):
        return value.isoformat(timespec="milliseconds")
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _json_safe(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False))


def _document_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n").encode("utf-8")


def _number(value: object, *, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _remove_tree_best_effort(root: Path) -> None:
    try:
        root = Path(os.path.abspath(root))
        if root.parent.name != "incidents" or not root.name.startswith(".OPEN-") or not root.name.endswith(".tmp"):
            return
        if not root.exists():
            return
        assert_plain_path(root)
        directories = [root]
        files: list[Path] = []
        for parent in directories:
            for path in parent.iterdir():
                if is_reparse(path) or not path.is_relative_to(root):
                    return
                if path.is_dir():
                    directories.append(path)
                elif path.is_file():
                    files.append(path)
        for path in files:
            assert_plain_path(path)
            path.unlink(missing_ok=True)
        for path in reversed(directories):
            assert_plain_path(path)
            path.rmdir()
    except OSError:
        return


__all__ = [
    "OPENING_EVIDENCE_SCHEMA",
    "OPENING_INCIDENT_SCHEMA",
    "OPENING_GEOMETRY_RECOVERY",
    "OpeningEvidenceMetrics",
    "OpeningEvidenceMonitor",
    "NonBlockingOpeningEvidenceSink",
    "NullOpeningEvidenceSink",
    "build_opening_evidence_monitor",
    "classify_opening_failure",
]
