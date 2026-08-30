from __future__ import annotations

"""Fail-open, pre-session evidence collection for opening-state failures.

The live controller intentionally owns this sidecar instead of the reducer.  A
diagnostic failure must never change a recognition decision, delay a capture,
or manufacture a game event.  Frames are retained by reference in a bounded
memory ring and are serialized only after an incident has been queued.
"""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from threading import BoundedSemaphore, RLock
from threading import Thread
from time import monotonic_ns
from time import monotonic as monotonic_seconds
from time import sleep
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4
from queue import Empty, Full, Queue

import cv2
import numpy as np

from .danzero.state import RANKS
from .image_io import save_image_unicode
from .runtime_identity import get_runtime_identity
from .startup_diagnostics import current_startup_diagnostics
from .storage import atomic_write_json


OPENING_EVIDENCE_SCHEMA = "guandan.opening-evidence/1"
OPENING_INCIDENT_SCHEMA = "guandan.opening-incident/1"
RECOGNITION_TRACE_SCHEMA = "guandan.recognition-trace/1"

OPENING_WINDOW_NOT_FOUND = "OPENING-WINDOW-NOT-FOUND"
OPENING_WINDOW_MINIMIZED = "OPENING-WINDOW-MINIMIZED"
OPENING_GEOMETRY_CHANGED = "OPENING-GEOMETRY-CHANGED"
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
    seq: int
    monotonic_ms: int
    wall_time: str
    snapshot: object
    byte_size: int
    frame_metadata: dict[str, object]
    recognition: dict[str, object] | None = None
    recognition_trace: dict[str, object] | None = None
    anchor_score: float | None = None


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
        }


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
        max_bytes: int = 256 * 1024 * 1024,
        field_timeout_seconds: float = 8.0,
        writer_queue_size: int = 16,
        max_persisted_frames: int = 40,
        max_persisted_image_bytes: int = 256 * 1024 * 1024,
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
        self._clock_ms = clock_ms or (lambda: monotonic_ns() // 1_000_000)
        self._ring: deque[_OpeningFrame] = deque()
        self._frame_by_identity: dict[int, _OpeningFrame] = {}
        self._retained_bytes = 0
        self._latest_seq = 0
        self._dropped_age = 0
        self._dropped_budget = 0
        self._dropped_writer_queue = 0
        self._incidents_queued = 0
        self._incidents_written = 0
        self._writer_failures = 0
        self._dedup: set[tuple[str, str]] = set()
        self._future_keys: dict[Future[None], tuple[str, str]] = {}
        self._levels: deque[str] = deque(maxlen=3)
        self._hands: deque[tuple[str, ...]] = deque(maxlen=3)
        self._anchor_ready = False
        self._listener_started_ms: int | None = None
        self._armed_ms: int | None = None
        self._field_blocked_since: dict[str, int] = {}
        self._lock = RLock()
        self._writer_slots = BoundedSemaphore(int(writer_queue_size))
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future[None]] = set()

    def begin(self, *, monotonic_ms: int | None = None) -> None:
        """Start a fresh listener interval without touching captured evidence."""

        try:
            with self._lock:
                self._listener_started_ms = int(monotonic_ms if monotonic_ms is not None else self._clock_ms())
                self._armed_ms = None
                self._field_blocked_since.clear()
                self._dedup.clear()
                self._levels.clear()
                self._hands.clear()
                self._anchor_ready = False
                self._ring.clear()
                self._frame_by_identity.clear()
                self._retained_bytes = 0
        except BaseException:
            return

    def observe_frame(self, snapshot: object, *, monotonic_ms: int | None = None) -> None:
        """Append one immutable capture reference in O(1), never encode it here."""

        try:
            now = int(monotonic_ms if monotonic_ms is not None else self._clock_ms())
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
            black = _is_black_frame(standard)
            with self._lock:
                self._latest_seq += 1
                item = _OpeningFrame(
                    seq=self._latest_seq,
                    monotonic_ms=now,
                    wall_time=_wall_time(snapshot),
                    snapshot=snapshot,
                    byte_size=byte_size,
                    frame_metadata=metadata,
                )
                self._ring.append(item)
                self._frame_by_identity[id(snapshot)] = item
                self._retained_bytes += byte_size
                self._trim_locked(now)
            if black:
                self.emit_incident(
                    OPENING_CAPTURE_BLACK_FRAME,
                    field="capture",
                    reason="captured standardized client frame is effectively black",
                    monotonic_ms=now,
                    evidence={"pixel_max": int(np.max(standard)) if isinstance(standard, np.ndarray) and standard.size else 0},
                )
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
                item = self._frame_by_identity.get(id(snapshot))
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
            hand = tuple(str(card) for card in document.get("my_hand", []))
            now = self._clock_ms()
            with self._lock:
                item = self._frame_by_identity.get(id(snapshot))
                if item is None and self._ring:
                    item = self._ring[-1]
                if item is not None:
                    item.recognition = document
                    item.recognition_trace = dict(trace) if isinstance(trace, Mapping) else None
                if level in RANKS:
                    self._levels.append(level)
                if len(hand) == 27:
                    self._hands.append(hand)
                if self._armed_ms is None and (level in RANKS or bool(hand)):
                    self._armed_ms = now
                    if not self._anchor_ready:
                        self._field_blocked_since.setdefault("table_anchor", now)
                self._set_field_state_locked("round_level", level in RANKS, now)
                self._set_field_state_locked("hand_count", len(hand) == 27, now)
                hand_stable = len(self._hands) >= 2 and len(set(self._hands)) == 1
                self._set_field_state_locked(
                    "hand_stability",
                    len(hand) == 27 and hand_stable,
                    now,
                )
                if len(hand) == 27 and hand_stable:
                    self._clear_field_dedup_locked("my_hand")
                level_conflict = len(set(self._levels)) > 1
                hand_unstable = len(self._hands) >= 2 and len(set(self._hands)) > 1
                if level in RANKS and len(hand) == 27 and opening_seed_valid is not None:
                    self._set_field_state_locked("lead_player", bool(opening_seed_valid), now)
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
    ) -> str:
        """Classify a window/capture/resource failure and queue it once."""

        code = classify_opening_failure(error)
        try:
            self.emit_incident(
                code,
                field=str(stage),
                reason=str(error),
                monotonic_ms=monotonic_ms,
                evidence={"error_type": type(error).__name__},
            )
        except BaseException:
            pass
        return code

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
        try:
            now = int(monotonic_ms if monotonic_ms is not None else self._clock_ms())
            normalized_code = str(code).strip().upper()
            normalized_field = str(field).strip().lower() or "unknown"
            key = (normalized_code, normalized_field)
            with self._lock:
                if key in self._dedup:
                    return False
                self._dedup.add(key)
                records = tuple(self._ring)
                metrics = self._metrics_locked().to_dict()
            if not self._writer_slots.acquire(blocking=False):
                with self._lock:
                    self._dropped_writer_queue += 1
                    self._dedup.discard(key)
                return False
            slot_acquired = True
            incident_id = f"OPEN-{now}-{uuid4().hex[:8]}"
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
                "runtime_identity": get_runtime_identity(),
            }
            with self._lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="opening-evidence",
                    )
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
                future.add_done_callback(self._writer_finished)
            return True
        except BaseException:
            if slot_acquired:
                try:
                    self._writer_slots.release()
                except (ValueError, RuntimeError):
                    pass
            return False

    def flush(self, timeout: float = 10.0) -> bool:
        try:
            with self._lock:
                futures = tuple(self._futures)
            deadline = max(0.0, float(timeout))
            for future in futures:
                future.result(timeout=deadline)
            with self._lock:
                return self._writer_failures == 0
        except BaseException:
            return False

    def close(self, timeout: float = 5.0) -> None:
        self.flush(timeout)
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)

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
        )

    def _trim_locked(self, now_ms: int) -> None:
        while self._ring and now_ms - self._ring[0].monotonic_ms > self.max_age_ms:
            self._drop_left_locked(age=True)
        while self._ring and self._retained_bytes > self.max_bytes:
            self._drop_left_locked(age=False)

    def _drop_left_locked(self, *, age: bool) -> None:
        item = self._ring.popleft()
        self._frame_by_identity.pop(id(item.snapshot), None)
        self._retained_bytes = max(0, self._retained_bytes - item.byte_size)
        if age:
            self._dropped_age += 1
        else:
            self._dropped_budget += 1

    def _emit_timeouts_if_due(self, *, result: object | None = None) -> None:
        now = self._clock_ms()
        with self._lock:
            armed = self._armed_ms
            blocked = dict(self._field_blocked_since)
        if armed is None:
            return
        if now - blocked.get("table_anchor", now) >= self.field_timeout_ms:
            self.emit_incident(
                OPENING_ANCHOR_TIMEOUT,
                field="table_anchor",
                reason="table anchor did not meet the readiness score before timeout",
                monotonic_ms=now,
            )
        if now - blocked.get("round_level", now) >= self.field_timeout_ms:
            self.emit_incident(
                OPENING_LEVEL_MISSING,
                field="round_level",
                reason="round level remained unresolved until opening timeout",
                monotonic_ms=now,
            )
        if now - blocked.get("hand_count", now) >= self.field_timeout_ms:
            hand = tuple(getattr(result, "my_hand", ()) or ()) if result is not None else ()
            self.emit_incident(
                OPENING_HAND_COUNT_MISMATCH,
                field="my_hand",
                reason="opening hand count did not equal 27 before timeout",
                monotonic_ms=now,
                evidence={"actual_count": len(hand), "required_count": 27},
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
        self._dedup = {
            key for key in self._dedup if key[1].casefold() != normalized
        }

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
                if failed and key is not None:
                    self._dedup.discard(key)
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
        self.root.mkdir(parents=True, exist_ok=True)
        incidents_root = self.root / "incidents"
        incidents_root.mkdir(exist_ok=True)
        target = incidents_root / incident_id
        staging = incidents_root / f".{incident_id}.{uuid4().hex}.tmp"
        staging.mkdir()
        try:
            frame_documents: list[dict[str, object]] = []
            trace_lines: list[str] = []
            selected = records[-self.max_persisted_frames :]
            image_bytes = 0
            for record in selected:
                document = {
                    "seq": record.seq,
                    "monotonic_ms": record.monotonic_ms,
                    "wall_time": record.wall_time,
                    "capture": record.frame_metadata,
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
                bytes_written, artifacts = self._write_record_images(
                    staging,
                    record,
                    remaining=max(0, self.max_persisted_image_bytes - image_bytes),
                )
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
                "frames": frame_documents,
            }
            atomic_write_json(staging / "incident.json", incident)
            atomic_write_json(staging / "opening_evidence.json", evidence)
            atomic_write_json(
                staging / "repro.json",
                {
                    "schema": "guandan.repro-manifest/1",
                    "incident_id": incident_id,
                    "field": incident.get("field"),
                    "symptom_code": incident.get("code"),
                    "expected_truth": None,
                    "truth_status": "not_provided",
                    "resource_identity": evidence["resource_identity"],
                    "frame_sequence": [
                        {
                            "seq": item["seq"],
                            "monotonic_ms": item["monotonic_ms"],
                            "artifacts": item.get("artifacts", []),
                        }
                        for item in frame_documents
                    ],
                },
            )
            (staging / "recognition_trace.jsonl").write_text(
                "\n".join(trace_lines) + ("\n" if trace_lines else ""),
                encoding="utf-8",
            )
            staging.replace(target)
            atomic_write_json(
                self.root / "latest.json",
                {
                    "schema": "guandan.opening-latest/1",
                    "incident_id": incident_id,
                    "relative_directory": f"incidents/{incident_id}",
                    "code": incident.get("code"),
                },
            )
            with self._lock:
                self._incidents_written += 1
        except BaseException:
            _remove_tree_best_effort(staging)
            raise

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
            estimated = int(image.nbytes)
            if estimated + written > remaining:
                continue
            save_image_unicode(staging / relative, image)
            path = staging / relative
            size = path.stat().st_size
            written += size
            artifacts.append(
                {
                    "frame_seq": record.seq,
                    "kind": kind,
                    "field": None,
                    "path": relative.replace("\\", "/"),
                    "bytes": size,
                    "sha256": _sha256_file(path),
                    "pixel_sha256": _array_sha256(image),
                }
            )
        if isinstance(standard, np.ndarray) and standard.size:
            for name, crop in self._opening_rois(standard):
                relative = Path("roi") / f"{name}_{record.seq:06d}.png"
                estimated = int(crop.nbytes)
                if estimated + written > remaining:
                    break
                save_image_unicode(staging / relative, crop)
                path = staging / relative
                size = path.stat().st_size
                written += size
                artifacts.append(
                    {
                        "frame_seq": record.seq,
                        "kind": "roi",
                        "field": name,
                        "path": relative.as_posix(),
                        "bytes": size,
                        "sha256": _sha256_file(path),
                        "pixel_sha256": _array_sha256(crop),
                    }
                )
        return written, artifacts

    def _opening_rois(self, image: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
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
            crops: list[tuple[str, np.ndarray]] = []
            for region in service.list_regions():
                if region.name not in allowed:
                    continue
                box = service._box_for_image(region, image)
                if not box.fits_within((image.shape[1], image.shape[0])):
                    continue
                crop = image[box.y : box.y + box.h, box.x : box.x + box.w]
                if crop.size:
                    crops.append((region.name, crop))
            return tuple(crops)
        except BaseException:
            return ()

    def _resource_identity(self) -> dict[str, object]:
        if self.profiles_root is None:
            return {"status": "unavailable", "files": []}
        root = self.profiles_root / self.profile_name
        try:
            selected = [
                root / "profile.json",
                root / "regions_config.json",
                root / "templates_config.json",
            ]
            selected.extend(sorted((root / "templates").rglob("*")))
            records = [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in selected
                if path.is_file()
            ]
            aggregate = hashlib.sha256(
                json.dumps(
                    records,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return {"status": "identified", "sha256": aggregate, "files": records}
        except BaseException as exc:
            return {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "files": [],
            }


class NonBlockingOpeningEvidenceSink:
    """Serialize observer calls on a bounded daemon queue.

    This protects the capture and recognition workers even when a test or a
    future plugin injects a slow/throwing evidence sink.  Queue saturation is
    observable but always fail-open.
    """

    def __init__(self, target: object, *, queue_size: int = 128) -> None:
        if queue_size <= 0:
            raise ValueError("evidence dispatch queue size must be positive")
        self.target = target
        self._queue: Queue[tuple[str, tuple[object, ...], dict[str, object]] | None] = Queue(
            maxsize=int(queue_size)
        )
        self._lock = RLock()
        self._worker: Thread | None = None
        self.dropped_calls = 0
        self.failed_calls = 0
        self.completed_calls = 0

    def begin(self, **kwargs: object) -> None:
        self._submit("begin", (), kwargs)

    def observe_frame(self, snapshot: object, **kwargs: object) -> None:
        self._submit("observe_frame", (snapshot,), kwargs)

    def observe_anchor(self, snapshot: object, score: float, **kwargs: object) -> None:
        self._submit("observe_anchor", (snapshot, score), kwargs)

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
        self.flush(timeout)
        close = getattr(self.target, "close", None)
        try:
            if callable(close):
                close(timeout=max(0.0, float(timeout)))
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


def _frame_metadata(snapshot: object) -> dict[str, object]:
    captured = getattr(snapshot, "frame", None)
    standardization = getattr(captured, "standardization", None)
    rect = getattr(captured, "rect", None)
    source_viewport = getattr(standardization, "source_viewport", None)
    content_box = getattr(standardization, "content_box", None)
    return {
        "backend": str(getattr(captured, "backend", "unknown")),
        "dpi": int(getattr(captured, "dpi", 0) or 0),
        "window_title": str(getattr(captured, "window_title", ""))[:256],
        "client_rect": _box_like(rect, client=True),
        "standardization": {
            "source_size": list(getattr(standardization, "source_size", ()) or ()),
            "source_viewport": _box_like(source_viewport),
            "content_box": _box_like(content_box),
            "scale": float(getattr(standardization, "scale", 0.0) or 0.0),
            "padding": list(getattr(standardization, "padding", ()) or ()),
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wall_time(snapshot: object) -> str:
    value = getattr(snapshot, "captured_at", None)
    if isinstance(value, datetime):
        return value.isoformat(timespec="milliseconds")
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _json_safe(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False))


def _remove_tree_best_effort(root: Path) -> None:
    try:
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        root.rmdir()
    except OSError:
        return


__all__ = [
    "OPENING_EVIDENCE_SCHEMA",
    "OPENING_INCIDENT_SCHEMA",
    "OpeningEvidenceMetrics",
    "OpeningEvidenceMonitor",
    "NonBlockingOpeningEvidenceSink",
    "NullOpeningEvidenceSink",
    "build_opening_evidence_monitor",
    "classify_opening_failure",
]
