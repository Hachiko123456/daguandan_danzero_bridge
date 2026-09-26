from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from time import monotonic_ns, perf_counter
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot

from ..application.listener_evidence import (
    EvidenceWriteHandle,
    ListenerEvidence,
    ListenerFrame as _DiagnosticFrameSaveContext,
)
from ..application.ports import (
    AdvicePort,
    CapturePort,
    LiveRuntimePort,
    RecognitionPort,
    SessionFactoryPort,
)
from ..advisor_strategy import (
    build_advisor,
    load_profile_advisor_strategy,
    load_profile_automatic_log_include_media,
    load_profile_recording_max_total_bytes,
    load_profile_recording_mode,
    normalize_advisor_strategy,
    recording_storage_summary as get_recording_storage_summary,
    save_profile_advisor_strategy,
    save_profile_automatic_log_include_media,
    save_profile_recording_max_total_bytes,
    save_profile_recording_mode,
)
from ..capture_service import FrameSnapshot
from ..dependencies import preload_live_worker_dependencies
from ..danzero.state import GuanDanState, RANKS
from ..domain.live_runtime import AdviceRequestKey, LiveAdvice, LiveUpdate
from ..live.frame_pipeline import analyze_frame_envelope
from ..live.latest_worker import LatestOnlyWorker
from ..live.pipeline_timing import PipelineTiming
from ..session_paths import resolve_sessions_root
from ..opening_evidence import (
    NonBlockingOpeningEvidenceSink,
    build_opening_evidence_monitor,
)
from ..application.opening_readiness import (
    OpeningReadinessReport,
    listening_report,
    report_for_error,
    report_for_phase,
)
from ..opening_gate import (
    OpeningActionSeed as _OpeningActionSeed,
    OpeningSessionSeed as _AutoSessionSeed,
    build_opening_seed,
    evaluate_opening_gate,
    OpeningTracker,
    ListeningPageSignal,
    OpeningSessionSeed,
    opening_semantic_key,
)
from ..infrastructure.win32_hand_preselector import Win32HandPreselector
from .hand_preselection import HandPreselectionPlanner, PreselectionResult
from .recording_cadence import RecordingCadenceGate
from .recording_dispatcher import (
    BoundedRecordingDispatcher,
    RecordingDropAccountingAdapter,
    RecordingFrame,
)
from .workers import OneShotThread, WorkerHandle


@dataclass(frozen=True)
class _PreselectionTask:
    request_id: str
    key: AdviceRequestKey
    advice_cards: tuple[str, ...]
    expected_hand: tuple[str, ...]
    frame: FrameSnapshot
    capture_generation: int


@dataclass(frozen=True)
class _LiveRunToken:
    orchestrator: LiveRuntimePort
    session_id: str
    nonce: int
    generation: int
    pipeline_timing: PipelineTiming = field(default_factory=PipelineTiming, compare=False)


@dataclass(frozen=True)
class _AnalysisFrameTask:
    token: _LiveRunToken
    snapshot: FrameSnapshot
    capture_seq: int
    captured_ms: int
    capture_started_ns: int = 0
    submitted_ns: int = 0
    evidence: _DiagnosticFrameSaveContext | None = None


@dataclass(frozen=True)
class _LiveFrameDelivery:
    """Internal capture-worker result carrying the sequence beside the frame."""

    snapshot: FrameSnapshot
    capture_seq: int


@dataclass(frozen=True)
class _AnalysisDelivery:
    token: _LiveRunToken
    update: LiveUpdate | None
    emitted_ns: int
    capture_started_ns: int = 0


@dataclass(frozen=True)
class _FatalWorkerFault:
    token: _LiveRunToken
    kind: str
    message: str


@dataclass(frozen=True)
class _WaitingAnalysisTask:
    snapshot: FrameSnapshot
    generation: int
    page: ListeningPageSignal | None = None
    evidence: _DiagnosticFrameSaveContext | None = None


@dataclass(frozen=True)
class _WaitingFrameDelivery:
    snapshot: FrameSnapshot
    generation: int
    capture_seq: int


class _DiagnosticFrameCaptureFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "CAPTURE-BACKEND-FAILED",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


@dataclass(frozen=True)
class _WaitingRecognitionEnvelope:
    snapshot: FrameSnapshot
    generation: int
    trace: object | None
    opening_seed_valid: bool | None
    page: ListeningPageSignal | None = None
    evidence: _DiagnosticFrameSaveContext | None = None


@dataclass(frozen=True)
class _GeometryRecoveryAttemptResult:
    generation: int
    attempt_count: int
    source: object | None
    snapshots: tuple[FrameSnapshot, ...]
    details: dict[str, object]
    error: Exception | None = None


class _WaitingRecognitionFailure(RuntimeError):
    """Carry the exact input snapshot without erasing the original exception."""

    def __init__(
        self,
        error: Exception,
        snapshot: FrameSnapshot,
        *,
        generation: int | None = None,
        evidence: _DiagnosticFrameSaveContext | None = None,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.snapshot = snapshot
        self.generation = generation
        self.evidence = evidence


class LiveAssistantController(QObject):
    """Qt signal adapter around the UI-independent live orchestrator."""

    initial_recognized = Signal(object, object)
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    session_finished = Signal(object)
    danzero_warmup_status = Signal(str)
    listening_status = Signal(object)
    preselection_result = Signal(object)
    log_delivery_status = Signal(object)
    recording_status = Signal(object)
    live_fault = Signal(object)
    diagnostic_frame_status = Signal(object)
    _diagnostic_save_completed = Signal(object)
    _waiting_recognized = Signal(object, object)
    _waiting_capture_stopped = Signal(object)
    _analysis_delivered = Signal()
    _fatal_worker_fault_requested = Signal(object)
    _TABLE_ANCHOR_READY_SCORE = 0.85
    _GEOMETRY_RECOVERABLE_CODES = frozenset(
        {"GEOMETRY-CHANGED", "WINDOW-MINIMIZED"}
    )
    _GEOMETRY_RECOVERY_DELAYS_MS = (0, 250, 500, 1_000)
    _GEOMETRY_STABLE_SAMPLE_DELAY_SEC = 0.15
    _GEOMETRY_ANALYSIS_DRAIN_TIMEOUT_SEC = 5.0
    _GEOMETRY_RECOVERY_MAX_CYCLES = 3
    _GEOMETRY_RECOVERY_RESET_AFTER_FRAMES = 5
    # A cheap page probe may be uncertain briefly.  This is a bounded
    # observation-quality recovery window, not a recognition threshold and it
    # must never manufacture an action event.
    _PAGE_UNKNOWN_MAX_RETRIES = 5
    _PAGE_UNKNOWN_RECOVERY_WINDOW_MS = 5_000
    _PAGE_UNKNOWN_SLOW_INTERVAL_SEC = 1.0

    def __init__(
        self,
        capture_service: CapturePort | None = None,
        *,
        profile_name: str = "tencent_daguandan",
        recognition_service: RecognitionPort | None = None,
        advisor: AdvicePort | None = None,
        session_factory: SessionFactoryPort | None = None,
        capture_interval_sec: float = 0.1,
        deduplicate_analysis_frames: bool = False,
        opening_evidence_monitor: object | None = None,
    ) -> None:
        super().__init__()
        if (
            capture_service is None
            or recognition_service is None
            or advisor is None
            or session_factory is None
        ):
            # Compatibility for direct construction. Production startup passes
            # the already assembled graph from ``bootstrap``.
            from ..bootstrap import build_live_controller_dependencies

            assembled = build_live_controller_dependencies(
                profile_name=profile_name,
                capture=capture_service,
                advisor=advisor,
            )
            capture_service = capture_service or assembled.capture
            recognition_service = recognition_service or assembled.recognizer
            advisor = advisor or assembled.advisor
            session_factory = session_factory or assembled.session_factory
        self.capture_service = capture_service
        self.profile_name = profile_name
        self.recognition_service = recognition_service
        # Keep one agent process-wide for this controller: DanZero resets its
        # per-hand cache before every recommendation, so it is safe to reuse
        # while avoiding a 10+ second model load on every new game.
        self.danzero_advisor = advisor
        self.advisor_strategy = (
            "fabledan"
            if type(advisor).__name__ == "FableDanAdvisor"
            else load_profile_advisor_strategy(
                self.capture_service.profiles_root,
                self.profile_name,
            )
        )
        self.session_factory = session_factory
        if capture_interval_sec <= 0:
            raise ValueError("capture_interval_sec must be positive")
        self.capture_interval_sec = float(capture_interval_sec)
        self.deduplicate_analysis_frames = bool(deduplicate_analysis_frames)
        evidence_target = opening_evidence_monitor or build_opening_evidence_monitor(
            profiles_root=self.capture_service.profiles_root,
            profile_name=self.profile_name,
        )
        self.opening_evidence = (
            evidence_target
            if isinstance(evidence_target, NonBlockingOpeningEvidenceSink)
            else NonBlockingOpeningEvidenceSink(evidence_target)
        )
        self.recording_mode = load_profile_recording_mode(
            self.capture_service.profiles_root,
            self.profile_name,
        )
        self.session_data_recording_enabled = self.recording_mode != "none"
        self.recording_max_total_bytes = load_profile_recording_max_total_bytes(
            self.capture_service.profiles_root, self.profile_name
        )
        self.automatic_log_include_media = load_profile_automatic_log_include_media(
            self.capture_service.profiles_root, self.profile_name
        )
        self.orchestrator: LiveRuntimePort | None = None
        self._live_source = None
        self._capture_worker: WorkerHandle | None = None
        self._analysis_worker: LatestOnlyWorker | None = None
        self._recording_dispatcher: BoundedRecordingDispatcher | None = None
        self._recording_cadence: RecordingCadenceGate | None = None
        self._initial_thread: OneShotThread | None = None
        self._danzero_warmup_thread: OneShotThread | None = None
        self._danzero_warmup_running = False
        self._danzero_warmup_complete = False
        self._finish_thread: OneShotThread | None = None
        self._log_export_thread: OneShotThread | None = None
        self._log_open_thread: OneShotThread | None = None
        self._diagnostic_frame_thread: EvidenceWriteHandle | None = None
        self._diagnostics_closed = False
        self._last_diagnostic_frame_directory: Path | None = None
        self._diagnostic_directory_lock = Lock()
        self._retained_diagnostic_frame: _DiagnosticFrameSaveContext | None = None
        self._diagnostic_run_id = uuid4().hex
        self._diagnostic_no_frame_failure = False
        self._diagnostic_runtime_evidence = None
        self._last_no_frame_fault: tuple[object, ...] | None = None
        self._listener_stop_requested = False
        self._listener_evidence = ListenerEvidence(
            self._persist_prepared_live_frame, self._diagnostic_write_completed,
        )
        self._diagnostic_save_completed.connect(
            self._diagnostic_frame_saved, Qt.ConnectionType.QueuedConnection,
        )
        self._manual_diagnostic_directory: Path | None = None
        self._manual_diagnostic_session_id = ""
        self._manual_capture_seq = 0
        self._manual_diagnostic_lock = Lock()
        self._last_log_delivery_result: dict[str, object] | None = None
        self._deferred_source_close = None
        self._capture_generation = 0
        self._live_session_nonce = 0
        self._active_live_token: _LiveRunToken | None = None
        self._gui_delivery_lock = Lock()
        self._pending_gui_delivery: _AnalysisDelivery | None = None
        self._gui_delivery_scheduled = False
        self._gui_accepted_version: tuple[_LiveRunToken, int, int, int] | None = None
        # The controller is the GUI-facing authority for recommendation
        # lifetime.  Preselection may prepare input, but it never owns the
        # recommendation currently visible to either UI.
        self._authoritative_advice_key: AdviceRequestKey | None = None
        self._authoritative_advice_generation = 0
        self._queued_fault_identity: tuple[str, int] | None = None
        self._fatal_capture_session_id: str | None = None
        self._resume_requested = False
        self._listening_enabled = False
        self._waiting_source = None
        self._waiting_capture_worker: WorkerHandle | None = None
        self._waiting_analysis_worker: LatestOnlyWorker | None = None
        self._draining_waiting_analysis_worker: LatestOnlyWorker | None = None
        self._waiting_generation = 0
        self._latest_waiting_frame: FrameSnapshot | None = None
        self._latest_waiting_frame_generation = -1
        self._latest_waiting_frame_capture_seq = -1
        self._waiting_capture_seq_counter = 0
        self._latest_waiting_frame_lock = Lock()
        self._preopening_diagnostic_directory: Path | None = None
        self._preopening_diagnostic_session_id = ""
        self._waiting_candidate: _AutoSessionSeed | None = None
        self._opening_tracker = OpeningTracker()
        self._listening_page = ListeningPageSignal("unknown", 0.0)
        self._last_stable_page_stage = "unknown"
        self._page_unknown_retry_count = 0
        self._page_unknown_started_ms: int | None = None
        self._page_unknown_last_reason = ""
        self._page_recovery_terminated = False
        self._page_recovery_slow = False
        self._last_opening_readiness: OpeningReadinessReport = listening_report()
        self._pending_auto_session: _AutoSessionSeed | None = None
        self._listener_recording: object | None = None
        self._listener_recording_stop_reason: str | None = None
        self._table_anchor_observed = False
        self._geometry_recovery_active = False
        self._geometry_recovery_attempt_count = 0
        self._geometry_recovery_cycle_count = 0
        self._geometry_post_recovery_frames_remaining = 0
        self._geometry_recovery_error_details: dict[str, object] = {}
        self._geometry_recovery_thread: OneShotThread | None = None
        self._geometry_recovery_cancel = Event()
        self._geometry_recovery_result: _GeometryRecoveryAttemptResult | None = None
        self._recovered_waiting_source: object | None = None
        self._preserve_opening_evidence_once = False
        self._geometry_recovery_timer = QTimer(self)
        self._geometry_recovery_timer.setSingleShot(True)
        self._geometry_recovery_timer.timeout.connect(
            self._start_geometry_recovery_attempt
        )
        self._recognition_strategy = "two_valid_streak"
        self._auto_finish_requested = False
        # Preselection is deliberately a GUI/infrastructure sidecar.  It has
        # no reducer access and receives only a completed, visible advice plus
        # the most recent immutable capture frame.
        self._latest_live_frame: FrameSnapshot | None = None
        self._latest_live_frame_generation = 0
        self._latest_live_frame_capture_seq = -1
        self._latest_live_frame_lock = Lock()
        self._preselection_thread: OneShotThread | None = None
        self._active_preselection_task: _PreselectionTask | None = None
        self._pending_preselection_task: _PreselectionTask | None = None
        self._handled_preselection_request_ids: set[str] = set()
        self._hand_preselector: object | None = None
        self._waiting_recognized.connect(self._consume_waiting_recognition)
        self._waiting_capture_stopped.connect(self._waiting_capture_finished)
        self.update_ready.connect(self._auto_finish_on_game_end)
        self.update_ready.connect(self._schedule_hand_preselection)
        self._analysis_delivered.connect(
            self._receive_queued_delivery, Qt.ConnectionType.QueuedConnection
        )
        self._fatal_worker_fault_requested.connect(
            self._deliver_fatal_worker_fault, Qt.ConnectionType.QueuedConnection
        )

    @property
    def is_running(self) -> bool:
        return bool(self._capture_worker and self._capture_worker.is_running)

    def _manual_diagnostic_capture_allowed(self) -> bool:
        return (not self._diagnostics_closed and not self._diagnostic_no_frame_failure
                and not self._listening_enabled and self.orchestrator is None
                and self._waiting_capture_worker is None and not self.is_running)

    def save_latest_live_frame_to_session(self) -> dict[str, object]:
        """Explicit synchronous API; GUI callers use the background variant."""
        context = self._prepare_cached_live_frame_save()
        if context is None and not self._manual_diagnostic_capture_allowed():
            raise self._cached_frame_unavailable_error()
        def operation():
            if context is None and not self._manual_diagnostic_capture_allowed():
                raise self._cached_frame_unavailable_error()
            prepared = context if context is not None else self._capture_manual_frame_context()
            return self._persist_prepared_live_frame(prepared)
        handle = self._listener_evidence.writer.submit(operation)
        if handle is None:
            raise RuntimeError("诊断截图写入队列已满或正在关闭")
        if not handle.wait(10_000):
            raise RuntimeError("诊断截图仍在后台保存，请稍后查看诊断目录")
        if handle.error is not None:
            raise handle.error
        return handle.result

    def save_latest_live_frame_to_session_background(self) -> bool:
        """Freeze the exact eligible frame and enqueue on the single writer."""
        current = self._diagnostic_frame_thread
        if self._diagnostics_closed or current is not None and current.isRunning():
            return False
        try:
            context = self._prepare_cached_live_frame_save()
            if context is None and not self._manual_diagnostic_capture_allowed():
                raise self._cached_frame_unavailable_error()
        except Exception as exc:
            self.diagnostic_frame_status.emit(self._diagnostic_failure_payload(exc))
            return False

        def operation():
            try:
                # Recheck idle status: a queued manual job cannot quietly open
                # another source after the user starts listening.
                if context is None and not self._manual_diagnostic_capture_allowed():
                    raise self._cached_frame_unavailable_error()
                prepared = context if context is not None else self._capture_manual_frame_context()
                return self._persist_prepared_live_frame(prepared)
            except Exception as exc:
                payload = self._diagnostic_failure_payload(exc)
                if context is not None:
                    payload.update(source=context.source, source_phase=context.source_phase)
                return payload

        handle = self._listener_evidence.writer.submit(operation, self._diagnostic_write_completed)
        if handle is None:
            self.diagnostic_frame_status.emit({
                "status": "FAILURE", "error_code": "DIAGNOSTIC-QUEUE-FULL",
                "message": "诊断截图队列已满或正在关闭，请稍后重试",
            })
            return False
        self._diagnostic_frame_thread = handle
        self.diagnostic_frame_status.emit({
            "status": "SAVING", "source": context.source if context else "manual_window_capture",
            "source_phase": context.source_phase if context else "manual_window_capture",
            "session_directory": str(context.session_directory) if context else "",
        })
        return True

    def _diagnostic_write_completed(self, value: object, error: Exception | None) -> None:
        if self._diagnostics_closed:
            return
        payload = dict(value) if isinstance(value, dict) else {}
        if error is not None:
            payload.update(status="FAILURE", source="live_listener_frame",
                           error_code="DIAGNOSTIC-SAVE-FAILED", message=str(error))
        from ..startup_diagnostics import initialized_startup_diagnostics, record_startup_event
        if initialized_startup_diagnostics() is not None:
            record_startup_event("listener_diagnostic_capture", {
                key: payload.get(key) for key in (
                    "status", "automatic", "reason", "no_frame", "error_code", "failure_code",
                    "message", "incident_manifest", "image_path", "capture_generation",
                    "capture_seq", "evidence_frame_id", "source", "source_phase",
                )
            })
        try:
            self._diagnostic_save_completed.emit(payload)
        except RuntimeError:
            # QObject destruction never turns a disk failure into a new incident.
            pass

    def _remember_diagnostic_directory(self, value: dict[str, object]) -> None:
        if value.get("status", "SUCCESS") != "SUCCESS" or not value.get("image_path"):
            return
        directory = Path(str(value["image_path"])).absolute().parent
        with self._diagnostic_directory_lock:
            self._last_diagnostic_frame_directory = directory

    def _diagnostic_frame_saved(self, value: object) -> None:
        payload = dict(value) if isinstance(value, dict) else {}
        payload.setdefault("status", "SUCCESS")
        self._remember_diagnostic_directory(payload)
        self.diagnostic_frame_status.emit(payload)

    def diagnostic_frame_directory(self) -> Path | None:
        """Resolve only known, existing directories; never allocate one here."""
        with self._diagnostic_directory_lock:
            last = self._last_diagnostic_frame_directory
        store = getattr(self.orchestrator, "store", None)
        candidates = [last]
        for directory in (getattr(store, "directory", None),
                          self._preopening_diagnostic_directory,
                          self._manual_diagnostic_directory):
            if directory is not None:
                candidates.append(Path(directory).absolute() / "diagnostic_frames")
        for directory in candidates:
            try:
                if directory is not None and directory.is_dir():
                    return directory
            except OSError:
                continue
        return None

    def _diagnostic_runtime_context(self) -> dict[str, object]:
        """Bind a listener run once; never scan templates/models on each frame."""
        cached = self._diagnostic_runtime_evidence
        if cached is not None:
            return dict(cached)
        from ..config import RUNTIME_LAYOUT
        from ..startup_diagnostics import initialized_startup_diagnostics
        root = Path(self.capture_service.profiles_root).resolve() / self.profile_name
        state = initialized_startup_diagnostics()
        context: dict[str, object] = {
            "build_id": RUNTIME_LAYOUT.build_id,
            "profile_name": self.profile_name, "profile_root": str(root),
            "fingerprint_scope": "config_at_listener_start_not_template_integrity_check",
        }
        if state is not None:
            context["run_id"] = state.run_id
            context["startup_report"] = (str(state.run_directory / "startup_report.json")
                                         if state.run_directory else None)
        digests = {}
        for name in ("profile.json", "regions_config.json", "templates_config.json"):
            try:
                path = root / name
                if path.stat().st_size > 1024 * 1024:
                    digests[name] = "over_size_limit"
                else:
                    digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                digests[name] = "unavailable"
        context["configuration_sha256"] = digests
        self._diagnostic_runtime_evidence = context
        return dict(context)

    def _diagnostic_scope_id(self, *, token: _LiveRunToken | None = None) -> str:
        session = token.session_id if token is not None else self._preopening_diagnostic_session_id
        return f"{self._diagnostic_run_id}:{session or 'unbound_listener'}"

    def _bind_diagnostic_context(self, context: _DiagnosticFrameSaveContext) -> _DiagnosticFrameSaveContext:
        if context.session_directory is None or not context.session_id:
            directory, session_id = self._ensure_manual_diagnostic_directory()
            context = replace(context, session_directory=directory, session_id=session_id)
        # Ordinary frames remain migratable once the single writer is idle.
        # Only accepted incidents pin their directory: their manifests must not
        # lose the exact paths while asynchronous evidence is being written.
        return replace(context, details=deepcopy(context.details))

    def _retain_listener_snapshot(
        self, snapshot: FrameSnapshot, generation: int, capture_seq: int, *,
        token: _LiveRunToken | None = None,
    ) -> _DiagnosticFrameSaveContext | None:
        """Called on capture, before any probe, submit, or Qt delivery can mutate it."""
        phase = "live_session" if token is not None else "preopening_listener"
        lock = self._latest_live_frame_lock if token is not None else self._latest_waiting_frame_lock
        with lock:
            if token is not None:
                if not self._live_token_is_current(token):
                    return None
                store = getattr(token.orchestrator, "store", None)
                directory = getattr(store, "directory", None)
                session_id = str(getattr(store, "session_id", "") or "")
                prefix = "_latest_live_frame"
            else:
                if not self._listening_enabled or generation != self._waiting_generation:
                    return None
                directory = self._preopening_diagnostic_directory
                session_id = self._preopening_diagnostic_session_id
                prefix = "_latest_waiting_frame"
            scope_id = self._diagnostic_scope_id(token=token)
            previous = self._listener_evidence.find(snapshot, generation, phase, scope_id=scope_id, capture_seq=capture_seq)
            if previous is not None and previous.capture_seq == capture_seq:
                return previous
            # Delayed UI frames must never replace the newer capture-side cache,
            # or recopy an old buffer which analysis may already have modified.
            if (getattr(self, prefix + "_generation") == generation
                    and getattr(self, prefix + "_capture_seq") >= capture_seq
                    and capture_seq >= 0):
                return None
            copied = self._copy_live_snapshot(snapshot)
            context = _DiagnosticFrameSaveContext(
                Path(directory) if directory is not None else None, session_id,
                copied, generation, capture_seq, phase,
                details={"geometry": self._snapshot_geometry(copied), "listener_phase": phase,
                         "runtime": self._diagnostic_runtime_context()},
                scope_id=scope_id,
            )
            context = self._listener_evidence.remember(context)
            setattr(self, prefix, context.snapshot)
            setattr(self, prefix + "_generation", generation)
            setattr(self, prefix + "_capture_seq", capture_seq)
            return context

    def _retain_stopped_listener_frame(self) -> None:
        try:
            context = self._prepare_cached_live_frame_save()
            if context is not None:
                phase = (context.source_phase if context.source_phase in {"failed_listener_frame", "last_listener_frame"}
                         else "listener_stopped")
                self._retained_diagnostic_frame = replace(context, source_phase=phase)
        except Exception:
            pass

    @staticmethod
    def _page_evidence(page: ListeningPageSignal) -> dict[str, object]:
        return {name: getattr(page, name, None) for name in (
            "stage", "anchor_score", "buttons", "table_anchor_1_score",
            "table_anchor_2_score", "game_logo_anchor_score",
        )}

    def _queue_listener_incident(
        self, reason: str, *, context: _DiagnosticFrameSaveContext | None = None,
        error: object = None, page: ListeningPageSignal | None = None,
        allow_latest: bool = True,
    ) -> None:
        """Freeze and enqueue before invalidation; never capture a substitute."""
        if self._diagnostics_closed:
            return
        try:
            explicit_failure_frame = context is not None
            if context is None and allow_latest:
                context = self._prepare_cached_live_frame_save()
            if context is None:
                token = self._active_live_token
                generation = token.generation if token is not None else self._waiting_generation
                identity = (self._diagnostic_run_id, generation, reason)
                if identity != self._last_no_frame_fault:
                    self._last_no_frame_fault = identity
                    self._diagnostic_no_frame_failure = True
                    self._diagnostic_write_completed({
                        **self._diagnostic_failure_payload(self._cached_frame_unavailable_error()),
                        "automatic": True, "reason": reason, "no_frame": True,
                        "capture_generation": generation, "capture_seq": None,
                        "evidence_frame_id": "", "failure_code": getattr(error, "code", ""),
                        "details": deepcopy(getattr(error, "details", {})),
                    }, None)
                return
            if context.source != "live_listener_frame":
                return
            token = self._active_live_token
            phase = context.capture_scope[0]
            if phase == "live_session":
                if (token is None or not self._live_token_is_current(token)
                        or context.capture_generation != token.generation
                        or context.capture_scope[2] != self._diagnostic_scope_id(token=token)):
                    return
            elif (not self._listening_enabled or self.orchestrator is not None
                  or context.capture_generation != self._waiting_generation
                  or context.capture_scope[2] != self._diagnostic_scope_id()):
                return
            context = self._bind_diagnostic_context(context)
            if page is not None:
                context = self._listener_evidence.annotate(context, page=self._page_evidence(page))
            role = "failure" if explicit_failure_frame else "context"
            self._retained_diagnostic_frame = replace(
                context, source_phase="failed_listener_frame" if explicit_failure_frame else "last_listener_frame",
            )
            self._listener_evidence.incident(reason, context, failure_role=role, details={
                "error": str(error or ""), "code": getattr(error, "code", ""),
                "capture_details": getattr(error, "details", {}),
                "has_failure_frame": explicit_failure_frame,
                "runtime": self._diagnostic_runtime_context(),
            })
        except Exception as exc:
            self._diagnostic_write_completed(None, exc)

    def _cached_diagnostic_context(
        self, snapshot: FrameSnapshot, generation: int, capture_seq: int,
        phase: str, directory: Path | None, session_id: str, *,
        token: _LiveRunToken | None = None,
    ) -> _DiagnosticFrameSaveContext:
        scope_id = self._diagnostic_scope_id(token=token)
        context = self._listener_evidence.find(snapshot, generation, phase, scope_id=scope_id, capture_seq=capture_seq)
        if context is None:
            # Compatibility for callers which explicitly prime the legacy cache.
            # Production captures always take the ring path, retaining annotations.
            context = _DiagnosticFrameSaveContext(
                directory, session_id, self._copy_live_snapshot(snapshot), generation,
                capture_seq, phase, details={
                    "geometry": self._snapshot_geometry(snapshot), "listener_phase": phase,
                }, scope_id=scope_id,
            )
        context = self._bind_diagnostic_context(replace(
            context, session_directory=directory, session_id=session_id,
        ))
        retained = self._retained_diagnostic_frame
        if (retained is not None and retained.source_phase in {"failed_listener_frame", "last_listener_frame"}
                and retained.identity == context.identity):
            return replace(context, source_phase=retained.source_phase)
        return context

    def _prepare_cached_live_frame_save(self) -> _DiagnosticFrameSaveContext | None:
        """Select a current capture, preserving ring provenance, or stopped evidence."""
        token = self._active_live_token
        orchestrator = self.orchestrator
        if token is not None and orchestrator is not None and self._live_token_is_current(token):
            store = getattr(orchestrator, "store", None)
            directory = getattr(store, "directory", None)
            session_id = str(getattr(store, "session_id", "") or "").strip()
            with self._latest_live_frame_lock:
                snapshot = self._latest_live_frame
                generation = self._latest_live_frame_generation
                capture_seq = self._latest_live_frame_capture_seq
            if (directory is not None and session_id == token.session_id and session_id
                    and snapshot is not None and generation == token.generation
                    and generation == self._capture_generation and capture_seq >= 0):
                return self._cached_diagnostic_context(
                    snapshot, generation, capture_seq, "live_session",
                    Path(directory), session_id, token=token,
                )
            # Never use a waiting/old-session frame for a formal current claim.
            return None

        if self._listening_enabled and orchestrator is None:
            with self._latest_waiting_frame_lock:
                snapshot = self._latest_waiting_frame
                generation = self._latest_waiting_frame_generation
                capture_seq = self._latest_waiting_frame_capture_seq
            if snapshot is not None and generation == self._waiting_generation and capture_seq >= 0:
                return self._cached_diagnostic_context(
                    snapshot, generation, capture_seq, "preopening_listener",
                    self._preopening_diagnostic_directory, self._preopening_diagnostic_session_id,
                )
            return None
        if token is None:
            retained = self._retained_diagnostic_frame
            if retained is not None:
                return self._bind_diagnostic_context(retained)
        return None

    def _cached_frame_unavailable_error(self) -> RuntimeError:
        return _DiagnosticFrameCaptureFailure(
            "当前监听代次尚无可保存帧，请等待下一帧后重试；若采集已失败，请先恢复窗口或重新连接",
            code="LISTENER-NO-FRAME",
        )

    def _ensure_manual_diagnostic_directory(self) -> tuple[Path, str]:
        """Allocate a manual fallback directory only when a frame is saved."""

        with self._manual_diagnostic_lock:
            if self._manual_diagnostic_directory is None:
                sessions_root = resolve_sessions_root(
                    self.capture_service.profiles_root,
                    self.profile_name,
                )
                stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
                diagnostic_id = f"diagnostic_{stamp}_{uuid4().hex[:8]}"
                self._manual_diagnostic_directory = (
                    Path(sessions_root) / "manual_diagnostic" / diagnostic_id
                )
                self._manual_diagnostic_session_id = diagnostic_id
                self._manual_capture_seq = 0
            return (
                Path(self._manual_diagnostic_directory),
                self._manual_diagnostic_session_id,
            )

    def _capture_manual_frame_context(self) -> _DiagnosticFrameSaveContext:
        """Capture one manual frame through CaptureService.open_live_source().capture()."""

        source = None
        try:
            source = self.capture_service.open_live_source(self.profile_name)
            snapshot = source.capture()
        except Exception as exc:
            raise self._manual_capture_failure(exc) from exc
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass

        if not isinstance(snapshot, FrameSnapshot) or getattr(snapshot, "image", None) is None:
            raise _DiagnosticFrameCaptureFailure(
                "截图失败：捕获源没有返回有效图片",
                code="CAPTURE-EMPTY",
            )
        image = snapshot.image
        if getattr(image, "size", 0) <= 0:
            raise _DiagnosticFrameCaptureFailure(
                "截图失败：捕获结果为空",
                code="CAPTURE-EMPTY",
            )

        directory, session_id = self._ensure_manual_diagnostic_directory()
        with self._manual_diagnostic_lock:
            self._manual_capture_seq += 1
            capture_seq = self._manual_capture_seq

        return _DiagnosticFrameSaveContext(
            Path(directory),
            session_id,
            self._copy_live_snapshot(snapshot),
            int(self._capture_generation),
            capture_seq,
            "manual_window_capture",
            "manual_window_capture",
        )

    @staticmethod
    def _manual_capture_failure(exc: Exception) -> _DiagnosticFrameCaptureFailure:
        code = str(getattr(exc, "code", "CAPTURE-BACKEND-FAILED") or "CAPTURE-BACKEND-FAILED")
        details = getattr(exc, "details", {})
        if not isinstance(details, dict):
            details = {}
        messages = {
            "WINDOW-NOT-FOUND": "截图失败：没有找到可捕获的大掼蛋窗口，请先打开程序并确保窗口可见",
            "WINDOW-MINIMIZED": "截图失败：大掼蛋窗口处于最小化状态，无法可靠截图",
            "WINDOW-AMBIGUOUS": "截图失败：找到多个可能的大掼蛋窗口，请收窄窗口匹配配置",
            "CAPTURE-OCCLUDED": "截图失败：大掼蛋窗口被其他窗口遮挡，请移开遮挡后重试",
            "GEOMETRY-CHANGED": "截图失败：大掼蛋窗口位置、大小或 DPI 发生变化，请保持窗口尺寸稳定后重试",
            "CAPTURE-UNSUPPORTED": "截图失败：当前环境不支持大掼蛋窗口捕获",
        }
        message = messages.get(code) or str(exc) or "截图失败：窗口捕获失败"
        return _DiagnosticFrameCaptureFailure(message, code=code, details=details)

    @classmethod
    def _diagnostic_failure_payload(cls, exc: Exception) -> dict[str, object]:
        failure = exc if isinstance(exc, _DiagnosticFrameCaptureFailure) else cls._manual_capture_failure(exc)
        listener = failure.code == "LISTENER-NO-FRAME"
        return {
            "status": "FAILURE",
            "source": "live_listener_frame" if listener else "manual_window_capture",
            "source_phase": "listener_no_frame" if listener else "manual_window_capture",
            "error_code": failure.code,
            "details": dict(failure.details),
            "message": str(failure),
        }

    def _persist_prepared_live_frame(
        self,
        context: _DiagnosticFrameSaveContext,
    ) -> dict[str, object]:
        from ..application.session_diagnostic_frames import SessionDiagnosticFrameStore

        persisted = SessionDiagnosticFrameStore().save_snapshot(
            context.session_directory,
            context.snapshot,
            session_id=context.session_id,
            capture_generation=context.capture_generation,
            capture_seq=context.capture_seq,
            source=context.source,
            source_phase=context.source_phase,
            diagnostic_context=context.details,
        )
        result = self._diagnostic_frame_store_result(persisted)
        store = SessionDiagnosticFrameStore()
        if "count" not in result:
            try:
                result["count"] = len(store.list_frames(context.session_directory))
            except Exception:
                result["count"] = result.get("sequence", 0)
        required = ("sequence", "image_path", "metadata_path", "count")
        missing = [key for key in required if key not in result]
        if missing:
            raise RuntimeError(
                "诊断截图存储服务返回字段不完整：" + ", ".join(missing)
            )
        result.setdefault("session_id", context.session_id)
        result.setdefault("session_directory", str(context.session_directory))
        result.setdefault("source", context.source)
        result.setdefault("source_phase", context.source_phase)
        result.setdefault("evidence_frame_id", context.snapshot.evidence_frame_id)
        result.setdefault("capture_seq", context.capture_seq)
        result.setdefault("capture_generation", context.capture_generation)
        result.setdefault("captured_at", context.snapshot.captured_at.isoformat())
        result.setdefault("captured_monotonic_ms", context.snapshot.captured_monotonic_ms)
        result["frame_age_ms"] = max(
            0, monotonic_ns() // 1_000_000 - context.snapshot.captured_monotonic_ms,
        )
        result.setdefault(
            "raw_sha256",
            hashlib.sha256(context.snapshot.image.tobytes(order="C")).hexdigest(),
        )
        result.setdefault(
            "message",
            "手动窗口截图已保存"
            if context.source == "manual_window_capture"
            else "实时监听截图已保存到当前对局"
            if context.source_phase == "live_session"
            else "实时监听截图已保存到本轮预开局诊断目录",
        )
        self._remember_diagnostic_directory(result)
        return result

    @staticmethod
    def _copy_live_snapshot(snapshot: FrameSnapshot) -> FrameSnapshot:
        """Copy mutable image buffers while retaining frame identity metadata."""

        image = snapshot.image.copy()
        raw_image = getattr(snapshot.frame, "raw_image", None)
        raw_image_copy = raw_image.copy() if hasattr(raw_image, "copy") else raw_image
        standardization = replace(snapshot.frame.standardization, image=image)
        frame = replace(
            snapshot.frame,
            standardization=standardization,
            raw_image=raw_image_copy,
        )
        return replace(snapshot, frame=frame)

    @staticmethod
    def _diagnostic_frame_store_result(value: object) -> dict[str, object]:
        if isinstance(value, dict):
            return dict(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                return dict(payload)
        fields = ("sequence", "image_path", "metadata_path", "count")
        payload = {field: getattr(value, field) for field in fields if hasattr(value, field)}
        if payload:
            return payload
        raise RuntimeError("诊断截图存储服务返回了无效结果")

    def target_client_rect(self):
        # Formal recommendation compact remains strict: only a confirmed
        # opening may request the recommendation surface.
        if not self.compact_request_allowed():
            return None
        return self.capture_service.target_client_rect(self.profile_name)

    def diagnostic_target_client_rect(self):
        # WAIT states (lobby/settlement/hand confirmation) may still show the
        # diagnostic waiting surface. This method exposes geometry only and
        # never authorizes recommendations or a live session.
        if not self.diagnostic_compact_request_allowed():
            return None
        return self.capture_service.target_client_rect(self.profile_name)

    def recognize_initial(self) -> None:
        self._start_danzero_warmup()
        if self._initial_thread is not None and self._initial_thread.isRunning():
            return

        def operation():
            snapshot = self.capture_service.capture_frame(self.profile_name)
            result = self._recognize_initial_image(snapshot.image)
            return result, snapshot

        thread = OneShotThread(operation, self)
        thread.result.connect(self._initial_result)
        thread.error.connect(self.error)
        thread.finished.connect(self._initial_finished)
        self._initial_thread = thread
        thread.start()

    def set_recognition_strategy(self, strategy: str) -> None:
        """Use the selected live recognition strategy for future auto sessions."""

        self._recognition_strategy = str(strategy)

    def set_advisor_strategy(self, strategy: str) -> None:
        """Select the advisor for the next session and persist the profile default."""

        normalized = normalize_advisor_strategy(strategy)
        if self.orchestrator is not None:
            raise RuntimeError("实时对局开始后建议模型已锁定")
        if normalized == self.advisor_strategy:
            save_profile_advisor_strategy(
                self.capture_service.profiles_root,
                self.profile_name,
                normalized,
            )
            return
        advisor = build_advisor(
            normalized,
            profiles_root=self.capture_service.profiles_root,
            profile_name=self.profile_name,
        )
        rebind = getattr(self.session_factory, "with_advisor", None)
        if not callable(rebind):
            raise RuntimeError("当前会话工厂不支持切换建议模型")
        session_factory = rebind(advisor)
        save_profile_advisor_strategy(
            self.capture_service.profiles_root,
            self.profile_name,
            normalized,
        )
        self.advisor_strategy = normalized
        self.danzero_advisor = advisor
        self.session_factory = session_factory
        self._danzero_warmup_complete = False
        if not self._danzero_warmup_running:
            self._start_danzero_warmup()

    def set_session_data_recording_enabled(self, enabled: bool) -> None:
        """Compatibility setter for the former binary recording setting."""

        self.set_recording_mode("game" if enabled else "none")

    def set_recording_mode(self, mode: str) -> None:
        """Persist the replay policy before a listener or live game begins."""

        if self.orchestrator is not None or self._listening_enabled:
            raise RuntimeError("开始监听页面后不能切换保存方式")
        self.recording_mode = save_profile_recording_mode(
            self.capture_service.profiles_root,
            self.profile_name,
            mode,
        )
        self.session_data_recording_enabled = self.recording_mode != "none"

    def set_recording_max_total_gb(self, gigabytes: object) -> int:
        """Persist the total media budget for future sessions."""

        if self.orchestrator is not None or self._listening_enabled:
            raise RuntimeError("开始监听页面后不能切换录像容量")
        try:
            value = float(gigabytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("录像容量必须是正数 GB") from exc
        if value <= 0:
            raise ValueError("录像容量必须是正数 GB")
        bytes_value = int(round(value * 1024 ** 3))
        self.recording_max_total_bytes = save_profile_recording_max_total_bytes(
            self.capture_service.profiles_root, self.profile_name, bytes_value
        )
        return self.recording_max_total_bytes

    def set_automatic_log_include_media(self, enabled: object) -> bool:
        """Persist whether automatic sealed-session ZIPs include media."""

        if self.orchestrator is not None or self._listening_enabled:
            raise RuntimeError("开始监听页面后不能切换自动诊断媒体设置")
        self.automatic_log_include_media = save_profile_automatic_log_include_media(
            self.capture_service.profiles_root, self.profile_name, enabled
        )
        return self.automatic_log_include_media

    def recording_storage_summary(self) -> dict[str, int | bool]:
        return get_recording_storage_summary(
            self.capture_service.profiles_root, self.profile_name
        )

    @property
    def opening_readiness(self) -> OpeningReadinessReport:
        """Latest unified report for the opening/listening boundary."""

        return self._last_opening_readiness

    def compact_request_report(self) -> OpeningReadinessReport:
        """Return the report a compact-window caller must inspect."""

        return self._last_opening_readiness

    def compact_request_allowed(self) -> bool:
        """Guard explicit compact requests without changing ``main_window``.

        Only PASS is allowed for recommendation compact mode.  WAIT is
        exposed separately through ``can_show_waiting_compact`` for a future
        diagnostic surface, and FAIL is denied so window/ROI/worker errors
        cannot be bypassed by asking for recommendations directly.
        """

        return self._last_opening_readiness.can_request_compact

    def can_show_compact_recommendation(self) -> bool:
        """Readable alias for UI callers checking recommendation compact mode."""

        return self.compact_request_allowed()

    def diagnostic_compact_request_allowed(self) -> bool:
        """Allow a future waiting-state diagnostic compact surface only."""

        return self._last_opening_readiness.can_show_waiting_compact

    def can_show_waiting_compact(self) -> bool:
        """Readable alias for the independent WAIT diagnostic contract."""

        return self.diagnostic_compact_request_allowed()

    def _emit_listening_status(
        self,
        state: str,
        report: OpeningReadinessReport,
        **payload: object,
    ) -> None:
        """Publish one structured report while retaining legacy top-level keys."""

        self._last_opening_readiness = report
        report_payload = report.to_dict()
        message = payload.pop("message", report.message)
        status_payload: dict[str, object] = {
            "state": state,
            "message": str(message),
            "report": report,
            "readiness": report_payload,
            **report_payload,
            **payload,
        }
        # ``message`` is the only legacy field that can intentionally override
        # the report's presentation text; the stable code fields never do.
        status_payload["message"] = str(message)
        self.listening_status.emit(status_payload)

    def _reset_page_recovery(self) -> None:
        self._page_unknown_retry_count = 0
        self._page_unknown_started_ms = None
        self._page_unknown_last_reason = ""
        self._page_recovery_slow = False

    def _page_recovery_payload(self) -> dict[str, object]:
        started = self._page_unknown_started_ms
        now = monotonic_ns() // 1_000_000
        elapsed = max(0, now - started) if started is not None else 0
        return {
            "stage": "page",
            "page_stage": self._listening_page.stage,
            "last_stable_page_stage": self._last_stable_page_stage,
            "reason": self._page_unknown_last_reason or "page_stable",
            "retry_count": int(self._page_unknown_retry_count),
            "retry_budget": int(self._PAGE_UNKNOWN_MAX_RETRIES),
            "recovery_window_ms": int(self._PAGE_UNKNOWN_RECOVERY_WINDOW_MS),
            "recovery_elapsed_ms": int(elapsed),
            "recovery_active": bool(self._page_unknown_retry_count),
            "low_frequency": self._page_recovery_slow,
        }

    def _update_listener_diagnostic_state(self, **changes: object) -> None:
        """Best-effort observability for the active opening recording."""

        recording = self._listener_recording
        store = getattr(recording, "store", None)
        update_metadata = getattr(store, "update_session_metadata", None)
        if not callable(update_metadata):
            return
        stage = str(changes.get("stage", "page"))
        page_stage = str(changes.get("page_stage", self._listening_page.stage))
        reason = str(changes.get("reason", ""))
        payload = {
            "listening_stage": stage,
            "listening_reason": reason,
            "listening_retry_count": int(changes.get("retry_count", 0) or 0),
            "listening_page_stage": page_stage,
            "listening_last_stable_page": str(
                changes.get("last_stable_page_stage", self._last_stable_page_stage)
            ),
            "page_recovery": dict(changes),
        }
        if stage == "page":
            payload.update({
                "lifecycle": "paused" if page_stage == "unknown" else "listening",
                "lifecycle_status": "paused" if page_stage == "unknown" else "listening",
                "lifecycle_reason": reason,
            })
        try:
            update_metadata(payload)
            append_trace = getattr(store, "append_recognition_trace", None)
            if callable(append_trace):
                append_trace({
                    "phase": "page_recovery",
                    "stage": stage,
                    "page_stage": page_stage,
                    "reason": reason,
                    "retry_count": payload["listening_retry_count"],
                    "recovery": dict(changes),
                })
        except Exception:
            # Observability must not change capture or recognition decisions.
            return

    def _handle_transient_unknown_page(
        self, page: ListeningPageSignal, snapshot: object
    ) -> bool:
        now = monotonic_ns() // 1_000_000
        if self._page_unknown_started_ms is None:
            self._page_unknown_started_ms = now
        self._page_unknown_retry_count += 1
        self._page_unknown_last_reason = "transient_page_unknown"
        elapsed = now - self._page_unknown_started_ms
        payload = self._page_recovery_payload()
        payload.update({
            "page_stage": page.stage,
            "anchor_score": float(page.anchor_score),
            "reason": self._page_unknown_last_reason,
        })
        self._update_listener_diagnostic_state(**payload)
        if (
            self._page_unknown_retry_count <= self._PAGE_UNKNOWN_MAX_RETRIES
            and elapsed <= self._PAGE_UNKNOWN_RECOVERY_WINDOW_MS
        ):
            # Keep the last stable table/episode context.  Unknown is not a
            # lobby, settlement, action, or new-game boundary.
            worker = getattr(self._waiting_capture_worker, "worker", None)
            if worker is not None:
                worker.interval_sec = 0.2
            return True

        entering_slow_recovery = not self._page_recovery_slow
        self._page_recovery_slow = True
        self._page_unknown_last_reason = "transient_page_unknown_budget_exhausted"
        payload = self._page_recovery_payload()
        payload.update({
            "page_stage": page.stage,
            "reason": self._page_unknown_last_reason,
            "failure_class": "transient_page_recovery_exhausted",
        })
        self._update_listener_diagnostic_state(**payload)
        if entering_slow_recovery:
            try:
                self.opening_evidence.observe_failure(
                    RuntimeError("page probe remained unknown beyond recovery budget"),
                    stage="page", snapshot=snapshot,
                )
            except Exception:
                pass
        worker = getattr(self._waiting_capture_worker, "worker", None)
        if worker is not None:
            worker.interval_sec = self._PAGE_UNKNOWN_SLOW_INTERVAL_SEC
        self._emit_listening_status(
            "recovering", report_for_phase("page_recovering", message="页面暂时无法确认，低频等待牌桌恢复"),
            **payload, message="页面暂时无法确认，低频等待牌桌恢复",
        )
        return True

    def _begin_preopening_diagnostic_round(self) -> bool:
        """Reset listener-round state without creating an empty parallel directory."""

        if not self._listener_evidence.begin_run():
            return False
        self._retained_diagnostic_frame = None
        self._diagnostic_run_id = uuid4().hex
        self._diagnostic_no_frame_failure = False
        self._diagnostic_runtime_evidence = None
        self._last_no_frame_fault = None
        self._listener_stop_requested = False
        with self._latest_waiting_frame_lock:
            self._latest_waiting_frame = None
            self._latest_waiting_frame_generation = -1
            self._latest_waiting_frame_capture_seq = -1
            self._waiting_capture_seq_counter = 0
        # The managed episode is allocated by start_listener_recording().
        # Until then these remain unset; a manual fallback is allocated only
        # when the user actually saves a frame.
        self._preopening_diagnostic_directory = None
        self._preopening_diagnostic_session_id = ""
        with self._manual_diagnostic_lock:
            self._manual_diagnostic_directory = None
            self._manual_diagnostic_session_id = ""
            self._manual_capture_seq = 0

        return True

    def start_listening(self) -> bool:
        """Continuously inspect the current page and start only on a stable deal."""

        if self._listening_enabled or self.orchestrator is not None:
            return True
        if not self._waiting_analysis_drain_complete():
            self.error.emit("上一次开局识别仍在停止中，请稍后重试")
            return False
        if not self._listener_evidence.writer.wait_idle(0):
            self.error.emit("上一轮诊断截图仍在保存，请稍后重新开始监听")
            return False
        try:
            preload_live_worker_dependencies()
        except Exception as exc:
            report = report_for_error(exc, stage="worker")
            self._emit_listening_status("failed", report)
            self.error.emit(f"实时依赖预加载失败：{exc}")
            return False
        self.opening_evidence.begin(monotonic_ms=monotonic_ns() // 1_000_000)
        lock_client = getattr(self.capture_service, "lock_target_client_size", None)
        if callable(lock_client):
            try:
                lock_client(self.profile_name)
            except Exception as exc:
                self.opening_evidence.observe_failure(exc, stage="window")
                report = report_for_error(exc, stage="window")
                self._emit_listening_status("failed", report)
                self.error.emit(f"无法锁定牌桌客户区尺寸：{exc}")
                return False
        if not self._begin_preopening_diagnostic_round():
            self.error.emit("上一轮诊断截图仍在保存，请稍后重新开始监听")
            return False
        self._listening_enabled = True
        self._geometry_recovery_cycle_count = 0
        self._geometry_post_recovery_frames_remaining = 0
        self._table_anchor_observed = False
        self._listener_recording_stop_reason = None
        self._opening_tracker.reset()
        self._listening_page = ListeningPageSignal("unknown", 0.0)
        self._last_stable_page_stage = "unknown"
        self._reset_page_recovery()
        self._page_recovery_terminated = False
        self._emit_listening_status("listening", listening_report())
        self._start_danzero_warmup()
        if self.orchestrator is None and self._finish_thread is None:
            self._start_waiting_workers()
        return True

    def stop_listening(self) -> None:
        self._listener_stop_requested = True
        self._retain_stopped_listener_frame()
        self._listening_enabled = False
        self._cancel_geometry_recovery()
        self._waiting_candidate = None
        self._opening_tracker.reset()
        self._pending_auto_session = None
        self._table_anchor_observed = False
        self._reset_page_recovery()
        self._page_recovery_terminated = False
        self._listener_recording_stop_reason = "listener_stopped"
        if self._stop_waiting_workers():
            self._close_listener_recording()

    def _start_waiting_workers(self) -> None:
        if (
            not self._listening_enabled
            or self.orchestrator is not None
            or self._waiting_capture_worker is not None
        ):
            return
        # Every new round owns a fresh field-timer/dedup scope. Geometry
        # recovery is the exception: its incident and pre-change ring must
        # remain exportable through the same opening evidence episode.
        preserve_evidence, self._preserve_opening_evidence_once = (
            self._preserve_opening_evidence_once,
            False,
        )
        if not preserve_evidence:
            self.opening_evidence.begin(monotonic_ms=monotonic_ns() // 1_000_000)
        source, self._recovered_waiting_source = self._recovered_waiting_source, None
        if source is None:
            try:
                source = self.capture_service.open_live_source(self.profile_name)
            except Exception as exc:
                self.opening_evidence.observe_failure(exc, stage="window")
                report = report_for_error(exc, stage="window")
                self._emit_listening_status("failed", report)
                self.error.emit(str(exc))
                self._listening_enabled = False
                return
        self._start_waiting_workers_with_source(source)

    def _start_waiting_workers_with_source(self, source: object) -> None:
        """Install one validated source under the current listening generation."""

        if not self._waiting_analysis_drain_complete():
            try:
                source.close()
            except Exception:
                pass
            self._listening_enabled = False
            report = report_for_error(
                RuntimeError("旧开局识别线程尚未退出，拒绝启动新的识别线程"),
                stage="worker",
            )
            self._emit_listening_status("failed", report)
            self.error.emit("旧开局识别线程尚未退出，拒绝启动新的识别线程")
            return
        self._waiting_source = source
        generation = self._waiting_generation
        analysis = LatestOnlyWorker(
            self._recognize_waiting_frame,
            on_result=lambda value: self._waiting_recognized.emit(value[0], value[1]),
            on_error=self._accept_waiting_recognition_error,
            on_discard=self._discard_waiting_analysis,
        )
        self._waiting_analysis_worker = analysis
        analysis.start()

        def operation() -> _WaitingFrameDelivery:
            snapshot: FrameSnapshot = source.capture()
            with self._latest_waiting_frame_lock:
                self._waiting_capture_seq_counter += 1
                capture_seq = self._waiting_capture_seq_counter
            delivery = _WaitingFrameDelivery(snapshot, generation, capture_seq)
            if generation != self._waiting_generation or not self._listening_enabled:
                return delivery
            evidence = self._retain_listener_snapshot(snapshot, generation, capture_seq)
            page_probe = getattr(self.recognition_service, "recognize_listening_page", None)
            page = page_probe(snapshot.image) if callable(page_probe) else None
            if generation != self._waiting_generation:
                return delivery
            if page is not None:
                if evidence is not None:
                    evidence = self._listener_evidence.annotate(evidence, page=self._page_evidence(page))
                # Gate every persisted frame using its own cheap page probe,
                # not a slow full-hand recognition from an earlier screen.
                active_capture = getattr(self._waiting_capture_worker, "worker", None)
                if active_capture is not None:
                    active_capture.interval_sec = (
                        self._PAGE_UNKNOWN_SLOW_INTERVAL_SEC
                        if page.stage == "unknown" and self._page_recovery_slow
                        else .2 if page.allows_media else 1.0
                    )
                if page.allows_media:
                    self._record_listener_frame(snapshot)
            if self._geometry_post_recovery_frames_remaining > 0:
                self._geometry_post_recovery_frames_remaining -= 1
                if self._geometry_post_recovery_frames_remaining == 0:
                    self._geometry_recovery_cycle_count = 0
            # Opening probes stay in memory.  Storage starts only after a
            # complete initial state has been confirmed, unless the user has
            # selected full recording for this listening pass.
            active_analysis = self._waiting_analysis_worker
            if active_analysis is not None:
                self.opening_evidence.observe_analysis_submitted(snapshot)
                task = _WaitingAnalysisTask(snapshot, generation, page, evidence)
                try:
                    active_analysis.submit(task)
                except Exception:
                    self.opening_evidence.observe_analysis_dropped(
                        snapshot,
                        reason="submit_failed",
                    )
                    raise
            return delivery

        worker = WorkerHandle(operation, 1.0)
        worker.frame_ready.connect(
            lambda value, current=generation: self._accept_waiting_frame(
                value,
                current,
            )
        )
        worker.error.connect(
            lambda error, current=generation: self._accept_waiting_error(
                error,
                current,
            )
        )
        worker.finished.connect(lambda: self._waiting_capture_stopped.emit(worker))
        self._waiting_capture_worker = worker
        worker.start()

    def _recognize_waiting_frame(
        self,
        value: FrameSnapshot | _WaitingAnalysisTask,
    ) -> tuple[object, FrameSnapshot | _WaitingRecognitionEnvelope]:
        task = value if isinstance(value, _WaitingAnalysisTask) else None
        snapshot = task.snapshot if task is not None else value
        generation = task.generation if task is not None else None
        self.opening_evidence.observe_analysis_started(snapshot)
        page = task.page if task is not None else None
        try:
            page_probe = getattr(self.recognition_service, "recognize_listening_page", None)
            if page is None and callable(page_probe):
                page = page_probe(snapshot.image)
            if page is not None:
                observe_page = getattr(self.opening_evidence, "observe_page", None)
                if (
                    callable(observe_page)
                    and page.stage != "unknown"
                    and (generation is None or generation == self._waiting_generation)
                ):
                    observe_page(page.stage, monotonic_ms=getattr(
                        snapshot, "captured_monotonic_ms", monotonic_ns() // 1_000_000))
            if page is not None and not page.allows_media:
                result = SimpleNamespace(
                    round_level=None, wild_rank=None, current_player=None,
                    lead_player=None, my_hand=(), events=(), buttons=page.buttons,
                    field_confidences={}, sources={}, unresolved_fields=(),
                    diagnostics=(), annotations=(), elapsed_ms=0.0,
                )
            else:
                self.opening_evidence.observe_frame(
                    snapshot, monotonic_ms=getattr(snapshot, "captured_monotonic_ms", None),
                )
                result = self._recognize_initial_image(snapshot.image)
        except Exception as exc:
            raise _WaitingRecognitionFailure(
                exc,
                snapshot,
                generation=generation,
                evidence=task.evidence if task is not None else None,
            ) from exc
        trace_reader = getattr(
            self.recognition_service,
            "get_last_diagnostic_trace",
            None,
        )
        trace = trace_reader() if callable(trace_reader) else None
        hand = tuple(str(card) for card in getattr(result, "my_hand", ()) or ())
        level = str(getattr(result, "round_level", "") or "")
        seed_valid: bool | None = None
        if level in RANKS and len(hand) == 27:
            try:
                normalizer = GuanDanState()
                normalizer.confirm_hand(hand)
                seed_valid = self._auto_session_seed(
                    result,
                    round_level=level,
                    hand=normalizer.my_hand,
                ) is not None
            except Exception:
                seed_valid = False
        if task is None:
            self.opening_evidence.observe_recognition(
                snapshot,
                result,
                trace,
                opening_seed_valid=seed_valid,
            )
            return result, snapshot
        evidence = task.evidence
        if evidence is not None:
            evidence = self._listener_evidence.annotate(
                evidence,
                recognition={
                    "buttons": tuple(getattr(result, "buttons", ()) or ()),
                    "hand_count": len(hand), "round_level": level,
                    "field_confidences": dict(getattr(result, "field_confidences", {}) or {}),
                    "opening_seed_valid": seed_valid,
                },
            )
        return result, _WaitingRecognitionEnvelope(
            snapshot, generation, trace, seed_valid, page, evidence,
        )

    def _recognize_initial_image(self, image: object) -> object:
        """Keep an occluded hand card as ``5?`` instead of losing the deal."""

        try:
            return self.recognition_service.recognize(
                image,
                allow_unknown_suit=True,
            )
        except TypeError as exc:
            # External recognizer plug-ins may still expose the older method
            # signature. The bundled recognizer always supports this flag.
            if "allow_unknown_suit" not in str(exc):
                raise
            return self.recognition_service.recognize(image)

    def _discard_waiting_analysis(self, value: object, reason: str) -> None:
        snapshot = (
            value.snapshot if isinstance(value, _WaitingAnalysisTask) else value
        )
        self.opening_evidence.observe_analysis_dropped(snapshot, reason=reason)

    def _accept_waiting_frame(
        self,
        value: object,
        generation: int | None = None,
        capture_seq: int | None = None,
    ) -> None:
        if isinstance(value, _WaitingFrameDelivery):
            snapshot = value.snapshot
            delivery_generation = value.generation
            delivery_seq = value.capture_seq
        elif isinstance(value, FrameSnapshot):
            snapshot = value
            delivery_generation = generation
            delivery_seq = capture_seq if capture_seq is not None else -1
        else:
            return
        current_generation = self._waiting_generation
        if (
            delivery_generation is None
            or delivery_generation != current_generation
            or (generation is not None and generation != current_generation)
            or not self._listening_enabled
        ):
            return
        self._retain_listener_snapshot(snapshot, current_generation, int(delivery_seq))
        self.frame_ready.emit(snapshot)

    def _accept_waiting_error(
        self,
        error: object,
        generation: int | None = None,
    ) -> None:
        if not self._listening_enabled or generation is not None and generation != self._waiting_generation:
            return
        self._queue_listener_incident("waiting_capture_failed", error=error)
        self.opening_evidence.observe_failure(error, stage="capture")
        code = str(getattr(error, "code", "") or "").upper()
        if (
            code in self._GEOMETRY_RECOVERABLE_CODES
            and self.orchestrator is None
            and self._listening_enabled
        ):
            self._begin_geometry_recovery(error)
            return
        report = report_for_error(error, stage="capture")
        hard_failure = {
            "stage": "capture",
            "reason": "window_or_capture_hard_failure",
            "failure_class": "window_capture_hard_failure",
            "retry_count": 0,
            "page_stage": self._listening_page.stage,
        }
        self._update_listener_diagnostic_state(**hard_failure)
        self._emit_listening_status("failed", report, **hard_failure)
        if self.orchestrator is None:
            self._listening_enabled = False
            self._waiting_candidate = None
            self._listener_recording_stop_reason = "waiting_capture_failed"
            self.live_fault.emit({
                "session_id": self._preopening_diagnostic_session_id,
                "capture_generation": self._waiting_generation,
                "kind": "capture",
                **hard_failure,
            })
        if self.orchestrator is None and self._stop_waiting_workers():
            self._close_listener_recording()
        self.error.emit(str(error))

    def _accept_waiting_recognition_error(self, error: Exception) -> None:
        """Preserve the typed failure and the already-buffered capture evidence."""

        original = error.error if isinstance(error, _WaitingRecognitionFailure) else error
        snapshot = error.snapshot if isinstance(error, _WaitingRecognitionFailure) else None
        generation = (
            error.generation if isinstance(error, _WaitingRecognitionFailure) else None
        )
        if generation is not None and generation != self._waiting_generation:
            if snapshot is not None:
                self.opening_evidence.observe_analysis_dropped(
                    snapshot,
                    reason="stale_generation_error",
                )
            return
        if self._listener_stop_requested:
            return
        if self._listening_enabled:
            self._queue_listener_incident(
                "waiting_recognition_failed",
                context=error.evidence if isinstance(error, _WaitingRecognitionFailure) else None,
                error=original, allow_latest=False,
            )
        self.opening_evidence.observe_failure(
            original,
            stage="recognition",
            snapshot=snapshot,
        )
        report = report_for_error(original, stage="recognition")
        self._emit_listening_status("failed", report)
        self.error.emit(str(original))

    def _consume_waiting_recognition(self, result: object, snapshot: object) -> None:
        """Publish current page/progress and confirm only semantic opening state."""

        generation: int | None = None
        envelope = (
            snapshot if isinstance(snapshot, _WaitingRecognitionEnvelope) else None
        )
        gate_eligible = bool(
            self._listening_enabled
            and self.orchestrator is None
            and not self._geometry_recovery_active
        )
        if not gate_eligible:
            return
        if envelope is not None:
            generation = envelope.generation
            snapshot = envelope.snapshot
        elif isinstance(snapshot, _WaitingAnalysisTask):
            generation = snapshot.generation
            snapshot = snapshot.snapshot
        if generation is not None and generation != self._waiting_generation:
            self.opening_evidence.observe_analysis_dropped(
                snapshot,
                reason="stale_generation_delivery",
            )
            return
        if envelope is not None and envelope.page is not None:
            self._apply_listening_page(envelope.page, snapshot, evidence=envelope.evidence)
            if not envelope.page.allows_media:
                self.initial_recognized.emit(result, snapshot)
                if envelope.page.stage == "unknown":
                    if not self._page_recovery_terminated:
                        recovery = self._page_recovery_payload()
                        recovery.update({
                            "page_stage": "unknown",
                            "reason": self._page_unknown_last_reason,
                            "retry_count": self._page_unknown_retry_count,
                        })
                        self._emit_listening_status(
                            "recovering",
                            listening_report(message="页面暂时无法确认，正在恢复"),
                            **recovery,
                        )
                else:
                    self._publish_opening_status(envelope.page.stage, result)
                return
        if envelope is not None:
            self.opening_evidence.observe_recognition(
                snapshot,
                result,
                envelope.trace,
                opening_seed_valid=envelope.opening_seed_valid,
            )

        recording = self._listener_recording
        record_recognition = getattr(recording, "record_recognition", None)
        if callable(record_recognition):
            try:
                record_recognition(result)
            except Exception as exc:
                self.error.emit(f"监听录像写入识别记录失败：{exc}")
        self.opening_evidence.observe_delivery(
            snapshot,
            gate_eligible=gate_eligible,
        )
        self.initial_recognized.emit(result, snapshot)
        if not gate_eligible:
            return
        buttons = set(getattr(result, "buttons", ()) or ())
        if buttons & {"change_table", "continue_game"}:
            # A settlement screen never becomes a session.  Reset the table
            # probe and keep listening for the next real opening.
            self._waiting_candidate = None
            self._table_anchor_observed = False
            self._opening_tracker.reset()
            self._apply_listening_page(ListeningPageSignal("settlement", 0.0), snapshot)
            self._publish_opening_status("settlement", result)
            return
        if not self._table_anchor_observed:
            if self._table_anchor_score(snapshot) < self._TABLE_ANCHOR_READY_SCORE:
                # Do not start a session from a lobby, settlement screen, or
                # a manually clicked late page.
                self._waiting_candidate = None
                self._opening_tracker.discard_candidates()
                self._publish_opening_status("waiting_table", result)
                return
            self._table_anchor_observed = True
        previous_waiting_candidate = self._waiting_candidate
        evaluation = self._opening_tracker.observe(
            result,
            anchor_score=self._TABLE_ANCHOR_READY_SCORE,
            generation=self._waiting_generation,
            monotonic_ms=getattr(snapshot, "captured_monotonic_ms", None) or monotonic_ns() // 1_000_000,
            observation_id=getattr(snapshot, "captured_monotonic_ms", None),
        )
        self._waiting_candidate = self._opening_tracker.candidate
        self._publish_opening_status(evaluation.reason, result)
        if not evaluation.ready or evaluation.seed is None:
            # A complete but temporarily suit-obscured self hand may start a
            # provisional listener session after two identical reads.  This is
            # deliberately not an OpeningTracker confirmation: the formal
            # runtime still carries the unresolved hand and waits for lead
            # evidence/advice-world resolution.
            hand_values = tuple(str(card) for card in getattr(result, "my_hand", ()) or ())
            if (
                evaluation.reason == "hand_unresolved"
                and len(hand_values) == 27
                and not getattr(result, "events", ())
                and getattr(result, "lead_player", None) is None
                and getattr(result, "current_player", None) is None
            ):
                provisional = OpeningSessionSeed(
                    round_level=str(getattr(result, "round_level", "") or ""),
                    hand=hand_values, lead_player=None,
                )
                if (
                    isinstance(previous_waiting_candidate, OpeningSessionSeed)
                    and opening_semantic_key(previous_waiting_candidate)
                    == opening_semantic_key(provisional)
                ):
                    self._waiting_candidate = None
                    self._start_detected_session(provisional)
                else:
                    self._waiting_candidate = provisional
            return
        self._waiting_candidate = None
        self._start_detected_session(result)

    def _apply_listening_page(
        self, page: ListeningPageSignal, snapshot: object, *,
        evidence: _DiagnosticFrameSaveContext | None = None,
    ) -> None:
        if self._listener_stop_requested or self._diagnostics_closed:
            return
        evidence = evidence or self._listener_evidence.find(
            snapshot, self._waiting_generation, "preopening_listener",
            scope_id=self._diagnostic_scope_id(),
        )
        previous = self._listening_page.stage
        self._listening_page = page
        worker = getattr(self._waiting_capture_worker, "worker", None)

        if page.stage == "unknown":
            # ``unknown`` is a transient page-probe miss.  Do not clear the
            # OpeningTracker candidate, table anchor, episode recording, or
            # any stable hand/lead evidence.  No action is produced from this
            # page because the recognizer returns an empty result for it.
            if evidence is not None:
                self._queue_listener_incident(
                    "page_unknown", context=evidence, page=page, allow_latest=False,
                )
            self._handle_transient_unknown_page(page, snapshot)
            return

        # A real page classification ends the unknown recovery window.
        recovered_unknown = self._page_unknown_retry_count > 0
        if page.stage in {"table", "lobby", "settlement"}:
            self._last_stable_page_stage = page.stage
        self._reset_page_recovery()
        if worker is not None:
            worker.interval_sec = 0.2 if page.allows_media else 1.0

        if not page.allows_media:
            if page.stage in {"lobby", "settlement"}:
                # These are explicit page boundaries, unlike transient
                # unknown.  Preserve the existing reset semantics.
                self._opening_tracker.reset()
                self._waiting_candidate = None
            self._table_anchor_observed = False
            self._listener_recording_stop_reason = "page_" + page.stage
            self._update_listener_diagnostic_state(
                stage="page",
                page_stage=page.stage,
                reason="page_" + page.stage,
                retry_count=0,
                recovered_unknown=recovered_unknown,
            )
            self._close_listener_recording()
            return

        if previous in {"lobby", "settlement"}:
            self._opening_tracker.reset()
        self._table_anchor_observed = page.anchor_score >= self._TABLE_ANCHOR_READY_SCORE
        self._update_listener_diagnostic_state(
            stage="page",
            page_stage=page.stage,
            reason=("page_recovered" if recovered_unknown else "page_table"),
            retry_count=0,
            recovered_unknown=recovered_unknown,
            anchor_score=float(page.anchor_score),
        )
        if recovered_unknown or page.stage == "table":
            if evidence is not None:
                evidence = self._listener_evidence.annotate(evidence, page=self._page_evidence(page))
            self._listener_evidence.recover(evidence)
        if recovered_unknown:
            recovery = self._page_recovery_payload()
            recovery.update({
                "page_stage": page.stage,
                "reason": "page_recovered",
                "retry_count": 0,
            })
            self._emit_listening_status(
                "recovered",
                listening_report(message="页面已恢复，继续监听"),
                **recovery,
            )
        if self._start_listener_recording() and snapshot is not None:
            self._record_listener_frame(snapshot)

    def _publish_opening_status(self, phase: str, result: object) -> None:
        count = len(tuple(getattr(result, "my_hand", ()) or ()))
        messages = {
            "unknown": "已连接，等待进入牌桌",
            "lobby": "已连接，等待进入牌桌",
            "waiting_table": "已连接，等待进入牌桌",
            "settlement": "本局已结束，等待下一局（未录像）",
            "round_level_unresolved": "当前级牌尚未确认",
            "hand_count_mismatch": f"起手牌数量尚未稳定，当前识别到{count}张",
            "hand_invalid": "起手牌识别仍在变化或存在冲突",
            "hand_unresolved": "起手牌存在未确认花色",
            "missed_opening": f"当前对局已进行，当前识别到{count}张手牌",
            "opening_seed_invalid": f"已识别{count}张，首出证据尚未确认",
            "confirming_hand": f"已识别{count}张，正在等待稳定起手牌",
            "confirming_opening": f"已识别{count}张，正在等待稳定首出",
            "ready": "完整开局已确认，正在建立对局",
        }
        if phase in {"already_started", "duplicate_frame"}:
            return
        report = report_for_phase(
            phase,
            hand_count=count,
            message=messages.get(phase),
            details={"lead_player": getattr(result, "lead_player", None)},
        )
        self._emit_listening_status(
            "opening",
            report,
            phase=phase,
            reason=phase,
            hand_count=count,
            generation=self._waiting_generation,
            page_stage=self._listening_page.stage,
            page_reason=(self._page_unknown_last_reason or phase),
            retry_count=self._page_unknown_retry_count,
        )

    def _start_detected_session(self, result: object) -> None:
        if isinstance(result, _AutoSessionSeed):
            seed = result
        elif self._opening_tracker.completed:
            seed = self._opening_tracker.candidate
        else:
            # A hand/level read without a completed opening action is not
            # enough to start normal turn listening.  The live-v2 runtime must
            # confirm the lead's first play itself; otherwise static hand
            # cards can enter the normal action pipeline as pre-opening plays.
            return
        if seed is None:
            return
        self._pending_auto_session = seed
        self._listener_recording_stop_reason = "initial_state_confirmed"
        if self._stop_waiting_workers():
            self._close_listener_recording()
            self._start_pending_auto_session()

    def _legacy_detected_seed(self, result: object) -> _AutoSessionSeed | None:
        hand = tuple(str(card) for card in getattr(result, "my_hand", ()))
        try:
            normalizer = GuanDanState()
            normalizer.confirm_hand(hand)
        except Exception:
            return None
        seed = self._auto_session_seed(
            result,
            round_level=str(getattr(result, "round_level", "")),
            hand=normalizer.my_hand,
        )
        return seed

    def _start_pending_auto_session(self) -> None:
        pending, self._pending_auto_session = self._pending_auto_session, None
        if (
            pending is None
            or not self._listening_enabled
            or self.orchestrator is not None
        ):
            return
        if not self.start_session(
            round_level=pending.round_level,
            hand=pending.hand,
            # The waiting tracker has already required bounded, independent
            # opening evidence. Preserve that seed when handing off to the
            # formal runtime so a visible self button cannot steal turn zero.
            lead_player=(
                pending.lead_player.value
                if hasattr(pending.lead_player, "value")
                else str(pending.lead_player)
                if pending.lead_player is not None else None
            ),
            recognition_strategy=self._recognition_strategy,
            opening_action=pending.opening_action,
        ):
            self._table_anchor_observed = False
            self._start_waiting_workers()

    def _table_anchor_score(self, snapshot: object) -> float:
        image = getattr(snapshot, "image", None)
        recognize_anchor = getattr(
            self.recognition_service,
            "recognize_table_anchor",
            None,
        )
        if image is None or not callable(recognize_anchor):
            return 0.0
        try:
            score = float(recognize_anchor(image))
            self.opening_evidence.observe_anchor(
                snapshot,
                score,
                required_score=self._TABLE_ANCHOR_READY_SCORE,
            )
            return score
        except Exception as exc:
            self.opening_evidence.observe_failure(exc, stage="anchor")
            self.error.emit(f"牌桌锚点识别失败：{exc}")
            return 0.0

    @staticmethod
    def _auto_session_seed(
        result: object,
        *,
        round_level: str,
        hand: tuple[str, ...],
    ) -> _AutoSessionSeed | None:
        return build_opening_seed(result, round_level=round_level, hand=hand)

    @staticmethod
    def _source_diagnostic_state(source: object | None) -> dict[str, object]:
        reader = getattr(source, "diagnostic_state", None)
        if not callable(reader):
            return {}
        try:
            value = reader()
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _geometry_recovery_requires_lock(details: dict[str, object]) -> bool:
        change_types = {
            str(item).strip().lower()
            for item in tuple(details.get("change_types", ()) or ())
            if str(item).strip()
        }
        backend = str(details.get("capture_backend", "") or "").strip().lower()
        return not (change_types == {"move"} and backend == "printwindow")

    @staticmethod
    def _snapshot_geometry(snapshot: FrameSnapshot | None) -> dict[str, object]:
        frame = getattr(snapshot, "frame", None)
        rect = getattr(frame, "rect", None)
        try:
            rect_payload = [
                int(getattr(rect, "left")),
                int(getattr(rect, "top")),
                int(getattr(rect, "width")),
                int(getattr(rect, "height")),
            ]
        except (TypeError, ValueError):
            rect_payload = None
        return {
            "new_rect": rect_payload,
            "new_dpi": getattr(frame, "dpi", None),
            "capture_backend": getattr(frame, "backend", None),
        }

    def _begin_geometry_recovery(self, error: object) -> None:
        if self._geometry_recovery_active:
            return
        self._geometry_recovery_active = True
        self._geometry_recovery_attempt_count = 0
        self._geometry_recovery_cycle_count += 1
        self._waiting_generation += 1
        self._stop_waiting_analysis_worker()
        self._waiting_candidate = None
        self._pending_auto_session = None
        self._table_anchor_observed = False
        details = getattr(error, "details", None)
        self._geometry_recovery_error_details = (
            dict(details) if isinstance(details, dict) else {}
        )
        self._geometry_recovery_error_details.setdefault(
            "source_error_code",
            str(getattr(error, "code", "") or ""),
        )
        self._geometry_recovery_error_details["recovery_cycle_count"] = (
            self._geometry_recovery_cycle_count
        )
        if self._geometry_recovery_cycle_count > self._GEOMETRY_RECOVERY_MAX_CYCLES:
            self._fail_geometry_recovery("牌桌窗口连续变化次数超过安全上限")
            return
        self.opening_evidence.observe_geometry_recovery(
            result="started",
            generation=self._waiting_generation,
            attempt_count=0,
            details=self._geometry_recovery_error_details,
            reason=str(error),
        )
        report = report_for_error(error, stage="capture", recovering=True)
        self._emit_listening_status(
            "recovering",
            report,
            generation=self._waiting_generation,
            attempt_count=0,
        )

    def _schedule_geometry_recovery_attempt(self) -> None:
        if not self._geometry_recovery_active or not self._listening_enabled:
            return
        index = self._geometry_recovery_attempt_count
        if index >= len(self._GEOMETRY_RECOVERY_DELAYS_MS):
            self._fail_geometry_recovery("牌桌窗口在限定时间内未恢复稳定")
            return
        self._geometry_recovery_timer.start(
            int(self._GEOMETRY_RECOVERY_DELAYS_MS[index])
        )

    def _start_geometry_recovery_attempt(self) -> None:
        if (
            not self._geometry_recovery_active
            or not self._listening_enabled
            or self.orchestrator is not None
            or self._geometry_recovery_thread is not None
        ):
            return
        self._geometry_recovery_attempt_count += 1
        attempt_count = self._geometry_recovery_attempt_count
        generation = self._waiting_generation
        prior_details = dict(self._geometry_recovery_error_details)
        prior_details["recovery_cycle_count"] = self._geometry_recovery_cycle_count
        draining_analysis = self._draining_waiting_analysis_worker
        cancel = Event()
        self._geometry_recovery_cancel = cancel

        def operation() -> _GeometryRecoveryAttemptResult:
            source = None
            details = dict(prior_details)
            try:
                if cancel.is_set():
                    raise RuntimeError("geometry recovery cancelled")
                if draining_analysis is not None:
                    drained = draining_analysis.stop(
                        timeout=self._GEOMETRY_ANALYSIS_DRAIN_TIMEOUT_SEC
                    )
                    details["analysis_worker_drain_timeout_sec"] = (
                        self._GEOMETRY_ANALYSIS_DRAIN_TIMEOUT_SEC
                    )
                    details["analysis_worker_drained"] = bool(drained)
                    if not drained:
                        raise RuntimeError(
                            "旧开局识别线程未在安全超时内退出，拒绝并发重启"
                        )
                else:
                    details["analysis_worker_drained"] = True
                lock_client = getattr(
                    self.capture_service,
                    "lock_target_client_size",
                    None,
                )
                relock_required = self._geometry_recovery_requires_lock(details)
                details["relock_required"] = relock_required
                if relock_required and callable(lock_client):
                    locked_rect = lock_client(self.profile_name)
                    if locked_rect is not None:
                        details["locked_rect"] = [
                            int(getattr(locked_rect, "left")),
                            int(getattr(locked_rect, "top")),
                            int(getattr(locked_rect, "width")),
                            int(getattr(locked_rect, "height")),
                        ]
                source = self.capture_service.open_live_source(self.profile_name)
                if cancel.is_set():
                    raise RuntimeError("geometry recovery cancelled")
                first = source.capture()
                if cancel.wait(self._GEOMETRY_STABLE_SAMPLE_DELAY_SEC):
                    raise RuntimeError("geometry recovery cancelled")
                second = source.capture()
                if cancel.is_set():
                    raise RuntimeError("geometry recovery cancelled")
                details.update(self._source_diagnostic_state(source))
                details.update(self._snapshot_geometry(second))
                details["stable_sample_count"] = 2
                return _GeometryRecoveryAttemptResult(
                    generation,
                    attempt_count,
                    source,
                    (first, second),
                    details,
                )
            except Exception as exc:
                failure_details = getattr(exc, "details", None)
                if isinstance(failure_details, dict):
                    details.update(failure_details)
                details["source_error_code"] = str(
                    getattr(exc, "code", "") or ""
                )
                details["error_type"] = type(exc).__name__
                details["last_failure_reason"] = str(exc)
                if source is not None:
                    try:
                        source.close()
                    except Exception:
                        pass
                return _GeometryRecoveryAttemptResult(
                    generation,
                    attempt_count,
                    None,
                    (),
                    details,
                    exc,
                )

        thread = OneShotThread(operation, self)
        thread.result.connect(
            lambda value, current=thread: self._store_geometry_recovery_result(
                current,
                value,
            ),
            Qt.ConnectionType.DirectConnection,
        )
        thread.finished.connect(
            lambda current=thread: self._geometry_recovery_thread_finished(current)
        )
        self._geometry_recovery_thread = thread
        thread.start()

    def _store_geometry_recovery_result(
        self,
        thread: OneShotThread,
        value: object,
    ) -> None:
        if (
            self._geometry_recovery_thread is thread
            and isinstance(value, _GeometryRecoveryAttemptResult)
        ):
            self._geometry_recovery_result = value
        elif isinstance(value, _GeometryRecoveryAttemptResult) and value.source is not None:
            try:
                value.source.close()
            except Exception:
                pass

    def _geometry_recovery_thread_finished(self, thread: OneShotThread) -> None:
        if self._geometry_recovery_thread is not thread:
            return
        self._geometry_recovery_thread = None
        result, self._geometry_recovery_result = self._geometry_recovery_result, None
        if result is None:
            self._fail_geometry_recovery("几何恢复工作线程未返回结果")
            return
        if bool(result.details.get("analysis_worker_drained")):
            self._waiting_analysis_drain_complete()
        if (
            not self._geometry_recovery_active
            or not self._listening_enabled
            or result.generation != self._waiting_generation
        ):
            if result.source is not None:
                try:
                    result.source.close()
                except Exception:
                    pass
            return
        if result.error is not None:
            self._geometry_recovery_error_details = dict(result.details)
            self.opening_evidence.observe_geometry_recovery(
                result="attempt_failed",
                generation=result.generation,
                attempt_count=result.attempt_count,
                details=result.details,
                reason=str(result.error),
            )
            if result.details.get("analysis_worker_drained") is False:
                self._fail_geometry_recovery(str(result.error))
                return
            self._schedule_geometry_recovery_attempt()
            return
        if result.source is None:
            self._fail_geometry_recovery("恢复采集源为空")
            return
        try:
            for snapshot in result.snapshots:
                self.opening_evidence.observe_frame(
                    snapshot,
                    monotonic_ms=snapshot.captured_monotonic_ms,
                )
                self._record_listener_frame(snapshot)
        except Exception as exc:
            try:
                result.source.close()
            except Exception:
                pass
            self._geometry_recovery_error_details = {
                **result.details,
                "recording_error": str(exc),
            }
            self._fail_geometry_recovery(str(exc))
            return
        self._geometry_recovery_active = False
        self._geometry_recovery_error_details = dict(result.details)
        self._recovered_waiting_source = result.source
        self._preserve_opening_evidence_once = True
        self._geometry_post_recovery_frames_remaining = (
            self._GEOMETRY_RECOVERY_RESET_AFTER_FRAMES
        )
        self.opening_evidence.observe_geometry_recovery(
            result="recovered",
            generation=result.generation,
            attempt_count=result.attempt_count,
            details=result.details,
            reason="target window stable across consecutive samples",
        )
        self._emit_listening_status(
            "recovered",
            listening_report(message="牌桌窗口已重新连接，继续监听"),
            generation=result.generation,
            attempt_count=result.attempt_count,
        )
        self._start_waiting_workers()

    def _fail_geometry_recovery(self, reason: str) -> None:
        if not self._geometry_recovery_active:
            return
        self._geometry_recovery_active = False
        self._geometry_recovery_timer.stop()
        self.opening_evidence.observe_geometry_recovery(
            result="failed",
            generation=self._waiting_generation,
            attempt_count=self._geometry_recovery_attempt_count,
            details=self._geometry_recovery_error_details,
            reason=reason,
        )
        self._listening_enabled = False
        self._waiting_candidate = None
        self._pending_auto_session = None
        self._listener_recording_stop_reason = "geometry_recovery_failed"
        self._close_listener_recording()
        message = "监听已停止，请打开完整助手"
        source_code = str(
            self._geometry_recovery_error_details.get("source_error_code", "")
            or "CAPTURE-BACKEND-FAILED"
        )
        source_error = RuntimeError(reason)
        source_error.code = source_code  # type: ignore[attr-defined]
        report = report_for_error(source_error, stage="capture")
        self._emit_listening_status(
            "failed",
            report,
            message=message,
            reason=reason,
            generation=self._waiting_generation,
            attempt_count=self._geometry_recovery_attempt_count,
        )
        self.error.emit(f"{message}：{reason}")

    def _cancel_geometry_recovery(self) -> None:
        self._geometry_recovery_active = False
        self._geometry_recovery_timer.stop()
        self._geometry_recovery_cancel.set()
        result, self._geometry_recovery_result = self._geometry_recovery_result, None
        if result is not None and result.source is not None:
            try:
                result.source.close()
            except Exception:
                pass
        source, self._recovered_waiting_source = self._recovered_waiting_source, None
        if source is not None:
            try:
                source.close()
            except Exception:
                pass
        self._preserve_opening_evidence_once = False
        self._geometry_post_recovery_frames_remaining = 0

    def _stop_waiting_workers(self) -> bool:
        self._retain_stopped_listener_frame()
        worker = self._waiting_capture_worker
        with self._latest_waiting_frame_lock:
            self._waiting_generation += 1
        self._opening_tracker.reset()
        self._listening_page = ListeningPageSignal("unknown", 0.0)
        self._stop_waiting_analysis_worker()
        if worker is None:
            self._close_waiting_source()
            return True
        worker.stop()
        stopped = worker.wait(0)
        if stopped:
            self._waiting_capture_worker = None
            self._close_waiting_source()
        return stopped

    def _waiting_analysis_drain_complete(self) -> bool:
        worker = self._draining_waiting_analysis_worker
        if worker is None:
            return True
        if worker.stop(timeout=0.0):
            self._draining_waiting_analysis_worker = None
            return True
        return False

    def _stop_waiting_analysis_worker(self) -> bool:
        worker, self._waiting_analysis_worker = self._waiting_analysis_worker, None
        if worker is None:
            return self._waiting_analysis_drain_complete()
        stopped = worker.stop(timeout=0.0)
        if stopped:
            return True
        self._draining_waiting_analysis_worker = worker
        return False

    def _close_waiting_source(self) -> None:
        source, self._waiting_source = self._waiting_source, None
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                self.error.emit(f"关闭监听采集源失败：{exc}")

    def _waiting_capture_finished(self, worker: WorkerHandle) -> None:
        if self._waiting_capture_worker is not worker:
            return
        self._waiting_capture_worker = None
        self._close_waiting_source()
        if self._geometry_recovery_active:
            self._schedule_geometry_recovery_attempt()
            return
        self._close_listener_recording()
        if self._pending_auto_session is not None:
            self._start_pending_auto_session()
            return

    def _start_listener_recording(self) -> bool:
        if not self._listening_page.allows_media:
            return True
        if self._listener_recording is not None:
            return True
        if self.recording_mode != "all":
            return True
        starter = getattr(self.session_factory, "start_listener_recording", None)
        if not callable(starter):
            self.error.emit("当前会话工厂不支持全程录制")
            return False
        try:
            recording = starter(recognition_strategy=self._recognition_strategy)
        except Exception as exc:
            self.error.emit(f"无法启动全程录制：{exc}")
            return False
        if recording is None:
            self.error.emit("全程录制未启动，请检查保存方式配置")
            return False
        self._listener_recording = recording
        self._bind_preopening_diagnostic_to_recording(recording)
        return True

    def _bind_preopening_diagnostic_to_recording(self, recording: object) -> None:
        """Bind diagnostic frames to the same managed episode as the recorder."""

        store = getattr(recording, "store", None)
        directory = getattr(store, "directory", None)
        session_id = str(getattr(store, "session_id", "") or "").strip()
        if directory is None or not session_id:
            return
        self._migrate_manual_diagnostic_frames_to_recording(store)
        self._preopening_diagnostic_directory = Path(directory)
        self._preopening_diagnostic_session_id = session_id

    def _rebind_diagnostic_directory(self, source: Path, target: Path, session_id: str) -> None:
        """Update saved-directory pointers only after all frame pairs migrated."""
        source_key = os.path.normcase(os.path.abspath(source))
        def matches(value):
            return value is not None and os.path.normcase(os.path.abspath(value)) == source_key
        self._listener_evidence.rebind_directory(source, target, session_id)
        retained = self._retained_diagnostic_frame
        if retained is not None and matches(retained.session_directory):
            self._retained_diagnostic_frame = replace(
                retained, session_directory=target, session_id=session_id,
                scope_id=retained.scope_id or retained.session_id,
            )
        if matches(self._preopening_diagnostic_directory):
            self._preopening_diagnostic_directory = target
            self._preopening_diagnostic_session_id = session_id
        with self._diagnostic_directory_lock:
            self._last_diagnostic_frame_directory = target / "diagnostic_frames"

    def _migrate_manual_diagnostic_frames_to_recording(self, target_store: object) -> None:
        """Move on-demand fallback frames into the active episode, best effort."""

        source_directory = self._manual_diagnostic_directory
        source_session_id = self._manual_diagnostic_session_id
        target_directory = getattr(target_store, "directory", None)
        target_session_id = str(getattr(target_store, "session_id", "") or "").strip()
        if source_directory is None or not source_session_id or target_directory is None or not target_session_id:
            return
        if (self._diagnostics_closed
                or self._listener_evidence.directory_is_pinned(source_directory)
                or not self._listener_evidence.writer.wait_idle(0)):
            return
        try:
            source_root = Path(resolve_sessions_root(
                self.capture_service.profiles_root, self.profile_name,
            )).resolve()
            source = Path(source_directory).resolve()
            source.relative_to(source_root / "manual_diagnostic")
            target = Path(target_directory).resolve()
            target.relative_to(source_root / ".preopening")
            from ..application.session_diagnostic_frames import SessionDiagnosticFrameStore

            frame_store = SessionDiagnosticFrameStore()
            records = tuple(frame_store.list_frames(source)) if source.exists() else ()
            if not records:
                return
            target_frames = frame_store.diagnostic_directory(Path(target_directory), create=True)
            for record in records:
                frame_store.copy_frame(record, Path(target_directory), session_id=target_session_id,
                                       provenance_field="original_manual_fallback_session_id")
                record.image_path.unlink(missing_ok=True)
                record.metadata_path.unlink(missing_ok=True)
            self._rebind_diagnostic_directory(source_directory, Path(target_directory), target_session_id)
            if source.exists():
                remaining = tuple(source.rglob("*"))
                if not any(path.is_file() for path in remaining):
                    shutil.rmtree(source, ignore_errors=True)
            self._manual_diagnostic_directory = None
            self._manual_diagnostic_session_id = ""
            self._manual_capture_seq = 0
        except Exception as exc:
            self.diagnostic_frame_status.emit({
                "status": "MIGRATION_SKIPPED",
                "source_phase": "preopening_listener",
                "session_directory": str(source_directory),
                "message": f"预开局截图保留在手动回退目录，迁移未完成：{exc}",
            })

    def _record_listener_frame(self, snapshot: FrameSnapshot) -> None:
        if not self._listening_page.allows_media:
            return
        recording = self._listener_recording
        record_frame = getattr(recording, "record_frame", None)
        if not callable(record_frame):
            return
        captured_ms = getattr(snapshot, "captured_monotonic_ms", None)
        last_ms = getattr(self, "_last_listener_frame_ms", None)
        if captured_ms is not None and last_ms is not None and captured_ms <= last_ms:
            return
        try:
            warning = record_frame(
                snapshot.image,
                monotonic_ms=captured_ms if captured_ms is not None else monotonic_ns() // 1_000_000,
                wall_time=snapshot.captured_at.isoformat(),
            )
            self._last_listener_frame_ms = captured_ms
            self._publish_recording_warning(warning)
        except Exception as exc:
            raise RuntimeError(f"全程录制写入失败：{exc}") from exc

    def _close_listener_recording(self) -> None:
        self._last_listener_frame_ms = None
        recording, self._listener_recording = self._listener_recording, None
        reason, self._listener_recording_stop_reason = (
            self._listener_recording_stop_reason or "listener_stopped",
            None,
        )
        close = getattr(recording, "close", None)
        if not callable(close):
            return
        try:
            close(reason=reason)
        except Exception as exc:
            self.error.emit(f"封存全程录像失败：{exc}")

    def _publish_recording_warning(self, warning: object) -> None:
        reason = str(getattr(warning, "reason", "") or "")
        if reason == "recording_capacity_reached":
            self.recording_status.emit({
                "reason": "recording_capacity_reached",
                "message": "录像容量已达上限，已停止录像，识别和推荐继续",
            })
        elif reason:
            self.recording_status.emit({
                "reason": reason,
                "message": f"录像写入异常，识别和推荐继续：{getattr(warning, 'details', '')}",
                "captured_ms": getattr(warning, "monotonic_ms", None),
            })

    def _initial_result(self, value: object) -> None:
        result, snapshot = value  # type: ignore[misc]
        self.initial_recognized.emit(result, snapshot)

    def _initial_finished(self) -> None:
        self._initial_thread = None

    def warm_danzero(self) -> None:
        """Begin the one-time local model warmup without blocking the UI."""
        self._start_danzero_warmup()

    def _start_danzero_warmup(self) -> None:
        if self._danzero_warmup_running or self._danzero_warmup_complete:
            return
        initializer = getattr(self.danzero_advisor, "initialize", None)
        if not callable(initializer):
            return
        try:
            preload_live_worker_dependencies()
        except Exception as exc:
            label = "FableDan" if self.advisor_strategy == "fabledan" else "DanZero"
            self.danzero_warmup_status.emit(
                f"{label} 模型预热失败，首次建议时将自动重试：{exc}"
            )
            return
        self._danzero_warmup_running = True
        advisor = self.danzero_advisor
        label = "FableDan" if self.advisor_strategy == "fabledan" else "DanZero"
        self.danzero_warmup_status.emit(f"{label} 模型预热中")

        def operation() -> tuple[object, float]:
            started = perf_counter()
            initializer()
            return advisor, (perf_counter() - started) * 1_000

        thread = OneShotThread(operation, self)
        thread.result.connect(self._danzero_warmup_succeeded)
        thread.error.connect(self._danzero_warmup_failed)
        thread.finished.connect(self._danzero_warmup_finished)
        self._danzero_warmup_thread = thread
        thread.start()

    def _danzero_warmup_succeeded(self, value: object) -> None:
        if isinstance(value, tuple):
            advisor, elapsed_ms = value
            if advisor is not self.danzero_advisor:
                return
        else:
            elapsed_ms = value
        self._danzero_warmup_complete = True
        label = "FableDan" if self.advisor_strategy == "fabledan" else "DanZero"
        self.danzero_warmup_status.emit(
            f"{label} 模型已就绪（首次预热 {float(elapsed_ms):.0f} ms）"
        )

    def _danzero_warmup_failed(self, message: str) -> None:
        label = "FableDan" if self.advisor_strategy == "fabledan" else "DanZero"
        self.danzero_warmup_status.emit(
            f"{label} 模型预热失败，首次建议时将自动重试：{message}"
        )

    def _danzero_warmup_finished(self) -> None:
        self._danzero_warmup_running = False
        self._danzero_warmup_thread = None

    def _migrate_preopening_diagnostic_frames(self, target_store: object | None) -> None:
        """Move completed pre-opening frame pairs into the formal session.

        This is best-effort by design: if a listener save is in flight or a
        filesystem safety check fails, the original pre-opening directory is
        left untouched and remains selectable for diagnosis.
        """

        source_directory = self._preopening_diagnostic_directory
        source_session_id = self._preopening_diagnostic_session_id
        target_directory = getattr(target_store, "directory", None)
        target_session_id = str(getattr(target_store, "session_id", "") or "").strip()
        if (
            source_directory is None
            or not source_session_id
            or target_directory is None
            or not target_session_id
        ):
            return
        if self._diagnostics_closed or self._listener_evidence.directory_is_pinned(source_directory):
            return
        diagnostic_thread = self._diagnostic_frame_thread
        if not self._listener_evidence.writer.wait_idle(0) or (
            diagnostic_thread is not None and diagnostic_thread.isRunning()
        ):
            return
        try:
            source_root = Path(resolve_sessions_root(
                self.capture_service.profiles_root,
                self.profile_name,
            )).resolve()
            source = source_directory.resolve()
            source.relative_to(source_root / ".preopening")
            target_session_directory = Path(target_directory).resolve()
            target_session_directory.relative_to(source_root)
            from ..application.session_diagnostic_frames import SessionDiagnosticFrameStore

            frame_store = SessionDiagnosticFrameStore()
            if not source.exists():
                return
            records = tuple(frame_store.list_frames(source))
            if not records:
                return
            target_frames = frame_store.diagnostic_directory(Path(target_directory), create=True)
            for record in records:
                frame_store.copy_frame(record, Path(target_directory), session_id=target_session_id,
                                       provenance_field="original_preopening_session_id")
                record.image_path.unlink(missing_ok=True)
                record.metadata_path.unlink(missing_ok=True)
            self._rebind_diagnostic_directory(source_directory, Path(target_directory), target_session_id)
            # Non-recursive cleanup only; retain episode manifests and traces.
            for empty in (source / "diagnostic_frames", source):
                try:
                    empty.rmdir()
                except OSError:
                    pass

        except Exception as exc:
            self.diagnostic_frame_status.emit({
                "status": "MIGRATION_SKIPPED",
                "source_phase": "preopening_listener",
                "session_directory": str(source_directory),
                "message": f"预开局截图保留在原目录，迁移未完成：{exc}",
            })

    def start_session(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: str | None,
        recognition_strategy: str = "two_valid_streak",
        opening_action: _OpeningActionSeed | None = None,
    ) -> bool:
        try:
            preload_live_worker_dependencies()
        except Exception as exc:
            self.error.emit(f"实时依赖预加载失败：{exc}")
            return False
        self._start_danzero_warmup()
        if self.orchestrator is not None:
            self.error.emit("当前已有实时对局")
            return False
        try:
            constructed = self.session_factory.start_session(
                round_level=round_level,
                hand=hand,
                lead_player=lead_player,
                recognition_strategy=recognition_strategy,
                on_update=self._queue_orchestrator_update,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return False
        self.orchestrator = constructed.orchestrator
        self._live_source = constructed.source
        token = self._activate_live_token(constructed.orchestrator)
        self._migrate_preopening_diagnostic_frames(
            getattr(constructed.orchestrator, "store", None)
        )
        try:
            # Binding must precede bootstrap/model notifications and capture,
            # including waiting-lead sessions which have not analyzed a frame.
            initial_update = self._bind_capture_token(token)
        except Exception as exc:
            self._abort_started_session(token, constructed.source)
            self.error.emit(f"绑定采集代次失败：{exc}")
            return False
        if opening_action is not None:
            try:
                initial_update = self.orchestrator.bootstrap_opening_action(
                    actor=opening_action.actor,
                    cards=opening_action.cards,
                    expected_next_player=opening_action.next_player,
                    monotonic_ms=monotonic_ns() // 1_000_000,
                    confidence=opening_action.confidence,
                    source=opening_action.source,
                    suit_options=opening_action.suit_options,
                )
            except Exception as exc:
                self._abort_started_session(token, constructed.source)
                self.error.emit(f"首出动作锚定失败：{exc}")
                return False
        self.opening_evidence.mark_session_started()
        self._auto_finish_requested = False
        with self._latest_live_frame_lock:
            self._latest_live_frame = None
            self._latest_live_frame_generation = 0
            self._latest_live_frame_capture_seq = -1
        self._pending_preselection_task = None
        self._handled_preselection_request_ids.clear()
        self._authoritative_advice_key = None
        self._authoritative_advice_generation = self._capture_generation
        self._emit_authoritative_update(initial_update)
        self._start_analysis_worker()
        self._start_capture_worker()
        return True

    def _bind_capture_token(self, token: _LiveRunToken) -> LiveUpdate:
        binder = getattr(token.orchestrator, "bind_capture_generation", None)
        if not callable(binder):
            raise RuntimeError("实时核心不支持采集代次绑定")
        bound = binder(token.generation)
        if (not isinstance(bound, LiveUpdate)
                or type(bound.capture_generation) is not int
                or bound.capture_generation != token.generation
                or str(getattr(bound.snapshot, "session_id", "")) != token.session_id
                or type(bound.update_sequence) is not int or bound.update_sequence <= 0):
            raise RuntimeError("实时核心返回了无效的采集代次绑定结果")
        return bound

    def _abort_started_session(self, token: _LiveRunToken, source: object) -> None:
        # No capture/analysis worker has been started at this point. Invalidate
        # first so queued callbacks from partial bootstrap/finish cannot publish.
        if self._active_live_token == token:
            self._invalidate_live_token()
        self._stop_analysis_worker()
        self._live_source = None
        if self.orchestrator is token.orchestrator:
            self.orchestrator = None
        try:
            finalizing = getattr(token.orchestrator, "begin_finalizing", None)
            if callable(finalizing):
                finalizing()
        except Exception:
            token.pipeline_timing.increment("failed_start_cancel_error")
        try:
            token.orchestrator.finish()
        except Exception:
            token.pipeline_timing.increment("failed_start_finish_error")
        try:
            source.close()
        except Exception:
            token.pipeline_timing.increment("failed_start_source_close_error")

    def _activate_live_token(self, orchestrator: LiveRuntimePort) -> _LiveRunToken:
        self._capture_generation += 1
        self._live_session_nonce += 1
        token = _LiveRunToken(
            orchestrator=orchestrator,
            session_id=str(orchestrator.snapshot.session_id),
            nonce=self._live_session_nonce,
            generation=self._capture_generation,
            pipeline_timing=getattr(
                getattr(orchestrator, "store", None), "pipeline_timing", PipelineTiming()
            ),
        )
        self._active_live_token = token
        self._fatal_capture_session_id = None
        recorder = getattr(orchestrator, "recorder", None)
        if hasattr(recorder, "pipeline_timing"):
            recorder.pipeline_timing = token.pipeline_timing
        return token

    def _invalidate_live_token(self) -> None:
        self._retain_stopped_listener_frame()
        self._capture_generation += 1
        self._live_session_nonce += 1
        self._active_live_token = None
        self._authoritative_advice_key = None
        self._authoritative_advice_generation = self._capture_generation
        self._pending_preselection_task = None

    def _live_token_is_current(self, token: _LiveRunToken) -> bool:
        return bool(
            self._active_live_token == token
            and self.orchestrator is token.orchestrator
            and str(token.orchestrator.snapshot.session_id) == token.session_id
            and token.generation == self._capture_generation
        )

    def _start_analysis_worker(self) -> None:
        token = self._active_live_token
        if token is None or self._analysis_worker is not None:
            return
        worker = LatestOnlyWorker(
            lambda value, current=token: self._run_analysis_task(current, value),
            on_result=lambda update, current=token: self._accept_analysis_update(
                current, update
            ),
            on_error=lambda exc, current=token: self._accept_analysis_error(
                current, exc
            ),
            on_discard=lambda value, reason, current=token: current.pipeline_timing.increment(
                f"analysis_discard_{reason}"
            ),
        )
        self._analysis_worker = worker
        worker.start()

    def _run_analysis_task(self, token: _LiveRunToken, value: object) -> _AnalysisDelivery:
        try:
            update = self._analyze_live_frame(token, value)
        except Exception as exc:
            if self._live_token_is_current(token):
                self._queue_listener_incident(
                    "live_analysis_failed",
                    context=value.evidence if isinstance(value, _AnalysisFrameTask) else None,
                    error=exc, allow_latest=False,
                )
            raise
        if (not self._diagnostics_closed and self._live_token_is_current(token)
                and isinstance(value, _AnalysisFrameTask)
                and getattr(update, "status", None) == "running"
                and not getattr(update, "block_reason", "")):
            processed_seq = getattr(update, "processed_capture_seq", None)
            processed_ms = getattr(update, "processed_captured_ms", None)
            if isinstance(processed_seq, int) and isinstance(processed_ms, int):
                processed = self._listener_evidence.find_processed(
                    generation=token.generation, capture_seq=processed_seq,
                    captured_ms=processed_ms, phase="live_session",
                    scope_id=self._diagnostic_scope_id(token=token),
                )
                if processed is not None:
                    processed = self._listener_evidence.annotate(processed,
                        processing={"status": "vision_completed", "capture_seq": processed_seq,
                                    "captured_ms": processed_ms})
                    self._listener_evidence.recover(processed)
        return _AnalysisDelivery(
            token, update, monotonic_ns(),
            value.capture_started_ns if isinstance(value, _AnalysisFrameTask) else 0,
        )

    def _analyze_live_frame(
        self,
        token: _LiveRunToken,
        value: object,
    ) -> LiveUpdate | None:
        task = value
        if not isinstance(task, _AnalysisFrameTask) or task.token != token:
            token.pipeline_timing.increment("analysis_invalid_task")
            return None
        if not self._live_token_is_current(token):
            token.pipeline_timing.increment("analysis_stale_before_start")
            return None
        timing = token.pipeline_timing
        dequeued_ns = monotonic_ns()
        if task.submitted_ns:
            timing.observe("analysis_queue", (dequeued_ns - task.submitted_ns) / 1_000_000)
        self._preview_local_controls(token, task)
        if not self._live_token_is_current(token):
            timing.increment("analysis_stale_before_compute")
            return None
        started_ns = monotonic_ns()
        timing.increment("analysis_started")
        try:
            envelope = task.snapshot.to_envelope(
                capture_seq=task.capture_seq,
                capture_generation=token.generation,
            )
            update = analyze_frame_envelope(
                token.orchestrator,
                envelope,
                trace_context={
                    "worker_token": {
                        "session_id": token.session_id,
                        "nonce": token.nonce,
                        "generation": token.generation,
                    },
                    "captured_ms": task.captured_ms,
                },
            )
        except Exception:
            timing.increment("analysis_failed")
            raise
        finally:
            timing.elapsed("analysis", started_ns)
        if not self._live_token_is_current(token):
            timing.increment("analysis_stale_after_finish")
            return None
        timing.increment("analysis_completed")
        advice = getattr(update, "advice", None)
        if advice is None or not getattr(advice, "visible", False):
            # Analysis results are not genuine self-turn opportunities.
            timing.increment("analysis_result_without_visible_advice")
        return update

    def _preview_local_controls(self, token: _LiveRunToken, task: _AnalysisFrameTask) -> None:
        """Run one read-only narrow scan before canonical frame analysis.

        Do not substitute this expected-self result for canonical recognition.
        Compatibility recognizers without this narrow API simply skip it.
        """
        service = getattr(token.orchestrator, "recognition_service", None)
        recognize = getattr(service, "recognize_local_controls", None)
        preview = getattr(token.orchestrator, "preview_controls", None)
        timing = token.pipeline_timing
        if not callable(recognize) or not callable(preview):
            timing.increment("controls_preview_unsupported")
            return
        if not self._live_token_is_current(token):
            timing.increment("controls_preview_stale")
            return
        started_ns = monotonic_ns()
        try:
            fast = recognize(task.snapshot.image)
            timing.increment("controls_preview_scanned")
            if not self._live_token_is_current(token):
                timing.increment("controls_preview_stale")
                return
            update = preview(
                fast, captured_ms=task.captured_ms, capture_generation=token.generation,
                frame_size=(task.snapshot.image.shape[1], task.snapshot.image.shape[0]),
            )
            if update is not None:
                timing.increment("controls_preview_changed")
        except Exception:
            # Narrow hint failures must not abort the recording receipt or
            # canonical analysis; stale hints have a bounded visual TTL.
            timing.increment("controls_preview_failed")
        finally:
            timing.elapsed("controls_preview", started_ns)

    def _accept_analysis_update(
        self,
        token: _LiveRunToken,
        update: object,
    ) -> None:
        delivery = update if isinstance(update, _AnalysisDelivery) else _AnalysisDelivery(
            token, update if isinstance(update, LiveUpdate) else None, monotonic_ns()
        )
        self._enqueue_gui_delivery(delivery)

    def _queue_orchestrator_update(self, update: LiveUpdate) -> None:
        token = self._active_live_token
        if token is not None and str(update.snapshot.session_id) == token.session_id:
            self._enqueue_gui_delivery(_AnalysisDelivery(token, update, monotonic_ns()))

    def _enqueue_gui_delivery(self, delivery: _AnalysisDelivery) -> None:
        if not self._live_token_is_current(delivery.token):
            delivery.token.pipeline_timing.increment("gui_stale_delivery")
            return
        timing = delivery.token.pipeline_timing
        update = delivery.update
        if not isinstance(update, LiveUpdate):
            # A completed no-result is telemetry, never a replacement for an
            # actionable queued update (especially a safety-blocked state).
            timing.increment("gui_no_result_delivery")
            return
        if not self._update_has_capture_identity(update, delivery.token):
            timing.increment("gui_wrong_update_identity")
            return
        sequence = update.update_sequence if type(update.update_sequence) is int else -1
        turn = getattr(update.snapshot, "turn_id", -1)
        revision = getattr(update.snapshot, "revision", -1)
        turn = turn if type(turn) is int else -1
        revision = revision if type(revision) is int else -1
        if sequence < 0:
            timing.increment("gui_invalid_update_sequence")
            return
        emit = False
        with self._gui_delivery_lock:
            previous = self._gui_accepted_version
            if previous is not None and previous[0] == delivery.token:
                _, prior_sequence, prior_turn, prior_revision = previous
                if ((prior_sequence > 0 and sequence <= prior_sequence)
                        or turn < prior_turn or revision < prior_revision):
                    timing.increment("gui_older_update_rejected")
                    return
            self._gui_accepted_version = (delivery.token, sequence, turn, revision)
            if self._pending_gui_delivery is not None:
                self._pending_gui_delivery.token.pipeline_timing.increment("gui_delivery_replaced")
            self._pending_gui_delivery = delivery
            if not self._gui_delivery_scheduled:
                self._gui_delivery_scheduled = True
                emit = True
        if emit:
            self._analysis_delivered.emit()

    @Slot()
    def _receive_queued_delivery(self) -> None:
        with self._gui_delivery_lock:
            delivery, self._pending_gui_delivery = self._pending_gui_delivery, None
            self._gui_delivery_scheduled = False
        self._deliver_analysis_update(delivery)

    @Slot(object)
    def _deliver_analysis_update(self, delivery: object) -> None:
        if not isinstance(delivery, _AnalysisDelivery):
            return
        timing = delivery.token.pipeline_timing
        if not self._live_token_is_current(delivery.token):
            timing.increment("gui_stale_delivery")
            return
        if not isinstance(delivery.update, LiveUpdate):
            timing.increment("gui_no_result_delivery")
            return
        if not self._update_has_capture_identity(delivery.update, delivery.token):
            timing.increment("gui_wrong_update_identity")
            return
        received_ns = monotonic_ns()
        timing.observe("gui_delivery", (received_ns - delivery.emitted_ns) / 1_000_000)
        if delivery.capture_started_ns:
            timing.observe(
                "capture_to_gui_receive", (received_ns - delivery.capture_started_ns) / 1_000_000
            )
        timing.increment("gui_update_received")
        self._emit_authoritative_update(delivery.update)

    @staticmethod
    def _advice_key_for_update(update: LiveUpdate) -> AdviceRequestKey:
        snapshot = update.snapshot
        return AdviceRequestKey(
            str(getattr(snapshot, "session_id", "") or ""),
            int(getattr(snapshot, "turn_id", 0) or 0),
            int(getattr(snapshot, "revision", 0) or 0),
        )

    @classmethod
    def _has_current_visible_advice(cls, update: LiveUpdate) -> bool:
        raw = update.advice
        if (
            update.status != "running"
            or getattr(update.snapshot, "current_player", None) != "self"
            or not isinstance(raw, LiveAdvice)
            or raw.status != "ready"
            or not raw.visible
            or raw.advice is None
        ):
            return False
        return raw.key == cls._advice_key_for_update(update)

    @staticmethod
    def _update_has_committed_action(update: LiveUpdate) -> bool:
        events = tuple(getattr(update, "events", ()) or ())
        event = getattr(update, "event", None)
        if event is not None:
            events = (event, *events)
        return any(
            str(getattr(item, "event_type", "") or "")
            in {"player_played", "player_passed", "action_committed"}
            for item in events
        )

    @classmethod
    def _advice_invalidation_reason(cls, update: LiveUpdate) -> str:
        status = str(getattr(update, "status", "") or "")
        if status in {"finalizing", "sealed"}:
            return "terminal"
        event = getattr(update, "event", None)
        events = tuple(getattr(update, "events", ()) or ())
        if event is not None:
            events = (event, *events)
        if any(
            str(getattr(item, "event_type", "") or "")
            in {"game_end_detected", "terminal_detected", "session_finished"}
            for item in events
        ):
            return "terminal"
        block_reason = str(getattr(update, "block_reason", "") or "")
        raw = update.advice
        withhold_reason = str(getattr(raw, "withhold_reason", "") or "")
        if block_reason or withhold_reason or getattr(update, "missing_player", None) is not None:
            return withhold_reason or block_reason or "recovery"
        if cls._update_has_committed_action(update):
            return "action_committed"
        if getattr(update.snapshot, "current_player", None) != "self":
            return "not_local_turn"
        return ""

    def _authoritative_update_for_ui(self, update: LiveUpdate) -> LiveUpdate:
        """Make the public update the only source of visible advice.

        A late/empty update must actively invalidate the previous visible
        recommendation.  The runtime remains untouched: this is only a GUI
        projection, and the synthetic withheld advice makes both the full and
        compact views clear their existing card in the same update turn.
        """
        key = self._advice_key_for_update(update)
        generation = int(getattr(update, "capture_generation", 0) or 0)
        reason = self._advice_invalidation_reason(update)
        current_visible = self._has_current_visible_advice(update) and not reason
        identity_changed = (
            self._authoritative_advice_key != key
            or self._authoritative_advice_generation != generation
        )
        had_visible = self._authoritative_advice_key is not None
        if current_visible:
            if identity_changed:
                self._pending_preselection_task = None
            self._authoritative_advice_key = key
            self._authoritative_advice_generation = generation
            return update

        if had_visible or identity_changed or reason:
            self._authoritative_advice_key = None
            self._authoritative_advice_generation = generation
            self._pending_preselection_task = None
            raw = update.advice
            if not isinstance(raw, LiveAdvice) or raw.status != "withheld" or raw.visible:
                withheld = LiveAdvice(
                    key=key,
                    status="withheld",
                    advice=None,
                    visible=False,
                    withhold_reason=reason or "advice_invalidated",
                    error="当前状态已变化，旧推荐已清除",
                )
                return replace(update, advice=withheld)
        return update

    def _emit_authoritative_update(self, update: object) -> None:
        if isinstance(update, LiveUpdate):
            update = self._authoritative_update_for_ui(update)
        self.update_ready.emit(update)

    @staticmethod
    def _update_has_capture_identity(update: LiveUpdate, token: _LiveRunToken) -> bool:
        if str(getattr(update.snapshot, "session_id", "")) != token.session_id:
            return False
        generation = update.capture_generation
        if type(generation) is not int or generation < 0:
            return False
        if generation > 0:
            return generation == token.generation
        # Only deliberately unversioned legacy test adapters remain accepted
        # unchanged. Production callbacks must already carry their generation;
        # never attach current authority to an unowned/old callback.
        return (not callable(getattr(token.orchestrator, "bind_capture_generation", None))
                and type(update.update_sequence) is int and update.update_sequence == 0)

    def _queue_fatal_worker_fault(self, token: _LiveRunToken, *, kind: str, message: str) -> None:
        if not self._live_token_is_current(token):
            token.pipeline_timing.increment("stale_worker_fault")
            return
        identity = (token.session_id, token.generation)
        with self._gui_delivery_lock:
            if self._queued_fault_identity == identity:
                return
            self._queued_fault_identity = identity
        # At most one fatal signal per capture identity. The GUI rechecks the
        # token before stopping workers, so a delayed failure cannot stop a
        # replacement capture or overwrite its recommendation.
        self._queue_listener_incident("worker_" + kind, error=message)
        self._fatal_worker_fault_requested.emit(_FatalWorkerFault(token, kind, message))

    @Slot(object)
    def _deliver_fatal_worker_fault(self, fault: object) -> None:
        if not isinstance(fault, _FatalWorkerFault):
            return
        token = fault.token
        if not self._live_token_is_current(token):
            token.pipeline_timing.increment("stale_worker_fault_delivery")
            return
        self.live_fault.emit({
            "session_id": token.session_id,
            "capture_generation": token.generation,
            "kind": fault.kind,
        })
        self._fatal_capture_session_id = token.session_id
        self._resume_requested = False
        self._stop_capture_worker()
        self._stop_analysis_worker()
        token.pipeline_timing.increment("fatal_worker_fault")
        self.error.emit(fault.message)

    def _accept_analysis_error(self, token: _LiveRunToken, exc: Exception) -> None:
        if not self._live_token_is_current(token):
            token.pipeline_timing.increment("stale_analysis_error")
            return
        message = str(exc)
        try:
            update = token.orchestrator.analysis_failed(
                message,
                monotonic_ms=monotonic_ns() // 1_000_000,
            )
        except Exception as incident_exc:
            self._queue_fatal_worker_fault(
                token, kind="analysis",
                message=f"{message}; 创建识别事故失败：{incident_exc}",
            )
            return
        self._accept_analysis_update(token, update)
        if self._live_token_is_current(token):
            self.error.emit(message)

    def _start_capture_worker(self) -> None:
        token = self._active_live_token
        if token is None or self._live_source is None or self.is_running:
            return
        source = self._live_source
        analysis = self._analysis_worker
        capture_seq = 0
        last_analysis_fingerprint: bytes | None = None

        def operation():
            nonlocal capture_seq, last_analysis_fingerprint
            timing = token.pipeline_timing
            capture_started_ns = monotonic_ns()
            timing.increment("capture_started")
            try:
                snapshot: FrameSnapshot = source.capture()
            except Exception:
                timing.increment("capture_failed")
                raise
            finally:
                timing.elapsed("capture", capture_started_ns)
            capture_seq += 1
            captured_ms = snapshot.captured_monotonic_ms
            if not self._live_token_is_current(token):
                timing.increment("capture_stale")
                return _LiveFrameDelivery(snapshot, capture_seq)
            evidence = self._retain_listener_snapshot(snapshot, token.generation, capture_seq, token=token)
            timing.increment("capture_completed")
            submit_for_analysis = True
            if self.deduplicate_analysis_frames:
                fingerprint = hashlib.blake2b(
                    memoryview(snapshot.image),
                    digest_size=16,
                ).digest()
                submit_for_analysis = fingerprint != last_analysis_fingerprint
                last_analysis_fingerprint = fingerprint
                if not submit_for_analysis:
                    timing.increment("analysis_deduplicated")
            if (
                submit_for_analysis
                and self._live_token_is_current(token)
                and analysis is not None
            ):
                submitted_ns = monotonic_ns()
                analysis.submit(
                    _AnalysisFrameTask(
                        token, snapshot, capture_seq, captured_ms,
                        capture_started_ns, submitted_ns, evidence,
                    ),
                    preserve=token.orchestrator.needs_first_action_frames,
                    max_preserved=8,
                )
                timing.elapsed("analysis_submit", submitted_ns)
                timing.observe("capture_to_submit", (submitted_ns - capture_started_ns) / 1_000_000)
                timing.increment("analysis_submitted")
            self._submit_recording_frame(
                token, snapshot, capture_sequence=capture_seq,
            )
            return _LiveFrameDelivery(snapshot, capture_seq)

        worker = WorkerHandle(operation, self.capture_interval_sec)
        worker.frame_ready.connect(
            lambda value, current=token: self._accept_live_frame(current, value)
        )
        worker.error.connect(
            lambda error, current=token: self._accept_live_error(current, error)
        )
        worker.finished.connect(lambda: self._capture_finished(worker, token))
        self._capture_worker = worker
        self._resume_requested = False
        worker.start()

    def _start_recording_dispatcher(self, token: _LiveRunToken | None) -> None:
        if self._recording_dispatcher is not None:
            return
        if token is None or not self._live_token_is_current(token):
            return
        orchestrator = token.orchestrator
        recorder = getattr(orchestrator, "recorder", None)
        if recorder is not None and not isinstance(
            recorder, RecordingDropAccountingAdapter
        ):
            recorder = RecordingDropAccountingAdapter(recorder)
            orchestrator.recorder = recorder
        target_fps = float(getattr(recorder, "fps", 10.0) or 10.0)
        self._recording_cadence = RecordingCadenceGate(target_fps)

        def operation(frame: RecordingFrame) -> object | None:
            current = frame.context if isinstance(frame.context, _LiveRunToken) else token
            timing = current.pipeline_timing
            started_ns = monotonic_ns()
            timing.increment("recording_started")
            try:
                warning = orchestrator.record_frame(
                    frame.image,
                    monotonic_ms=frame.captured_ms,
                    wall_time=frame.wall_time,
                )
            finally:
                timing.elapsed("record_frame", started_ns)
            timing.increment("recording_written")
            self._flush_pipeline_timing(current)
            return warning

        dispatcher = BoundedRecordingDispatcher(
            operation,
            capacity=16,
            on_warning=lambda frame, warning: self._recording_warning(frame, warning),
            on_error=lambda frame, error: self._recording_error(frame, error),
            on_drop=lambda frame, reason: self._recording_dropped(frame, reason),
        )
        self._recording_dispatcher = dispatcher
        dispatcher.start()

    def _submit_recording_frame(
        self,
        token: _LiveRunToken,
        snapshot: FrameSnapshot,
        *,
        capture_sequence: int,
    ) -> None:
        if not self._live_token_is_current(token):
            token.pipeline_timing.increment("recording_dropped_stale")
            return
        dispatcher = self._recording_dispatcher
        if dispatcher is None:
            self._start_recording_dispatcher(token)
            dispatcher = self._recording_dispatcher
        if dispatcher is None or not self._live_token_is_current(token):
            token.pipeline_timing.increment("recording_dispatcher_unavailable")
            return
        cadence = self._recording_cadence
        if cadence is not None and not cadence.admit(snapshot.captured_monotonic_ms):
            token.pipeline_timing.increment("recording_sampled_out")
            return
        frame = RecordingFrame(
            snapshot.image,
            snapshot.captured_monotonic_ms,
            snapshot.captured_at.isoformat(),
            capture_sequence,
            token,
        )
        try:
            no_eviction = dispatcher.submit(frame)
        except RuntimeError:
            token.pipeline_timing.increment("recording_dropped_closing")
            self._flush_pipeline_timing(token)
            return
        token.pipeline_timing.increment("recording_submitted")
        if not no_eviction:
            token.pipeline_timing.increment("recording_dropped_capacity")

    def _recording_warning(self, frame: RecordingFrame, warning: object) -> None:
        self._publish_recording_warning(warning)

    def _recording_error(self, frame: RecordingFrame, error: Exception) -> None:
        token = frame.context if isinstance(frame.context, _LiveRunToken) else None
        if token is not None:
            token.pipeline_timing.increment("recording_failed")
            self._flush_pipeline_timing(token)
        self.recording_status.emit({
            "reason": "recording_failed",
            "message": f"录像写入失败，识别和推荐继续：{error}",
            "capture_sequence": frame.capture_sequence,
            "captured_ms": frame.captured_ms,
        })

    def _recording_dropped(self, frame: RecordingFrame, reason: str) -> None:
        token = frame.context if isinstance(frame.context, _LiveRunToken) else None
        if token is not None:
            token.pipeline_timing.increment(f"recording_dispatcher_drop_{reason}")
            self._flush_pipeline_timing(token)

    @staticmethod
    def _flush_pipeline_timing(token: _LiveRunToken) -> None:
        flush = getattr(getattr(token.orchestrator, "store", None), "flush_pipeline_timing", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                # Diagnostic persistence is never an admission dependency for
                # capture, recording, or recognition.
                pass

    def _close_recording_dispatcher(
        self,
        token: _LiveRunToken | None = None,
    ) -> bool:
        dispatcher = self._recording_dispatcher
        if dispatcher is None:
            return True
        # ProcessSessionRecorder bounds a write timeout plus terminate/kill.
        # Wait long enough for that fail-closed path; never seal while the
        # dispatcher may still be inside record_frame().
        stats = dispatcher.close(drain_timeout=0.25, stop_timeout=7.0)
        cadence, self._recording_cadence = self._recording_cadence, None
        cadence_document = cadence.stats.to_dict() if cadence is not None else None
        token = token or self._active_live_token
        if token is not None:
            accounting = stats.to_dict()
            handoff = getattr(
                getattr(token.orchestrator, "recorder", None),
                "accept_dispatcher_stats",
                None,
            )
            if callable(handoff):
                supplied = handoff(stats)
                if isinstance(supplied, dict):
                    accounting = supplied
            cadence_handoff = getattr(
                getattr(token.orchestrator, "recorder", None),
                "accept_recording_cadence", None,
            )
            if callable(cadence_handoff) and cadence_document is not None:
                cadence_handoff(cadence_document)
            update_metadata = getattr(
                getattr(token.orchestrator, "store", None),
                "update_session_metadata",
                None,
            )
            if callable(update_metadata):
                try:
                    update_metadata({"recording_dispatcher": accounting})
                    if cadence_document is not None:
                        update_metadata({"recording_cadence": cadence_document})
                except Exception:
                    token.pipeline_timing.increment(
                        "recording_dispatcher_metadata_failed"
                    )
        if token is not None:
            token.pipeline_timing.increment("recording_dispatcher_closed")
            if stats.dropped_close:
                token.pipeline_timing.increment(
                    "recording_dispatcher_close_discarded", stats.dropped_close
                )
            if stats.running:
                token.pipeline_timing.increment("recording_dispatcher_stop_timeout")
            self._flush_pipeline_timing(token)
        if stats.running:
            self.recording_status.emit({
                "reason": "recording_stop_timeout",
                "message": "录像线程未在限定时间内停止；本局录像已标记失败，daemon线程不会阻止程序退出",
                "pending": stats.pending,
                "inflight": stats.inflight,
            })
            return False
        if self._recording_dispatcher is dispatcher:
            self._recording_dispatcher = None
        return True

    def _accept_live_frame(self, token: _LiveRunToken, value: object) -> None:
        if isinstance(value, _LiveFrameDelivery):
            snapshot = value.snapshot
            capture_seq = value.capture_seq
        elif isinstance(value, FrameSnapshot):
            # Keep compatibility with tests/plugins that still deliver only a
            # FrameSnapshot.  Do not invent a sequence for that path.
            snapshot = value
            capture_seq = -1
        else:
            return
        if self._live_token_is_current(token):
            self._retain_listener_snapshot(snapshot, token.generation, int(capture_seq), token=token)
            # The public signal remains FrameSnapshot-only for existing UI
            # consumers; the sequence is kept in controller sidecar state.
            self.frame_ready.emit(snapshot)

    def _schedule_hand_preselection(self, update: object) -> None:
        """Sidecar entry point: queue one all-or-nothing hand selection.

        This intentionally consumes only the public ``LiveUpdate``.  It never
        writes a live event or state, and every failed guard returns before an
        OS input adapter is reached.
        """

        if not isinstance(update, LiveUpdate):
            return
        raw = update.advice
        snapshot = update.snapshot
        if (
            update.status != "running"
            or getattr(snapshot, "current_player", None) != "self"
            or not isinstance(raw, LiveAdvice)
            or raw.status != "ready"
            or not raw.visible
            or raw.advice is None
            or raw.advice.is_pass
        ):
            return
        expected_key = AdviceRequestKey(
            str(getattr(snapshot, "session_id", "")),
            int(getattr(snapshot, "turn_id", 0) or 0),
            int(getattr(snapshot, "revision", 0) or 0),
        )
        if raw.key != expected_key:
            return
        request_id = raw.key.request_id
        if request_id in self._handled_preselection_request_ids:
            return
        self._handled_preselection_request_ids.add(request_id)
        frame = self._latest_live_frame
        if frame is None or self._latest_live_frame_generation != self._capture_generation:
            self._publish_preselection_result(
                PreselectionResult(
                    request_id=request_id,
                    status="rejected",
                    detail="没有与当前对局匹配的新截图，已拒绝预选",
                )
            )
            return
        advice_request_id = str(getattr(raw.advice, "request_id", "") or "")
        advice_revision = int(getattr(raw.advice, "state_revision", -1))
        if (
            advice_request_id != request_id
            or advice_revision != raw.key.state_revision
        ):
            self._publish_preselection_result(
                PreselectionResult(
                    request_id=request_id,
                    status="rejected",
                    detail="推荐结果与当前请求不一致，已拒绝预选",
                )
            )
            return
        task = _PreselectionTask(
            request_id=request_id,
            key=raw.key,
            advice_cards=tuple(str(card) for card in raw.advice.cards),
            expected_hand=tuple(str(card) for card in snapshot.my_hand),
            frame=self._copy_live_snapshot(frame),
            capture_generation=self._capture_generation,
        )
        if self._preselection_thread is not None and self._preselection_thread.isRunning():
            # A previous full-hand scan cannot be interrupted safely.  Keep
            # only the newest eligible turn and still mark it handled so a
            # repeated UI update cannot cause duplicate input.
            self._pending_preselection_task = task
            return
        self._start_hand_preselection_recognition(task)

    def _start_hand_preselection_recognition(self, task: _PreselectionTask) -> None:
        if not self._preselection_task_is_current(task):
            self._publish_preselection_result(
                PreselectionResult(
                    request_id=task.request_id,
                    status="rejected",
                    detail="推荐或牌局状态已过期，已拒绝预选",
                )
            )
            return
        self._active_preselection_task = task

        def operation() -> object:
            return self.recognition_service.recognize(task.frame.image)

        thread = OneShotThread(operation, self)
        thread.result.connect(
            lambda recognition, value=task: self._complete_hand_preselection_recognition(
                value, recognition
            )
        )
        thread.error.connect(
            lambda message, value=task: self._fail_hand_preselection_recognition(
                value, message
            )
        )
        thread.finished.connect(
            lambda value=thread: self._hand_preselection_thread_finished(value)
        )
        self._preselection_thread = thread
        thread.start()

    def _complete_hand_preselection_recognition(
        self,
        task: _PreselectionTask,
        recognition: object,
    ) -> None:
        if not self._preselection_task_is_current(task):
            self._publish_preselection_result(
                PreselectionResult(
                    request_id=task.request_id,
                    status="rejected",
                    detail="推荐、牌局状态或截图已过期，已拒绝预选",
                )
            )
            return
        planner = HandPreselectionPlanner()
        try:
            planned = planner.plan(
                request_id=task.request_id,
                advice_cards=task.advice_cards,
                expected_hand=task.expected_hand,
                recognition=recognition,  # type: ignore[arg-type]
                frame=task.frame,
            )
        except Exception as exc:
            self._publish_preselection_result(
                PreselectionResult(
                    request_id=task.request_id,
                    status="failed",
                    detail=f"手牌定位失败：{exc}",
                )
            )
            return
        if planned.status != "planned" or planned.plan is None:
            self._publish_preselection_result(planned)
            return
        if not self._preselection_task_is_current(task):
            self._publish_preselection_result(
                PreselectionResult(
                    request_id=task.request_id,
                    status="rejected",
                    detail="推荐、牌局状态或截图已过期，已拒绝预选",
                )
            )
            return
        self._publish_preselection_result(planned)
        adapter = self._get_hand_preselector()
        if adapter is None:
            self._publish_preselection_result(
                PreselectionResult(
                    request_id=task.request_id,
                    status="rejected",
                    detail="无法初始化游戏窗口预选适配器",
                )
            )
            return
        try:
            result = adapter.preselect_hand_cards(planned.plan)
        except Exception as exc:
            result = PreselectionResult(
                request_id=task.request_id,
                status="failed",
                detail=f"自动预选失败：{exc}",
            )
        self._publish_preselection_result(result)

    def _fail_hand_preselection_recognition(
        self,
        task: _PreselectionTask,
        message: str,
    ) -> None:
        self._publish_preselection_result(
            PreselectionResult(
                request_id=task.request_id,
                status="failed",
                detail=f"手牌定位失败：{message}",
            )
        )

    def _hand_preselection_thread_finished(self, thread: OneShotThread) -> None:
        if self._preselection_thread is not thread:
            return
        self._preselection_thread = None
        self._active_preselection_task = None
        pending, self._pending_preselection_task = self._pending_preselection_task, None
        if pending is not None:
            self._start_hand_preselection_recognition(pending)

    def _preselection_task_is_current(self, task: _PreselectionTask) -> bool:
        if task.capture_generation != self._capture_generation:
            return False
        frame_age = (datetime.now().astimezone() - task.frame.captured_at).total_seconds()
        if frame_age < -1.0 or frame_age > 2.0:
            return False
        orchestrator = self.orchestrator
        if orchestrator is None or orchestrator.status != "running":
            return False
        snapshot = orchestrator.snapshot
        if (
            snapshot.current_player != "self"
            or AdviceRequestKey(snapshot.session_id, snapshot.turn_id, snapshot.revision)
            != task.key
        ):
            return False
        # Only the controller's last authoritative update can keep a
        # preselection task alive.  The orchestrator's mutable sidecar advice
        # is deliberately not a GUI authority.
        return bool(
            self._authoritative_advice_key is not None
            and self._authoritative_advice_generation == task.capture_generation
            and self._authoritative_advice_key == task.key
        )

    def _get_hand_preselector(self):
        if self._hand_preselector is not None:
            return self._hand_preselector
        try:
            loaded = self.capture_service.load_profile(self.profile_name)
            keywords = tuple(loaded.config.window_title_keywords)
        except Exception:
            return None
        self._hand_preselector = Win32HandPreselector(keywords)
        return self._hand_preselector

    def _publish_preselection_result(self, result: PreselectionResult) -> None:
        # Preselection is an execution/diagnostic signal only.  It is never
        # retained as a second recommendation cache.
        self.preselection_result.emit(result)

    def _accept_live_error(self, token: _LiveRunToken, error: object) -> None:
        if not self._live_token_is_current(token):
            return
        self._queue_listener_incident("live_capture_failed", error=error)
        message = str(error)
        self._stop_analysis_worker()
        try:
            if token.orchestrator.status not in {"finalizing", "sealed"}:
                update = token.orchestrator.capture_interrupted(
                    message,
                    monotonic_ms=monotonic_ns() // 1_000_000,
                )
                self._accept_analysis_update(token, update)
        except Exception as incident_exc:
            self._queue_fatal_worker_fault(
                token,
                kind="occluded" if getattr(error, "code", "") == "CAPTURE-OCCLUDED" else "capture",
                message=f"{message}; 创建采集中断事故失败：{incident_exc}",
            )
            return
        self.error.emit(message)

    def _capture_finished(
        self,
        worker: WorkerHandle,
        token: _LiveRunToken | None = None,
    ) -> None:
        if self._capture_worker is worker:
            self._capture_worker = None
        deferred, self._deferred_source_close = self._deferred_source_close, None
        if deferred is not None:
            deferred.close()
        if self._resume_requested:
            self._resume_after_capture_stopped()

    def confirm_candidate(self, candidate_id: str) -> None:
        self._invoke(lambda value: value.confirm_candidate(candidate_id))

    def confirm_manual_action(
        self,
        *,
        cards: tuple[str, ...] = (),
        is_pass: bool,
    ) -> None:
        self._invoke(
            lambda value: value.confirm_manual_action(cards=cards, is_pass=is_pass)
        )

    def correct_latest(
        self,
        *,
        cards: tuple[str, ...] = (),
        is_pass: bool,
    ) -> None:
        self._invoke(lambda value: value.correct_latest(cards=cards, is_pass=is_pass))

    def confirm_lead_player(self, seat: str) -> None:
        self._invoke(lambda value: value.confirm_lead_player(seat))

    def _invoke(self, operation) -> None:
        if self.orchestrator is None:
            self.error.emit("实时对局尚未开始")
            return
        try:
            update = operation(self.orchestrator)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self._emit_authoritative_update(update)

    def pause(self) -> None:
        self._resume_requested = False
        self._stop_capture_worker()
        self._stop_analysis_worker()
        self._invoke(lambda value: value.pause())

    def resume(self) -> None:
        if self.orchestrator is None:
            return
        # A persistent source pins one HWND and one client geometry.  After a
        # move, resize, minimization, or target recreation it must never be
        # reused under the old state token.
        self._resume_requested = True
        capture_stopped = self._stop_capture_worker()
        self._stop_analysis_worker()
        old_source, self._live_source = self._live_source, None
        if old_source is not None:
            if capture_stopped:
                try:
                    old_source.close()
                except Exception as exc:
                    self._resume_requested = False
                    self.error.emit(f"关闭旧采集源失败：{exc}")
                    return
            else:
                self._deferred_source_close = old_source
        if capture_stopped:
            self._resume_after_capture_stopped()

    def _resume_after_capture_stopped(self) -> None:
        """Open a fresh validated source before state or workers resume."""

        if not self._resume_requested:
            return
        orchestrator = self.orchestrator
        if orchestrator is None:
            self._resume_requested = False
            return
        try:
            source = self.capture_service.open_live_source(self.profile_name)
        except Exception as exc:
            self._resume_requested = False
            self.error.emit(f"重新打开采集源失败：{exc}")
            return
        token: _LiveRunToken | None = None
        try:
            if self._fatal_capture_session_id == str(getattr(getattr(orchestrator, "snapshot", None), "session_id", "")):
                # The original error may have prevented normal pause/incident
                # construction. A user reconnect must restore a paused state
                # before resume, never reuse the failed generation.
                if orchestrator.status != "paused":
                    orchestrator.pause()
            if callable(getattr(orchestrator, "bind_capture_generation", None)):
                token = self._activate_live_token(orchestrator)
                self._bind_capture_token(token)
            update = orchestrator.resume(
                monotonic_ms=monotonic_ns() // 1_000_000,
            )
            if token is not None and (not isinstance(update, LiveUpdate)
                    or not self._update_has_capture_identity(update, token)
                    or update.capture_generation != token.generation):
                raise RuntimeError("恢复结果不属于当前采集代次")
        except Exception as exc:
            if token is not None and self._active_live_token == token:
                self._invalidate_live_token()
                self._fatal_capture_session_id = token.session_id
                # A failed resume may already have changed status or scheduled
                # work. Cancel again, but never let cleanup revive its token.
                try:
                    orchestrator.pause()
                except Exception:
                    token.pipeline_timing.increment("failed_resume_pause_error")
            try:
                source.close()
            except Exception:
                pass
            self._resume_requested = False
            self.error.emit(f"恢复实时对局失败：{exc}")
            return
        self._live_source = source
        self._resume_requested = False
        if update is not None:
            self._emit_authoritative_update(update)
        self._start_analysis_worker()
        self._start_capture_worker()

    def finish(self) -> None:
        orchestrator = self.orchestrator
        if orchestrator is None or (
            self._finish_thread is not None and self._finish_thread.isRunning()
        ):
            return
        token = self._active_live_token
        self._resume_requested = False
        capture_stopped = self._stop_capture_worker()
        self._stop_analysis_worker()
        if self._live_source is not None:
            if capture_stopped:
                self._live_source.close()
            else:
                self._deferred_source_close = self._live_source
            self._live_source = None

        def finalize_session() -> object:
            if not self._close_recording_dispatcher(token):
                raise RuntimeError("录像进程未能确认终止，本局拒绝封存")
            orchestrator.begin_finalizing()
            return orchestrator.finish()

        thread = OneShotThread(finalize_session, self)
        thread.result.connect(
            lambda update, value=orchestrator: self._finish_result(value, update)
        )
        thread.error.connect(self.error)
        thread.finished.connect(self._finish_thread_finished)
        self._finish_thread = thread
        thread.start()

    def _finish_result(
        self,
        orchestrator: LiveRuntimePort,
        update: object,
    ) -> None:
        if self.orchestrator is orchestrator:
            self.orchestrator = None
        delivery = getattr(orchestrator, "automatic_log_delivery_result", None)
        if isinstance(delivery, dict):
            self._last_log_delivery_result = dict(delivery)
            self.log_delivery_status.emit(dict(delivery))
        self._emit_authoritative_update(update)
        self.session_finished.emit(update)

    def automatic_log_directory(self) -> Path:
        result = self._last_log_delivery_result
        if isinstance(result, dict):
            raw = str(result.get("output_directory") or "").strip()
            if raw:
                path = Path(raw)
                if path.exists():
                    return path
        latest = self._latest_sealed_session()
        if latest is not None:
            evidence = latest / "automatic_log_delivery.json"
            if evidence.is_file():
                try:
                    document = json.loads(evidence.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    document = {}
                raw = str(document.get("output_directory") or "").strip()
                if raw and Path(raw).exists():
                    return Path(raw)
        profile = str(os.environ.get("USERPROFILE") or "").strip()
        preferred = (Path(profile) if profile else Path.home()) / "Documents" / "掼蛋助手日志"
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred

    def request_full_diagnostic_export(self) -> None:
        if self._log_export_thread is not None and self._log_export_thread.isRunning():
            return
        self.log_delivery_status.emit(
            {
                "schema": "guandan.auto-log-delivery/1",
                "status": "RUNNING",
                "include_media": True,
                "message": "正在导出最近一局完整诊断",
            }
        )
        thread = OneShotThread(self._export_recent_full_diagnostic, self)
        thread.result.connect(self._full_diagnostic_exported)
        thread.error.connect(
            lambda message: self.log_delivery_status.emit(
                {
                    "schema": "guandan.auto-log-delivery/1",
                    "status": "FAIL",
                    "include_media": True,
                    "error": str(message),
                }
            )
        )
        thread.finished.connect(self._log_export_finished)
        self._log_export_thread = thread
        thread.start()

    def open_automatic_log_directory(self) -> Path:
        path = self.automatic_log_directory()
        if os.name != "nt" or not hasattr(os, "startfile"):
            raise RuntimeError("当前系统不支持直接打开日志目录")
        os.startfile(str(path))
        return path

    def request_open_automatic_log_directory(self) -> None:
        if self._log_open_thread is not None and self._log_open_thread.isRunning():
            return
        self.log_delivery_status.emit(
            {
                "schema": "guandan.auto-log-delivery/1",
                "status": "RUNNING",
                "action": "open_directory",
                "message": "正在打开日志目录",
            }
        )
        thread = OneShotThread(self.open_automatic_log_directory, self)
        thread.result.connect(
            lambda path: self.log_delivery_status.emit(
                {
                    "schema": "guandan.auto-log-delivery/1",
                    "status": "PASS",
                    "action": "open_directory",
                    "output_directory": str(path),
                }
            )
        )
        thread.error.connect(
            lambda message: self.log_delivery_status.emit(
                {
                    "schema": "guandan.auto-log-delivery/1",
                    "status": "FAIL",
                    "action": "open_directory",
                    "error": str(message),
                }
            )
        )
        thread.finished.connect(self._log_open_finished)
        self._log_open_thread = thread
        thread.start()

    def _log_open_finished(self) -> None:
        self._log_open_thread = None

    def _export_recent_full_diagnostic(self) -> dict[str, object]:
        from ..automatic_log_delivery import export_automatic_session_log

        session = self._latest_sealed_session()
        if session is None:
            raise RuntimeError("没有可导出的已封存对局")
        return export_automatic_session_log(
            session,
            include_media=True,
            profiles_root=getattr(self.capture_service, "profiles_root", None),
            profile_name=self.profile_name,
        ).to_dict()

    def _full_diagnostic_exported(self, result: object) -> None:
        if isinstance(result, dict):
            self._last_log_delivery_result = dict(result)
            self.log_delivery_status.emit(dict(result))

    def _log_export_finished(self) -> None:
        self._log_export_thread = None

    def _latest_sealed_session(self) -> Path | None:
        profiles_root = Path(getattr(self.capture_service, "profiles_root", ""))
        sessions = profiles_root / self.profile_name / "sessions"
        candidates: list[tuple[int, Path]] = []
        if not sessions.is_dir():
            return None
        for manifest_path in sessions.glob("*/manifest.json"):
            try:
                document = json.loads(manifest_path.read_text(encoding="utf-8"))
                if document.get("status") != "sealed":
                    continue
                modified = manifest_path.stat().st_mtime_ns
            except (OSError, json.JSONDecodeError):
                continue
            candidates.append((modified, manifest_path.parent))
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def _finish_thread_finished(self) -> None:
        self._finish_thread = None
        self._auto_finish_requested = False
        if self._listening_enabled and self.orchestrator is None:
            self._start_waiting_workers()

    def _auto_finish_on_game_end(self, update: object) -> None:
        event = getattr(update, "event", None)
        events = tuple(getattr(update, "events", ()) or ())
        game_end_detected = getattr(event, "event_type", None) == "game_end_detected" or any(
            getattr(item, "event_type", None) == "game_end_detected"
            for item in events
        )
        if (
            not game_end_detected
            or self._auto_finish_requested
        ):
            return
        self._auto_finish_requested = True
        self.finish()

    def _stop_capture_worker(self) -> bool:
        worker = self._capture_worker
        self._invalidate_live_token()
        if worker is None:
            return True
        worker.stop()
        stopped = worker.wait(0)
        if stopped:
            self._capture_worker = None
        return stopped

    def _stop_analysis_worker(self) -> None:
        worker, self._analysis_worker = self._analysis_worker, None
        if worker is not None:
            worker.stop(timeout=0.0)

    def shutdown(self) -> None:
        self._diagnostics_closed = True
        self.stop_listening()
        self._listener_evidence.writer.close(timeout=2.0)
        self._diagnostic_frame_thread = None
        recovery_thread = self._geometry_recovery_thread
        if recovery_thread is not None and recovery_thread.isRunning():
            recovery_thread.wait(10_000)
        result, self._geometry_recovery_result = self._geometry_recovery_result, None
        if result is not None and result.source is not None:
            try:
                result.source.close()
            except Exception:
                pass
        self._geometry_recovery_thread = None
        draining_analysis, self._draining_waiting_analysis_worker = (
            self._draining_waiting_analysis_worker,
            None,
        )
        if draining_analysis is not None:
            draining_analysis.stop(timeout=10.0)
        self.finish()
        if self._finish_thread is not None and self._finish_thread.isRunning():
            self._finish_thread.wait(30_000)
        self._close_recording_dispatcher()
        if self._log_export_thread is not None and self._log_export_thread.isRunning():
            self._log_export_thread.wait(30_000)
        if self._log_open_thread is not None and self._log_open_thread.isRunning():
            self._log_open_thread.wait(30_000)
        if self._capture_worker is not None and self._capture_worker.is_running:
            self._capture_worker.stop()
            self._capture_worker.wait(10_000)
        if self._waiting_capture_worker is not None:
            if self._waiting_capture_worker.is_running:
                self._waiting_capture_worker.stop()
                self._waiting_capture_worker.wait(10_000)
            self._waiting_capture_worker = None
            self._close_waiting_source()
        self._listener_recording_stop_reason = "application_shutdown"
        self._close_listener_recording()
        if self._initial_thread is not None and self._initial_thread.isRunning():
            self._initial_thread.wait(10_000)
        if (
            self._danzero_warmup_thread is not None
            and self._danzero_warmup_thread.isRunning()
        ):
            self._danzero_warmup_thread.wait(30_000)
        self.opening_evidence.close(timeout=5.0)
