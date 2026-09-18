from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from time import monotonic_ns, perf_counter
from types import SimpleNamespace
from typing import Any

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot

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
from ..opening_evidence import (
    NonBlockingOpeningEvidenceSink,
    build_opening_evidence_monitor,
)
from ..opening_gate import (
    OpeningActionSeed as _OpeningActionSeed,
    OpeningSessionSeed as _AutoSessionSeed,
    build_opening_seed,
    evaluate_opening_gate,
    OpeningTracker,
    ListeningPageSignal,
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


@dataclass(frozen=True)
class _WaitingRecognitionEnvelope:
    snapshot: FrameSnapshot
    generation: int
    trace: object | None
    opening_seed_valid: bool | None
    page: ListeningPageSignal | None = None


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
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.snapshot = snapshot
        self.generation = generation


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
        self._last_log_delivery_result: dict[str, object] | None = None
        self._deferred_source_close = None
        self._capture_generation = 0
        self._live_session_nonce = 0
        self._active_live_token: _LiveRunToken | None = None
        self._gui_delivery_lock = Lock()
        self._pending_gui_delivery: _AnalysisDelivery | None = None
        self._gui_delivery_scheduled = False
        self._gui_accepted_version: tuple[_LiveRunToken, int, int, int] | None = None
        self._queued_fault_identity: tuple[str, int] | None = None
        self._fatal_capture_session_id: str | None = None
        self._resume_requested = False
        self._listening_enabled = False
        self._waiting_source = None
        self._waiting_capture_worker: WorkerHandle | None = None
        self._waiting_analysis_worker: LatestOnlyWorker | None = None
        self._draining_waiting_analysis_worker: LatestOnlyWorker | None = None
        self._waiting_generation = 0
        self._waiting_candidate: _AutoSessionSeed | None = None
        self._opening_tracker = OpeningTracker()
        self._listening_page = ListeningPageSignal("unknown", 0.0)
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
        self._preselection_thread: OneShotThread | None = None
        self._active_preselection_task: _PreselectionTask | None = None
        self._pending_preselection_task: _PreselectionTask | None = None
        self._handled_preselection_request_ids: set[str] = set()
        self._hand_preselector: object | None = None
        self.latest_preselection_result: PreselectionResult | None = None
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

    def target_client_rect(self):
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

    def start_listening(self) -> bool:
        """Continuously inspect the current page and start only on a stable deal."""

        if self._listening_enabled or self.orchestrator is not None:
            return True
        if not self._waiting_analysis_drain_complete():
            self.error.emit("上一次开局识别仍在停止中，请稍后重试")
            return False
        try:
            preload_live_worker_dependencies()
        except Exception as exc:
            self.error.emit(f"实时依赖预加载失败：{exc}")
            return False
        self.opening_evidence.begin(monotonic_ms=monotonic_ns() // 1_000_000)
        lock_client = getattr(self.capture_service, "lock_target_client_size", None)
        if callable(lock_client):
            try:
                lock_client(self.profile_name)
            except Exception as exc:
                self.opening_evidence.observe_failure(exc, stage="window")
                self.error.emit(f"无法锁定牌桌客户区尺寸：{exc}")
                return False
        self._listening_enabled = True
        self._geometry_recovery_cycle_count = 0
        self._geometry_post_recovery_frames_remaining = 0
        self._table_anchor_observed = False
        self._listener_recording_stop_reason = None
        self._opening_tracker.reset()
        self._listening_page = ListeningPageSignal("unknown", 0.0)
        self.listening_status.emit(
            {"state": "listening", "message": "持续监听页面中"}
        )
        self._start_danzero_warmup()
        if self.orchestrator is None and self._finish_thread is None:
            self._start_waiting_workers()
        return True

    def stop_listening(self) -> None:
        self._listening_enabled = False
        self._cancel_geometry_recovery()
        self._waiting_candidate = None
        self._opening_tracker.reset()
        self._pending_auto_session = None
        self._table_anchor_observed = False
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

        def operation() -> FrameSnapshot:
            snapshot: FrameSnapshot = source.capture()
            if generation != self._waiting_generation:
                return snapshot
            page_probe = getattr(self.recognition_service, "recognize_listening_page", None)
            page = page_probe(snapshot.image) if callable(page_probe) else None
            if generation != self._waiting_generation:
                return snapshot
            if page is not None:
                # Gate every persisted frame using its own cheap page probe,
                # not a slow full-hand recognition from an earlier screen.
                active_capture = getattr(self._waiting_capture_worker, "worker", None)
                if active_capture is not None:
                    active_capture.interval_sec = .2 if page.allows_media else 1.0
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
                task = _WaitingAnalysisTask(snapshot, generation, page)
                try:
                    active_analysis.submit(task)
                except Exception:
                    self.opening_evidence.observe_analysis_dropped(
                        snapshot,
                        reason="submit_failed",
                    )
                    raise
            return snapshot

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
                if callable(observe_page) and (generation is None or generation == self._waiting_generation):
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
        return result, _WaitingRecognitionEnvelope(
            snapshot,
            generation,
            trace,
            seed_valid,
            page,
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
        snapshot: object,
        generation: int | None = None,
    ) -> None:
        if generation is None or generation == self._waiting_generation:
            self.frame_ready.emit(snapshot)

    def _accept_waiting_error(
        self,
        error: object,
        generation: int | None = None,
    ) -> None:
        if generation is not None and generation != self._waiting_generation:
            return
        self.opening_evidence.observe_failure(error, stage="capture")
        code = str(getattr(error, "code", "") or "").upper()
        if (
            code in self._GEOMETRY_RECOVERABLE_CODES
            and self.orchestrator is None
            and self._listening_enabled
        ):
            self._begin_geometry_recovery(error)
            return
        self.error.emit(str(error))
        if self.orchestrator is None:
            self._listening_enabled = False
            self._waiting_candidate = None
            self._listener_recording_stop_reason = "waiting_capture_failed"

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
        self.opening_evidence.observe_failure(
            original,
            stage="recognition",
            snapshot=snapshot,
        )
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
            self._apply_listening_page(envelope.page, snapshot)
            if not envelope.page.allows_media:
                self.initial_recognized.emit(result, snapshot)
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
            return
        self._waiting_candidate = None
        self._start_detected_session(result)

    def _apply_listening_page(self, page: ListeningPageSignal, snapshot: object) -> None:
        previous = self._listening_page.stage
        self._listening_page = page
        worker = getattr(self._waiting_capture_worker, "worker", None)
        if worker is not None:
            worker.interval_sec = 0.2 if page.allows_media else 1.0
        if not page.allows_media:
            if page.stage in {"lobby", "settlement"}:
                self._opening_tracker.reset()
                self._waiting_candidate = None
            self._table_anchor_observed = False
            self._listener_recording_stop_reason = "page_" + page.stage
            self._close_listener_recording()
            return
        if previous in {"lobby", "settlement"}:
            self._opening_tracker.reset()
        self._table_anchor_observed = page.anchor_score >= self._TABLE_ANCHOR_READY_SCORE
        if self._start_listener_recording() and snapshot is not None:
            self._record_listener_frame(snapshot)

    def _publish_opening_status(self, phase: str, result: object) -> None:
        count = len(tuple(getattr(result, "my_hand", ()) or ()))
        messages = {
            "unknown": "已连接，等待进入牌桌", "lobby": "已连接，等待进入牌桌",
            "waiting_table": "已连接，等待进入牌桌",
            "settlement": "本局已结束，等待下一局（未录像）",
            "round_level_unresolved": "正在确认当前级牌",
            "hand_count_mismatch": f"正在确认起手牌，已识别{count}张",
            "hand_invalid": "起手牌识别有冲突，正在重新确认",
            "hand_unresolved": "起手牌存在未确认花色，正在等待清晰画面",
            "missed_opening": f"错过完整开局，当前{count}张，本局暂无法推荐",
            "opening_seed_invalid": f"已识别{count}张，等待首出确认",
            "confirming_hand": f"已识别{count}张，正在确认起手牌",
            "confirming_opening": f"已识别{count}张，等待首出确认",
            "ready": "完整开局已确认，正在建立对局",
        }
        if phase in {"already_started", "duplicate_frame"}:
            return
        self.listening_status.emit({
            "state": "opening", "phase": phase, "reason": phase,
            "hand_count": count, "generation": self._waiting_generation,
            "message": messages.get(phase, "正在确认完整开局"),
        })

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
            # Automatic sessions always re-confirm lead + opening play in the
            # live-v2 opening barrier.  The waiting tracker seed is only a
            # trigger, never authoritative action history.
            lead_player=None,
            recognition_strategy=self._recognition_strategy,
            opening_action=None,
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
        message = "牌桌窗口发生变化，正在重新连接"
        self.listening_status.emit(
            {
                "state": "recovering",
                "message": message,
                "generation": self._waiting_generation,
                "attempt_count": 0,
            }
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
        self.listening_status.emit(
            {
                "state": "recovered",
                "message": "牌桌窗口已重新连接，继续监听",
                "generation": result.generation,
                "attempt_count": result.attempt_count,
            }
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
        self.listening_status.emit(
            {
                "state": "failed",
                "message": message,
                "reason": reason,
                "generation": self._waiting_generation,
                "attempt_count": self._geometry_recovery_attempt_count,
            }
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
        worker = self._waiting_capture_worker
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
        return True

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
                )
            except Exception as exc:
                self._abort_started_session(token, constructed.source)
                self.error.emit(f"首出动作锚定失败：{exc}")
                return False
        self.opening_evidence.mark_session_started()
        self._auto_finish_requested = False
        self._latest_live_frame = None
        self._latest_live_frame_generation = 0
        self._pending_preselection_task = None
        self._handled_preselection_request_ids.clear()
        self.latest_preselection_result = None
        self.update_ready.emit(initial_update)
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
        self._capture_generation += 1
        self._live_session_nonce += 1
        self._active_live_token = None

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
        update = self._analyze_live_frame(token, value)
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
        self.update_ready.emit(delivery.update)

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
                return snapshot
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
                        capture_started_ns, submitted_ns,
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
            return snapshot

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
        if self._live_token_is_current(token) and isinstance(value, FrameSnapshot):
            self._latest_live_frame = value
            self._latest_live_frame_generation = self._capture_generation
            self.frame_ready.emit(value)

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
            frame=frame,
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
        current = orchestrator.latest_advice
        return bool(
            isinstance(current, LiveAdvice)
            and current.key == task.key
            and current.status == "ready"
            and current.visible
            and current.advice is not None
            and not current.advice.is_pass
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
        self.latest_preselection_result = result
        self.preselection_result.emit(result)

    def _accept_live_error(self, token: _LiveRunToken, error: object) -> None:
        if not self._live_token_is_current(token):
            return
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
        self.update_ready.emit(update)

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
            self.update_ready.emit(update)
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
        self.update_ready.emit(update)
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
        self.stop_listening()
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
