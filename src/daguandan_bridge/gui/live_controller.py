from __future__ import annotations

from time import monotonic_ns, perf_counter
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..application.ports import (
    AdvicePort,
    CapturePort,
    RecognitionPort,
    SessionFactoryPort,
)
from ..capture_service import FrameSnapshot
from ..danzero.state import GuanDanState, RANKS
from ..live.orchestrator import LiveOrchestrator, LiveUpdate
from ..live.latest_worker import LatestOnlyWorker
from .workers import OneShotThread, WorkerHandle


class LiveAssistantController(QObject):
    """Qt signal adapter around the UI-independent live orchestrator."""

    initial_recognized = Signal(object, object)
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    session_finished = Signal(object)
    danzero_warmup_status = Signal(str)
    _waiting_recognized = Signal(object, object)
    _waiting_capture_stopped = Signal(object)

    def __init__(
        self,
        capture_service: CapturePort | None = None,
        *,
        profile_name: str = "tencent_daguandan",
        recognition_service: RecognitionPort | None = None,
        advisor: AdvicePort | None = None,
        session_factory: SessionFactoryPort | None = None,
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
        self.session_factory = session_factory
        self.orchestrator: LiveOrchestrator | None = None
        self._live_source = None
        self._capture_worker: WorkerHandle | None = None
        self._analysis_worker: LatestOnlyWorker | None = None
        self._initial_thread: OneShotThread | None = None
        self._danzero_warmup_thread: OneShotThread | None = None
        self._danzero_warmup_running = False
        self._danzero_warmup_complete = False
        self._finish_thread: OneShotThread | None = None
        self._deferred_source_close = None
        self._capture_generation = 0
        self._resume_requested = False
        self._listening_enabled = False
        self._waiting_source = None
        self._waiting_capture_worker: WorkerHandle | None = None
        self._waiting_analysis_worker: LatestOnlyWorker | None = None
        self._waiting_generation = 0
        self._waiting_candidate: tuple[str, tuple[str, ...]] | None = None
        self._pending_auto_session: tuple[str, tuple[str, ...]] | None = None
        self._recognition_strategy = "two_valid_streak"
        self._auto_finish_requested = False
        self._waiting_recognized.connect(self._consume_waiting_recognition)
        self._waiting_capture_stopped.connect(self._waiting_capture_finished)
        self.update_ready.connect(self._auto_finish_on_game_end)

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
            result = self.recognition_service.recognize(snapshot.image)
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

    def start_listening(self) -> None:
        """Continuously inspect the current page and start only on a stable deal."""

        self._listening_enabled = True
        self._start_danzero_warmup()
        if self.orchestrator is None and self._finish_thread is None:
            self._start_waiting_workers()

    def stop_listening(self) -> None:
        self._listening_enabled = False
        self._waiting_candidate = None
        self._pending_auto_session = None
        self._stop_waiting_workers()

    def _start_waiting_workers(self) -> None:
        if (
            not self._listening_enabled
            or self.orchestrator is not None
            or self._waiting_capture_worker is not None
        ):
            return
        try:
            source = self.capture_service.open_live_source(self.profile_name)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self._waiting_source = source
        generation = self._waiting_generation
        analysis = LatestOnlyWorker(
            self._recognize_waiting_frame,
            on_result=lambda value: self._waiting_recognized.emit(value[0], value[1]),
            on_error=lambda exc: self.error.emit(str(exc)),
        )
        self._waiting_analysis_worker = analysis
        analysis.start()

        def operation() -> FrameSnapshot:
            snapshot: FrameSnapshot = source.capture()
            if generation != self._waiting_generation:
                return snapshot
            active_analysis = self._waiting_analysis_worker
            if active_analysis is not None:
                active_analysis.submit(snapshot)
            return snapshot

        worker = WorkerHandle(operation, 0.2)
        worker.frame_ready.connect(self._accept_waiting_frame)
        worker.error.connect(self._accept_waiting_error)
        worker.finished.connect(lambda: self._waiting_capture_stopped.emit(worker))
        self._waiting_capture_worker = worker
        worker.start()

    def _recognize_waiting_frame(
        self,
        snapshot: FrameSnapshot,
    ) -> tuple[object, FrameSnapshot]:
        return self.recognition_service.recognize(snapshot.image), snapshot

    def _accept_waiting_frame(self, snapshot: object) -> None:
        self.frame_ready.emit(snapshot)

    def _accept_waiting_error(self, message: str) -> None:
        self.error.emit(message)

    def _consume_waiting_recognition(self, result: object, snapshot: object) -> None:
        """Require two identical normalized 27-card results before starting."""

        self.initial_recognized.emit(result, snapshot)
        if not self._listening_enabled or self.orchestrator is not None:
            return
        hand = tuple(str(card) for card in getattr(result, "my_hand", ()))
        round_level = str(getattr(result, "round_level", ""))
        if round_level not in RANKS or len(hand) != 27:
            self._waiting_candidate = None
            return
        try:
            normalizer = GuanDanState()
            normalizer.confirm_hand(hand)
        except Exception:
            self._waiting_candidate = None
            return
        candidate = (round_level, normalizer.my_hand)
        if candidate != self._waiting_candidate:
            self._waiting_candidate = candidate
            return
        self._waiting_candidate = None
        self._start_detected_session(result)

    def _start_detected_session(self, result: object) -> None:
        self._pending_auto_session = (
            str(getattr(result, "round_level")),
            tuple(str(card) for card in getattr(result, "my_hand", ())),
        )
        if self._stop_waiting_workers():
            self._start_pending_auto_session()

    def _start_pending_auto_session(self) -> None:
        pending, self._pending_auto_session = self._pending_auto_session, None
        if (
            pending is None
            or not self._listening_enabled
            or self.orchestrator is not None
        ):
            return
        round_level, hand = pending
        if not self.start_session(
            round_level=round_level,
            hand=hand,
            lead_player=None,
            recognition_strategy=self._recognition_strategy,
        ):
            self._start_waiting_workers()

    def _stop_waiting_workers(self) -> bool:
        worker = self._waiting_capture_worker
        self._waiting_generation += 1
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

    def _stop_waiting_analysis_worker(self) -> None:
        worker, self._waiting_analysis_worker = self._waiting_analysis_worker, None
        if worker is not None:
            worker.stop(timeout=0.0)

    def _close_waiting_source(self) -> None:
        source, self._waiting_source = self._waiting_source, None
        if source is not None:
            source.close()

    def _waiting_capture_finished(self, worker: WorkerHandle) -> None:
        if self._waiting_capture_worker is worker:
            self._waiting_capture_worker = None
        self._close_waiting_source()
        if self._pending_auto_session is not None:
            self._start_pending_auto_session()

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
        self._danzero_warmup_running = True
        self.danzero_warmup_status.emit("DanZero 模型预热中")

        def operation() -> float:
            started = perf_counter()
            initializer()
            return (perf_counter() - started) * 1_000

        thread = OneShotThread(operation, self)
        thread.result.connect(self._danzero_warmup_succeeded)
        thread.error.connect(self._danzero_warmup_failed)
        thread.finished.connect(self._danzero_warmup_finished)
        self._danzero_warmup_thread = thread
        thread.start()

    def _danzero_warmup_succeeded(self, elapsed_ms: float) -> None:
        self._danzero_warmup_complete = True
        self.danzero_warmup_status.emit(
            f"DanZero 模型已就绪（首次预热 {float(elapsed_ms):.0f} ms）"
        )

    def _danzero_warmup_failed(self, message: str) -> None:
        self.danzero_warmup_status.emit(
            f"DanZero 模型预热失败，首次建议时将自动重试：{message}"
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
    ) -> bool:
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
                on_update=self.update_ready.emit,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return False
        self.orchestrator = constructed.orchestrator
        self._live_source = constructed.source
        self._auto_finish_requested = False
        self.update_ready.emit(constructed.initial_update)
        self._start_analysis_worker()
        self._start_capture_worker()
        return True

    def _start_analysis_worker(self) -> None:
        if self.orchestrator is None or self._analysis_worker is not None:
            return
        worker = LatestOnlyWorker(
            self._analyze_live_frame,
            on_result=self.update_ready.emit,
            on_error=self._accept_analysis_error,
        )
        self._analysis_worker = worker
        worker.start()

    def _analyze_live_frame(self, value: object) -> LiveUpdate:
        snapshot, monotonic_ms = value  # type: ignore[misc]
        assert self.orchestrator is not None
        return self.orchestrator.analyze_frame(
            snapshot.image,
            monotonic_ms=monotonic_ms,
        )

    def _accept_analysis_error(self, exc: Exception) -> None:
        message = str(exc)
        if self.orchestrator is not None:
            try:
                update = self.orchestrator.analysis_failed(
                    message,
                    monotonic_ms=monotonic_ns() // 1_000_000,
                )
            except Exception as incident_exc:
                self.error.emit(f"{message}; 创建识别事故失败：{incident_exc}")
                return
            self.update_ready.emit(update)
        self.error.emit(message)

    def _start_capture_worker(self) -> None:
        if self.orchestrator is None or self._live_source is None or self.is_running:
            return
        orchestrator = self.orchestrator
        generation = self._capture_generation

        def operation():
            snapshot: FrameSnapshot = self._live_source.capture()
            if generation != self._capture_generation:
                return snapshot
            captured_ms = monotonic_ns() // 1_000_000
            orchestrator.record_frame(
                snapshot.image,
                monotonic_ms=captured_ms,
                wall_time=snapshot.captured_at.isoformat(),
            )
            analysis = self._analysis_worker
            if analysis is not None:
                analysis.submit(
                    (snapshot, captured_ms),
                    preserve=orchestrator.needs_first_action_frames,
                    max_preserved=8,
                )
            return snapshot

        worker = WorkerHandle(operation, 0.1)
        worker.frame_ready.connect(self._accept_live_frame)
        worker.error.connect(self._accept_live_error)
        worker.finished.connect(lambda: self._capture_finished(worker))
        self._capture_worker = worker
        self._resume_requested = False
        worker.start()

    def _accept_live_frame(self, value: object) -> None:
        self.frame_ready.emit(value)

    def _accept_live_error(self, message: str) -> None:
        self._stop_analysis_worker()
        if self.orchestrator is not None and self.orchestrator.status not in {
            "finalizing",
            "sealed",
        }:
            update = self.orchestrator.capture_interrupted(
                message,
                monotonic_ms=monotonic_ns() // 1_000_000,
            )
            self.update_ready.emit(update)
        self.error.emit(message)

    def _capture_finished(self, worker: WorkerHandle) -> None:
        if self._capture_worker is worker:
            self._capture_worker = None
        deferred, self._deferred_source_close = self._deferred_source_close, None
        if deferred is not None:
            deferred.close()
        if (
            self._resume_requested
            and self.orchestrator is not None
            and self.orchestrator.status == "running"
        ):
            self._start_capture_worker()

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
        self._invoke(
            lambda value: value.resume(monotonic_ms=monotonic_ns() // 1_000_000)
        )
        self._resume_requested = True
        self._start_analysis_worker()
        self._start_capture_worker()

    def finish(self) -> None:
        orchestrator = self.orchestrator
        if orchestrator is None or (
            self._finish_thread is not None and self._finish_thread.isRunning()
        ):
            return
        orchestrator.begin_finalizing()
        self._resume_requested = False
        capture_stopped = self._stop_capture_worker()
        self._stop_analysis_worker()
        if self._live_source is not None:
            if capture_stopped:
                self._live_source.close()
            else:
                self._deferred_source_close = self._live_source
            self._live_source = None
        thread = OneShotThread(orchestrator.finish, self)
        thread.result.connect(
            lambda update, value=orchestrator: self._finish_result(value, update)
        )
        thread.error.connect(self.error)
        thread.finished.connect(self._finish_thread_finished)
        self._finish_thread = thread
        thread.start()

    def _finish_result(
        self,
        orchestrator: LiveOrchestrator,
        update: object,
    ) -> None:
        if self.orchestrator is orchestrator:
            self.orchestrator = None
        self.update_ready.emit(update)
        self.session_finished.emit(update)

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
        if worker is None:
            return True
        self._capture_generation += 1
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
        self.finish()
        if self._finish_thread is not None and self._finish_thread.isRunning():
            self._finish_thread.wait(30_000)
        if self._capture_worker is not None and self._capture_worker.is_running:
            self._capture_worker.stop()
            self._capture_worker.wait(10_000)
        if (
            self._waiting_capture_worker is not None
            and self._waiting_capture_worker.is_running
        ):
            self._waiting_capture_worker.stop()
            self._waiting_capture_worker.wait(10_000)
            self._waiting_capture_worker = None
            self._close_waiting_source()
        if self._initial_thread is not None and self._initial_thread.isRunning():
            self._initial_thread.wait(10_000)
        if (
            self._danzero_warmup_thread is not None
            and self._danzero_warmup_thread.isRunning()
        ):
            self._danzero_warmup_thread.wait(30_000)
