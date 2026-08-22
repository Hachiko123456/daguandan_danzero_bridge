from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import monotonic_ns, perf_counter
from typing import Any

from PySide6.QtCore import QObject, Signal

from ..application.ports import (
    AdvicePort,
    CapturePort,
    RecognitionPort,
    SessionFactoryPort,
)
from ..advisor_strategy import (
    build_advisor,
    load_profile_advisor_strategy,
    load_profile_session_data_recording_enabled,
    normalize_advisor_strategy,
    save_profile_advisor_strategy,
    save_profile_session_data_recording_enabled,
)
from ..capture_service import FrameSnapshot
from ..danzero.state import GuanDanState, RANKS, Seat
from ..live.orchestrator import AdviceRequestKey, LiveAdvice, LiveOrchestrator, LiveUpdate
from ..live.latest_worker import LatestOnlyWorker
from ..live.turns import TURN_ORDER, next_active_seat
from ..infrastructure.win32_hand_preselector import Win32HandPreselector
from .hand_preselection import HandPreselectionPlanner, PreselectionResult
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
    orchestrator: LiveOrchestrator
    session_id: str
    nonce: int
    generation: int


@dataclass(frozen=True)
class _AnalysisFrameTask:
    token: _LiveRunToken
    snapshot: FrameSnapshot
    capture_seq: int
    captured_ms: int


@dataclass(frozen=True)
class _OpeningActionSeed:
    """A fully visual, already-observed first action at listener startup."""

    actor: Seat
    cards: tuple[str, ...]
    next_player: Seat
    confidence: float
    source: str


@dataclass(frozen=True)
class _AutoSessionSeed:
    """The stable opening state passed from the listener to the live session."""

    round_level: str
    hand: tuple[str, ...]
    lead_player: Seat | None
    opening_action: _OpeningActionSeed | None = None


class LiveAssistantController(QObject):
    """Qt signal adapter around the UI-independent live orchestrator."""

    initial_recognized = Signal(object, object)
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    session_finished = Signal(object)
    danzero_warmup_status = Signal(str)
    preselection_result = Signal(object)
    _waiting_recognized = Signal(object, object)
    _waiting_capture_stopped = Signal(object)
    _TABLE_ANCHOR_READY_SCORE = 0.85

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
        self.advisor_strategy = (
            "fabledan"
            if type(advisor).__name__ == "FableDanAdvisor"
            else load_profile_advisor_strategy(
                self.capture_service.profiles_root,
                self.profile_name,
            )
        )
        self.session_factory = session_factory
        self.session_data_recording_enabled = (
            load_profile_session_data_recording_enabled(
                self.capture_service.profiles_root,
                self.profile_name,
            )
        )
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
        self._live_session_nonce = 0
        self._active_live_token: _LiveRunToken | None = None
        self._resume_requested = False
        self._listening_enabled = False
        self._waiting_source = None
        self._waiting_capture_worker: WorkerHandle | None = None
        self._waiting_analysis_worker: LatestOnlyWorker | None = None
        self._waiting_generation = 0
        self._waiting_candidate: _AutoSessionSeed | None = None
        self._pending_auto_session: _AutoSessionSeed | None = None
        self._table_anchor_observed = False
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
        """Persist whether the next live game writes any session artifacts."""

        if self.orchestrator is not None:
            raise RuntimeError("实时对局开始后不能切换对局数据保存")
        self.session_data_recording_enabled = (
            save_profile_session_data_recording_enabled(
                self.capture_service.profiles_root,
                self.profile_name,
                enabled,
            )
        )

    def start_listening(self) -> bool:
        """Continuously inspect the current page and start only on a stable deal."""

        if self._listening_enabled or self.orchestrator is not None:
            return True
        lock_client = getattr(self.capture_service, "lock_target_client_size", None)
        if callable(lock_client):
            try:
                lock_client(self.profile_name)
            except Exception as exc:
                self.error.emit(f"无法锁定牌桌客户区尺寸：{exc}")
                return False
        self._listening_enabled = True
        self._table_anchor_observed = False
        self._start_danzero_warmup()
        if self.orchestrator is None and self._finish_thread is None:
            self._start_waiting_workers()
        return True

    def stop_listening(self) -> None:
        self._listening_enabled = False
        self._waiting_candidate = None
        self._pending_auto_session = None
        self._table_anchor_observed = False
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
            self._listening_enabled = False
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
            # Opening probes stay in memory.  Storage starts only after a
            # complete initial state has been confirmed.
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
        return self._recognize_initial_image(snapshot.image), snapshot

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

    def _accept_waiting_frame(self, snapshot: object) -> None:
        self.frame_ready.emit(snapshot)

    def _accept_waiting_error(self, message: str) -> None:
        self.error.emit(message)
        if self.orchestrator is None:
            self._listening_enabled = False
            self._waiting_candidate = None

    def _consume_waiting_recognition(self, result: object, snapshot: object) -> None:
        """Require two identical normalized 27-card results before starting."""

        self.initial_recognized.emit(result, snapshot)
        if not self._listening_enabled or self.orchestrator is not None:
            return
        buttons = set(getattr(result, "buttons", ()) or ())
        if buttons & {"change_table", "continue_game"}:
            # A settlement screen never becomes a session.  Reset the table
            # probe and keep listening for the next real opening.
            self._waiting_candidate = None
            self._table_anchor_observed = False
            return
        if not self._table_anchor_observed:
            if self._table_anchor_score(snapshot) < self._TABLE_ANCHOR_READY_SCORE:
                # Do not start a session from a lobby, settlement screen, or
                # a manually clicked late page.
                self._waiting_candidate = None
                return
            self._table_anchor_observed = True
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
        candidate = self._auto_session_seed(
            result,
            round_level=round_level,
            hand=normalizer.my_hand,
        )
        if candidate is None:
            self._waiting_candidate = None
            return
        if candidate != self._waiting_candidate:
            self._waiting_candidate = candidate
            return
        self._waiting_candidate = None
        self._start_detected_session(result)

    def _start_detected_session(self, result: object) -> None:
        hand = tuple(str(card) for card in getattr(result, "my_hand", ()))
        try:
            normalizer = GuanDanState()
            normalizer.confirm_hand(hand)
        except Exception:
            return
        seed = self._auto_session_seed(
            result,
            round_level=str(getattr(result, "round_level", "")),
            hand=normalizer.my_hand,
        )
        if seed is None:
            return
        self._pending_auto_session = seed
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
        if not self.start_session(
            round_level=pending.round_level,
            hand=pending.hand,
            lead_player=pending.lead_player,
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
            return float(recognize_anchor(image))
        except Exception as exc:
            self.error.emit(f"牌桌锚点识别失败：{exc}")
            return 0.0

    @staticmethod
    def _auto_session_seed(
        result: object,
        *,
        round_level: str,
        hand: tuple[str, ...],
    ) -> _AutoSessionSeed | None:
        lead_player = getattr(result, "lead_player", None)
        current_player = getattr(result, "current_player", None)
        events = tuple(getattr(result, "events", ()) or ())
        if not events:
            # Before the first play, the lead marker may be present while the
            # active indicator is either absent or still points at that lead.
            if lead_player is None and current_player is None:
                return _AutoSessionSeed(round_level, hand, None)
            if (
                lead_player in TURN_ORDER
                and current_player in {None, lead_player}
            ):
                return _AutoSessionSeed(round_level, hand, lead_player)
            return None
        if len(events) != 1 or lead_player not in TURN_ORDER:
            return None
        event = events[0]
        actor = getattr(event, "player", None)
        cards = tuple(str(card) for card in getattr(event, "cards", ()) or ())
        if (
            actor != lead_player
            or actor not in TURN_ORDER
            or bool(getattr(event, "is_pass", False))
            or not cards
            or any("?" in card for card in cards)
            or current_player != next_active_seat(actor, frozenset())
        ):
            # A complete hand on an already-running table is not a valid
            # opening anchor.  Do not invent a history or request FableDan
            # from that unknown state.
            return None
        opening_action = _OpeningActionSeed(
            actor=actor,
            cards=cards,
            next_player=current_player,
            confidence=float(getattr(event, "confidence", 0.0)),
            source=str(getattr(event, "source", "visual_opening_anchor")),
        )
        return _AutoSessionSeed(
            round_level,
            hand,
            lead_player,
            opening_action,
        )

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
            return

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
        initial_update = constructed.initial_update
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
                finish = getattr(self.orchestrator, "finish", None)
                if callable(finish):
                    try:
                        finish()
                    except Exception:
                        pass
                self._live_source.close()
                self._live_source = None
                self.orchestrator = None
                self.error.emit(f"首出动作锚定失败：{exc}")
                return False
        self._activate_live_token(constructed.orchestrator)
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

    def _activate_live_token(self, orchestrator: LiveOrchestrator) -> _LiveRunToken:
        self._capture_generation += 1
        self._live_session_nonce += 1
        token = _LiveRunToken(
            orchestrator=orchestrator,
            session_id=str(orchestrator.snapshot.session_id),
            nonce=self._live_session_nonce,
            generation=self._capture_generation,
        )
        self._active_live_token = token
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
            lambda value, current=token: self._analyze_live_frame(current, value),
            on_result=lambda update, current=token: self._accept_analysis_update(
                current, update
            ),
            on_error=lambda exc, current=token: self._accept_analysis_error(
                current, exc
            ),
        )
        self._analysis_worker = worker
        worker.start()

    def _analyze_live_frame(
        self,
        token: _LiveRunToken,
        value: object,
    ) -> LiveUpdate | None:
        task = value
        if not isinstance(task, _AnalysisFrameTask) or task.token != token:
            return None
        if not self._live_token_is_current(token):
            return None
        update = token.orchestrator.analyze_frame(
            task.snapshot.image,
            monotonic_ms=task.captured_ms,
            trace_context={
                "worker_token": {
                    "session_id": token.session_id,
                    "nonce": token.nonce,
                    "generation": token.generation,
                },
                "capture_seq": task.capture_seq,
                "captured_ms": task.captured_ms,
            },
        )
        return update if self._live_token_is_current(token) else None

    def _accept_analysis_update(
        self,
        token: _LiveRunToken,
        update: object,
    ) -> None:
        if self._live_token_is_current(token) and isinstance(update, LiveUpdate):
            self.update_ready.emit(update)

    def _accept_analysis_error(self, token: _LiveRunToken, exc: Exception) -> None:
        message = str(exc)
        if self._live_token_is_current(token):
            try:
                update = token.orchestrator.analysis_failed(
                    message,
                    monotonic_ms=monotonic_ns() // 1_000_000,
                )
            except Exception as incident_exc:
                self.error.emit(f"{message}; 创建识别事故失败：{incident_exc}")
                return
            self.update_ready.emit(update)
        self.error.emit(message)

    def _start_capture_worker(self) -> None:
        token = self._active_live_token
        if token is None or self._live_source is None or self.is_running:
            return
        source = self._live_source
        analysis = self._analysis_worker
        capture_seq = 0

        def operation():
            nonlocal capture_seq
            snapshot: FrameSnapshot = source.capture()
            capture_seq += 1
            captured_ms = monotonic_ns() // 1_000_000
            if not self._live_token_is_current(token):
                return snapshot
            token.orchestrator.record_frame(
                snapshot.image,
                monotonic_ms=captured_ms,
                wall_time=snapshot.captured_at.isoformat(),
            )
            if self._live_token_is_current(token) and analysis is not None:
                analysis.submit(
                    _AnalysisFrameTask(token, snapshot, capture_seq, captured_ms),
                    preserve=token.orchestrator.needs_first_action_frames,
                    max_preserved=8,
                )
            return snapshot

        worker = WorkerHandle(operation, 0.1)
        worker.frame_ready.connect(
            lambda value, current=token: self._accept_live_frame(current, value)
        )
        worker.error.connect(
            lambda message, current=token: self._accept_live_error(current, message)
        )
        worker.finished.connect(lambda: self._capture_finished(worker, token))
        self._capture_worker = worker
        self._resume_requested = False
        worker.start()

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

    def _accept_live_error(self, token: _LiveRunToken, message: str) -> None:
        if not self._live_token_is_current(token):
            return
        self._stop_analysis_worker()
        if token.orchestrator.status not in {
            "finalizing",
            "sealed",
        }:
            update = token.orchestrator.capture_interrupted(
                message,
                monotonic_ms=monotonic_ns() // 1_000_000,
            )
            self.update_ready.emit(update)
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
        if (
            self._resume_requested
            and (
                token is None
                or (
                    self._live_token_is_current(token)
                    and token.orchestrator.status == "running"
                )
            )
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
        if isinstance(self.orchestrator, LiveOrchestrator):
            self._activate_live_token(self.orchestrator)
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
        self.finish()
        if self._finish_thread is not None and self._finish_thread.isRunning():
            self._finish_thread.wait(30_000)
        if self._capture_worker is not None and self._capture_worker.is_running:
            self._capture_worker.stop()
            self._capture_worker.wait(10_000)
        if self._waiting_capture_worker is not None:
            if self._waiting_capture_worker.is_running:
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
