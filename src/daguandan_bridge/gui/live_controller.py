from __future__ import annotations

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic_ns

from PySide6.QtCore import QObject, Signal

from ..annotation_service import AnnotationService
from ..capture_service import CaptureService, FrameSnapshot
from ..danzero import DanzeroAdvisor
from ..live.orchestrator import LiveOrchestrator, LiveUpdate
from ..live.latest_worker import LatestOnlyWorker
from ..live.recorder import SessionRecorder
from ..live.reducer import LiveReducer
from ..live.session_store import LiveSessionStore
from ..recognition_service import ScreenshotRecognitionService
from ..template_service import TemplateService
from .workers import OneShotThread, WorkerHandle


class LiveAssistantController(QObject):
    """Qt signal adapter around the UI-independent live orchestrator."""

    initial_recognized = Signal(object, object)
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    session_finished = Signal(object)

    def __init__(
        self,
        capture_service: CaptureService | None = None,
        *,
        profile_name: str = "tencent_daguandan",
    ) -> None:
        super().__init__()
        self.capture_service = capture_service or CaptureService()
        self.profile_name = profile_name
        LiveSessionStore.recover_incomplete_sessions(
            self.capture_service.profiles_root,
            self.profile_name,
        )
        annotation = AnnotationService(
            self.capture_service.profiles_root,
            profile_name,
        )
        templates = TemplateService(
            self.capture_service.profiles_root,
            profile_name,
        )
        self.recognition_service = ScreenshotRecognitionService(annotation, templates)
        self.orchestrator: LiveOrchestrator | None = None
        self._live_source = None
        self._capture_worker: WorkerHandle | None = None
        self._analysis_worker: LatestOnlyWorker | None = None
        self._initial_thread: OneShotThread | None = None
        self._finish_thread: OneShotThread | None = None
        self._deferred_source_close = None
        self._capture_generation = 0

    @property
    def is_running(self) -> bool:
        return bool(self._capture_worker and self._capture_worker.is_running)

    def recognize_initial(self) -> None:
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

    def _initial_result(self, value: object) -> None:
        result, snapshot = value  # type: ignore[misc]
        self.initial_recognized.emit(result, snapshot)

    def _initial_finished(self) -> None:
        self._initial_thread = None

    def start_session(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: str,
    ) -> bool:
        if self.orchestrator is not None:
            self.error.emit("当前已有实时对局")
            return False
        store = None
        recorder = None
        source = None
        try:
            loaded = self.capture_service.load_profile(self.profile_name)
            store = LiveSessionStore(
                self.capture_service.profiles_root,
                self.profile_name,
            )
            store.start(self._manifest(loaded.paths.profile_config_path, loaded.paths.templates_config_path))
            recorder = SessionRecorder(
                store.directory,
                size=loaded.config.base_size,
                fps=10,
            )
            orchestrator = LiveOrchestrator(
                reducer=LiveReducer(store.session_id),
                store=store,
                recorder=recorder,
                recognition_service=self.recognition_service,
                advisor=DanzeroAdvisor(),
            )
            source = self.capture_service.open_live_source(self.profile_name)
            started_ms = monotonic_ns() // 1_000_000
            update = orchestrator.start(
                round_level=round_level,
                hand=hand,
                lead_player=lead_player,  # type: ignore[arg-type]
                monotonic_ms=started_ms,
            )
        except Exception as exc:
            if source is not None:
                source.close()
            if recorder is not None:
                recording = recorder.close()
                if store is not None:
                    store.seal(
                        frame_count=recording.frame_count,
                        dropped_frames=recording.dropped_frames,
                        incident_media_failures=(
                            failure.to_dict()
                            for failure in recording.incident_media_failures
                        ),
                    )
            self.error.emit(str(exc))
            return False
        self.orchestrator = orchestrator
        self._live_source = source
        self.update_ready.emit(update)
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
                analysis.submit((snapshot, captured_ms))
            return snapshot

        worker = WorkerHandle(operation, 0.1)
        worker.frame_ready.connect(self._accept_live_frame)
        worker.error.connect(self._accept_live_error)
        worker.finished.connect(lambda: self._capture_finished(worker))
        self._capture_worker = worker
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
        self._stop_capture_worker()
        self._stop_analysis_worker()
        self._invoke(lambda value: value.pause())

    def resume(self) -> None:
        if self.orchestrator is None:
            return
        self._invoke(
            lambda value: value.resume(monotonic_ms=monotonic_ns() // 1_000_000)
        )
        self._start_analysis_worker()
        self._start_capture_worker()

    def finish(self) -> None:
        orchestrator = self.orchestrator
        if orchestrator is None or (
            self._finish_thread is not None and self._finish_thread.isRunning()
        ):
            return
        orchestrator.begin_finalizing()
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
        self.finish()
        if self._finish_thread is not None and self._finish_thread.isRunning():
            self._finish_thread.wait(30_000)
        if self._capture_worker is not None and self._capture_worker.is_running:
            self._capture_worker.stop()
            self._capture_worker.wait(10_000)
        if self._initial_thread is not None and self._initial_thread.isRunning():
            self._initial_thread.wait(10_000)

    @staticmethod
    def _manifest(config_path: Path, templates_path: Path) -> dict[str, object]:
        try:
            application_version = version("daguandan-danzero-bridge")
        except PackageNotFoundError:
            application_version = "0.1.0"
        return {
            "application_version": application_version,
            "owner_pid": os.getpid(),
            "configuration_hash": LiveAssistantController._file_hash(config_path),
            "template_manifest_hash": LiveAssistantController._file_hash(templates_path),
            "target_fps": 10,
            "codec": "MJPG",
        }

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes() if path.is_file() else b"").hexdigest()
