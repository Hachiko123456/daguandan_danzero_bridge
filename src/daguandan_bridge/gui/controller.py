from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from ..capture_service import CaptureService, FrameSnapshot, ScreenshotSession
from .workers import WorkerHandle


class CaptureController(QObject):
    frame_ready = Signal(object)
    frame_cleared = Signal()
    state_changed = Signal(str)
    error = Signal(str)
    screenshot_session_updated = Signal(object)
    screenshot_session_finished = Signal(object)

    def __init__(self, service: CaptureService | None = None):
        super().__init__()
        self.service = service or CaptureService()
        self.profile_name = "tencent_daguandan"
        self.current_frame: FrameSnapshot | None = None
        self._worker: WorkerHandle | None = None
        self._session: ScreenshotSession | None = None

    @property
    def is_running(self) -> bool:
        return bool(self._worker and self._worker.is_running)

    def recording_interval_ms(self) -> int:
        return int(round(self.service.recording_interval_seconds(self.profile_name) * 1000))

    def start_capture(self) -> None:
        if self.is_running:
            return
        interval = self.service.load_profile(self.profile_name).config.auto_capture_interval_sec
        self._worker = WorkerHandle(lambda: self.service.capture_frame(self.profile_name), max(0.05, min(0.5, interval / 5)))
        self._worker.frame_ready.connect(self._accept_frame)
        self._worker.error.connect(self._accept_error)
        self._worker.finished.connect(lambda: self.state_changed.emit("已停止"))
        self._worker.start()
        self.state_changed.emit("正在采集")

    def _accept_frame(self, snapshot: FrameSnapshot) -> None:
        self.current_frame = snapshot
        self.frame_ready.emit(snapshot)

    def _accept_error(self, message: str) -> None:
        self.current_frame = None
        self.frame_cleared.emit()
        self.error.emit(message)

    def stop_capture(self) -> None:
        worker, self._worker = self._worker, None
        if worker:
            worker.stop()
            worker.wait()
        self.state_changed.emit("已停止")

    def save_current_frame(self) -> Path | None:
        if self.current_frame is None:
            self.error.emit("还没有可保存的截图")
            return None
        try:
            if self._session:
                path = self.service.save_session_frame(self._session, self.current_frame)
                self.screenshot_session_updated.emit(self._session)
            else:
                path = self.service.save_frame(self.profile_name, self.current_frame)
            return path
        except Exception as exc:
            self.error.emit(str(exc))
            return None

    def start_screenshot_session(self, interval_ms: int) -> ScreenshotSession | None:
        if self._session:
            self.error.emit("当前已有进行中的对局录制")
            return None
        try:
            self._session = self.service.start_screenshot_session(self.profile_name, interval_ms)
        except Exception as exc:
            self.error.emit(str(exc))
            return None
        self.screenshot_session_updated.emit(self._session)
        if not self.is_running:
            self.start_capture()
        return self._session

    def finish_screenshot_session(self) -> ScreenshotSession | None:
        if self._session is None:
            return None
        session = self.service.finish_screenshot_session(self._session)
        self._session = None
        self.screenshot_session_finished.emit(session)
        return session

    def open_screenshot_folder(self) -> None:
        path = self.service.screenshot_folder(self.profile_name)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.error.emit(f"无法打开保存文件夹：{path}")

    def shutdown(self) -> bool:
        self.finish_screenshot_session()
        self.stop_capture()
        return True
