from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    ComboBox,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    TextEdit,
    TitleLabel,
    isDarkTheme,
    qconfig,
)

from ..annotation_service import AnnotationService
from ..config import PROFILES_ROOT
from ..live.models import LiveEvent
from ..live.reducer import LiveReducer
from ..live.replay import EventReplayer, FrameIndexRecord, VideoReplaySource
from ..live.session_store import read_json_lines
from ..recognition_service import ScreenshotRecognitionService
from ..storage import append_json_line
from ..template_service import TemplateService


class ReplayDecodeThread(QThread):
    frame_ready = Signal(object, object)
    failed = Signal(str)

    def __init__(self, video_path: Path, index_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.index_path = index_path
        self._condition = threading.Condition()
        self._playing = False
        self._step_requested = False
        self._stop_requested = False
        self._speed = 1.0

    def play(self) -> None:
        with self._condition:
            self._playing = True
            self._condition.notify_all()

    def pause(self) -> None:
        with self._condition:
            self._playing = False

    def step(self) -> None:
        with self._condition:
            self._step_requested = True
            self._condition.notify_all()

    def set_speed(self, speed: float) -> None:
        with self._condition:
            self._speed = max(0.1, float(speed))
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    def run(self) -> None:
        previous_ms: int | None = None
        try:
            for record, frame in VideoReplaySource(
                self.video_path, self.index_path
            ).frames():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_requested
                        or self._playing
                        or self._step_requested
                    )
                    if self._stop_requested:
                        return
                    stepping = self._step_requested and not self._playing
                    self._step_requested = False
                    speed = self._speed
                if previous_ms is not None and not stepping:
                    delay = max(0.0, (record.monotonic_ms - previous_ms) / 1000 / speed)
                    with self._condition:
                        self._condition.wait(timeout=min(delay, 2.0))
                        if self._stop_requested:
                            return
                previous_ms = record.monotonic_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(record, image)
                if stepping:
                    self.pause()
        except Exception as exc:
            self.failed.emit(str(exc))


class VisualRecognitionReplayThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, session: Path, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        output = self.session / "visual_replay.jsonl"
        try:
            output.unlink(missing_ok=True)
            profile_root = self.session.parents[2]
            profile_name = self.session.parents[1].name
            recognition = ScreenshotRecognitionService(
                AnnotationService(profile_root, profile_name),
                TemplateService(profile_root, profile_name),
            )
            source = VideoReplaySource(
                self.session / "video" / "game.avi",
                self.session / "video" / "frame_index.jsonl",
            )
            count = 0
            for record, frame in source.frames():
                if self._stop_requested.is_set():
                    return
                result = recognition.recognize(frame)
                append_json_line(
                    output,
                    {
                        "frame_index": record.frame_index,
                        "monotonic_ms": record.monotonic_ms,
                        "round_level": result.round_level,
                        "current_player": result.current_player,
                        "lead_player": result.lead_player,
                        "my_hand": list(result.my_hand),
                        "events": [
                            {
                                "player": event.player,
                                "cards": list(event.cards),
                                "is_pass": event.is_pass,
                                "confidence": event.confidence,
                            }
                            for event in result.events
                        ],
                        "unresolved_fields": list(result.unresolved_fields),
                        "elapsed_ms": result.elapsed_ms,
                    },
                )
                count += 1
            self.completed.emit((output, count, source.warnings))
        except Exception as exc:
            self.failed.emit(str(exc))


class ReplayPage(QWidget):
    def __init__(self, sessions_root: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.sessions_root = Path(
            sessions_root
            or (PROFILES_ROOT / "tencent_daguandan" / "sessions")
        )
        self.current_session: Path | None = None
        self._decode_thread: ReplayDecodeThread | None = None
        self._visual_thread: VisualRecognitionReplayThread | None = None
        self.setObjectName("replayPage")
        self._build_ui()
        qconfig.themeChanged.connect(self._apply_theme)
        self._apply_theme()
        self.refresh_sessions()

    def _apply_theme(self, *_args) -> None:
        background = "#202020" if isDarkTheme() else "#f3f3f3"
        foreground = "#f5f5f5" if isDarkTheme() else "#1f1f1f"
        self.setStyleSheet(
            f"QWidget#replayPage {{ background: {background}; color: {foreground}; }}"
            f" QWidget#replayPage QLabel {{ color: {foreground}; }}"
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(14)
        root.addWidget(TitleLabel("对局回放与复测"))

        selector = CardWidget()
        selector_layout = QVBoxLayout(selector)
        selector_layout.setContentsMargins(16, 14, 16, 14)
        selector_layout.addWidget(StrongBodyLabel("选择已隔离的对局会话"))
        selector_row = QHBoxLayout()
        self.session_combo = ComboBox()
        self.refresh_button = PushButton("刷新")
        self.incident_combo = ComboBox()
        self.incident_combo.addItem("事故跳转", userData=None)
        selector_row.addWidget(self.session_combo, 1)
        selector_row.addWidget(self.refresh_button)
        selector_row.addWidget(self.incident_combo)
        selector_layout.addLayout(selector_row)
        self.session_summary = BodyLabel("尚未选择对局")
        self.session_summary.setWordWrap(True)
        selector_layout.addWidget(self.session_summary)
        root.addWidget(selector)

        content = QHBoxLayout()
        video_card = CardWidget()
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(16, 14, 16, 16)
        video_layout.addWidget(StrongBodyLabel("录像（使用逐帧原始时间戳）"))
        self.preview = QLabel("选择包含录像的对局后可播放")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet(
            "background:#20252b;color:#e8eaed;border-radius:6px;"
        )
        video_layout.addWidget(self.preview, 1)
        playback = QHBoxLayout()
        self.play_button = PrimaryPushButton("播放")
        self.pause_button = PushButton("暂停")
        self.step_button = PushButton("单帧")
        self.speed_combo = ComboBox()
        for label, speed in (("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0)):
            self.speed_combo.addItem(label, userData=speed)
        self.speed_combo.setCurrentIndex(1)
        self.frame_status = BodyLabel("帧：—")
        playback.addWidget(self.play_button)
        playback.addWidget(self.pause_button)
        playback.addWidget(self.step_button)
        playback.addWidget(self.speed_combo)
        playback.addWidget(self.frame_status, 1)
        video_layout.addLayout(playback)
        content.addWidget(video_card, 3)

        diagnostics_card = CardWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_card)
        diagnostics_layout.setContentsMargins(16, 14, 16, 16)
        diagnostics_layout.addWidget(StrongBodyLabel("复测与诊断"))
        self.state_replay_button = PushButton("确定性状态重放")
        self.visual_replay_button = PushButton("重新视觉识别")
        self.export_button = PrimaryPushButton("导出大模型诊断包")
        self.diagnostics = TextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setPlaceholderText("状态哈希、视觉复测和事故索引会显示在这里。")
        diagnostics_layout.addWidget(self.state_replay_button)
        diagnostics_layout.addWidget(self.visual_replay_button)
        diagnostics_layout.addWidget(self.export_button)
        diagnostics_layout.addWidget(self.diagnostics, 1)
        content.addWidget(diagnostics_card, 2)
        root.addLayout(content, 1)

        self.refresh_button.clicked.connect(self.refresh_sessions)
        self.session_combo.currentIndexChanged.connect(self._session_selected)
        self.incident_combo.currentIndexChanged.connect(self._incident_selected)
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.step_button.clicked.connect(self.step)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.state_replay_button.clicked.connect(self.replay_state)
        self.visual_replay_button.clicked.connect(self.replay_visual)
        self.export_button.clicked.connect(self.export_diagnostics)
        self._set_session_actions(False)

    def refresh_sessions(self) -> None:
        selected = self.current_session
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        if self.sessions_root.is_dir():
            sessions = sorted(
                (
                    path
                    for path in self.sessions_root.iterdir()
                    if path.is_dir() and (path / "manifest.json").is_file()
                ),
                reverse=True,
            )
            for path in sessions:
                self.session_combo.addItem(path.name, userData=str(path))
        self.session_combo.blockSignals(False)
        if selected is not None and selected.is_dir():
            self.select_session(selected)
        elif self.session_combo.count():
            self._session_selected(0)

    def select_session(self, session: Path) -> None:
        session = Path(session).resolve()
        self._stop_decode()
        self.current_session = session
        index = self.session_combo.findData(str(session))
        if index >= 0:
            self.session_combo.blockSignals(True)
            self.session_combo.setCurrentIndex(index)
            self.session_combo.blockSignals(False)
        try:
            manifest = json.loads((session / "manifest.json").read_text("utf-8"))
        except Exception as exc:
            self.session_summary.setText(f"会话清单不可读：{exc}")
            self._set_session_actions(False)
            return
        video = session / "video" / "game.avi"
        index_path = session / "video" / "frame_index.jsonl"
        playable = video.is_file() and index_path.is_file()
        self.session_summary.setText(
            f"{manifest.get('session_id', session.name)}　|　状态 {manifest.get('status', '未知')}　|　"
            f"录像帧 {manifest.get('frame_count', 0)}　|　丢帧 {manifest.get('dropped_frames', 0)}"
        )
        self._set_session_actions(True, playable=playable)
        self._load_incidents()

    def _session_selected(self, _index: int) -> None:
        value = self.session_combo.currentData()
        if value:
            self.select_session(Path(str(value)))

    def _set_session_actions(self, enabled: bool, *, playable: bool = False) -> None:
        for widget in (
            self.state_replay_button,
            self.visual_replay_button,
            self.export_button,
        ):
            widget.setEnabled(enabled)
        for widget in (self.play_button, self.pause_button, self.step_button, self.speed_combo):
            widget.setEnabled(enabled and playable)

    def _ensure_decode(self) -> ReplayDecodeThread | None:
        if self.current_session is None:
            return None
        if self._decode_thread is None:
            thread = ReplayDecodeThread(
                self.current_session / "video" / "game.avi",
                self.current_session / "video" / "frame_index.jsonl",
                self,
            )
            thread.frame_ready.connect(self._show_frame)
            thread.failed.connect(self._show_error)
            thread.finished.connect(self._decode_finished)
            thread.set_speed(float(self.speed_combo.currentData() or 1.0))
            self._decode_thread = thread
            thread.start()
        return self._decode_thread

    def play(self) -> None:
        thread = self._ensure_decode()
        if thread is not None:
            thread.play()

    def pause(self) -> None:
        if self._decode_thread is not None:
            self._decode_thread.pause()

    def step(self) -> None:
        thread = self._ensure_decode()
        if thread is not None:
            thread.pause()
            thread.step()

    def _speed_changed(self, _index: int) -> None:
        if self._decode_thread is not None:
            self._decode_thread.set_speed(float(self.speed_combo.currentData() or 1.0))

    def _show_frame(self, record: FrameIndexRecord, image: QImage) -> None:
        self.preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.frame_status.setText(
            f"帧 {record.frame_index}　{record.monotonic_ms} ms　此前丢帧 {record.dropped_before}"
        )

    def _decode_finished(self) -> None:
        self._decode_thread = None

    def _stop_decode(self) -> None:
        thread, self._decode_thread = self._decode_thread, None
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait(5_000)

    def replay_state(self) -> None:
        if self.current_session is None:
            return
        try:
            raw_events = read_json_lines(self.current_session / "timeline.jsonl")
            events = [LiveEvent.from_dict(raw) for raw in raw_events]
            semantic = [
                event
                for event in events
                if event.event_type
                in {
                    "initial_state_confirmed",
                    "player_played",
                    "player_passed",
                    "manual_confirmed_event",
                    "event_correction",
                }
            ]
            session_id = str(
                json.loads((self.current_session / "manifest.json").read_text("utf-8"))[
                    "session_id"
                ]
            )
            result = EventReplayer(lambda: LiveReducer(session_id)).replay(semantic)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.diagnostics.setPlainText(
            "确定性状态重放完成\n"
            f"事件数：{len(result.ordered_event_ids)}\n"
            f"最终状态版本：{result.final_snapshot.revision}\n"
            f"最终哈希：{result.snapshot_hashes[-1] if result.snapshot_hashes else '无事件'}"
        )

    def replay_visual(self) -> None:
        if self.current_session is None or (
            self._visual_thread is not None and self._visual_thread.isRunning()
        ):
            return
        thread = VisualRecognitionReplayThread(self.current_session, self)
        thread.completed.connect(self._visual_completed)
        thread.failed.connect(self._show_error)
        thread.finished.connect(self._visual_finished)
        self._visual_thread = thread
        self.visual_replay_button.setEnabled(False)
        self.diagnostics.setPlainText("正在用当前模板逐帧重新识别…")
        thread.start()

    def _visual_completed(self, value: object) -> None:
        output, count, warnings = value  # type: ignore[misc]
        self.diagnostics.setPlainText(
            f"视觉重新识别完成\n帧数：{count}\n结果：{output}\n"
            f"录像警告：{warnings or '无'}"
        )

    def _visual_finished(self) -> None:
        self._visual_thread = None
        self.visual_replay_button.setEnabled(self.current_session is not None)

    def _load_incidents(self) -> None:
        self.incident_combo.blockSignals(True)
        self.incident_combo.clear()
        self.incident_combo.addItem("事故跳转", userData=None)
        if self.current_session is not None:
            root = self.current_session / "incidents"
            if root.is_dir():
                for path in sorted(root.iterdir()):
                    if path.is_dir():
                        self.incident_combo.addItem(path.name, userData=str(path))
        self.incident_combo.blockSignals(False)

    def _incident_selected(self, _index: int) -> None:
        value = self.incident_combo.currentData()
        if not value:
            return
        path = Path(str(value))
        report = path / "llm_report.md"
        self.diagnostics.setPlainText(
            report.read_text("utf-8") if report.is_file() else f"事故目录：{path}"
        )

    def export_diagnostics(self) -> None:
        if self.current_session is None:
            return
        destination = self.current_session / "diagnostic_export.zip"
        include_names = {
            "manifest.json",
            "timeline.jsonl",
            "timeline.md",
            "advice.jsonl",
            "observations.jsonl.gz",
            "observations.jsonl.part",
            "visual_replay.jsonl",
        }
        try:
            with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in self.current_session.rglob("*"):
                    if not path.is_file() or path == destination:
                        continue
                    relative = path.relative_to(self.current_session)
                    if path.name in include_names or "incidents" in relative.parts:
                        archive.write(path, relative.as_posix())
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.diagnostics.setPlainText(f"诊断包已导出：{destination}")

    def _show_error(self, message: str) -> None:
        self.diagnostics.setPlainText(f"错误：{message}")

    def shutdown(self) -> None:
        self._stop_decode()
        if self._visual_thread is not None and self._visual_thread.isRunning():
            self._visual_thread.stop()
            self._visual_thread.wait(30_000)
