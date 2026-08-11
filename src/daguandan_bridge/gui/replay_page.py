from __future__ import annotations

import json
import threading
import time
import zipfile
from bisect import bisect_right
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CheckBox,
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
from ..danzero import DanzeroAdvisor
from ..image_io import save_image_unicode
from ..live.models import LiveEvent
from ..live.reducer import LiveReducer
from ..live.recognition_strategy import RECOGNITION_STRATEGY_OPTIONS
from ..live.replay import (
    EventReplayer,
    FrameIndexRecord,
    VideoReplaySource,
    replay_truth_through_live_advisor,
    replay_video_through_live_pipeline,
)
from ..live.session_store import read_json_lines
from ..live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    load_truth_log,
    save_truth_log,
)
from ..recognition_service import ScreenshotRecognitionService
from ..template_service import TemplateService
from .single_image_danzero_page import SingleImageDanzeroPage
from .truth_log_editor import TruthLogEditor
from .video_playback import ReplayDecodeThread, SessionPlaybackToolbar
from .workers import OneShotThread


class FrameInspectDialog(QDialog):
    """Inspect the paused frame: right-click copy + full single-image annotation."""

    def __init__(
        self,
        frame: np.ndarray,
        recognition: ScreenshotRecognitionService,
        info_text: str,
        parent=None,
        *,
        session: Path | None = None,
        frame_number: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("单帧查看与标注")
        self.resize(980, 900)
        self._frame = frame
        self._recognition_service = recognition
        self._danzero_advisor = DanzeroAdvisor()
        self._danzero_thread: OneShotThread | None = None
        self._session = session
        self._frame_number = frame_number
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.save_frame_button = PushButton("保存当前帧为截图")
        self.save_frame_button.setToolTip(
            "保存后可在『区域标注』页选择该图片，框选并裁剪模板"
        )
        toolbar.addWidget(self.save_frame_button)
        self.save_frame_hint = BodyLabel("")
        toolbar.addWidget(self.save_frame_hint, 1)
        layout.addLayout(toolbar)
        self.page = SingleImageDanzeroPage(parent=self)
        self.page.setWindowFlags(Qt.WindowType.Widget)
        layout.addWidget(self.page)
        self.page.recognize_requested.connect(self._recognize)
        self.page.test_requested.connect(self._run_danzero_test)
        self.page.set_frame_image(frame, info_text)
        self.save_frame_button.clicked.connect(self._save_frame)
        self._recognize()

    def _save_frame(self) -> None:
        try:
            if self._session is not None:
                target_dir = (
                    self._session.parent.parent
                    / "screenshots"
                    / self._session.name
                )
            else:
                target_dir = PROFILES_ROOT / "tencent_daguandan" / "screenshots" / "frame_inspect"
            target_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{self._frame_number:05d}" if self._frame_number is not None else ""
            path = target_dir / f"frame{suffix}.png"
            save_image_unicode(path, self._frame)
        except Exception as exc:
            self.save_frame_hint.setText(f"保存失败：{exc}")
            return
        self.save_frame_hint.setText(
            f"已保存：{path}（去『区域标注』页选择该图即可框选裁剪模板）"
        )

    def _recognize(self) -> None:
        self.page.set_recognition_busy(True)
        try:
            try:
                result = self._recognition_service.recognize(
                    self._frame,
                    allow_unknown_suit=True,
                )
            except TypeError as exc:
                if "allow_unknown_suit" not in str(exc):
                    raise
                result = self._recognition_service.recognize(self._frame)
        except Exception as exc:
            self.page.show_recognition_error(str(exc))
            self.page.set_recognition_busy(False)
            return
        self.page.apply_recognition(result)
        self.page.set_recognition_busy(False)

    def _run_danzero_test(self, state: object) -> None:
        if self._danzero_thread is not None and self._danzero_thread.isRunning():
            self.page.status.setText("正在调用 DanZero……")
            return
        operation = lambda: self._danzero_advisor.recommend(state)
        self.page.set_test_busy(True)
        thread = OneShotThread(operation, self)
        thread.result.connect(self._danzero_succeeded)
        thread.error.connect(self._danzero_failed)
        thread.finished.connect(self._danzero_finished)
        self._danzero_thread = thread
        thread.start()

    def _danzero_succeeded(self, advice: object) -> None:
        cards = "、".join(str(card) for card in advice.cards) or "不出"
        self.page.show_test_result(
            "\n".join(
                (
                    f"推荐动作：{cards}",
                    f"牌型：{advice.play_type}",
                    f"是否不出：{'是' if advice.is_pass else '否'}",
                    f"耗时：{advice.elapsed_ms:.2f} ms",
                )
            )
        )

    def _danzero_failed(self, message: str) -> None:
        self.page.show_test_error(f"调用失败：{message}")

    def _danzero_finished(self) -> None:
        self.page.set_test_busy(False)
        self._danzero_thread = None

    def shutdown(self) -> None:
        if self._danzero_thread is not None and self._danzero_thread.isRunning():
            self._danzero_thread.wait(30_000)


class VisualRecognitionReplayThread(QThread):
    completed = Signal(object)
    failed = Signal(str)
    turn_result = Signal(object)

    def __init__(
        self,
        session: Path,
        truth_log: TruthLog | None = None,
        parent=None,
        *,
        position_provider: Callable[[], int | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.truth_log = truth_log
        self._position_provider = position_provider
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()

    def _wait_for_position(self, target_frame: int) -> bool:
        """等待播放位置到达目标帧；暂停时阻塞，停止时返回 False。"""
        while not self._stop_requested.is_set():
            if self._position_provider is None:
                return True
            position = self._position_provider()
            if position is None or position >= target_frame:
                return True
            time.sleep(0.02)
        return False

    def run(self) -> None:
        try:
            profile_root = self.session.parents[2]
            profile_name = self.session.parents[1].name
            recognition = ScreenshotRecognitionService(
                AnnotationService(profile_root, profile_name),
                TemplateService(profile_root, profile_name),
            )
            result = replay_video_through_live_pipeline(
                self.session,
                recognition,
                truth_log=self.truth_log,
                stop_requested=self._stop_requested.is_set,
                on_turn=lambda data: self.turn_result.emit(data),
                wait_for_position=self._wait_for_position,
                use_live_pipeline=getattr(self, "replay_mode", "pipeline") == "pipeline",
                recognition_strategy=getattr(self, "recognition_strategy", "two_valid_streak"),
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrustedAdviceReplayThread(QThread):
    completed = Signal(object)
    failed = Signal(str)
    advice_result = Signal(object)

    def __init__(
        self,
        session: Path,
        truth_log: TruthLog,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.session = Path(session)
        self.truth_log = truth_log
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        try:
            result = replay_truth_through_live_advisor(
                self.session,
                DanzeroAdvisor(),
                truth_log=self.truth_log,
                stop_requested=self._stop_requested.is_set,
                on_advice=lambda data: self.advice_result.emit(data),
            )
            self.completed.emit(result)
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
        self._trusted_thread: TrustedAdviceReplayThread | None = None
        self._seek_frame: int | None = None
        self.truth_log: TruthLog | None = None
        self._truth_scan_base: TruthLog | None = None
        self._truth_scan_turns: list[dict[str, object]] = []
        self._current_record: FrameIndexRecord | None = None
        self._current_image: QImage | None = None
        self._recognition_service: ScreenshotRecognitionService | None = None
        self._playing = False
        self._replay_overlay: dict[int, dict[str, object]] = {}
        self._replay_overlay_frames: list[int] = []
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
        self.playback_toolbar = SessionPlaybackToolbar(self)
        # Existing callers use these page attributes; all of them now point to
        # the shared toolbar used by the annotation page as well.
        self.play_button = self.playback_toolbar.play_button
        self.step_button = self.playback_toolbar.step_button
        self.rewind_button = self.playback_toolbar.rewind_button
        self.forward_button = self.playback_toolbar.forward_button
        self.frame_spin = self.playback_toolbar.frame_spin
        self.frame_jump_button = self.playback_toolbar.frame_jump_button
        self.speed_combo = self.playback_toolbar.speed_combo
        self.frame_status = self.playback_toolbar.frame_status
        video_layout.addWidget(self.playback_toolbar)
        self.inspect_frame_button = PushButton("查看 / 标注当前帧")
        self.inspect_frame_button.setToolTip("在当前暂停帧打开单图识别与模板标注")
        video_layout.addWidget(self.inspect_frame_button)
        content.addWidget(video_card, 3)

        diagnostics_card = CardWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_card)
        diagnostics_layout.setContentsMargins(16, 14, 16, 16)
        self.diagnostics_stack = QStackedWidget()
        diag_page = QWidget()
        diag_layout = QVBoxLayout(diag_page)
        diag_layout.setContentsMargins(0, 0, 0, 0)
        diag_layout.addWidget(StrongBodyLabel("复测与诊断"))
        self.state_replay_button = PushButton("确定性状态重放")
        self.visual_replay_button = PrimaryPushButton("逐帧分析并编辑出牌日志")
        self.truth_import_button = PushButton("导入标准日志")
        self.truth_export_button = PushButton("导出标准日志")
        self.truth_replay_button = PrimaryPushButton("开始复测")
        self.truth_edit_button = PushButton("手动编辑出牌日志")
        self.diagnostics = TextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setPlaceholderText("复测结果会显示在这里。")
        diag_layout.addWidget(self.visual_replay_button)
        diag_layout.addWidget(self.truth_edit_button)
        diag_layout.addWidget(self.truth_replay_button)
        self.truth_status = BodyLabel("出牌日志：未维护")
        diag_layout.addWidget(self.truth_status)
        mode_row = QHBoxLayout()
        mode_row.addWidget(BodyLabel("复测模式"))
        self.replay_mode_combo = ComboBox()
        self.replay_mode_combo.addItem("状态机管线（实时同核心）", userData="pipeline")
        self.replay_mode_combo.addItem(
            "可信日志驱动（测试实时 DanZero）",
            userData="trusted_advisor",
        )
        mode_row.addWidget(self.replay_mode_combo)
        mode_row.addWidget(BodyLabel("识别策略"))
        self.recognition_strategy_combo = ComboBox()
        for value, label in RECOGNITION_STRATEGY_OPTIONS:
            self.recognition_strategy_combo.addItem(label, userData=value)
        default_strategy = self.recognition_strategy_combo.findData("two_valid_streak")
        if default_strategy >= 0:
            self.recognition_strategy_combo.setCurrentIndex(default_strategy)
        self.recognition_strategy_combo.setToolTip(
            "只在“状态机管线”复测中生效；与实时助手使用同一策略。"
        )
        mode_row.addWidget(self.recognition_strategy_combo)
        mode_row.addStretch(1)
        diag_layout.addLayout(mode_row)
        self.replay_overlay_check = CheckBox("播放时叠加复测识别框")
        self.replay_overlay_check.setChecked(True)
        self.replay_overlay_check.toggled.connect(self._rerender_current_frame)
        diag_layout.addWidget(self.replay_overlay_check)
        diag_layout.addWidget(self.diagnostics, 1)
        editor_page = QWidget()
        editor_layout = QVBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_header = QHBoxLayout()
        editor_header.addWidget(
            StrongBodyLabel("出牌日志 · 逐帧分析生成草稿后，在左侧录像中逐条校验和修正")
        )
        self.back_to_diagnostics_button = PushButton("返回复测")
        editor_header.addStretch(1)
        editor_header.addWidget(self.back_to_diagnostics_button)
        editor_layout.addLayout(editor_header)
        self.truth_editor_host = QVBoxLayout()
        editor_layout.addLayout(self.truth_editor_host)
        self.diagnostics_stack.addWidget(diag_page)
        self.diagnostics_stack.addWidget(editor_page)
        diagnostics_layout.addWidget(self.diagnostics_stack)
        content.addWidget(diagnostics_card, 2)
        root.addLayout(content, 1)

        self.refresh_button.clicked.connect(self.refresh_sessions)
        self.session_combo.currentIndexChanged.connect(self._session_selected)
        self.incident_combo.currentIndexChanged.connect(self._incident_selected)
        self.playback_toolbar.play_pause_requested.connect(self._toggle_play_pause)
        self.playback_toolbar.step_requested.connect(self.step)
        self.playback_toolbar.seek_requested.connect(self.seek_to_frame)
        self.playback_toolbar.seek_seconds_requested.connect(self._seek_by_seconds)
        self.playback_toolbar.speed_changed.connect(self._speed_changed)
        self.inspect_frame_button.clicked.connect(self.open_frame_inspect)
        self.replay_mode_combo.currentIndexChanged.connect(self._replay_mode_changed)
        self.state_replay_button.clicked.connect(self.replay_state)
        self.visual_replay_button.clicked.connect(self.analyze_video_to_truth_log)
        self.truth_edit_button.clicked.connect(self.edit_truth_log)
        self.truth_replay_button.clicked.connect(self.replay_truth)
        self.back_to_diagnostics_button.clicked.connect(self._back_to_diagnostics)
        self._set_session_actions(False)
        self._replay_mode_changed()

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
        self._current_record = None
        self._current_image = None
        self._replay_overlay = {}
        self._replay_overlay_frames = []
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
        frame_count = max(0, int(manifest.get("frame_count", 0) or 0))
        if frame_count <= 0:
            frame_count = len(self._index_records())
        self.playback_toolbar.set_frame_count(frame_count)
        self.session_summary.setText(
            f"{manifest.get('session_id', session.name)}　|　状态 {manifest.get('status', '未知')}　|　"
            f"录像帧 {manifest.get('frame_count', 0)}　|　丢帧 {manifest.get('dropped_frames', 0)}"
        )
        self._load_truth_log_for_session(manifest)
        self._set_session_actions(True, playable=playable)
        self._load_incidents()

    def _session_selected(self, _index: int) -> None:
        value = self.session_combo.currentData()
        if value:
            self.select_session(Path(str(value)))

    def _replay_mode_changed(self, _index: int = -1) -> None:
        trusted = self.replay_mode_combo.currentData() == "trusted_advisor"
        pipeline = self.replay_mode_combo.currentData() == "pipeline"
        self.recognition_strategy_combo.setEnabled(pipeline)
        self.truth_replay_button.setText(
            "开始实时助手测试" if trusted else "开始复测"
        )

    def _set_session_actions(self, enabled: bool, *, playable: bool = False) -> None:
        for widget in (
            self.visual_replay_button,
            self.truth_edit_button,
            self.truth_replay_button,
        ):
            widget.setEnabled(enabled and playable)
        trusted_mode = self.replay_mode_combo.currentData() == "trusted_advisor"
        self.truth_replay_button.setEnabled(
            enabled
            and (playable or trusted_mode)
            and self.truth_log is not None
            and bool(self.truth_log.turns)
        )
        for widget in (
            self.play_button,
            self.step_button,
            self.frame_spin,
            self.frame_jump_button,
            self.rewind_button,
            self.forward_button,
            self.speed_combo,
            self.inspect_frame_button,
        ):
            widget.setEnabled(enabled and playable)

    def _ensure_decode(self) -> ReplayDecodeThread | None:
        if self.current_session is None:
            return None
        if self._decode_thread is None:
            thread = ReplayDecodeThread(
                self.current_session / "video" / "game.avi",
                self.current_session / "video" / "frame_index.jsonl",
                self,
                start_frame=self._seek_frame,
            )
            self._seek_frame = None
            thread.frame_ready.connect(self._show_frame)
            thread.failed.connect(self._show_error)
            thread.finished.connect(lambda: self._decode_finished(thread))
            thread.set_speed(float(self.speed_combo.currentData() or 1.0))
            self._decode_thread = thread
            thread.start()
        return self._decode_thread

    def _index_records(self) -> tuple[FrameIndexRecord, ...]:
        try:
            return tuple(
                FrameIndexRecord.from_dict(raw)
                for raw in read_json_lines(
                    self.current_session / "video" / "frame_index.jsonl"
                )
            )
        except Exception:
            return ()

    def _restart_decode_at(
        self,
        record: FrameIndexRecord | None,
        *,
        play: bool,
    ) -> None:
        self._stop_decode()
        self._seek_frame = record.frame_index if record is not None else 0
        thread = self._ensure_decode()
        if thread is not None:
            if play:
                thread.play()
            else:
                thread.step()
        self._playing = play
        self._update_play_button()

    def _update_play_button(self) -> None:
        self.playback_toolbar.set_playing(self._playing)

    def _toggle_play_pause(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        thread = self._ensure_decode()
        if thread is not None:
            thread.play()
            self._playing = True
            self._update_play_button()

    def pause(self) -> None:
        if self._decode_thread is not None:
            self._decode_thread.pause()
        self._playing = False
        self._update_play_button()

    def step(self) -> None:
        thread = self._ensure_decode()
        if thread is not None:
            thread.pause()
            thread.step()
            self._playing = False
            self._update_play_button()

    def seek_to_frame(self, target: int | None = None) -> None:
        if self.current_session is None:
            return
        target = self.frame_spin.value() if target is None else int(target)
        records = self._index_records()
        if not records:
            self._show_error("帧索引不可读")
            return
        record = next(
            (item for item in records if item.frame_index >= target),
            None,
        )
        if record is None:
            self._show_error(f"没有找到帧 {target} 及之后的记录")
            return
        self._restart_decode_at(record, play=True)
        self.frame_status.setText(f"已跳转到帧 {target}，正在播放")

    def open_frame_inspect(self) -> None:
        if self.current_session is None:
            return
        if self._current_image is None:
            self._show_error("请先播放或暂停到目标帧，再点单帧")
            return
        frame_bgr = self._qimage_to_bgr(self._current_image)
        record = self._current_record
        info = (
            f"当前帧：{self.current_session.name}　"
            f"帧 {record.frame_index if record is not None else '—'}"
            f"　{record.monotonic_ms if record is not None else 0} ms"
        )
        dialog = FrameInspectDialog(
            frame_bgr,
            self._recognition(),
            info,
            self,
            session=self.current_session,
            frame_number=record.frame_index if record is not None else None,
        )
        dialog.exec()
        dialog.shutdown()
        dialog.deleteLater()

    def _seek_by_seconds(self, delta_seconds: float) -> None:
        if self.current_session is None:
            return
        was_playing = self._playing
        current_ms = (
            self._current_record.monotonic_ms
            if self._current_record is not None
            else 0
        )
        target_ms = max(0, current_ms + int(delta_seconds * 1000))
        records = self._index_records()
        record = next(
            (item for item in records if item.monotonic_ms >= target_ms),
            records[-1] if records else None,
        )
        self._restart_decode_at(record, play=was_playing)

    def _speed_changed(self, speed: float | int = 1.0) -> None:
        if self._decode_thread is not None:
            self._decode_thread.set_speed(
                float(speed if isinstance(speed, float) else self.speed_combo.currentData() or 1.0)
            )

    def _show_frame(self, record: FrameIndexRecord, image: QImage) -> None:
        self._current_record = record
        self._current_image = image
        if self.replay_overlay_check.isChecked():
            image = self._draw_replay_overlay(image, record)
        self.preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.playback_toolbar.set_current_frame(record.frame_index)
        self.frame_status.setText(
            f"帧 {record.frame_index}　{record.monotonic_ms} ms　此前丢帧 {record.dropped_before}"
        )

    def _add_replay_overlay(self, data: dict[str, object]) -> None:
        frame_index = data.get("frame_index")
        if frame_index is None:
            return
        frame_index = int(frame_index)
        if frame_index not in self._replay_overlay:
            self._replay_overlay_frames.append(frame_index)
            self._replay_overlay_frames.sort()
        self._replay_overlay[frame_index] = data

    def _overlay_for_frame(self, frame_index: int) -> dict[str, object] | None:
        frames = self._replay_overlay_frames
        if not frames:
            return None
        index = bisect_right(frames, frame_index) - 1
        if index < 0:
            return None
        return self._replay_overlay[frames[index]]

    def _draw_replay_overlay(self, image: QImage, record: FrameIndexRecord) -> QImage:
        data = self._overlay_for_frame(record.frame_index)
        if data is None or not data.get("boxes"):
            return image
        rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
        height, width = rgba.height(), rgba.width()
        bits = rgba.constBits()
        raw = bits.tobytes() if hasattr(bits, "tobytes") else bytes(bits)
        arr = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGBA2BGR)
        matched = bool(data.get("matched"))
        color = (70, 200, 90) if matched else (70, 70, 235)
        for box in data["boxes"]:
            x, y, box_w, box_h = (int(value) for value in box["box"])
            cv2.rectangle(bgr, (x, y), (x + box_w, y + box_h), color, 2)
        seat = self._REPLAY_SEAT_LABELS.get(
            str(data.get("actor", "")), str(data.get("actor", ""))
        )
        cards = (
            "不出"
            if bool(data.get("expected_pass"))
            else " ".join(str(card) for card in data.get("expected_cards", ()))
        )
        label = f"第{data.get('turn_id')}条 {seat} {cards} {'OK' if matched else 'X'}"
        cv2.putText(
            bgr,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    def _decode_finished(self, thread: ReplayDecodeThread) -> None:
        if thread is not self._decode_thread:
            return
        self._decode_thread = None
        self._playing = False
        self._update_play_button()

    def _stop_decode(self) -> None:
        thread, self._decode_thread = self._decode_thread, None
        self._playing = False
        self._update_play_button()
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait(5_000)

    def _load_truth_log_for_session(self, manifest: dict[str, object]) -> None:
        if self.current_session is None:
            return
        path = self.current_session / "truth_log.json"
        try:
            if path.is_file():
                self.truth_log = load_truth_log(
                    path,
                    session_id=str(manifest.get("session_id", self.current_session.name)),
                )
            else:
                self.truth_log = self._truth_log_from_session(manifest)
            self.truth_status.setText(
                f"出牌日志：{'已维护 ' + str(len(self.truth_log.turns)) + ' 条' if self.truth_log else '未维护'}"
            )
        except Exception as exc:
            self.truth_log = None
            self.truth_status.setText(f"标准日志不可用：{exc}")

    def _truth_log_from_session(self, manifest: dict[str, object]) -> TruthLog | None:
        if self.current_session is None:
            return None
        events = [
            LiveEvent.from_dict(raw)
            for raw in read_json_lines(self.current_session / "timeline.jsonl")
        ]
        initial = next(
            (event for event in events if event.event_type == "initial_state_confirmed"),
            None,
        )
        if initial is None:
            return None
        lead = initial.payload.get("lead_player")
        if lead not in {"self", "right", "opposite", "left"}:
            return None
        return TruthLog(
            source_session_id=str(manifest.get("session_id", self.current_session.name)),
            initial_state=TruthInitialState(
                str(initial.payload.get("round_level", "")),
                lead,
                tuple(str(card) for card in initial.payload.get("hand", ())),
            ),
            turns=(),
        )

    def _render_truth_log(self) -> None:
        return

    def edit_truth_log(self) -> None:
        if self.current_session is None:
            return
        if self.truth_log is None:
            _frame_index, image = self._current_frame_bgr_and_index()
            if image is None:
                image = self._first_frame_bgr()
            if image is None:
                self._show_error(
                    "请先播放或单帧到有牌局的画面（推荐开局画面），再点编辑出牌日志"
                )
                return
            try:
                self.truth_log = self._truth_log_from_recognition(image)
            except Exception as exc:
                self._show_error(f"无法从画面识别初始状态：{exc}")
                return
        self._show_truth_log_editor()

    def _show_truth_log_editor(self) -> None:
        if self.current_session is None or self.truth_log is None:
            return
        while self.truth_editor_host.count():
            item = self.truth_editor_host.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        editor = TruthLogEditor(
            self.current_session,
            self.truth_log,
            frame_provider=self._current_frame_bgr_and_index,
        )
        editor.log_saved.connect(self._on_truth_log_saved)
        self.truth_editor_host.addWidget(editor)
        self.diagnostics_stack.setCurrentIndex(1)

    def _back_to_diagnostics(self) -> None:
        self.diagnostics_stack.setCurrentIndex(0)

    def _on_truth_log_saved(self, log: object) -> None:
        self.truth_log = log
        self.truth_status.setText(f"出牌日志：已维护 {len(self.truth_log.turns)} 条")
        self.truth_replay_button.setEnabled(bool(self.truth_log.turns))

    @staticmethod
    def _qimage_to_bgr(image: QImage) -> np.ndarray:
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        height, width = image.height(), image.width()
        bits = image.constBits()
        raw = bits.tobytes() if hasattr(bits, "tobytes") else bytes(bits)
        rgba = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _current_frame_bgr_and_index(self) -> tuple[int | None, np.ndarray | None]:
        if self._current_image is None:
            return None, None
        record = self._current_record
        return (
            record.frame_index if record is not None else None,
            self._qimage_to_bgr(self._current_image),
        )

    def _recognition(self) -> ScreenshotRecognitionService:
        if self._recognition_service is None:
            if self.current_session is None:
                raise RuntimeError("尚未选择对局")
            profile_root = self.current_session.parents[2]
            profile_name = self.current_session.parents[1].name
            self._recognition_service = ScreenshotRecognitionService(
                AnnotationService(profile_root, profile_name),
                TemplateService(profile_root, profile_name),
            )
        return self._recognition_service

    def _first_frame_bgr(self) -> np.ndarray | None:
        if self.current_session is None:
            return None
        try:
            source = VideoReplaySource(
                self.current_session / "video" / "game.avi",
                self.current_session / "video" / "frame_index.jsonl",
            )
            for _record, frame in source.frames():
                return frame
        except Exception:
            return None
        return None

    def _truth_log_from_recognition(
        self,
        image: np.ndarray,
        *,
        recognition: ScreenshotRecognitionService | None = None,
    ) -> TruthLog:
        result = (recognition or self._recognition()).recognize(image)
        missing = [
            name
            for name, value in (
                ("级牌", result.round_level),
                ("我方手牌", result.my_hand),
            )
            if not value
        ]
        if missing:
            raise ValueError("未识别到" + "、".join(missing))
        lead = (
            result.lead_player
            if result.lead_player in {"self", "right", "opposite", "left"}
            else "self"
        )
        if self.current_session is not None:
            try:
                session_id = str(
                    json.loads(
                        (self.current_session / "manifest.json").read_text("utf-8")
                    ).get("session_id", self.current_session.name)
                )
            except Exception:
                session_id = self.current_session.name
        else:
            session_id = "single-image"
        return TruthLog(
            source_session_id=session_id,
            initial_state=TruthInitialState(
                str(result.round_level),
                lead,
                tuple(result.my_hand),
            ),
            turns=(),
        )

    def _truth_log_from_editor(self) -> TruthLog:
        if self.truth_log is None:
            raise ValueError("请先选择包含初始状态的录像会话")
        return self.truth_log

    def replay_truth(self) -> None:
        if self.current_session is None or self.truth_log is None:
            return
        try:
            log = self._truth_log_from_editor()
        except Exception as exc:
            self._show_error(str(exc))
            return
        if self._visual_thread is not None and self._visual_thread.isRunning():
            return
        if self.replay_mode_combo.currentData() == "trusted_advisor":
            if self._trusted_thread is not None and self._trusted_thread.isRunning():
                return
            self.replay_trusted_advisor(log)
            return
        thread = VisualRecognitionReplayThread(
            self.current_session,
            log,
            self,
            position_provider=lambda: (
                self._current_record.frame_index
                if self._current_record is not None
                else 0
            ),
        )
        thread.completed.connect(self._visual_completed)
        thread.failed.connect(self._show_error)
        thread.turn_result.connect(self._append_replay_turn_line)
        thread.finished.connect(lambda: self._visual_finished(thread))
        thread.replay_mode = str(self.replay_mode_combo.currentData() or "pipeline")
        thread.recognition_strategy = str(self.recognition_strategy_combo.currentData())
        self._visual_thread = thread
        self.truth_replay_button.setEnabled(False)
        mode_label = "状态机管线"
        self.diagnostics.setPlainText(
            f"正在播放录像并逐帧识别出牌（模式：{mode_label}），识别结果会实时输出："
        )
        thread.start()
        # 自动开始播放，边播放边输出每条识别结果
        self.play()

    def replay_trusted_advisor(self, truth_log: TruthLog) -> None:
        if self.current_session is None or self._trusted_thread is not None:
            return
        thread = TrustedAdviceReplayThread(
            self.current_session,
            truth_log,
            self,
        )
        thread.completed.connect(self._trusted_completed)
        thread.failed.connect(self._show_error)
        thread.advice_result.connect(self._append_trusted_advice_line)
        thread.finished.connect(lambda: self._trusted_finished(thread))
        self._trusted_thread = thread
        self.truth_replay_button.setEnabled(False)
        self.diagnostics.setPlainText(
            "正在使用可信出牌日志驱动实时 DanZero；不重新识别视频，"
            "结果会写入当前对局的 replay_runs：\n"
        )
        thread.start()

    _REPLAY_SEAT_LABELS = {"self": "自己", "right": "右家", "opposite": "对家", "left": "左家"}

    def _append_replay_turn_line(self, data: object) -> None:
        turn_id = int(data["turn_id"])
        seat = self._REPLAY_SEAT_LABELS.get(str(data["actor"]), str(data["actor"]))
        expected_pass = bool(data["expected_pass"])
        expected_cards = tuple(str(card) for card in data["expected_cards"])
        expected = "不出" if expected_pass else "出牌 " + " ".join(expected_cards)
        if bool(data.get("matched")):
            status = f"匹配正确（置信度 {float(data['confidence']):.2f}）"
        else:
            reason = str(data.get("reason") or "")
            if reason == "search_window_exceeded":
                status = "匹配异常：录像未识别到该回合（搜索超时）"
            elif reason == "video_end":
                status = "匹配异常：录像未识别到该回合（录像未覆盖）"
            else:
                got = (
                    "不出"
                    if bool(data.get("recognized_pass"))
                    else (" ".join(str(card) for card in data["recognized_cards"]) or "未识别到")
                )
                status = f"匹配异常：录像识别为「{got}」"
        line = (
            f"[第 {turn_id:>2} 条] {seat} {expected} ｜ 日志：{expected} ｜ {status}"
        )
        self.diagnostics.insertPlainText(line + "\n")
        scrollbar = self.diagnostics.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        frame_index = data.get("frame_index")
        if frame_index is not None:
            # 播放到该帧时立即叠加识别框（框持续到下一手出现）
            self._add_replay_overlay(data)
            if (
                self._current_record is not None
                and int(frame_index) == self._current_record.frame_index
            ):
                self._rerender_current_frame()

    def _append_trusted_advice_line(self, data: object) -> None:
        record = dict(data)
        status = str(record.get("status", "unknown"))
        request_id = str(record.get("request_id", ""))
        turn_id = record.get("turn_id", "?")
        if status == "ready":
            cards = " ".join(str(card) for card in record.get("cards", ())) or "不出"
            detail = (
                f"推荐：{cards}；牌型：{record.get('play_type', '未知')}；"
                f"耗时：{float(record.get('elapsed_ms', 0.0)):.0f} ms；"
                f"可见：{'是' if record.get('visible') else '否'}"
            )
        elif status == "failed":
            detail = f"失败：{record.get('error', '')}"
        elif status == "timeout":
            detail = "等待超时"
        else:
            detail = status
        self.diagnostics.insertPlainText(
            f"[实时 DanZero] 第 {turn_id} 手　请求 {request_id}　{detail}\n"
        )
        scrollbar = self.diagnostics.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def replay_state(self) -> None:
        if self.current_session is None:
            return
        try:
            session_id = str(
                json.loads((self.current_session / "manifest.json").read_text("utf-8"))[
                    "session_id"
                ]
            )
            if self.truth_log is not None and self.truth_log.turns:
                events = list(self.truth_log.to_events(session_id=session_id))
                source = "出牌日志（人工编制）"
            else:
                raw_events = read_json_lines(self.current_session / "timeline.jsonl")
                events = [
                    event
                    for event in (LiveEvent.from_dict(raw) for raw in raw_events)
                    if event.event_type
                    in {
                        "initial_state_confirmed",
                        "lead_player_confirmed",
                        "player_played",
                        "player_passed",
                        "player_finished",
                        "manual_confirmed_event",
                        "event_correction",
                    }
                ]
                source = "对局时间线"
            ordered = sorted(events, key=lambda event: (event.monotonic_ms, event.seq))
            reducer = LiveReducer(session_id)
            for index, event in enumerate(ordered):
                try:
                    reducer.apply(event)
                except Exception as exc:
                    previous = ordered[index - 1] if index > 0 else None
                    context = (
                        f"上一条：第 {previous.turn_id} 条"
                        if previous is not None
                        else "（这是第一条动作）"
                    )
                    self._show_error(
                        f"确定性重放在第 {event.turn_id} 条被状态机拒绝：{exc}\n"
                        f"{context}"
                    )
                    return
            result = EventReplayer(lambda: LiveReducer(session_id)).replay(events)
        except Exception as exc:
            self._show_error(str(exc))
            return
        lines = [
            f"确定性状态重放完成（基线：{source}）",
            f"事件数：{len(result.ordered_event_ids)}",
            f"最终状态版本：{result.final_snapshot.revision}",
            f"最终哈希：{result.snapshot_hashes[-1] if result.snapshot_hashes else '无事件'}",
        ]
        if result.snapshot_hashes:
            lines.append("状态轨迹哈希（每次提交后，供逐点比对）：")
            for index, digest in enumerate(result.snapshot_hashes):
                lines.append(f"  rev{index}: {digest}")
        self.diagnostics.setPlainText("\n".join(lines))

    def analyze_video_to_truth_log(self) -> None:
        if self.current_session is None or (
            self._visual_thread is not None and self._visual_thread.isRunning()
        ):
            return
        try:
            baseline = self._truth_log_for_video_scan()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._truth_scan_base = baseline
        self._truth_scan_turns = []
        thread = VisualRecognitionReplayThread(
            self.current_session,
            baseline,
            parent=self,
        )
        thread.completed.connect(self._truth_scan_completed)
        thread.failed.connect(self._show_error)
        thread.turn_result.connect(self._collect_truth_scan_turn)
        thread.finished.connect(lambda: self._visual_finished(thread))
        thread.replay_mode = "pipeline"
        thread.recognition_strategy = str(self.recognition_strategy_combo.currentData())
        self._visual_thread = thread
        self.visual_replay_button.setEnabled(False)
        self.truth_edit_button.setEnabled(False)
        self.truth_replay_button.setEnabled(False)
        self.diagnostics.setPlainText(
            "正在逐帧分析视频并生成出牌日志草稿；使用与实时助手相同的状态机和识别策略。\n"
            "完成后会自动打开编辑器，请逐条校验并手动保存。"
        )
        thread.start()

    def replay_visual(self) -> None:
        """Compatibility entry point for the former standalone visual scan."""
        self.analyze_video_to_truth_log()

    def _truth_log_for_video_scan(self) -> TruthLog:
        if self.truth_log is None:
            _frame_index, image = self._current_frame_bgr_and_index()
            if image is None:
                image = self._first_frame_bgr()
            if image is None:
                raise ValueError("请先选择包含开局画面的录像，再逐帧分析出牌日志")
            self.truth_log = self._truth_log_from_recognition(image)
        return TruthLog(
            source_session_id=self.truth_log.source_session_id,
            initial_state=self.truth_log.initial_state,
            turns=(),
            source_video=self.truth_log.source_video,
            frame_index_path=self.truth_log.frame_index_path,
        )

    def _collect_truth_scan_turn(self, data: object) -> None:
        record = dict(data)
        cards = tuple(str(card) for card in record.get("recognized_cards", ()))
        is_pass = bool(record.get("recognized_pass"))
        if not is_pass and not cards:
            return
        self._truth_scan_turns.append(record)
        seat = self._REPLAY_SEAT_LABELS.get(str(record.get("actor")), "未知座位")
        action = "不出" if is_pass else "出牌 " + " ".join(cards)
        self.diagnostics.insertPlainText(
            f"\n[逐帧分析] 第 {record.get('turn_id', '?')} 手　{seat}{action}"
        )
        scrollbar = self.diagnostics.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _truth_log_from_scan_turns(
        baseline: TruthLog,
        records: tuple[dict[str, object], ...] | list[dict[str, object]],
    ) -> TruthLog:
        turns: list[TruthTurn] = []
        seen_turns: set[int] = set()
        for raw in records:
            turn_id = int(raw.get("turn_id", 0) or 0)
            actor = str(raw.get("actor", ""))
            is_pass = bool(raw.get("recognized_pass"))
            cards = tuple(str(card) for card in raw.get("recognized_cards", ()))
            if turn_id <= 0 or turn_id in seen_turns:
                continue
            if actor not in {"self", "right", "opposite", "left"}:
                continue
            if not is_pass and not cards:
                continue
            turns.append(
                TruthTurn(
                    turn_id,
                    actor,
                    is_pass,
                    () if is_pass else cards,
                    frame_index=(
                        int(raw["frame_index"])
                        if raw.get("frame_index") is not None
                        else None
                    ),
                )
            )
            seen_turns.add(turn_id)
        return TruthLog(
            source_session_id=baseline.source_session_id,
            initial_state=baseline.initial_state,
            turns=tuple(turns),
            source_video=baseline.source_video,
            frame_index_path=baseline.frame_index_path,
        )

    def _truth_scan_completed(self, _value: object) -> None:
        if self._truth_scan_base is None:
            return
        self.truth_log = self._truth_log_from_scan_turns(
            self._truth_scan_base,
            self._truth_scan_turns,
        )
        self._truth_scan_base = None
        self.truth_status.setText(
            f"出牌日志：逐帧分析生成 {len(self.truth_log.turns)} 条，待校验保存"
        )
        self._show_truth_log_editor()

    def _visual_completed(self, value: object) -> None:
        result = value
        comparison = result.comparison
        self._replay_overlay = self._load_replay_overlay(result.output_path)
        self._rerender_current_frame()
        summary = (
            f"逐帧复测完成｜帧数：{result.frame_count}\n"
            f"一致回合：{len(comparison.identical_turn_ids)}　"
            f"缺失：{len(comparison.missing)}　新增：{len(comparison.added)}　"
            f"变化：{len(comparison.changed)}\n"
            f"逐帧结果：{result.output_path}\n"
            f"比较结果：{result.comparison_path}"
        )
        if self.current_session is not None:
            report_path = self.current_session / "truth_replay_report.txt"
            if not report_path.is_file():
                report_path = self.current_session / "visual_replay_report.txt"
            if report_path.is_file():
                # 逐条结果已在播放时实时输出，这里只补汇总
                self.diagnostics.insertPlainText("\n" + summary)
                scrollbar = self.diagnostics.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                return
        self.diagnostics.setPlainText(summary)

    def _trusted_completed(self, value: object) -> None:
        result = value
        self.diagnostics.insertPlainText(
            "\n实时 DanZero 可信日志测试完成\n"
            f"源对局：{result.source_session_id}\n"
            f"动作进度：{result.processed_turn_count}/{result.turn_count}\n"
            f"DanZero 请求：{result.advice_requested}；成功：{result.advice_ready}；"
            f"失败：{result.advice_failed}；过期：{result.advice_stale}；"
            f"超时：{result.advice_timeouts}\n"
            f"未知花色仅在 DanZero 输入副本中补全：{result.unknown_card_resolutions} 条\n"
            f"结果目录：{result.run_directory}\n"
            f"建议明细：{result.output_path}\n"
            f"汇总：{result.summary_path}\n"
        )
        scrollbar = self.diagnostics.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _load_replay_overlay(
        self,
        output_path: Path,
    ) -> dict[int, dict[str, object]]:
        overlay: dict[int, dict[str, object]] = {}
        frames: list[int] = []
        try:
            for raw in read_json_lines(output_path):
                data = raw if isinstance(raw, dict) else json.loads(str(raw))
                frame_index = data.get("frame_index")
                if frame_index is not None:
                    frame_index = int(frame_index)
                    if frame_index not in overlay:
                        frames.append(frame_index)
                    overlay[frame_index] = data
        except Exception:
            self._replay_overlay_frames = []
            return {}
        frames.sort()
        self._replay_overlay_frames = frames
        return overlay

    def _rerender_current_frame(self) -> None:
        if self._current_record is not None and self._current_image is not None:
            self._show_frame(self._current_record, self._current_image)

    def _visual_finished(self, thread: VisualRecognitionReplayThread) -> None:
        if thread is not self._visual_thread:
            return
        self._visual_thread = None
        self.visual_replay_button.setEnabled(self.current_session is not None)
        self.truth_edit_button.setEnabled(self.current_session is not None)
        self.truth_replay_button.setEnabled(
            self.current_session is not None
            and self.truth_log is not None
            and bool(self.truth_log.turns)
        )

    def _trusted_finished(self, thread: TrustedAdviceReplayThread) -> None:
        if thread is not self._trusted_thread:
            return
        self._trusted_thread = None
        self._set_session_actions(
            self.current_session is not None,
            playable=(
                self.current_session is not None
                and (self.current_session / "video" / "game.avi").is_file()
                and (self.current_session / "video" / "frame_index.jsonl").is_file()
            ),
        )

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
        trigger_ms = None
        incident_path = path / "incident.json"
        media_path = path / "media.json"
        try:
            if incident_path.is_file():
                trigger_ms = json.loads(incident_path.read_text("utf-8")).get(
                    "trigger_ms"
                )
            if trigger_ms is None and media_path.is_file():
                trigger_ms = json.loads(media_path.read_text("utf-8")).get(
                    "trigger_ms"
                )
        except (OSError, json.JSONDecodeError):
            trigger_ms = None
        if trigger_ms is not None:
            records = self._index_records()
            record = next(
                (item for item in records if item.monotonic_ms >= int(trigger_ms)),
                records[-1] if records else None,
            )
            self._restart_decode_at(record, play=False)
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
            "visual_replay_comparison.json",
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
        if self._trusted_thread is not None and self._trusted_thread.isRunning():
            self._trusted_thread.stop()
            self._trusted_thread.wait(30_000)
