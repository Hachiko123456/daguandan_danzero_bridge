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
from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QScrollArea,
    QStyle,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CheckBox,
    ComboBox,
    InfoBar,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
    TextEdit,
    TitleLabel,
    isDarkTheme,
    qconfig,
)

from ..annotation_service import AnnotationService
from ..application.replay_turn_draft import ReplayTurnDraftAssembler
from ..advisor_strategy import (
    ADVISOR_OPTIONS,
    build_advisor,
    load_profile_advisor_strategy,
    normalize_advisor_strategy,
    save_profile_advisor_strategy,
)
from ..config import PROFILES_ROOT
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
        self.setWindowTitle("单图标注")
        self.resize(980, 900)
        self._frame = frame
        self._recognition_service = recognition
        if session is not None:
            self._profiles_root = Path(session).parents[2]
            self._profile_name = Path(session).parents[1].name
        else:
            self._profiles_root = PROFILES_ROOT
            self._profile_name = "tencent_daguandan"
        self._advisor_strategy = load_profile_advisor_strategy(
            self._profiles_root,
            self._profile_name,
        )
        self._danzero_advisor = build_advisor(
            self._advisor_strategy,
            profiles_root=self._profiles_root,
            profile_name=self._profile_name,
        )
        self._danzero_thread: OneShotThread | None = None
        self._session = session
        self._frame_number = frame_number
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.save_frame_button = PushButton("保存当前画面为截图")
        self.save_frame_button.setToolTip(
            "保存后可在『区域标注』页选择该图片，框选并裁剪模板"
        )
        toolbar.addWidget(self.save_frame_button)
        self.save_frame_hint = BodyLabel("")
        toolbar.addWidget(self.save_frame_hint, 1)
        layout.addLayout(toolbar)
        self.page = SingleImageDanzeroPage(
            parent=self,
            advisor_strategy=self._advisor_strategy,
        )
        self.page.setWindowFlags(Qt.WindowType.Widget)
        layout.addWidget(self.page)
        self.page.recognize_requested.connect(self._recognize)
        self.page.test_requested.connect(self._run_danzero_test)
        self.page.advisor_strategy_changed.connect(self._advisor_changed)
        self.page.set_frame_image(frame, info_text)
        self.save_frame_button.clicked.connect(self._save_frame)
        self._recognize()

    def _advisor_changed(self, strategy: str) -> None:
        self._advisor_strategy = normalize_advisor_strategy(strategy)
        profile_path = self._profiles_root / self._profile_name / "profile.json"
        if profile_path.is_file():
            self._advisor_strategy = save_profile_advisor_strategy(
                self._profiles_root,
                self._profile_name,
                self._advisor_strategy,
            )
        self._danzero_advisor = build_advisor(
            self._advisor_strategy,
            profiles_root=self._profiles_root,
            profile_name=self._profile_name,
        )

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
            self.page.status.setText("正在调用策略模型……")
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
    cancelled = Signal()
    turn_result = Signal(object)
    frame_progress = Signal(int, int, int)

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
        self._last_progress_percent: int | None = None

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

    def _emit_frame_progress(
        self,
        processed: int,
        total: int,
        frame_index: int,
    ) -> None:
        """Forward only percentage changes to the UI thread."""

        safe_total = max(1, total)
        percent = min(100, max(0, processed) * 100 // safe_total)
        if (
            processed not in {0, total}
            and percent == self._last_progress_percent
        ):
            return
        self._last_progress_percent = percent
        self.frame_progress.emit(processed, total, frame_index)

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
                on_progress=self._emit_frame_progress,
                wait_for_position=self._wait_for_position,
                use_live_pipeline=getattr(self, "replay_mode", "pipeline") == "pipeline",
                recognition_strategy=getattr(self, "recognition_strategy", "two_valid_streak"),
                use_saved_baseline=bool(getattr(self, "use_saved_baseline", False)),
            )
            if self._stop_requested.is_set():
                self.cancelled.emit()
            else:
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
        *,
        advisor_strategy: str = "danzero",
    ) -> None:
        super().__init__(parent)
        self.session = Path(session)
        self.truth_log = truth_log
        self.advisor_strategy = normalize_advisor_strategy(advisor_strategy)
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        try:
            profiles_root = self.session.parents[2]
            profile_name = self.session.parents[1].name
            result = replay_truth_through_live_advisor(
                self.session,
                build_advisor(
                    self.advisor_strategy,
                    profiles_root=profiles_root,
                    profile_name=profile_name,
                ),
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
        self.profiles_root = self.sessions_root.parent.parent
        self.profile_name = self.sessions_root.parent.name
        self.advisor_strategy = load_profile_advisor_strategy(
            self.profiles_root,
            self.profile_name,
        )
        self.current_session: Path | None = None
        self._decode_thread: ReplayDecodeThread | None = None
        self._visual_thread: VisualRecognitionReplayThread | None = None
        self._trusted_thread: TrustedAdviceReplayThread | None = None
        self._seek_frame: int | None = None
        self.truth_log: TruthLog | None = None
        self._truth_scan_base: TruthLog | None = None
        self._truth_scan_log: TruthLog | None = None
        self._truth_scan_turns: list[dict[str, object]] = []
        self._truth_draft_assembler: ReplayTurnDraftAssembler | None = None
        self._truth_scan_next_source_turn_id = 1
        self._truth_scan_failure: str | None = None
        self._truth_scan_status: str = "idle"
        self._truth_scan_status_reason: str = ""
        self._truth_scan_untrusted_passes: list[dict[str, object]] = []
        self._truth_scan_discard_actions = False
        self._truth_scan_session: Path | None = None
        self._truth_scan_progress_processed = 0
        self._truth_scan_progress_total = 0
        self._truth_scan_progress_frame_index = 0
        self._truth_editor: TruthLogEditor | None = None
        self._content_vertical: bool | None = None
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
        dark = bool(isDarkTheme())
        colors = (
            {
                "background": "#202020",
                "surface": "#292929",
                "surface_alt": "#242424",
                "base": "#1e1e1e",
                "border": "#454545",
                "foreground": "#f5f5f5",
                "muted": "#c8c8c8",
                "selection": "#0f6cbd",
                "selection_text": "#ffffff",
            }
            if dark
            else {
                "background": "#f3f3f3",
                "surface": "#ffffff",
                "surface_alt": "#f8f8f8",
                "base": "#ffffff",
                "border": "#d6d6d6",
                "foreground": "#1f1f1f",
                "muted": "#606060",
                "selection": "#0f6cbd",
                "selection_text": "#ffffff",
            }
        )
        self.setStyleSheet(
            f"QWidget#replayPage {{ background:{colors['background']};"
            f" color:{colors['foreground']}; }}"
            f" QWidget#replayPage QLabel {{ color:{colors['foreground']}; }}"
            f" QScrollArea#replayContentScroll, QWidget#replayContentViewport,"
            f" QWidget#replayContentHost {{ background:{colors['background']};"
            " border:0; }}"
            f" QWidget#replaySelectorCard, QWidget#replayVideoCard,"
            f" QWidget#replayDiagnosticsCard, QStackedWidget#replayDiagnosticsStack,"
            f" QWidget#replayDiagnosticsPage, QWidget#replayEditorPage {{"
            f" background:{colors['surface']}; color:{colors['foreground']}; }}"
            f" QTextEdit#replayDiagnostics {{ background:{colors['base']};"
            f" color:{colors['foreground']}; border:1px solid {colors['border']};"
            f" selection-background-color:{colors['selection']};"
            f" selection-color:{colors['selection_text']}; }}"
            f" QWidget#truthLogEditor {{ background:{colors['surface']};"
            f" color:{colors['foreground']}; }}"
            f" QWidget#truthLogEditor QTableWidget {{ background:{colors['base']};"
            f" alternate-background-color:{colors['surface_alt']};"
            f" color:{colors['foreground']}; gridline-color:{colors['border']};"
            f" border:1px solid {colors['border']};"
            f" selection-background-color:{colors['selection']};"
            f" selection-color:{colors['selection_text']}; }}"
            f" QWidget#truthLogEditor QHeaderView::section {{"
            f" background:{colors['surface_alt']}; color:{colors['foreground']};"
            f" border:0; border-right:1px solid {colors['border']};"
            f" border-bottom:1px solid {colors['border']}; padding:5px; }}"
            f" QWidget#truthLogEditor QScrollArea {{ background:{colors['base']};"
            f" border:1px solid {colors['border']}; }}"
            f" QWidget#truthLogEditor QComboBox,"
            f" QWidget#truthLogEditor QPushButton {{"
            f" background:{colors['surface_alt']}; color:{colors['foreground']};"
            f" border:1px solid {colors['border']}; border-radius:4px;"
            " padding:4px 8px; }}"
            f" QWidget#truthLogEditor QComboBox:disabled,"
            f" QWidget#truthLogEditor QPushButton:disabled {{"
            f" color:{colors['muted']}; }}"
            f" QWidget#truthLogEditor QComboBox QAbstractItemView {{"
            f" background:{colors['base']}; color:{colors['foreground']};"
            f" selection-background-color:{colors['selection']};"
            f" selection-color:{colors['selection_text']}; }}"
        )
        foreground = QColor(colors["foreground"])
        muted = QColor(colors["muted"])
        selection = QColor(colors["selection"])
        selection_text = QColor(colors["selection_text"])

        def apply_palette(widget: QWidget, *, window: str, base: str) -> None:
            palette = widget.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor(window))
            palette.setColor(QPalette.ColorRole.Base, QColor(base))
            palette.setColor(
                QPalette.ColorRole.AlternateBase,
                QColor(colors["surface_alt"]),
            )
            for role in (
                QPalette.ColorRole.WindowText,
                QPalette.ColorRole.Text,
                QPalette.ColorRole.ButtonText,
            ):
                palette.setColor(role, foreground)
                palette.setColor(QPalette.ColorGroup.Disabled, role, muted)
            palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
            palette.setColor(QPalette.ColorRole.Highlight, selection)
            palette.setColor(QPalette.ColorRole.HighlightedText, selection_text)
            widget.setPalette(palette)

        apply_palette(
            self,
            window=colors["background"],
            base=colors["base"],
        )
        for widget in (
            self.content_scroll,
            self.content_scroll.viewport(),
            self.content_host,
        ):
            apply_palette(
                widget,
                window=colors["background"],
                base=colors["background"],
            )
            widget.setAutoFillBackground(True)
        for widget in (
            self.selector_card,
            self.video_card,
            self.diagnostics_card,
            self.diagnostics_stack,
            self.diagnostics_page,
            self.editor_page,
        ):
            apply_palette(
                widget,
                window=colors["surface"],
                base=colors["base"],
            )
        apply_palette(
            self.diagnostics,
            window=colors["base"],
            base=colors["base"],
        )
        for editor in self.findChildren(TruthLogEditor):
            apply_palette(
                editor,
                window=colors["surface"],
                base=colors["base"],
            )
            for table in editor.findChildren(QTableWidget):
                apply_palette(
                    table,
                    window=colors["base"],
                    base=colors["base"],
                )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(14)
        root.addWidget(TitleLabel("对局回放与复测"))

        selector = CardWidget()
        self.selector_card = selector
        selector.setObjectName("replaySelectorCard")
        selector_layout = QVBoxLayout(selector)
        selector_layout.setContentsMargins(16, 14, 16, 14)
        selector_layout.addWidget(StrongBodyLabel("选择已隔离的对局会话"))
        selector_row = QHBoxLayout()
        self.session_combo = ComboBox()
        self.refresh_button = PushButton("刷新")
        selector_row.addWidget(self.session_combo, 1)
        selector_row.addWidget(self.refresh_button)
        selector_layout.addLayout(selector_row)
        self.session_summary = BodyLabel("尚未选择对局")
        self.session_summary.setWordWrap(True)
        selector_layout.addWidget(self.session_summary)
        root.addWidget(selector)

        self.content_scroll = QScrollArea()
        self.content_scroll.setObjectName("replayContentScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.content_host = QWidget()
        self.content_host.setObjectName("replayContentHost")
        self.content_scroll.viewport().setObjectName("replayContentViewport")
        self.content_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(14)
        self.content_host.setLayout(self.content_layout)
        self.content_scroll.setWidget(self.content_host)
        video_card = CardWidget()
        self.video_card = video_card
        video_card.setObjectName("replayVideoCard")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(16, 14, 16, 16)
        video_layout.addWidget(StrongBodyLabel("录像（使用逐帧原始时间戳）"))
        self.preview = QLabel("选择包含录像的对局后可播放")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        self.preview.setStyleSheet(
            "background:#20252b;color:#e8eaed;border-radius:6px;"
        )
        preview_host = QWidget()
        preview_grid = QGridLayout(preview_host)
        preview_grid.setContentsMargins(0, 0, 0, 0)
        preview_grid.addWidget(self.preview, 0, 0)
        self.rewind_overlay_button = QToolButton()
        self.rewind_overlay_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft)
        )
        self.rewind_overlay_button.setToolTip("后退 5 秒")
        self.rewind_overlay_button.setAccessibleName("后退 5 秒")
        self.forward_overlay_button = QToolButton()
        self.forward_overlay_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight)
        )
        self.forward_overlay_button.setToolTip("前进 5 秒")
        self.forward_overlay_button.setAccessibleName("前进 5 秒")
        for button in (self.rewind_overlay_button, self.forward_overlay_button):
            button.setFixedSize(42, 58)
            button.setStyleSheet(
                "QToolButton{background:rgba(15,15,15,150);color:white;"
                "border:1px solid rgba(255,255,255,90);border-radius:8px;"
                "font-size:34px;} QToolButton:hover{background:rgba(0,120,212,190);}"
            )
        preview_grid.addWidget(
            self.rewind_overlay_button,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        preview_grid.addWidget(
            self.forward_overlay_button,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        video_layout.addWidget(preview_host, 1)
        self.playback_toolbar = SessionPlaybackToolbar(self, overlay_seek=True)
        # Existing callers use these page attributes; all of them now point to
        # the shared toolbar used by the annotation page as well.
        self.play_button = self.playback_toolbar.play_button
        self.step_button = self.playback_toolbar.step_button
        self.step_button.setText("单图标注")
        self.step_button.setToolTip(
            "打开当前显示画面的单图识别与模板标注；不会前进视频"
        )
        self.step_button.setAccessibleName("单图标注")
        self.rewind_button = self.playback_toolbar.rewind_button
        self.forward_button = self.playback_toolbar.forward_button
        self.frame_spin = self.playback_toolbar.frame_spin
        self.frame_jump_button = self.playback_toolbar.frame_jump_button
        self.speed_combo = self.playback_toolbar.speed_combo
        self.frame_status = self.playback_toolbar.frame_status
        video_layout.addWidget(self.playback_toolbar)
        self.content_layout.addWidget(video_card, 3)

        diagnostics_card = CardWidget()
        self.diagnostics_card = diagnostics_card
        diagnostics_card.setObjectName("replayDiagnosticsCard")
        diagnostics_layout = QVBoxLayout(diagnostics_card)
        diagnostics_layout.setContentsMargins(16, 14, 16, 16)
        self.diagnostics_stack = QStackedWidget()
        self.diagnostics_stack.setObjectName("replayDiagnosticsStack")
        diag_page = QWidget()
        self.diagnostics_page = diag_page
        diag_page.setObjectName("replayDiagnosticsPage")
        diag_layout = QVBoxLayout(diag_page)
        diag_layout.setContentsMargins(0, 0, 0, 0)
        diag_layout.addWidget(StrongBodyLabel("复测与诊断"))
        self.state_replay_button = PushButton("状态重放")
        self.visual_replay_button = PrimaryPushButton("扫描出牌")
        self.truth_import_button = PushButton("导入日志")
        self.truth_export_button = PushButton("导出日志")
        self.truth_replay_button = PrimaryPushButton("复测")
        self.truth_edit_button = PushButton("编辑日志")
        self.diagnostics = TextEdit()
        self.diagnostics.setObjectName("replayDiagnostics")
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setPlaceholderText("复测结果会显示在这里。")
        diag_layout.addWidget(self.visual_replay_button)
        diag_layout.addWidget(self.truth_edit_button)
        diag_layout.addWidget(self.truth_replay_button)
        self.truth_status = BodyLabel("出牌日志：未维护")
        diag_layout.addWidget(self.truth_status)
        self.truth_scan_status = BodyLabel("扫描：未开始")
        self.truth_scan_status.setWordWrap(True)
        self.truth_scan_progress = QProgressBar()
        self.truth_scan_progress.setObjectName("truthScanProgress")
        self.truth_scan_progress.setRange(0, 1)
        self.truth_scan_progress.setValue(0)
        self.truth_scan_progress.setVisible(False)
        self.truth_scan_progress.setEnabled(False)
        mode_row = QHBoxLayout()
        mode_row.addWidget(BodyLabel("复测模式"))
        self.replay_mode_combo = ComboBox()
        self.replay_mode_combo.addItem("状态机管线（实时同核心）", userData="pipeline")
        self.replay_mode_combo.addItem(
            "可信日志驱动（测试实时策略）",
            userData="trusted_advisor",
        )
        mode_row.addWidget(self.replay_mode_combo)
        mode_row.addWidget(BodyLabel("建议模型"))
        self.advisor_strategy_combo = ComboBox()
        for value, label in ADVISOR_OPTIONS:
            self.advisor_strategy_combo.addItem(label, userData=value)
        advisor_index = self.advisor_strategy_combo.findData(self.advisor_strategy)
        if advisor_index >= 0:
            self.advisor_strategy_combo.setCurrentIndex(advisor_index)
        self.advisor_strategy_combo.setToolTip(
            "用于可信日志驱动复测，并持久化为当前 profile 默认值。"
        )
        mode_row.addWidget(self.advisor_strategy_combo)
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
        self.editor_page = editor_page
        editor_page.setObjectName("replayEditorPage")
        editor_layout = QVBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_header = QHBoxLayout()
        editor_header.addWidget(
            StrongBodyLabel("出牌日志 · 扫描完成后，在左侧录像中逐条校验和修正")
        )
        self.back_to_diagnostics_button = PushButton("返回")
        editor_header.addStretch(1)
        editor_header.addWidget(self.back_to_diagnostics_button)
        editor_layout.addLayout(editor_header)
        self.truth_editor_host = QVBoxLayout()
        editor_layout.addLayout(self.truth_editor_host)
        self.diagnostics_stack.addWidget(diag_page)
        self.diagnostics_stack.addWidget(editor_page)
        diagnostics_layout.addWidget(self.diagnostics_stack, 1)
        diagnostics_layout.addWidget(self.truth_scan_status)
        diagnostics_layout.addWidget(self.truth_scan_progress)
        self.content_layout.addWidget(diagnostics_card, 2)
        root.addWidget(self.content_scroll, 1)

        self.refresh_button.clicked.connect(self.refresh_sessions)
        self.session_combo.currentIndexChanged.connect(self._session_selected)
        self.playback_toolbar.play_pause_requested.connect(self._toggle_play_pause)
        self.playback_toolbar.step_requested.connect(self.open_frame_inspect)
        self.playback_toolbar.seek_requested.connect(self.seek_to_frame)
        self.playback_toolbar.seek_seconds_requested.connect(self._seek_by_seconds)
        self.rewind_overlay_button.clicked.connect(
            lambda: self.playback_toolbar.seek_seconds_requested.emit(-5.0)
        )
        self.forward_overlay_button.clicked.connect(
            lambda: self.playback_toolbar.seek_seconds_requested.emit(5.0)
        )
        self.playback_toolbar.speed_changed.connect(self._speed_changed)
        self.replay_mode_combo.currentIndexChanged.connect(self._replay_mode_changed)
        self.advisor_strategy_combo.currentIndexChanged.connect(
            self._advisor_strategy_changed
        )
        self.state_replay_button.clicked.connect(self.replay_state)
        self.visual_replay_button.clicked.connect(self.analyze_video_to_truth_log)
        self.truth_edit_button.clicked.connect(self.edit_truth_log)
        self.truth_replay_button.clicked.connect(self.replay_truth)
        self.back_to_diagnostics_button.clicked.connect(self._back_to_diagnostics)
        self._set_session_actions(False)
        self._replay_mode_changed()
        self._arrange_content(vertical=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not hasattr(self, "content_layout"):
            return
        width = event.size().width()
        if width <= 1160:
            self._arrange_content(vertical=True)
        elif width >= 1220:
            self._arrange_content(vertical=False)

    def _arrange_content(self, *, vertical: bool) -> None:
        if vertical == self._content_vertical:
            return
        self._content_vertical = vertical
        self.content_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if vertical
            else QBoxLayout.Direction.LeftToRight
        )
        self.content_layout.setStretch(0, 1 if vertical else 3)
        self.content_layout.setStretch(1, 1 if vertical else 2)

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
                label = path.name
                try:
                    manifest = json.loads(
                        (path / "manifest.json").read_text(encoding="utf-8")
                    )
                    metrics = manifest.get("performance_metrics", {})
                    if (
                        isinstance(metrics, dict)
                        and metrics.get("recording_mode") == "listening_only"
                    ):
                        label = f"监听录像 · {path.name}"
                except (OSError, json.JSONDecodeError):
                    pass
                self.session_combo.addItem(label, userData=str(path))
        self.session_combo.blockSignals(False)
        if selected is not None and selected.is_dir():
            self.select_session(selected)
        elif self.session_combo.count():
            self._session_selected(0)

    def select_session(self, session: Path) -> None:
        session = Path(session).resolve()
        if self._visual_thread is not None and self._visual_thread.isRunning():
            self._visual_thread.stop()
        self._truth_scan_base = None
        self._truth_scan_log = None
        self._truth_draft_assembler = None
        self._truth_scan_next_source_turn_id = 1
        self._truth_scan_failure = None
        self._truth_scan_session = None
        if hasattr(self, "truth_scan_status"):
            self.truth_scan_status.setText("扫描：未开始")
        self._reset_truth_scan_progress()
        if self._truth_editor is not None:
            self._truth_editor.shutdown()
        self._truth_editor = None
        if hasattr(self, "truth_editor_host"):
            while self.truth_editor_host.count():
                item = self.truth_editor_host.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.diagnostics_stack.setCurrentIndex(0)
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

    def _session_selected(self, _index: int) -> None:
        value = self.session_combo.currentData()
        if value:
            self.select_session(Path(str(value)))

    def _replay_mode_changed(self, _index: int = -1) -> None:
        trusted = self.replay_mode_combo.currentData() == "trusted_advisor"
        pipeline = self.replay_mode_combo.currentData() == "pipeline"
        self.recognition_strategy_combo.setEnabled(pipeline)
        self.advisor_strategy_combo.setEnabled(trusted)
        self.truth_replay_button.setText("助手复测" if trusted else "复测")

    def _advisor_strategy_changed(self, _index: int = -1) -> None:
        self.advisor_strategy = normalize_advisor_strategy(
            self.advisor_strategy_combo.currentData()
        )
        profile_path = self.profiles_root / self.profile_name / "profile.json"
        if not profile_path.is_file():
            return
        try:
            self.advisor_strategy = save_profile_advisor_strategy(
                self.profiles_root,
                self.profile_name,
                self.advisor_strategy,
            )
        except Exception as exc:
            self._show_error(str(exc))

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
            self.rewind_overlay_button,
            self.forward_overlay_button,
            self.speed_combo,
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
            self._show_error("请先播放或暂停到目标画面，再点『单图标注』")
            return
        frame_bgr = self._qimage_to_bgr(self._current_image)
        record = self._current_record
        info = (
            f"当前画面：{self.current_session.name}　"
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
            first_action_index = next(
                (
                    index
                    for index, event in enumerate(events)
                    if event.event_type
                    in {"player_played", "player_passed", "manual_confirmed_event"}
                ),
                len(events),
            )
            confirmed_lead = next(
                (
                    event
                    for index, event in enumerate(events)
                    if index < first_action_index
                    and event.event_type == "lead_player_confirmed"
                ),
                None,
            )
            if confirmed_lead is not None:
                candidate = confirmed_lead.payload.get("lead_player")
                lead = (
                    candidate
                    if candidate in {"self", "right", "opposite", "left"}
                    else confirmed_lead.actor
                )
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
                    "请先播放或点『下一帧』到有牌局的画面（推荐开局画面），"
                    "再点编辑出牌日志"
                )
                return
            try:
                self.truth_log = self._truth_log_from_recognition(image)
            except Exception as exc:
                self._show_error(f"无法从画面识别初始状态：{exc}")
                return
        self._show_truth_log_editor()

    def _show_truth_log_editor(self, log: TruthLog | None = None) -> None:
        log = log or self.truth_log
        if self.current_session is None or log is None:
            return
        while self.truth_editor_host.count():
            item = self.truth_editor_host.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        editor = TruthLogEditor(
            self.current_session,
            log,
            frame_provider=self._current_frame_bgr_and_index,
            frame_scan_provider=self._frames_after_current,
        )
        editor.log_saved.connect(self._on_truth_log_saved)
        self._truth_editor = editor
        self.truth_editor_host.addWidget(editor)
        self.diagnostics_stack.setCurrentIndex(1)
        self._apply_theme()

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

    def _frames_after_current(self, frame_index: int):
        """Read a small editor-only look-ahead without changing playback."""

        if self.current_session is None:
            return
        source = VideoReplaySource(
            self.current_session / "video" / "game.avi",
            self.current_session / "video" / "frame_index.jsonl",
        )
        for count, (record, image) in enumerate(
            source.frames(start_frame=int(frame_index) + 1), start=1
        ):
            yield record.frame_index, image
            if count >= 24:
                return

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
        thread.finished.connect(self._on_visual_thread_finished)
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
            advisor_strategy=self.advisor_strategy,
        )
        thread.completed.connect(self._trusted_completed)
        thread.failed.connect(self._show_error)
        thread.advice_result.connect(self._append_trusted_advice_line)
        thread.finished.connect(lambda: self._trusted_finished(thread))
        self._trusted_thread = thread
        self.truth_replay_button.setEnabled(False)
        self.diagnostics.setPlainText(
            f"正在使用可信出牌日志驱动实时 {self.advisor_strategy_combo.currentText()}；不重新识别视频，"
            "结果会写入当前对局的 replay_runs：\n"
        )
        thread.start()

    def _truth_scan_failed(self, message: str) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._visual_thread:
            return
        if self._truth_scan_session != self.current_session:
            return
        self._fail_truth_scan(str(message))

    @Slot()
    def _truth_scan_cancelled(self) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._visual_thread:
            return
        if self._truth_scan_session != self.current_session:
            return
        if self._truth_scan_failure is None:
            self._fail_truth_scan("扫描已取消")
        self._reset_truth_scan_progress()

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
            f"[实时策略] 第 {turn_id} 手　请求 {request_id}　{detail}\n"
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
            self._begin_truth_scan(baseline)
        except Exception as exc:
            message = str(exc)
            self.truth_scan_status.setText(f"扫描失败：{message}")
            self._show_error(message)
            return
        thread = VisualRecognitionReplayThread(
            self.current_session,
            baseline,
            parent=self,
        )
        thread.completed.connect(self._truth_scan_completed)
        thread.failed.connect(self._truth_scan_failed)
        thread.cancelled.connect(self._truth_scan_cancelled)
        thread.turn_result.connect(self._collect_truth_scan_turn)
        thread.frame_progress.connect(self._truth_scan_progress_changed)
        thread.finished.connect(self._on_visual_thread_finished)
        thread.replay_mode = "pipeline"
        thread.use_saved_baseline = True
        thread.recognition_strategy = str(self.recognition_strategy_combo.currentData())
        self._visual_thread = thread
        self.visual_replay_button.setEnabled(False)
        self.truth_edit_button.setEnabled(False)
        self.truth_replay_button.setEnabled(False)
        self.truth_scan_status.setText("扫描中：正在逐帧识别出牌，编辑与保存暂不可用")
        self.diagnostics.setPlainText(
            "正在逐帧识别出牌；使用与实时助手相同的状态机和识别策略。\n"
            "识别结果仅保留在内存；完成后可编辑并点击『保存日志』写入 truth_log.json。"
        )
        thread.start()

    def replay_visual(self) -> None:
        """Compatibility entry point for the former standalone visual scan."""
        self.analyze_video_to_truth_log()

    def _truth_log_for_video_scan(self) -> TruthLog:
        if self.truth_log is None:
            self.truth_log = self._truth_log_from_session({})
        if self.truth_log is None:
            raise ValueError("会话缺少已确认的首出玩家，无法安全扫描出牌日志")
        return TruthLog(
            source_session_id=self.truth_log.source_session_id,
            initial_state=self.truth_log.initial_state,
            turns=(),
            source_video=self.truth_log.source_video,
            frame_index_path=self.truth_log.frame_index_path,
            label_status=self.truth_log.label_status,
            provenance=self.truth_log.provenance,
            outcome=self.truth_log.outcome,
        )

    def _begin_truth_scan(self, baseline: TruthLog) -> None:
        """Prepare an in-memory scan; only the later explicit save writes truth."""

        if self.current_session is None:
            raise RuntimeError("尚未选择对局")
        self._truth_scan_base = baseline
        self._truth_scan_log = baseline
        self._truth_scan_turns = []
        self._truth_draft_assembler = ReplayTurnDraftAssembler(baseline)
        self._truth_scan_next_source_turn_id = 1
        self._truth_scan_failure = None
        self._truth_scan_status = "running"
        self._truth_scan_status_reason = ""
        self._truth_scan_untrusted_passes = []
        self._truth_scan_discard_actions = False
        self._truth_scan_session = self.current_session
        self._start_truth_scan_progress()
        self._show_truth_log_editor(baseline)
        if self._truth_editor is not None:
            self._truth_editor.setEnabled(False)

    def _start_truth_scan_progress(self) -> None:
        self._truth_scan_progress_processed = 0
        self._truth_scan_progress_total = 0
        self._truth_scan_progress_frame_index = 0
        self.truth_scan_progress.setRange(0, 1)
        self.truth_scan_progress.setValue(0)
        self.truth_scan_progress.setEnabled(True)
        self.truth_scan_progress.setVisible(True)

    def _reset_truth_scan_progress(self) -> None:
        if not hasattr(self, "truth_scan_progress"):
            return
        self._truth_scan_progress_processed = 0
        self._truth_scan_progress_total = 0
        self._truth_scan_progress_frame_index = 0
        self.truth_scan_progress.setRange(0, 1)
        self.truth_scan_progress.setValue(0)
        self.truth_scan_progress.setEnabled(False)
        self.truth_scan_progress.setVisible(False)

    @Slot(int, int, int)
    def _truth_scan_progress_changed(
        self,
        processed: int,
        total: int,
        frame_index: int,
    ) -> None:
        sender = self.sender()
        if (
            sender is not None
            and sender is not self._visual_thread
        ):
            return
        if (
            self._truth_scan_base is None
            or self._truth_scan_session != self.current_session
            or self._truth_scan_failure is not None
        ):
            return
        safe_total = max(1, int(total))
        safe_processed = min(safe_total, max(0, int(processed)))
        self._truth_scan_progress_processed = safe_processed
        self._truth_scan_progress_total = safe_total
        self._truth_scan_progress_frame_index = int(frame_index)
        self.truth_scan_progress.setRange(0, safe_total)
        self.truth_scan_progress.setValue(safe_processed)
        self.truth_scan_progress.setVisible(True)
        percent = (safe_processed * 100 + safe_total // 2) // safe_total
        self.truth_scan_status.setText(
            f"扫描中：{safe_processed}/{safe_total} 帧（{percent}%）"
        )

    def _fail_truth_scan(self, message: str) -> None:
        """Stop a malformed stream before it can be silently reindexed."""

        if self._truth_scan_failure is not None:
            return
        self._truth_scan_failure = message
        self.truth_scan_status.setText(f"扫描失败：{message}；已保留现有正式日志")
        self._reset_truth_scan_progress()
        self.truth_status.setText("出牌日志：扫描未完成，未自动保存")
        InfoBar.error(
            title="扫描失败",
            content=f"{message}；未保存，正式日志保持不变",
            parent=self,
        )
        self.diagnostics.insertPlainText(f"\n[扫描失败] {message}\n")
        if self._truth_editor is not None:
            self._truth_editor.setEnabled(True)
        thread = self._visual_thread
        if thread is not None and thread.isRunning():
            thread.stop()

    def _collect_truth_scan_turn(self, data: object) -> None:
        if (
            self._truth_scan_failure is not None
            or self._truth_scan_base is None
            or self._truth_scan_session != self.current_session
        ):
            return
        record = dict(data)
        kind = str(record.get("kind", "action"))
        if kind in {"suit_corrected", "event_correction"}:
            if self._truth_draft_assembler is None and self._truth_scan_base is not None:
                self._truth_draft_assembler = ReplayTurnDraftAssembler(
                    self._truth_scan_base
                )
            if self._truth_draft_assembler is None:
                return
            corrected = (
                self._truth_draft_assembler.apply_suit_correction(record)
                if kind == "suit_corrected"
                else self._truth_draft_assembler.apply_event_correction(record)
            )
            if not corrected.accepted or corrected.turn is None:
                return
            self._truth_scan_turns.append(record)
            self._truth_scan_log = corrected.truth_log
            if self._truth_editor is not None:
                self._truth_editor.replace_confirmed_turn(
                    corrected.turn,
                    status=corrected.status,
                )
            seat = self._REPLAY_SEAT_LABELS.get(
                str(record.get("actor")),
                "未知座位",
            )
            cards = " ".join(str(card) for card in record.get("recognized_cards", ()))
            self.diagnostics.insertPlainText(
                f"\n[逐帧分析] {'花色修正' if kind == 'suit_corrected' else '动作修正'}"
                f"　第 {record.get('target_turn_id', '?')} 手"
                f"　{seat} {cards}"
            )
            scrollbar = self.diagnostics.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            return
        if self._truth_scan_discard_actions:
            return
        if (
            bool(record.get("recognized_pass"))
            and str(record.get("pass_audit", "")) == "inferred"
        ):
            self._truth_scan_untrusted_passes.append(record)
            self._truth_scan_discard_actions = True
            reason = (
                "检测到无充分视觉证据的推导不出；后续动作不写入可靠草稿"
            )
            self._truth_scan_status = "partial"
            self._truth_scan_status_reason = reason
            self.diagnostics.insertPlainText(f"\n[扫描待复核] {reason}\n")
            return
        cards = tuple(str(card) for card in record.get("recognized_cards", ()))
        is_pass = bool(record.get("recognized_pass"))
        if not is_pass and not cards:
            self._fail_truth_scan(
                f"第 {record.get('turn_id', '?')} 手缺少有效出牌，未写入"
            )
            return
        if self._truth_draft_assembler is None and self._truth_scan_base is not None:
            self._truth_draft_assembler = ReplayTurnDraftAssembler(self._truth_scan_base)
        if self._truth_draft_assembler is None:
            self._fail_truth_scan("扫描状态未初始化，无法写入动作")
            return
        try:
            source_turn_id = int(record.get("turn_id", 0) or 0)
        except (TypeError, ValueError):
            self._fail_truth_scan("收到无效的来源 turn_id，未写入动作")
            return
        if source_turn_id != self._truth_scan_next_source_turn_id:
            self._fail_truth_scan(
                f"来源 turn_id 不连续：应为 {self._truth_scan_next_source_turn_id}，"
                f"实际为 {source_turn_id}"
            )
            return
        appended = self._truth_draft_assembler.append(record)
        if not appended.accepted or appended.turn is None:
            self._fail_truth_scan(
                f"第 {source_turn_id} 手被拒绝：{appended.reason or appended.status}"
            )
            return
        self._truth_scan_turns.append(record)
        self._truth_scan_next_source_turn_id += 1
        self._truth_scan_log = appended.truth_log
        if self._truth_editor is not None:
            self._truth_editor.append_confirmed_turn(
                appended.turn,
                status=appended.status,
            )
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
                    trick_id=(
                        int(raw["trick_id"])
                        if raw.get("trick_id") is not None
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
            label_status=baseline.label_status,
            provenance=baseline.provenance,
            outcome=baseline.outcome,
        )

    def _truth_scan_completed(self, _value: object) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._visual_thread:
            return
        if (
            self._truth_scan_base is None
            or self._truth_scan_session != self.current_session
        ):
            return
        if self._truth_scan_failure is not None:
            self._truth_scan_base = None
            return
        result_status = str(getattr(_value, "status", "complete") or "complete")
        result_reason = str(getattr(_value, "status_reason", "") or "")
        if self._truth_scan_untrusted_passes and result_status == "complete":
            result_status = "partial"
            result_reason = (
                "存在无充分证据的推导不出，不能直接作为可靠 truth log"
            )
        self._truth_scan_status = result_status
        self._truth_scan_status_reason = result_reason
        frame_count = getattr(_value, "frame_count", None)
        if (
            isinstance(frame_count, int)
            and self._truth_scan_progress_total > 0
            and frame_count < self._truth_scan_progress_total
        ):
            self._fail_truth_scan(
                "扫描未完成：仅处理 "
                f"{frame_count}/{self._truth_scan_progress_total} 帧"
            )
            return
        if self._truth_scan_progress_total > 0:
            self._truth_scan_progress_processed = self._truth_scan_progress_total
            self.truth_scan_progress.setValue(self._truth_scan_progress_total)
            self.truth_scan_progress.setEnabled(False)
            self.truth_scan_progress.setVisible(True)
        if self._truth_draft_assembler is not None:
            self._truth_scan_log = self._truth_draft_assembler.truth_log
        else:
            self._truth_scan_log = self._truth_log_from_scan_turns(
                self._truth_scan_base,
                self._truth_scan_turns,
            )
        assert self._truth_scan_log is not None
        self.truth_log = self._truth_scan_log
        self.truth_status.setText(
            f"出牌日志：扫描得到 {len(self._truth_scan_log.turns)} 条（未保存）"
        )
        if result_status == "complete":
            self.truth_scan_status.setText(
                "扫描完成：动作链已闭合，可编辑并点击『保存日志』写入 truth_log.json"
            )
            InfoBar.success(
                title="扫描完成",
                content="动作链已闭合，可编辑并点击『保存日志』写入 truth_log.json",
                parent=self,
            )
        else:
            detail = result_reason or "动作链未闭合或证据不足"
            self.truth_scan_status.setText(
                f"扫描{result_status}：{detail}；未生成可直接保存的可靠日志"
            )
            InfoBar.warning(
                title=f"扫描{result_status}",
                content=f"{detail}；请人工核对后再决定是否保存，正式日志保持不变",
                parent=self,
            )
            self.diagnostics.insertPlainText(
                f"\n[扫描{result_status}] {detail}\n"
                f"处理帧数：{getattr(_value, 'frame_count', '?')}；"
                f"动作数：{len(self._truth_scan_log.turns)}\n"
            )
        self._truth_scan_base = None
        if self._truth_editor is None:
            self._show_truth_log_editor(self._truth_scan_log)
        if self._truth_editor is not None:
            self._truth_editor.setEnabled(True)

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
            "\n实时策略可信日志测试完成\n"
            f"源对局：{result.source_session_id}\n"
            f"动作进度：{result.processed_turn_count}/{result.turn_count}\n"
            f"策略请求：{result.advice_requested}；成功：{result.advice_ready}；"
            f"失败：{result.advice_failed}；过期：{result.advice_stale}；"
            f"超时：{result.advice_timeouts}\n"
            f"未知花色仅在策略输入副本中补全：{result.unknown_card_resolutions} 条\n"
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

    @Slot()
    def _on_visual_thread_finished(self) -> None:
        thread = self.sender()
        if isinstance(thread, VisualRecognitionReplayThread):
            self._visual_finished(thread)

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
        if self._truth_editor is not None:
            self._truth_editor.shutdown()
        if self._visual_thread is not None and self._visual_thread.isRunning():
            self._visual_thread.stop()
            self._visual_thread.wait(30_000)
        if self._trusted_thread is not None and self._trusted_thread.isRunning():
            self._trusted_thread.stop()
            self._trusted_thread.wait(30_000)
