from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any
from time import monotonic_ns

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, Signal, Slot, QTimer, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPlainTextEdit, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    PushButton,
    PrimaryPushButton,
    LineEdit,
    ToolButton,
    ToolTipFilter,
    ToolTipPosition,
    FluentIcon,
    StrongBodyLabel,
    TitleLabel,
    isDarkTheme,
    qconfig,
)

from ..advisor_strategy import ADVISOR_OPTIONS
from ..live.display_text import compact_cards_text, live_status_text, seat_text
from ..domain.live_runtime import LiveAdvice, LiveUpdate
from ..application.opening_readiness import (
    OpeningReadinessCode,
    OpeningReadinessStatus,
    coerce_report,
)
from .single_image_danzero_page import CardBadge
from .compact_view_state import (
    CompactUpdateGate, CompactViewState, advice_matches_snapshot,
    project_compact_view,
)


_PLAY_TYPE_LABELS = {
    "Single": "单张",
    "Pair": "对子",
    "Trips": "三张",
    "ThreePair": "三连对",
    "ThreeWithTwo": "三带二",
    "TwoTrips": "钢板",
    "Straight": "顺子",
    "StraightFlush": "同花顺",
    "Bomb": "炸弹",
    "PASS": "不出",
}

_TRICK_SEATS = (
    ("self", "自己"),
    ("right", "右家"),
    ("opposite", "对家"),
    ("left", "左家"),
)

_PREVIOUS_ACTION_REREAD_PENDING = "previous_action_reread_pending"
_TURN_RECOVERY_PENDING = "turn_recovery_pending"
_WIND_CATCH_PASS_RECOVERY_PENDING = "wind_catch_pass_recovery_pending"
_CANNOT_BEAT_MIN_CONFIDENCE = 0.80
_TRANSIENT_WITHHOLD_DELAY_MS = 500


def _rank_token(card: str) -> str:
    if card == "small_joker":
        return "小王"
    if card == "big_joker":
        return "大王"
    return card[:-1] if len(card) >= 2 else card


class _TrickSeatCell(QWidget):
    """One compact, theme-aware seat in the current-trick strip."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("trickSeatCell")
        self.setMinimumWidth(72)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(0)
        self.title_label = CaptionLabel(title, self)
        self.action_label = StrongBodyLabel("—", self)
        self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.action_label, 0, Qt.AlignmentFlag.AlignCenter)
        self._current = False

    def set_action(self, text: str, *, tooltip: str, current: bool) -> None:
        self.action_label.setText(text)
        self.setToolTip(tooltip)
        self._current = bool(current)

    def apply_palette(self, *, dark: bool, accent: str) -> None:
        if self._current:
            background = "#153f3c" if dark else "#dff7f3"
            border = accent
            text = "#a7f3d0" if dark else "#0f5f58"
        else:
            background = "#292929" if dark else "#ffffff"
            border = "#454545" if dark else "#d9e4e2"
            text = "#f2f2f2" if dark else "#253331"
        self.setStyleSheet(
            "QWidget#trickSeatCell {"
            f"background:{background}; border:1px solid {border}; border-radius:6px;"
            "}"
            f"QWidget#trickSeatCell QLabel {{ color:{text}; background:transparent; border:none; }}"
        )


class _CurrentTrickStrip(QWidget):
    """Read-only current-trick summary backed only by ``LiveSnapshot``."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(44)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.turn_label = CaptionLabel("等待", self)
        self.turn_label.setFixedWidth(42)
        self.turn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.turn_label)
        self.cells: dict[str, _TrickSeatCell] = {}
        for seat, title in _TRICK_SEATS:
            cell = _TrickSeatCell(title, self)
            self.cells[seat] = cell
            layout.addWidget(cell, 1)
        self._dark = False
        self._accent = "#0f766e"

    def set_snapshot(self, snapshot: object, status: str) -> None:
        current_player = getattr(snapshot, "current_player", None)
        turn_id = int(getattr(snapshot, "turn_id", 0) or 0)
        self.turn_label.setText(f"第{turn_id}手" if turn_id else live_status_text(status))
        self.turn_label.setToolTip(live_status_text(status))
        latest_by_seat = {
            getattr(play, "player", None): play
            for play in getattr(snapshot, "trick_plays", ())
        }
        for seat, _title in _TRICK_SEATS:
            play = latest_by_seat.get(seat)
            if play is None:
                text = "等待" if seat == current_player else "—"
                tooltip = f"{seat_text(seat)}：{text}"
            elif bool(getattr(play, "is_pass", False)):
                text = "不出"
                tooltip = f"{seat_text(seat)}：不出"
            else:
                cards = tuple(str(card) for card in getattr(play, "cards", ()))
                ranks = "".join(_rank_token(card) for card in cards)
                text = ranks if len(ranks) <= 8 else f"{ranks[:6]}…·{len(cards)}张"
                tooltip = f"{seat_text(seat)}：{compact_cards_text(cards, getattr(play, 'suit_options', ()))}"
            self.cells[seat].set_action(
                text,
                tooltip=tooltip,
                current=seat == current_player,
            )
        self.apply_palette(dark=self._dark, accent=self._accent)

    def apply_palette(self, *, dark: bool, accent: str) -> None:
        self._dark = bool(dark)
        self._accent = accent
        for cell in self.cells.values():
            cell.apply_palette(dark=dark, accent=accent)


def _absolute_local_path(value: object) -> Path | None:
    """Accept filesystem paths only; never reinterpret a URL or cwd as evidence."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    try:
        path = Path(value)
        if not path.is_absolute() or path == Path(path.anchor) or ".." in path.parts:
            return None
        if "\x00" in str(path):
            return None
        return path
    except (OSError, ValueError):
        return None


def _case_frames_directory(path: Path) -> bool:
    try:
        return path.name.casefold() == "frames" and (path.parent / "case.json").is_file()
    except (OSError, ValueError):
        return False


class _ProblemDialog(QDialog):
    """Native dialogs need their own background when Fluent dark labels are used."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("problemExportDialog")
        self._apply_problem_theme()
        qconfig.themeChanged.connect(self._apply_problem_theme)

    def _apply_problem_theme(self, *_args) -> None:
        dark = isDarkTheme()
        background, foreground = ("#202020", "#f3f3f3") if dark else ("#f7f7f7", "#202020")
        entry, border = ("#292929", "#555555") if dark else ("#ffffff", "#cccccc")
        self.setStyleSheet(
            "QDialog#problemExportDialog {"
            f"background-color:{background}; color:{foreground};"
            "} QDialog#problemExportDialog QPlainTextEdit {"
            f"background-color:{entry}; color:{foreground}; border:1px solid {border};"
            "}"
        )


class _ProblemExportConfirmation(_ProblemDialog):
    """A small modeless consent dialog; it never interrupts the live listener."""

    def __init__(self, parent: QWidget, case_directory: Path | None) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出问题包")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint,
                           bool(parent.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))
        self.setMinimumWidth(420)
        self.choice: bool | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.addWidget(StrongBodyLabel("导出问题包", self))
        self.privacy_label = CaptionLabel(
            "游戏截图可能包含昵称、头像等个人信息。\n"
            "问题包仅保存到本机，不会自动上传；监听和推荐继续。", self
        )
        self.privacy_label.setWordWrap(True)
        layout.addWidget(self.privacy_label)
        self.target_label = CaptionLabel(
            "范围：已选择的历史对局" if case_directory is not None else "范围：当前或最近一次对局", self
        )
        self.target_label.setToolTip(str(case_directory or ""))
        layout.addWidget(self.target_label)
        buttons = QHBoxLayout()
        self.images_button = PrimaryPushButton("包含截图导出（推荐）", self)
        self.images_button.setDefault(True)
        self.images_button.clicked.connect(lambda: self._choose(True))
        buttons.addWidget(self.images_button)
        self.logs_button = PushButton("仅日志", self)
        self.logs_button.clicked.connect(lambda: self._choose(False))
        buttons.addWidget(self.logs_button)
        self.cancel_button = PushButton("取消", self)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def _choose(self, include_images: bool) -> None:
        self.choice = include_images
        self.accept()


class _ProblemExportResultDialog(_ProblemDialog):
    """Bounded result presentation with a single, explicitly local ZIP path."""

    def __init__(self, payload: Mapping[str, object], parent: QWidget) -> None:
        super().__init__(parent)
        status = str(payload.get("status") or "FAILURE")
        self.archive_path = _absolute_local_path(payload.get("archive_path"))
        title = "问题包已生成" if status in {"SUCCESS", "PARTIAL"} else "问题包导出失败"
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint,
                           bool(parent.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))
        self.setFixedWidth(460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.addWidget(StrongBodyLabel(title, self))
        self.message_label = CaptionLabel(
            "已保存可用证据，部分信息缺失；可以直接发送此 ZIP。" if status == "PARTIAL" else
            "仅保存在本机，未自动上传。" if status == "SUCCESS" else "未生成可用的问题包，请重试。", self
        )
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        details = []
        if payload.get("message"):
            details.append(str(payload["message"]))
        for key, label in (("missing", "缺失"), ("omitted", "未包含")):
            value = payload.get(key)
            if value:
                text = "、".join(str(item) for item in value) if isinstance(value, (list, tuple)) else str(value)
                details.append(f"{label}：{text}")
        if payload.get("include_images") is False:
            details.append("本次为仅日志导出，不包含游戏截图。")
        self.details_view = QPlainTextEdit(self)
        self.details_view.setReadOnly(True)
        self.details_view.setPlainText("\n".join(details))
        self.details_view.setFixedHeight(76)
        self.details_view.hide()
        self.details_toggle = PushButton("查看详情", self)
        self.details_toggle.setCheckable(True)
        self.details_toggle.setVisible(bool(details))
        self.details_toggle.toggled.connect(self._toggle_details)
        layout.addWidget(self.details_toggle)
        layout.addWidget(self.details_view)
        self.path_edit = LineEdit(self)
        self.path_edit.setReadOnly(True)
        self.path_edit.setAccessibleName("问题包 ZIP 文件路径")
        self.path_edit.setText(str(self.archive_path) if self.archive_path is not None else "")
        self.path_edit.setCursorPosition(0)
        self.path_edit.setToolTip(self.path_edit.text())
        layout.addWidget(self.path_edit)
        buttons = QHBoxLayout()
        self.open_folder_button = PushButton("打开所在文件夹", self)
        self.open_folder_button.clicked.connect(self._open_folder)
        buttons.addWidget(self.open_folder_button)
        self.copy_path_button = PushButton("复制文件路径", self)
        self.copy_path_button.clicked.connect(self._copy_path)
        buttons.addWidget(self.copy_path_button)
        self.close_button = PrimaryPushButton("关闭", self)
        self.close_button.clicked.connect(self.accept)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        usable = status in {"SUCCESS", "PARTIAL"} and self._archive_exists()
        self.open_folder_button.setEnabled(usable)
        self.copy_path_button.setEnabled(usable)
        self.path_edit.setVisible(usable)
        if not usable:
            self.archive_path = None

    def _toggle_details(self, checked: bool) -> None:
        self.details_view.setVisible(checked)
        self.details_toggle.setText("收起详情" if checked else "查看详情")
        self.adjustSize()

    def _archive_exists(self) -> bool:
        try:
            return bool(self.archive_path is not None and self.archive_path.suffix.casefold() == ".zip"
                        and self.archive_path.is_file())
        except (OSError, ValueError):
            return False

    def _check_archive(self) -> bool:
        if self._archive_exists():
            return True
        self.message_label.setText("问题包文件已不存在，请重新导出。")
        self.open_folder_button.setEnabled(False)
        self.copy_path_button.setEnabled(False)
        return False

    def _open_folder(self) -> None:
        if not self._check_archive():
            return
        try:
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.archive_path.parent)))
        except Exception:
            opened = False
        if not opened:
            self.message_label.setText("无法打开文件夹，可复制文件路径后手动查看。")

    def _copy_path(self) -> None:
        if self._check_archive():
            QGuiApplication.clipboard().setText(str(self.archive_path))
            self.message_label.setText("文件路径已复制；问题包未自动上传。")


class ProblemExportUi(QObject):
    """Shared UI-only consent, duplicate-click guard and result presentation.

    The controller owns all case selection, collection and background work.
    The main window shares this presenter across both export entry points.
    """

    def __init__(self, runtime: object, parent: QWidget) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self._buttons: list[QWidget] = []
        self._running = False
        self._pending = False
        self._include_images = True
        self._result_parent: QWidget = parent
        self.confirmation_dialog: _ProblemExportConfirmation | None = None
        self.result_dialog: _ProblemExportResultDialog | None = None
        signal = getattr(runtime, "problem_export_status", None)
        self._has_status_signal = signal is not None and callable(getattr(signal, "connect", None))
        if self._has_status_signal:
            signal.connect(self.apply_status)

    def bind_button(self, button: QWidget) -> None:
        self._buttons.append(button)
        button.setEnabled(not self._running)
        button.setToolTip("导出问题包：仅保存本机，可选择包含截图或仅日志")

    def _set_busy(self, busy: bool, notice: str = "") -> None:
        self._running = busy
        for button in self._buttons:
            button.setEnabled(not busy)
            button.setToolTip(notice or "导出问题包：仅保存本机，可选择包含截图或仅日志")

    def request(self, *, parent: QWidget, case_directory: Path | None = None) -> None:
        if self._running:
            return
        self._result_parent = parent
        if not callable(getattr(self.runtime, "request_problem_export", None)) or not self._has_status_signal:
            self._show_result({"status": "FAILURE", "message": "当前运行时不支持问题包导出，请升级后重试。"})
            return
        if case_directory is not None and _absolute_local_path(case_directory) is None:
            self._show_result({"status": "FAILURE", "message": "所选对局必须是本机绝对目录；未改为导出当前对局。"})
            return
        self._set_busy(True, "请确认问题包截图隐私选项")
        dialog = _ProblemExportConfirmation(parent, case_directory)
        self.confirmation_dialog = dialog
        dialog.finished.connect(lambda _result: self._confirmed(dialog, case_directory))
        dialog.show()

    def _confirmed(self, dialog: _ProblemExportConfirmation, case_directory: Path | None) -> None:
        self.confirmation_dialog = None
        include_images = dialog.choice
        dialog.deleteLater()
        if include_images is None:
            self._set_busy(False)
            return
        self._pending = True
        self._include_images = include_images
        self._set_busy(True, "正在后台导出问题包 · 监听和推荐继续")
        try:
            accepted = self.runtime.request_problem_export(
                include_images=include_images, case_directory=case_directory,
            )
        except Exception as exc:
            if self._pending:
                self.apply_status({"status": "FAILURE", "message": str(exc) or type(exc).__name__})
            return
        # Some small runtimes emit the completion signal synchronously.
        if not accepted and self._pending:
            self.apply_status({"status": "FAILURE", "message": "未能启动导出，可能已有导出任务，请稍后重试。"})

    @Slot(object)
    def apply_status(self, value: object) -> None:
        def field(name: str, default: object = None) -> object:
            return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

        raw_status = field("status", "")
        status = str(getattr(raw_status, "value", raw_status)).upper()
        if status == "RUNNING":
            # A task started elsewhere while consent was open owns the guard.
            # Dismiss that stale consent rather than submit a second request.
            if self.confirmation_dialog is not None:
                self.confirmation_dialog.reject()
            self._set_busy(True, "正在后台导出问题包 · 监听和推荐继续")
            return
        if status not in {"SUCCESS", "PARTIAL", "FAILURE"}:
            return
        if self.confirmation_dialog is not None and not self._pending:
            return  # An unrelated late result must not unlock active consent.
        self._set_busy(False, {"SUCCESS": "问题包已导出", "PARTIAL": "问题包部分导出", "FAILURE": "问题包导出失败，请重试"}[status])
        if not self._pending:
            return
        self._pending = False
        payload = {name: field(name) for name in ("archive_path", "message", "missing", "omitted")}
        payload.update(status=status, include_images=field("include_images", self._include_images))
        if status in {"SUCCESS", "PARTIAL"}:
            archive = _absolute_local_path(payload["archive_path"])
            try:
                valid = archive is not None and archive.suffix.casefold() == ".zip" and archive.is_file()
            except (OSError, ValueError):
                valid = False
            if not valid:
                payload.update(status="FAILURE", archive_path=None,
                               message="未找到导出的 ZIP 文件，不能确认导出成功，请重试。")
                self._set_busy(False, "问题包导出失败：未找到 ZIP 文件")
        self._show_result(payload)

    def _show_result(self, payload: Mapping[str, object]) -> None:
        if self.result_dialog is not None:
            self.result_dialog.close()
            self.result_dialog.deleteLater()
        self.result_dialog = _ProblemExportResultDialog(payload, self._result_parent)
        self.result_dialog.show()


class RecommendationFloatWindow(QWidget):
    """A read-only companion view over the shared live controller."""

    open_full_assistant_requested = Signal()
    open_diagnostic_requested = Signal()
    capture_diagnostic_requested = Signal()
    copy_issue_requested = Signal()
    copy_summary_requested = Signal()
    stop_listening_requested = Signal()

    def __init__(self, runtime: Any, parent=None) -> None:
        super().__init__(parent)
        self.last_placement_diagnostic: dict[str, object] = {}
        self.runtime = runtime
        self._last_request_id = ""
        self._preselection_by_request_id: dict[str, object] = {}
        self._backend = ""
        self._card_badges: list[CardBadge] = []
        self._transient_withhold_request_id = ""
        self._transient_withhold_timer = QTimer(self)
        self._transient_withhold_timer.setSingleShot(True)
        self._transient_withhold_timer.timeout.connect(
            self._show_delayed_transient_withhold
        )
        self._update_gate = CompactUpdateGate()
        self._view_state: CompactViewState | None = None
        self._latest_update: object | None = None
        self._opening_readiness = None
        self._fault_identity: tuple[str, int] | None = None
        self._diagnostic_frame_saving = False
        self._last_diagnostic_image_path: Path | None = None
        self._last_diagnostic_session_directory: Path | None = None
        self._hint_expiry_timer = QTimer(self)
        self._hint_expiry_timer.setSingleShot(True)
        self._hint_expiry_timer.timeout.connect(self._expire_local_hint)
        self.setObjectName("recommendationFloatWindow")
        self.setWindowTitle(f"{self._advisor_display_name()} 极简推荐")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setMinimumSize(420, 235)
        self.resize(500, 245)
        self._build_ui()
        self.problem_export_ui = ProblemExportUi(runtime, self)
        self.problem_export_ui.bind_button(self.problem_export_button)
        self._connect_runtime()
        qconfig.themeChanged.connect(self._apply_theme)
        self._apply_theme()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        self.card = CardWidget(self)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(12, 8, 12, 8)
        card_layout.setSpacing(4)

        self.trick_strip = _CurrentTrickStrip(self.card)
        self.state_label = self.trick_strip.turn_label
        card_layout.addWidget(self.trick_strip)

        self.suggestion_label = TitleLabel("等待建议")
        self.suggestion_label.setWordWrap(True)
        card_layout.addWidget(self.suggestion_label)

        self.cards_host = QWidget(self.card)
        self.cards_host.setFixedHeight(56)
        self.cards_layout = QHBoxLayout(self.cards_host)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(4)
        self.cards_layout.addStretch(1)
        self.cards_host.hide()
        card_layout.addWidget(self.cards_host)

        self.detail_label = CaptionLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        card_layout.addWidget(self.detail_label)
        root.addWidget(self.card, 1)

        actions = QHBoxLayout()
        self.capture_label = CaptionLabel("")
        self.capture_label.setWordWrap(True)
        self.capture_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        actions.addWidget(self.capture_label, 1)

        self.capture_button = self._make_action_button(
            FluentIcon.CAMERA,
            "截取当前画面：优先保存最近监听帧（停止后为保留帧），无监听帧时手动截取窗口",
            self.capture_diagnostic_requested.emit,
        )
        actions.addWidget(self.capture_button)
        self.screenshot_folder_button = self._make_action_button(
            FluentIcon.FOLDER,
            "打开截图目录",
            self.open_screenshot_folder,
        )
        actions.addWidget(self.screenshot_folder_button)
        self.debug_button = self._make_action_button(
            FluentIcon.SEARCH,
            "打开窗口与牌局诊断：打开完整助手中的窗口与牌局诊断页",
            self.open_diagnostic_requested.emit,
        )
        actions.addWidget(self.debug_button)
        self.copy_issue_button = self._make_action_button(
            FluentIcon.INFO,
            "复制当前问题说明到剪贴板",
            self.copy_issue_requested.emit,
        )
        actions.addWidget(self.copy_issue_button)
        self.problem_export_button = self._make_action_button(
            FluentIcon.SAVE,
            "导出问题包：包含截图或仅日志，不自动上传",
            lambda: self.problem_export_ui.request(parent=self),
        )
        self.problem_export_button.setObjectName("compactProblemExportButton")
        actions.addWidget(self.problem_export_button)
        self.open_button = self._make_action_button(
            FluentIcon.SETTING,
            "打开完整助手",
            self.open_full_assistant_requested.emit,
        )
        actions.addWidget(self.open_button)
        self.stop_button = self._make_action_button(
            FluentIcon.CLOSE,
            "停止实时监听",
            self.stop_listening_requested.emit,
        )
        actions.addWidget(self.stop_button)
        root.addLayout(actions)

    @staticmethod
    def _make_action_button(icon: object, tooltip: str, callback: object) -> ToolButton:
        button = ToolButton()
        button.setIcon(icon)
        button.setFixedSize(32, 32)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        # QFluentWidgets ToolButton does not consistently show the native Qt
        # tooltip on every platform/theme. Install its own filter while
        # retaining the native tooltip and accessibility name as fallbacks.
        tooltip_filter = ToolTipFilter(
            button, showDelay=250, position=ToolTipPosition.TOP
        )
        button.installEventFilter(tooltip_filter)
        button._compact_tooltip_filter = tooltip_filter
        if callable(callback):
            button.clicked.connect(callback)
        return button

    def _connect_runtime(self) -> None:
        update_signal = getattr(self.runtime, "update_ready", None)
        if update_signal is not None:
            update_signal.connect(self.apply_update)
        frame_signal = getattr(self.runtime, "frame_ready", None)
        if frame_signal is not None:
            frame_signal.connect(self.apply_frame)
        error_signal = getattr(self.runtime, "error", None)
        if error_signal is not None:
            error_signal.connect(self.show_error)
        live_fault = getattr(self.runtime, "live_fault", None)
        if live_fault is not None:
            live_fault.connect(self.apply_live_fault)
        preselection_signal = getattr(self.runtime, "preselection_result", None)
        if preselection_signal is not None:
            preselection_signal.connect(self.apply_preselection_result)
        latest = getattr(self.runtime, "latest_preselection_result", None)
        if latest is not None:
            self.apply_preselection_result(latest)
        listening_status = getattr(self.runtime, "listening_status", None)
        if listening_status is not None:
            listening_status.connect(self.apply_listening_status)
        log_delivery_status = getattr(self.runtime, "log_delivery_status", None)
        if log_delivery_status is not None:
            log_delivery_status.connect(self.apply_log_delivery_status)
        recording_status = getattr(self.runtime, "recording_status", None)
        if recording_status is not None:
            recording_status.connect(self.apply_recording_status)
        diagnostic_frame_status = getattr(self.runtime, "diagnostic_frame_status", None)
        if diagnostic_frame_status is not None and hasattr(diagnostic_frame_status, "connect"):
            diagnostic_frame_status.connect(self.apply_diagnostic_frame_status)

    def _apply_theme(self, *_args) -> None:
        if isDarkTheme():
            background = "#202020"
            border = "#454545"
            accent = "#5eead4"
        else:
            background = "#f7fbfa"
            border = "#b8d9d3"
            accent = "#0f766e"
        self.setStyleSheet(
            "QWidget#recommendationFloatWindow {"
            f"background:{background}; border:1px solid {border};"
            "}"
        )
        self.suggestion_label.setStyleSheet(
            f"color:{accent}; font-size:26px; font-weight:700;"
        )
        self.trick_strip.apply_palette(dark=isDarkTheme(), accent=accent)

    def apply_update(self, update: LiveUpdate) -> None:
        orchestrator = getattr(self.runtime, "orchestrator", None)
        expected_session = str(getattr(getattr(orchestrator, "snapshot", None), "session_id", "") or "")
        if not self._update_gate.accept(update, expected_session_id=expected_session):
            return
        current_identity = (self._update_gate.session_id, self._update_gate.generation)
        if self._fault_identity is not None:
            if self._fault_identity == current_identity and update.status == "running":
                return  # A fatal worker failure needs a new capture generation.
            if self._fault_identity != current_identity:
                self._fault_identity = None
        self._latest_update = update
        self.setWindowTitle(f"{self._advisor_display_name()} 极简推荐")
        self.trick_strip.set_snapshot(update.snapshot, update.status)
        now_ms = monotonic_ns() // 1_000_000
        state = project_compact_view(update, now_ms=now_ms)
        self._render_view(state)
        if state.kind == "local_rule_hint":
            hint = getattr(update, "local_rule_hint", None)
            self._hint_expiry_timer.start(max(1, int(hint.expires_ms) - now_ms + 1))

    def _render_view(self, state: CompactViewState) -> None:
        self._clear_transient_withhold()
        self._hint_expiry_timer.stop()
        # The *entire* rendered state is the cache key. A request can safely
        # become ready again after an intervening hold without changing ID.
        if state == self._view_state:
            return
        self._view_state = state
        self._last_request_id = state.request_id
        self.suggestion_label.setText(state.title)
        self.detail_label.setText(state.detail)
        self.detail_label.setVisible(bool(state.detail))
        self._render_cards(state.cards)

    def _expire_local_hint(self) -> None:
        if self._view_state is not None and self._view_state.kind == "local_rule_hint":
            # No new capture arrived: do not resurrect a model result from
            # before this visual control lifecycle when its hint expires.
            self._render_view(CompactViewState("waiting", "等待建议"))

    @staticmethod
    def _advice_matches_snapshot(advice: LiveAdvice, snapshot: object) -> bool:
        return advice_matches_snapshot(advice, snapshot)

    def _show_delayed_transient_withhold(self) -> None:
        # Kept as an inert slot for previously queued timers. Internal reread
        # phases no longer schedule callbacks that compete with recommendations.
        self._clear_transient_withhold()

    def _clear_transient_withhold(self) -> None:
        self._transient_withhold_timer.stop()
        self._transient_withhold_request_id = ""

    def apply_preselection_result(self, result: object) -> None:
        # Results have no session/generation identity. Never let old callbacks
        # mutate advice or append unbounded success messages to compact detail.
        # The complete assistant retains execution diagnostics and failures.
        return

    def _advisor_display_name(self) -> str:
        strategy = str(getattr(self.runtime, "advisor_strategy", "") or "")
        return dict(ADVISOR_OPTIONS).get(strategy, "建议模型")

    def apply_frame(self, snapshot: object) -> None:
        # Backend/ROI/capture telemetry belongs to the complete assistant.
        return

    def _has_live_view(self) -> bool:
        return bool(
            getattr(self.runtime, "orchestrator", None) is not None
            or (self._update_gate.session_id and not self._update_gate.terminal)
        )

    def show_error(self, message: str) -> None:
        # Untagged error strings can arrive from an old export/capture worker.
        # During a live session, only versioned LiveUpdate may change safety
        # state. The controller already emits its paused/failed update first.
        if self._has_live_view():
            return
        text = str(message)
        if "遮挡" in text or "屏幕采集已暂停" in text:
            self._render_view(CompactViewState("paused", "窗口遮挡，已暂停", "请移开遮挡后点击继续"))

    def apply_live_fault(self, fault: object) -> None:
        """Fail closed for authenticated fatal worker failures.

        Controller emits this if creating the normal paused/error LiveUpdate
        itself fails. Required fields: session_id, capture_generation, kind
        (capture/analysis/occluded). Untagged error strings remain diagnostic.
        Recovery must activate a fresh capture generation before cards return.
        """
        if not isinstance(fault, dict):
            return
        session_id = str(fault.get("session_id", "") or "")
        generation = fault.get("capture_generation")
        if not session_id or not isinstance(generation, int):
            return
        if (session_id, generation) != (self._update_gate.session_id, self._update_gate.generation):
            return
        sequence = fault.get("update_sequence")
        if isinstance(sequence, int) and sequence > 0 and sequence <= self._update_gate.update_sequence:
            return
        if isinstance(sequence, int) and sequence > 0:
            self._update_gate.update_sequence = sequence
        self._fault_identity = (session_id, generation)
        kind = str(fault.get("kind", ""))
        detail = {
            "capture": "画面采集失败，请重新连接牌桌",
            "analysis": "识别失败，请重新连接牌桌",
            "occluded": "请移开遮挡后点击继续",
        }.get(kind, "识别已停止，请重新连接牌桌")
        self._render_view(CompactViewState("paused", "识别已暂停", detail))

    def apply_listening_status(self, status: object) -> None:
        if not isinstance(status, dict):
            return
        generation = status.get("generation")
        if isinstance(generation, int):
            latest = max(getattr(self, "_listening_generation", -1), self._update_gate.generation)
            if generation < latest:
                return
            self._listening_generation = generation
        if self._has_live_view():
            return

        report = coerce_report(status)
        if report is not None:
            self._opening_readiness = report
            self._render_opening_readiness(report)
            return

        # Compatibility for older controller payloads and third-party runtimes.
        # New controller emissions always take the structured path above.
        state = str(status.get("state", "") or "")
        if state == "listening":
            self._update_gate.begin_listening()
            self._latest_update = None
            self._render_view(CompactViewState("listening", "等待开局"))
        elif state == "opening":
            phase = str(status.get("phase", "") or "")
            if phase in {"ready_waiting_first_action", "ready_waiting_lead"}:
                title = "已进入牌桌，等待自己首出"
            else:
                title = "等待开局" if phase in {"unknown", "lobby", "settlement", "waiting_table"} else "确认开局中…"
            self._render_view(CompactViewState("opening", title))
        elif state == "recovering":
            self._render_view(CompactViewState("recovering", "重新连接中…"))
        elif state == "recovered":
            self._render_view(CompactViewState("listening", "等待开局"))
        elif state == "failed":
            self._render_view(CompactViewState("failed", "监听已停止", "请打开完整助手重新连接牌桌"))

    def _render_opening_readiness(self, report: object) -> None:
        """Render the shared report without collapsing every WAIT into one label."""

        reason = getattr(report, "primary_reason", None)
        reason_value = getattr(reason, "value", reason)
        reason_value = str(reason_value or "OPENING_UNRESOLVED")
        status = getattr(report, "status", None)
        status_value = getattr(status, "value", status)
        status_value = str(status_value or "WAIT")
        titles = {
            OpeningReadinessCode.READY.value: "开局已就绪",
            OpeningReadinessCode.READY_WAITING_FIRST_ACTION.value: "已进入牌桌，等待自己首出",
            OpeningReadinessCode.LISTENING.value: "等待开局",
            OpeningReadinessCode.LOBBY.value: "等待进入牌桌",
            OpeningReadinessCode.DEAL_IN_PROGRESS.value: "对局已进行",
            OpeningReadinessCode.MID_GAME_HAND_COUNT.value: "起手牌数量未稳定",
            OpeningReadinessCode.HAND_UNSTABLE.value: "起手牌识别不稳定",
            OpeningReadinessCode.OPENING_UNRESOLVED.value: "开局证据未确认",
            OpeningReadinessCode.WINDOW_NOT_FOUND.value: "未找到牌桌窗口",
            OpeningReadinessCode.MULTIPLE_WINDOWS.value: "检测到多个牌桌窗口",
            OpeningReadinessCode.WINDOW_MINIMIZED.value: "牌桌窗口已最小化",
            OpeningReadinessCode.CAPTURE_FAILED.value: "画面捕获失败",
            OpeningReadinessCode.ROI_FATAL.value: "识别区域配置错误",
            OpeningReadinessCode.WORKER_FAULT.value: "后台识别失败",
        }
        title = titles.get(reason_value, "开局状态不可用")
        details = getattr(report, "details", {})
        phase = details.get("phase", "") if isinstance(details, Mapping) else ""
        message = str(getattr(report, "message", "") or "")
        if phase == "doubling":
            title = "等待加倍结束"
        elif phase == "page_recovering":
            title = "等待画面恢复"
        elif reason_value == OpeningReadinessCode.READY_WAITING_FIRST_ACTION.value and message:
            title = message
        suggested_action = str(getattr(report, "suggested_action", "") or "")
        detail_parts = [part for part in (message, f"建议：{suggested_action}" if suggested_action else "") if part]
        detail = "\n".join(detail_parts)

        if status_value == OpeningReadinessStatus.PASS.value:
            kind = "opening"
        elif status_value == OpeningReadinessStatus.FAIL.value:
            kind = "failed"
        else:
            kind = "opening"
        if reason_value == OpeningReadinessCode.LISTENING.value:
            kind = "listening"
        self._render_view(CompactViewState(kind, title, detail))

    _absolute_diagnostic_path = staticmethod(_absolute_local_path)

    @staticmethod
    def _known_frame_directory(path: Path) -> bool:
        return path.name.casefold() == "diagnostic_frames" or _case_frames_directory(path)

    def _remember_diagnostic_frame(self, value: Mapping[str, object]) -> None:
        image = self._absolute_diagnostic_path(value.get("image_path"))
        session = self._absolute_diagnostic_path(value.get("session_directory"))
        if image is not None and self._known_frame_directory(image.parent):
            # The saved PNG is stronger evidence than a stale session directory.
            self._last_diagnostic_image_path = image
            self._last_diagnostic_session_directory = image.parent.parent
        elif session is not None:
            self._last_diagnostic_image_path = None
            self._last_diagnostic_session_directory = session

    def open_screenshot_folder(self) -> None:
        """Open saved evidence in Explorer, never via full-assistant navigation."""
        candidates: list[Path] = []
        getter = getattr(self.runtime, "diagnostic_frame_directory", None)
        if callable(getter):
            try:
                directory = self._absolute_diagnostic_path(getter())
            except Exception:
                directory = None
            if directory is not None and (
                self._known_frame_directory(directory)
                or (directory.name.casefold() == "cases" and directory.parent.name.casefold() == "diagnostics")
            ):
                candidates.append(directory)
        image = self._last_diagnostic_image_path
        session = self._last_diagnostic_session_directory
        if image is not None:
            candidates.append(image.parent)
        elif session is not None:
            if self._known_frame_directory(session):
                candidates.append(session)
            else:
                candidates.append(session / "frames" if _case_frames_directory(session / "frames")
                                  else session / "diagnostic_frames")

        def existing_directory(path: Path) -> bool:
            try:
                return path.is_dir()
            except (OSError, ValueError):
                return False

        directory = next((path for path in candidates if existing_directory(path)), None)
        fallback = directory is None
        if fallback:
            # A removed frame directory may still have its session or sessions
            # parent. Do not walk arbitrarily up to a drive root or create paths.
            directory = next((
                parent
                for path in candidates
                for parent in ((path.parent,) if path.name.casefold() == "cases"
                               else (path.parent, path.parent.parent))
                if parent != Path(parent.anchor) and existing_directory(parent)
            ), None)
        if directory is None:
            self._set_capture_notice(
                "截图目录已不存在，请重新截图" if candidates else "尚无截图，请先截取一帧"
            )
            return
        try:
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
        except Exception:
            opened = False
        if opened:
            self._set_capture_notice(
                "截图目录已移除，已打开上级目录" if fallback else "已打开截图目录",
                str(directory),
            )
        else:
            self._set_capture_notice("无法打开截图目录，请稍后重试", str(directory))

    def _set_capture_notice(self, text: str, detail: str = "") -> None:
        self.capture_label.setText(text)
        self.capture_label.setToolTip(detail)

    @staticmethod
    def _diagnostic_source(value: Mapping[str, object]) -> tuple[str, bool]:
        metadata = value.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source = str(value.get("source") or metadata.get("source") or "")
        phase = str(value.get("source_phase") or metadata.get("source_phase") or "")
        if source == "manual_window_capture" or phase == "manual_window_capture":
            return "手动窗口截图", False
        if phase == "last_listener_frame":
            return "故障前最近监听帧", True
        if phase == "listener_stopped":
            return "停止前监听帧", True
        if phase in {"failed_listener_frame", "listener_failed", "waiting_capture_failed", "geometry_recovery_failed"}:
            return "失败前监听帧", True
        if source == "live_listener_frame":
            return "实时监听截图", False
        return "截图", False

    @staticmethod
    def _retained_frame_time(value: Mapping[str, object]) -> str:
        metadata = value.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        captured = value.get("captured_at") or metadata.get("captured_at")
        try:
            captured = captured if isinstance(captured, datetime) else datetime.fromisoformat(str(captured))
        except (TypeError, ValueError):
            captured = None
        age = value.get("frame_age_seconds", metadata.get("frame_age_seconds"))
        if age is None and value.get("frame_age_ms") is not None:
            try:
                age = float(value["frame_age_ms"]) / 1000.0
            except (TypeError, ValueError, OverflowError):
                age = None
        try:
            age = float(age)
            if not isfinite(age):
                age = None
        except (TypeError, ValueError, OverflowError):
            age = None
        if age is None and captured is not None:
            age = (datetime.now().astimezone() - captured.astimezone()).total_seconds()
        parts = [f"采集 {captured:%H:%M:%S}" if captured else "采集时间未知"]
        if age is not None:
            seconds = max(0, int(age))
            if seconds < 60:
                parts.append(f"{seconds} 秒前")
            elif seconds < 3600:
                parts.append(f"{seconds // 60} 分钟前")
            elif seconds < 86400:
                parts.append(f"{seconds // 3600} 小时前")
            else:
                parts.append(f"{seconds // 86400} 天前")
        return " · ".join(parts)

    def apply_diagnostic_frame_status(self, value: object) -> None:
        """Render save provenance and retain only successful screenshot paths."""
        if not isinstance(value, Mapping):
            return
        status = str(value.get("status") or value.get("state") or "").upper()
        message = str(value.get("message") or "")
        if status in {"PASS", "SUCCESS", "SAVED", "COMPLETED", "DONE"} and "复制" in message:
            # Clipboard feedback is not a save and cannot complete one in flight.
            self._set_capture_notice(message)
            return
        source_label, retained = self._diagnostic_source(value)
        if status in {"RUNNING", "SAVING", "PENDING", "STARTED", "LOADING"}:
            self._diagnostic_frame_saving = True
            self.capture_button.setEnabled(False)
            self._set_capture_notice(f"正在保存{source_label}…")
            return

        self._diagnostic_frame_saving = False
        self.capture_button.setEnabled(True)
        if status in {"PASS", "SUCCESS", "SAVED", "COMPLETED", "DONE"}:
            self._remember_diagnostic_frame(value)
            count = value.get("count", value.get("frame_count", value.get("saved_count")))
            detail = f"{source_label}已保存"
            if retained:
                detail += "\n" + self._retained_frame_time(value)
            elif count is not None:
                detail += f" · {count} 张"
            tooltip = "\n".join(str(part) for part in (
                message,
                value.get("image_path"),
                value.get("session_directory"),
                "保留的历史监听帧，非当前画面" if retained else "",
            ) if part)
            self._set_capture_notice(detail, tooltip)
        elif status in {"FAIL", "FAILURE", "ERROR", "FAILED"}:
            message = message or str(value.get("error") or "保存截图失败")
            self._set_capture_notice("截图保存失败", message)
        elif status:
            self._set_capture_notice(message or status)

    def apply_recording_status(self, value: object) -> None:
        if isinstance(value, dict) and value.get("reason") == "recording_capacity_reached":
            self._recording_capacity_notice = str(value.get("message", ""))
            self.capture_label.setText("录像已达容量上限 · 识别和推荐继续")

    def apply_log_delivery_status(self, value: object) -> None:
        if not isinstance(value, dict):
            return
        # Export can finish after the next game starts and these messages
        # historically carry no reliable generation. They never own the main
        # recommendation area, even if there is currently no live game.
        session_id = str(value.get("session_id", "") or "")
        if self._has_live_view():
            return
        if session_id and self._update_gate.session_id and session_id != self._update_gate.session_id:
            return
        status = str(value.get("status", "") or "").upper()
        notices = {
            "PASS": "日志已保存", "SUCCESS": "日志已保存",
            "FAIL": "日志保存失败，请打开完整助手", "FAILURE": "日志保存失败，请打开完整助手",
            "RUNNING": "日志整理中", "LOADING": "日志整理中",
            "DISABLED": "未保存对局日志",
        }
        notice = notices.get(status)
        if notice:
            self.capture_label.setText(notice)

    def _render_cards(self, cards: tuple[str, ...]) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._card_badges = []
        if not cards:
            self.cards_host.hide()
            self.cards_layout.addStretch(1)
            return
        for card in cards:
            badge = CardBadge(str(card), compact=True)
            badge.setFixedSize(38, 54)
            self._card_badges.append(badge)
            self.cards_layout.addWidget(badge)
        self.cards_layout.addStretch(1)
        self.cards_host.show()

    def place_beside(self, target: QRect) -> bool:
        """Place in a work area without ever accepting an overlapping fallback."""

        self.last_placement_diagnostic = {"target": target.getRect()}
        screens = tuple(QGuiApplication.screens())
        if not screens:
            self.last_placement_diagnostic["reason"] = "no_screen"
            return False
        source = QGuiApplication.screenAt(target.center())
        ordered = tuple(
            screen for screen in screens if screen is source
        ) + tuple(screen for screen in screens if screen is not source)
        for screen in ordered:
            available = screen.availableGeometry()
            width = max(self.minimumWidth(), min(max(self.width(), self.minimumWidth()), available.width()))
            height = max(self.minimumHeight(), min(max(self.height(), self.minimumHeight()), available.height()))
            candidates = [
                QRect(target.right() + 8, target.top(), width, height),
                QRect(target.left() - width - 8, target.top(), width, height),
                QRect(target.left(), target.bottom() + 8, width, height),
                QRect(target.left(), target.top() - height - 8, width, height),
            ]
            if screen is not source:
                candidates.insert(0, QRect(available.topLeft(), QSize(width, height)))
            for candidate in candidates:
                if available.contains(candidate) and not candidate.intersects(target):
                    self.setGeometry(candidate)
                    self.last_placement_diagnostic.update({
                        "screen": screen.name(),
                        "available": available.getRect(),
                        "placed": candidate.getRect(),
                        "dpi": screen.devicePixelRatio(),
                        "reason": "placed",
                    })
                    return True
        self.last_placement_diagnostic.update({
            "reason": "no_safe_slot",
            "screen_count": len(screens),
            "dpi": [screen.devicePixelRatio() for screen in screens],
        })
        return False

    def closeEvent(self, event: QCloseEvent) -> None:
        event.ignore()
        self.hide()
        self.open_full_assistant_requested.emit()
