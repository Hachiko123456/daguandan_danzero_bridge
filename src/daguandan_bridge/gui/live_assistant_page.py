from __future__ import annotations

import html
import json
import re
from typing import Any

import cv2
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    CheckBox,
    ComboBox,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    SpinBox,
    StrongBodyLabel,
    TitleLabel,
    isDarkTheme,
    qconfig,
)

from ..advisor_strategy import ADVISOR_OPTIONS, RECORDING_MODE_OPTIONS
from ..danzero.state import RANKS
from ..live.display_text import (
    event_action_text,
    event_prefix,
    live_status_text,
    reasons_text,
    seat_text,
)
from ..live.models import LiveEvent
from ..domain.live_runtime import LiveAdvice, LiveUpdate, ReviewRequest
from ..live.recognition_strategy import RECOGNITION_STRATEGY_OPTIONS
from ..live.truth_log import card_code_to_text, card_text_to_code
from .single_image_danzero_page import CardBadge
from .truth_log_editor import CardPickerDialog
from .live_controller import LiveAssistantController


_SEAT_LABELS = {
    "self": "自己",
    "right": "右家",
    "opposite": "对家",
    "left": "左家",
}
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

_TURN_RECOVERY_PENDING = "turn_recovery_pending"
_WIND_CATCH_PASS_RECOVERY_PENDING = "wind_catch_pass_recovery_pending"
_CANNOT_BEAT_MIN_CONFIDENCE = 0.80


class _ClickableCardStrip(QWidget):
    """A compact hand preview that opens the existing card picker on click."""

    clicked = Signal()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class LiveAssistantPage(ScrollArea):
    """Semi-automatic live assistant; all game decisions stay in the orchestrator."""

    compact_mode_requested = Signal()

    def __init__(self, runtime: Any | None = None, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime or LiveAssistantController()
        self.review_candidate_buttons: list[PushButton] = []
        self._last_event_id = ""
        self._shown_event_ids: set[str] = set()
        self._last_advice_timeline_key: tuple[object, ...] | None = None
        self._session_active = False
        self._geometry_terminal_error_visible = False
        self.setObjectName("liveAssistantPage")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.viewport().setStyleSheet("background: transparent;")
        self._content = QWidget()
        self._content.setObjectName("liveAssistantContent")
        self._content.setStyleSheet("QWidget#liveAssistantContent { background: transparent; }")
        self.setWidget(self._content)
        self._build_ui()
        self._connect_runtime()
        warmup = getattr(self.runtime, "warm_danzero", None)
        if callable(warmup):
            warmup()
        qconfig.themeChanged.connect(self._apply_theme)
        self._apply_theme()
        self._refresh_initialization()

    def _apply_theme(self, *_args) -> None:
        if isDarkTheme():
            background, foreground = "#202020", "#f5f5f5"
        else:
            background, foreground = "#f3f3f3", "#1f1f1f"
        self._content.setStyleSheet(
            "QWidget#liveAssistantContent {"
            f"background: {background}; color: {foreground};"
            "} QWidget#liveAssistantContent QLabel {"
            f"color: {foreground};"
            "}"
        )
        self.timeline.setStyleSheet(self._timeline_style())

    def _advisor_display_name(self) -> str:
        strategy = str(getattr(self.runtime, "advisor_strategy", "") or "")
        combo = getattr(self, "advisor_strategy_combo", None)
        if combo is not None and combo.currentData():
            strategy = str(combo.currentData())
        return dict(ADVISOR_OPTIONS).get(strategy, "建议模型")

    @staticmethod
    def _timeline_style() -> str:
        if isDarkTheme():
            return (
                "QTextBrowser#liveTimeline { background: #202020; color: #f5f5f5; "
                "border: 1px solid #3f3f3f; border-radius: 6px; padding: 6px; }"
            )
        return (
            "QTextBrowser#liveTimeline { background: #ffffff; color: #1f1f1f; "
            "border: 1px solid #d7d7d7; border-radius: 6px; padding: 6px; }"
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self._content)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(16)
        root.addWidget(TitleLabel("实时出牌助手"))
        subtitle = BodyLabel(
            "持续监听页面：程序只观察与建议，不会点击游戏；确认起手牌和首出信息后自动开始。"
        )
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        self.initial_card = CardWidget()
        initial_layout = QVBoxLayout(self.initial_card)
        initial_layout.setContentsMargins(18, 16, 18, 16)
        initial_layout.setSpacing(10)
        initial_layout.addWidget(StrongBodyLabel("1. 当前页面识别"))
        form = QFormLayout()
        self.round_level_combo = ComboBox()
        for rank in RANKS:
            self.round_level_combo.addItem(rank, userData=rank)
        self.lead_player_combo = ComboBox()
        self.lead_player_combo.addItem("待自动识别", userData="")
        for seat in ("self", "right", "opposite", "left"):
            self.lead_player_combo.addItem(_SEAT_LABELS[seat], userData=seat)
        self.lead_player_combo.setToolTip("仅作提示；实时开局时会再次确认。")
        self.recognition_strategy_combo = ComboBox()
        for value, label in RECOGNITION_STRATEGY_OPTIONS:
            self.recognition_strategy_combo.addItem(label, userData=value)
        default_strategy = self.recognition_strategy_combo.findData("two_valid_streak")
        if default_strategy >= 0:
            self.recognition_strategy_combo.setCurrentIndex(default_strategy)
        self.recognition_strategy_combo.setToolTip(
            "实时与“状态机管线”复测使用同一策略；可用同一录像横向比较。"
        )
        self.advisor_strategy_combo = ComboBox()
        for value, label in ADVISOR_OPTIONS:
            self.advisor_strategy_combo.addItem(label, userData=value)
        selected_advisor = str(getattr(self.runtime, "advisor_strategy", "danzero"))
        selected_index = self.advisor_strategy_combo.findData(selected_advisor)
        if selected_index >= 0:
            self.advisor_strategy_combo.setCurrentIndex(selected_index)
        self.advisor_strategy_combo.setToolTip(
            "只影响下一局；实时对局开始后锁定，结束后可重新选择。"
        )
        self.recording_mode_combo = ComboBox()
        for value, label in RECORDING_MODE_OPTIONS:
            self.recording_mode_combo.addItem(label, userData=value)
        recording_mode = str(
            getattr(
                self.runtime,
                "recording_mode",
                "game"
                if bool(getattr(self.runtime, "session_data_recording_enabled", True))
                else "none",
            )
        )
        recording_index = self.recording_mode_combo.findData(recording_mode)
        if recording_index >= 0:
            self.recording_mode_combo.setCurrentIndex(recording_index)
        self.recording_mode_combo.setToolTip(
            "不保存：仅实时建议。\n"
            "对局录制：确认起手牌后开始保存。\n"
            "完整牌桌录制：保存准备、发牌与对局；大厅和结算仅监听。"
        )
        self.recording_capacity_spin = SpinBox(self)
        self.recording_capacity_spin.setRange(1, 1024)
        self.recording_capacity_spin.setSuffix(" GB")
        configured_bytes = int(
            getattr(self.runtime, "recording_max_total_bytes", 20 * 1024 ** 3)
        )
        self.recording_capacity_spin.setValue(
            max(1, int(round(configured_bytes / 1024 ** 3)))
        )
        self.recording_capacity_spin.setToolTip(
            "限制 sessions 中视频、截图和事故媒体的总容量。"
            "达到上限后识别与推荐继续，但录像会停止。"
        )
        self.automatic_log_media_check = CheckBox("自动诊断包含视频和截图", self)
        self.automatic_log_media_check.setChecked(
            bool(getattr(self.runtime, "automatic_log_include_media", False))
        )
        self.automatic_log_media_check.setToolTip(
            "开启后每局封存时自动生成含 game.avi 的完整诊断 ZIP；"
            "文件较大。关闭时仍可点击“导出最近一局完整诊断”。"
        )
        self.recording_storage_status = CaptionLabel()
        self.recording_storage_status.setWordWrap(True)
        self.recording_storage_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        # Internal normalized codes remain here for the state machine; the
        # user edits the visible card strip through the existing picker.
        self.hand_edit = LineEdit(self)
        self.hand_edit.hide()
        self.initial_hand_badges: list[CardBadge] = []
        self.initial_hand_scroll = ScrollArea()
        self.initial_hand_scroll.setObjectName("liveInitialHandCards")
        self.initial_hand_scroll.setToolTip("点击牌面可修改。")
        self.initial_hand_scroll.setWidgetResizable(False)
        # CardBadge(compact=True) is 36 x 50.  The old 50-pixel scroll area
        # then forced every badge down to 27 x 40, clipping ranks and suits
        # even though the recognized 27-card data was correct.
        self.initial_hand_scroll.setFixedHeight(64)
        self.initial_hand_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.initial_hand_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.initial_hand_cards = _ClickableCardStrip()
        self.initial_hand_cards.setCursor(Qt.CursorShape.PointingHandCursor)
        self.initial_hand_cards_layout = QHBoxLayout(self.initial_hand_cards)
        self.initial_hand_cards_layout.setContentsMargins(4, 4, 4, 4)
        self.initial_hand_cards_layout.setSpacing(3)
        self.initial_hand_scroll.setWidget(self.initial_hand_cards)
        self._render_initial_hand_cards(())
        form.addRow("当前级牌", self.round_level_combo)
        form.addRow("首发候选", self.lead_player_combo)
        form.addRow("识别策略", self.recognition_strategy_combo)
        form.addRow("建议模型", self.advisor_strategy_combo)
        form.addRow("保存方式", self.recording_mode_combo)
        form.addRow("录像总容量", self.recording_capacity_spin)
        form.addRow("自动诊断", self.automatic_log_media_check)
        form.addRow("录像容量状态", self.recording_storage_status)
        form.addRow("起手牌", self.initial_hand_scroll)
        initial_layout.addLayout(form)
        initial_actions = QHBoxLayout()
        self.recognize_initial_button = PushButton("识别当前页面（持续监听）")
        initial_actions.addWidget(self.recognize_initial_button)
        initial_actions.addStretch(1)
        initial_layout.addLayout(initial_actions)
        self.initialization_status = CaptionLabel()
        self.initialization_status.setWordWrap(True)
        initial_layout.addWidget(self.initialization_status)
        self.danzero_warmup_status = CaptionLabel(
            f"{self._advisor_display_name()} 模型准备中"
        )
        self.danzero_warmup_status.setWordWrap(True)
        initial_layout.addWidget(self.danzero_warmup_status)
        root.addWidget(self.initial_card)

        columns = QHBoxLayout()
        columns.setSpacing(16)
        preview_card = CardWidget()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 14, 16, 16)
        preview_layout.addWidget(StrongBodyLabel("实时画面"))
        self.preview = QLabel("尚未开始采集")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        self.preview.setStyleSheet(
            "background: #20252b; color: #e8eaed; border-radius: 6px;"
        )
        preview_layout.addWidget(self.preview, 1)
        columns.addWidget(preview_card, 3)

        state_card = CardWidget()
        state_layout = QVBoxLayout(state_card)
        state_layout.setContentsMargins(16, 14, 16, 16)
        self.timeline_title = StrongBodyLabel("对局动态")
        state_layout.addWidget(self.timeline_title)
        self.live_status = BodyLabel("状态：等待开局")
        self.turn_status = BodyLabel("当前回合：—")
        state_layout.addWidget(self.live_status)
        state_layout.addWidget(self.turn_status)
        state_layout.addSpacing(8)
        self.timeline = QTextBrowser()
        self.timeline.setObjectName("liveTimeline")
        self.timeline.setReadOnly(True)
        self.timeline.setOpenExternalLinks(False)
        self.timeline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.timeline.setMinimumHeight(280)
        self.timeline.setPlaceholderText(
            "动作、模型建议与异常会依次显示在这里；可直接框选并复制。"
        )
        state_layout.addWidget(self.timeline, 1)
        control_row = QHBoxLayout()
        self.pause_button = PushButton("暂停")
        self.resume_button = PushButton("继续")
        self.finish_button = PrimaryPushButton("结束并封存")
        self.compact_button = PushButton("进入极简推荐浮窗")
        control_row.addWidget(self.pause_button)
        control_row.addWidget(self.resume_button)
        control_row.addWidget(self.compact_button)
        control_row.addWidget(self.finish_button)
        state_layout.addLayout(control_row)
        log_row = QHBoxLayout()
        self.open_log_directory_button = PushButton("打开日志目录")
        self.export_full_diagnostic_button = PushButton("导出最近一局完整诊断")
        log_row.addWidget(self.open_log_directory_button)
        log_row.addWidget(self.export_full_diagnostic_button)
        state_layout.addLayout(log_row)
        self.log_delivery_status = CaptionLabel(self._automatic_log_description())
        self.log_delivery_status.setWordWrap(True)
        self.log_delivery_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        state_layout.addWidget(self.log_delivery_status)
        self._apply_log_delivery_availability()
        columns.addWidget(state_card, 2)
        root.addLayout(columns)

        self.fabledan_debug_card = CardWidget()
        debug_layout = QVBoxLayout(self.fabledan_debug_card)
        debug_layout.setContentsMargins(18, 16, 18, 16)
        debug_layout.setSpacing(10)
        debug_header = QHBoxLayout()
        debug_header.addWidget(StrongBodyLabel("FableDan 模型评分"))
        debug_header.addStretch(1)
        self.fabledan_detail_button = PushButton("查看模型诊断")
        debug_header.addWidget(self.fabledan_detail_button)
        debug_layout.addLayout(debug_header)

        debug_summary = QGridLayout()
        debug_summary.setHorizontalSpacing(20)
        debug_summary.setVerticalSpacing(6)
        self.fabledan_recommendation = TitleLabel("-")
        self.fabledan_q_value = BodyLabel("Q值：-")
        self.fabledan_q_gap = BodyLabel("Top1 - Top2：-")
        self.fabledan_context = CaptionLabel("当前需要压：-　|　当前级牌：-")
        debug_summary.addWidget(CaptionLabel("推荐"), 0, 0)
        debug_summary.addWidget(self.fabledan_recommendation, 1, 0)
        debug_summary.addWidget(self.fabledan_q_value, 0, 1)
        debug_summary.addWidget(self.fabledan_q_gap, 1, 1)
        debug_summary.addWidget(self.fabledan_context, 2, 0, 1, 2)
        debug_layout.addLayout(debug_summary)
        debug_layout.addWidget(StrongBodyLabel("模型评分前三（Q 值由高到低）"))
        self.fabledan_top_three_hint = CaptionLabel(
            "Q 值用于模型排序，并不是可直接解读为百分比的真实胜率。"
        )
        self.fabledan_top_three_hint.setWordWrap(True)
        debug_layout.addWidget(self.fabledan_top_three_hint)
        self.fabledan_candidates = BodyLabel("-")
        self.fabledan_candidates.setWordWrap(True)
        self.fabledan_candidates.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        debug_layout.addWidget(self.fabledan_candidates)
        self.fabledan_detail = QTextBrowser()
        self.fabledan_detail.setReadOnly(True)
        self.fabledan_detail.setMinimumHeight(260)
        self.fabledan_detail.hide()
        debug_layout.addWidget(self.fabledan_detail)
        self.fabledan_debug_card.hide()
        root.addWidget(self.fabledan_debug_card)

        self.review_bar = CardWidget()
        review_layout = QVBoxLayout(self.review_bar)
        review_layout.setContentsMargins(16, 14, 16, 16)
        self.review_title = StrongBodyLabel("需要确认")
        self.review_reason = CaptionLabel()
        self.review_reason.setWordWrap(True)
        self.review_buttons_layout = QHBoxLayout()
        review_layout.addWidget(self.review_title)
        review_layout.addWidget(self.review_reason)
        review_layout.addLayout(self.review_buttons_layout)
        self.manual_editor = QWidget()
        manual_layout = QHBoxLayout(self.manual_editor)
        manual_layout.setContentsMargins(0, 6, 0, 0)
        self.manual_action_combo = ComboBox()
        self.manual_action_combo.addItem("出牌", userData="play")
        self.manual_action_combo.addItem("不出", userData="pass")
        self.manual_cards_edit = LineEdit()
        self.manual_cards_edit.setPlaceholderText("都不对时填写正确牌面，例如黑桃7、红桃7")
        self.manual_confirm_button = PrimaryPushButton("确认补录")
        manual_layout.addWidget(self.manual_action_combo)
        manual_layout.addWidget(self.manual_cards_edit, 1)
        manual_layout.addWidget(self.manual_confirm_button)
        self.manual_editor.hide()
        review_layout.addWidget(self.manual_editor)
        self.review_bar.hide()
        root.addWidget(self.review_bar)

        self.error_status = CaptionLabel()
        self.error_status.setWordWrap(True)
        root.addWidget(self.error_status)
        root.addStretch(1)

        self.recognize_initial_button.clicked.connect(self._start_listening)
        self.pause_button.clicked.connect(self.runtime.pause)
        self.resume_button.clicked.connect(self.runtime.resume)
        self.finish_button.clicked.connect(self.runtime.finish)
        self.compact_button.clicked.connect(self.compact_mode_requested.emit)
        self.open_log_directory_button.clicked.connect(self._open_log_directory)
        self.export_full_diagnostic_button.clicked.connect(
            self._export_full_diagnostic
        )
        self.fabledan_detail_button.clicked.connect(
            self._toggle_fabledan_details
        )
        self.manual_confirm_button.clicked.connect(self._confirm_manual)
        self.hand_edit.textChanged.connect(self._refresh_initialization)
        self.initial_hand_cards.clicked.connect(self._edit_initial_hand)
        self.round_level_combo.currentIndexChanged.connect(self._refresh_initialization)
        self.lead_player_combo.currentIndexChanged.connect(self._refresh_initialization)
        self.recognition_strategy_combo.currentIndexChanged.connect(
            self._update_recognition_strategy
        )
        self.advisor_strategy_combo.currentIndexChanged.connect(
            self._update_advisor_strategy
        )
        self.recording_mode_combo.currentIndexChanged.connect(
            self._update_recording_mode
        )
        self.recording_capacity_spin.editingFinished.connect(
            self._update_recording_capacity
        )
        self.automatic_log_media_check.toggled.connect(
            self._update_automatic_log_media
        )
        self._refresh_recording_storage_status()
        self._update_recognition_strategy()
        self._set_live_controls(False)

    def _connect_runtime(self) -> None:
        for name, handler in (
            ("initial_recognized", self.apply_initial_recognition),
            ("update_ready", self.apply_update),
            ("frame_ready", self.show_frame),
            ("error", self.show_error),
            ("session_finished", self._session_finished),
            ("danzero_warmup_status", self.show_danzero_warmup_status),
            ("listening_status", self.show_listening_status),
            ("log_delivery_status", self.show_log_delivery_status),
            ("recording_status", self.show_recording_status),
        ):
            signal = getattr(self.runtime, name, None)
            if signal is not None:
                signal.connect(handler)

    @staticmethod
    def _parse_cards(raw: str) -> tuple[str, ...]:
        cards: list[str] = []
        for value in re.split(r"[\s,，、]+", raw.strip()):
            if not value:
                continue
            try:
                cards.append(card_text_to_code(value))
            except ValueError:
                cards.append(value)
        return tuple(cards)

    def _refresh_initialization(self, *_args) -> None:
        cards = self._parse_cards(self.hand_edit.text())
        if self._session_active:
            self.initialization_status.setText("实时对局已开始；初始字段已锁定。")
        elif len(cards) == 27:
            self.initialization_status.setText(
                "已识别 27 张初始手牌；连续两次识别一致后会自动开始。"
            )
        else:
            self.initialization_status.setText(
                f"持续监听页面中；需要稳定识别 27 张初始手牌，当前为 {len(cards)} 张。"
            )

    @staticmethod
    def _recognized_seat_text(value: object) -> str:
        seat = str(value or "")
        return _SEAT_LABELS.get(seat, "未识别")

    def _waiting_initialization_status(self, result: object) -> str:
        buttons = set(getattr(result, "buttons", ()) or ())
        if buttons & {"change_table", "continue_game"}:
            return (
                "持续监听页面中；当前为结算页（换桌 / 再来一局）；"
                "不会创建新的对局录制，等待下一局牌桌出现。"
            )
        level = str(getattr(result, "round_level", "") or "")
        wild_rank = str(getattr(result, "wild_rank", "") or "")
        hand_count = len(tuple(getattr(result, "my_hand", ()) or ()))
        current_player = self._recognized_seat_text(
            getattr(result, "current_player", None)
        )
        lead_player = self._recognized_seat_text(
            getattr(result, "lead_player", None)
        )
        recognized_level = level if level in RANKS else "未识别"
        recognized_wild_rank = wild_rank if wild_rank in RANKS else "未识别"
        if level not in RANKS:
            progress = "建局状态：等待级牌识别。"
        elif hand_count != 27:
            progress = "建局状态：等待稳定的 27 张起手牌。"
        else:
            progress = "建局状态：等待下一帧确认同一副起手牌。"
        return (
            f"持续监听页面中；识别级牌：{recognized_level}；"
            f"百搭级牌：{recognized_wild_rank}；起手牌：{hand_count}/27 张；"
            f"当前行动：{current_player}；首发候选：{lead_player}；{progress}"
        )

    def _start_listening(self) -> None:
        self._update_advisor_strategy()
        setter = getattr(self.runtime, "set_recognition_strategy", None)
        if callable(setter):
            setter(str(self.recognition_strategy_combo.currentData()))
        start = getattr(self.runtime, "start_listening", None)
        if callable(start):
            if start() is False:
                return
            self.compact_mode_requested.emit()
            self.initialization_status.setText(
                "已锁定牌桌标准画面；识别级牌：等待首帧；起手牌：0/27 张。"
            )
            return
        # Compatibility only for an older embedded controller.  The shipped
        # controller always exposes persistent listening.
        self.runtime.recognize_initial()

    def _update_recognition_strategy(self, *_args) -> None:
        setter = getattr(self.runtime, "set_recognition_strategy", None)
        if callable(setter):
            setter(str(self.recognition_strategy_combo.currentData()))

    def _update_advisor_strategy(self, *_args) -> None:
        if self._session_active:
            return
        setter = getattr(self.runtime, "set_advisor_strategy", None)
        if not callable(setter):
            return
        try:
            setter(str(self.advisor_strategy_combo.currentData()))
        except Exception as exc:
            self.show_error(str(exc))

    def _update_recording_mode(self, *_args) -> None:
        if self._session_active:
            return
        mode = str(self.recording_mode_combo.currentData())
        mode_setter = getattr(self.runtime, "set_recording_mode", None)
        legacy_setter = getattr(
            self.runtime,
            "set_session_data_recording_enabled",
            None,
        )
        if not callable(mode_setter) and (mode == "all" or not callable(legacy_setter)):
            return
        try:
            if callable(mode_setter):
                mode_setter(mode)
            else:
                legacy_setter(mode != "none")
            self._apply_log_delivery_availability()
        except Exception as exc:
            self.show_error(str(exc))
            restored = str(
                getattr(
                    self.runtime,
                    "recording_mode",
                    "game"
                    if bool(
                        getattr(self.runtime, "session_data_recording_enabled", True)
                    )
                    else "none",
                )
            )
            index = self.recording_mode_combo.findData(restored)
            if index >= 0:
                self.recording_mode_combo.blockSignals(True)
                self.recording_mode_combo.setCurrentIndex(index)
                self.recording_mode_combo.blockSignals(False)

    def _update_recording_capacity(self) -> None:
        if self._session_active:
            return
        setter = getattr(self.runtime, "set_recording_max_total_gb", None)
        if not callable(setter):
            return
        try:
            setter(self.recording_capacity_spin.value())
            self._refresh_recording_storage_status()
        except Exception as exc:
            self.show_error(str(exc))
            configured = int(
                getattr(self.runtime, "recording_max_total_bytes", 20 * 1024 ** 3)
            )
            self.recording_capacity_spin.blockSignals(True)
            self.recording_capacity_spin.setValue(
                max(1, int(round(configured / 1024 ** 3)))
            )
            self.recording_capacity_spin.blockSignals(False)

    def _update_automatic_log_media(self, checked: bool) -> None:
        if self._session_active:
            return
        setter = getattr(self.runtime, "set_automatic_log_include_media", None)
        if not callable(setter):
            return
        try:
            setter(bool(checked))
            self.log_delivery_status.setText(self._automatic_log_description())
            self._apply_log_delivery_availability()
        except Exception as exc:
            self.show_error(str(exc))
            restored = bool(
                getattr(self.runtime, "automatic_log_include_media", False)
            )
            self.automatic_log_media_check.blockSignals(True)
            self.automatic_log_media_check.setChecked(restored)
            self.automatic_log_media_check.blockSignals(False)

    def _refresh_recording_storage_status(self) -> None:
        provider = getattr(self.runtime, "recording_storage_summary", None)
        if not callable(provider):
            self.recording_storage_status.setText("当前运行时未提供容量统计")
            return
        try:
            value = provider()
        except Exception as exc:
            self.recording_storage_status.setText(f"容量统计失败：{exc}")
            return
        if not isinstance(value, dict):
            self.recording_storage_status.setText("容量统计不可用")
            return
        used = int(value.get("used_bytes", 0) or 0)
        limit = int(value.get("limit_bytes", 0) or 0)
        remaining = int(value.get("remaining_bytes", 0) or 0)
        exhausted = bool(value.get("capacity_exhausted", False))
        prefix = "⚠ 录像配额已用完；新对局将没有完整视频。" if exhausted else "录像配额正常。"
        self.recording_storage_status.setText(
            f"{prefix} 已用 {self._format_gib(used)} / "
            f"上限 {self._format_gib(limit)}，剩余 {self._format_gib(remaining)}"
        )

    def _automatic_log_description(self) -> str:
        return (
            "封局后会自动生成包含截图和视频的完整诊断 ZIP"
            if bool(getattr(self.runtime, "automatic_log_include_media", False))
            else "封局后会自动生成不含截图和视频的诊断 ZIP"
        )

    @staticmethod
    def _format_gib(value: int) -> str:
        return f"{max(0, int(value)) / 1024 ** 3:.2f} GB"

    def _apply_log_delivery_availability(self) -> None:
        """Keep log actions honest when the user selected no persistence."""

        enabled = self._log_delivery_enabled()
        self.open_log_directory_button.setEnabled(enabled)
        if not enabled:
            self.export_full_diagnostic_button.setEnabled(False)
            self.log_delivery_status.setText("当前已关闭对局数据保存，日志功能不可用")
        elif "日志功能不可用" in self.log_delivery_status.text():
            self.export_full_diagnostic_button.setEnabled(True)
            self.log_delivery_status.setText(self._automatic_log_description())

    def _log_delivery_enabled(self) -> bool:
        mode = str(
            getattr(
                self.runtime,
                "recording_mode",
                "game"
                if bool(getattr(self.runtime, "session_data_recording_enabled", True))
                else "none",
            )
        )
        return mode != "none"

    def apply_initial_recognition(self, result: object, snapshot: object | None) -> None:
        level = getattr(result, "round_level", None)
        lead = getattr(result, "lead_player", None) or getattr(
            result, "current_player", None
        )
        self._set_combo_data(self.round_level_combo, level)
        # A single image is a useful display hint, but never a commitment:
        # the live opening state machine must reconfirm it after any doubling
        # controls have cleared.  Reset first so a missing detection cannot
        # silently reuse the previous game's candidate.
        self.lead_player_combo.setCurrentIndex(0)
        buttons = set(getattr(result, "buttons", ()))
        if not (buttons & {"super_double", "double"}):
            self._set_combo_data(self.lead_player_combo, lead)
        self.hand_edit.setText(" ".join(getattr(result, "my_hand", ())))
        self._render_initial_hand_cards(self._parse_cards(self.hand_edit.text()))
        if snapshot is not None:
            self.show_frame(snapshot)
        diagnostics = tuple(getattr(result, "diagnostics", ()))
        self.error_status.setText("；".join(diagnostics[:4]))
        if not self._session_active:
            self.initialization_status.setText(
                self._waiting_initialization_status(result)
            )

    @staticmethod
    def _set_combo_data(combo: ComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_initial_fields_enabled(self, enabled: bool) -> None:
        self.hand_edit.setEnabled(enabled)
        self.initial_hand_scroll.setEnabled(enabled)
        self.round_level_combo.setEnabled(enabled)
        self.lead_player_combo.setEnabled(enabled)
        self.recognition_strategy_combo.setEnabled(enabled)
        self.advisor_strategy_combo.setEnabled(enabled)
        self.recording_mode_combo.setEnabled(enabled)
        self.recording_capacity_spin.setEnabled(enabled)
        self.automatic_log_media_check.setEnabled(enabled)

    def _render_initial_hand_cards(self, cards: tuple[str, ...]) -> None:
        while self.initial_hand_cards_layout.count():
            item = self.initial_hand_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.initial_hand_badges = []
        if not cards:
            label = CaptionLabel("尚未识别手牌")
            self.initial_hand_cards_layout.addWidget(label)
            self.initial_hand_cards.setMinimumWidth(240)
            self.initial_hand_cards.setToolTip("尚未识别手牌")
            return
        for card in cards:
            badge = CardBadge(card, compact=True)
            badge.setFixedSize(36, 50)
            badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.initial_hand_badges.append(badge)
            self.initial_hand_cards_layout.addWidget(badge)
        self.initial_hand_cards_layout.addStretch(1)
        self.initial_hand_cards.setMinimumWidth(max(240, len(cards) * 39 + 12))
        self.initial_hand_cards.setToolTip("、".join(card_code_to_text(card) for card in cards))

    def _edit_initial_hand(self) -> None:
        if self._session_active:
            return
        dialog = CardPickerDialog(self._parse_cards(self.hand_edit.text()), self)
        if dialog.exec():
            self.hand_edit.setText(" ".join(dialog.cards()))
            self._render_initial_hand_cards(self._parse_cards(self.hand_edit.text()))

    def _reset_transient_session_ui(self) -> None:
        """Forget only the previous live-page view before a fresh game."""

        self._last_event_id = ""
        self._shown_event_ids.clear()
        self._last_advice_timeline_key = None
        self.timeline.clear()
        self.review_bar.hide()
        self.lead_player_combo.setCurrentIndex(0)

    def _set_live_controls(self, active: bool) -> None:
        self.pause_button.setEnabled(active)
        self.resume_button.setEnabled(active)
        self.finish_button.setEnabled(active)

    def _open_log_directory(self) -> None:
        requester = getattr(
            self.runtime,
            "request_open_automatic_log_directory",
            None,
        )
        if callable(requester):
            self.open_log_directory_button.setEnabled(False)
            self.open_log_directory_button.setText("正在打开…")
            requester()
            return
        opener = getattr(self.runtime, "open_automatic_log_directory", None)
        if not callable(opener):
            self.show_error("当前运行时不支持打开日志目录")
            return
        try:
            path = opener()
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.log_delivery_status.setText(f"已打开日志目录：{path}")

    def _export_full_diagnostic(self) -> None:
        exporter = getattr(self.runtime, "request_full_diagnostic_export", None)
        if not callable(exporter):
            self.show_error("当前运行时不支持完整诊断导出")
            return
        self.export_full_diagnostic_button.setEnabled(False)
        self.export_full_diagnostic_button.setText("正在导出完整诊断…")
        exporter()

    def show_log_delivery_status(self, value: object) -> None:
        if not isinstance(value, dict):
            return
        status = str(value.get("status", "") or "").upper()
        status = {
            "LOADING": "RUNNING",
            "SUCCESS": "PASS",
            "FAILURE": "FAIL",
        }.get(status, status)
        action = str(value.get("action", "") or "")
        include_media = bool(value.get("include_media", False))
        if status != "DISABLED" and not self._log_delivery_enabled():
            self._apply_log_delivery_availability()
            return
        if status == "DISABLED":
            self.open_log_directory_button.setEnabled(False)
            self.export_full_diagnostic_button.setEnabled(False)
            self.log_delivery_status.setText(
                str(value.get("message") or "当前已关闭对局数据保存，日志功能不可用")
            )
            return
        if action == "open_directory":
            if status == "RUNNING":
                self.open_log_directory_button.setEnabled(False)
                self.open_log_directory_button.setText("正在打开…")
                self.log_delivery_status.setText("正在打开日志目录")
                return
            self.open_log_directory_button.setEnabled(True)
            self.open_log_directory_button.setText("打开日志目录")
            if status == "PASS":
                self.log_delivery_status.setText(
                    f"已打开日志目录：{value.get('output_directory') or ''}"
                )
            elif status == "FAIL":
                self.log_delivery_status.setText(
                    f"打开日志目录失败：{value.get('error') or '未知错误'}"
                )
            return
        if status == "RUNNING":
            self.export_full_diagnostic_button.setEnabled(False)
            self.export_full_diagnostic_button.setText("正在导出完整诊断…")
            self.log_delivery_status.setText(
                str(value.get("message") or "正在后台生成完整诊断")
            )
            return
        self.export_full_diagnostic_button.setEnabled(True)
        self.export_full_diagnostic_button.setText("导出最近一局完整诊断")
        if status == "PASS":
            path = str(value.get("diagnostic_zip_path") or "")
            prefix = "完整诊断已生成" if include_media else "本局日志已自动生成（不含截图和视频）"
            self.log_delivery_status.setText(f"{prefix}：{path}")
        elif status == "FAIL":
            self.log_delivery_status.setText(
                f"日志导出失败：{value.get('error') or '未知错误'}"
            )

    def show_frame(self, snapshot: object) -> None:
        image = getattr(snapshot, "image", None)
        if image is None:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        self.preview.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def apply_update(self, update: LiveUpdate) -> None:
        if (
            update.status in {"waiting_lead", "running", "review_required", "paused"}
            and not self._session_active
        ):
            # A controller-started automatic session must reset only the
            # transient view.  Historical session folders and logs are never
            # touched here.
            self._reset_transient_session_ui()
            self._session_active = True
            self._set_initial_fields_enabled(False)
            self._set_live_controls(True)
        if update.fast_signals is not None and update.fast_signals.super_double_visible:
            self.live_status.setText("状态：正在决定是否加倍")
            self.turn_status.setText("加倍按钮显示期间不进行首出或出牌识别")
        elif update.status == "waiting_lead":
            self.live_status.setText("状态：等待首发标志（检测到首发后自动开始）")
            self.turn_status.setText(
                "等待加倍结束与首出标志；加倍按钮显示期间不会进行首出或出牌识别。"
            )
        else:
            self.live_status.setText(f"状态：{live_status_text(update.status)}")
            player = update.snapshot.current_player
            if player is None and len(getattr(update.snapshot, "finished_seats", ())) >= 3:
                self.turn_status.setText("赛果已确定，等待结算界面自动封存本局")
            else:
                self.turn_status.setText(
                    f"当前行动：{seat_text(player, unknown='等待确认')}"
                    f"　|　第 {update.snapshot.turn_id} 手"
                )
        events = tuple(getattr(update, "events", ()))
        if not events and update.event is not None:
            events = (update.event,)
        for event in events:
            if event.event_id in self._shown_event_ids:
                continue
            self._shown_event_ids.add(event.event_id)
            self._last_event_id = event.event_id
            self._append_event_to_timeline(event)
        if update.review is not None:
            self.show_review(update.review)
        else:
            self.review_bar.hide()
        self._show_advice(
            update.advice,
            update.fast_signals,
            live_status=update.status,
            snapshot=update.snapshot,
        )
        if update.status == "sealed":
            self._session_finished(update)

    @staticmethod
    def _cannot_beat_candidate(fast_signals: object | None) -> bool:
        try:
            confidence = float(
                getattr(fast_signals, "cannot_beat_confidence", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            confidence = 0.0
        return bool(
            getattr(fast_signals, "cannot_beat_visible", False)
            and confidence >= _CANNOT_BEAT_MIN_CONFIDENCE
            and getattr(fast_signals, "active_player", None) in (None, "self")
            and getattr(fast_signals, "self_action_buttons_visible", False)
            and not getattr(fast_signals, "effect_visible", False)
        )

    @staticmethod
    def _advice_matches_snapshot(advice: LiveAdvice, snapshot: object) -> bool:
        turn_id = getattr(snapshot, "turn_id", None)
        revision = getattr(snapshot, "revision", None)
        if turn_id is None or revision is None:
            return True
        return bool(
            advice.key.turn_id == int(turn_id)
            and advice.key.state_revision == int(revision)
        )

    def _show_fast_cannot_beat_status(self) -> None:
        """Show a provisional local-button status without committing PASS."""

        self.fabledan_debug_card.hide()
        self.live_status.setText("状态：检测到要不起，正在确认不出")
        self.turn_status.setText(
            "确认后直接提示不出；单帧信号不会直接提交动作"
        )

    def _show_advice(
        self,
        raw: object | None,
        fast_signals: object | None = None,
        *,
        live_status: object | None = None,
        snapshot: object | None = None,
    ) -> None:
        cannot_beat_visible = bool(
            live_status == "running"
            and getattr(snapshot, "current_player", None) == "self"
            and self._cannot_beat_candidate(fast_signals)
        )
        # Withheld states describe canonical recovery/history state and always
        # outrank a raw button match.  Outside those states, the current-frame
        # candidate outranks any advice object left from the preceding turn.
        if isinstance(raw, LiveAdvice) and raw.status == "withheld":
            if raw.withhold_reason == _WIND_CATCH_PASS_RECOVERY_PENDING:
                self.live_status.setText("状态：正在确认接风前的不出")
                self.turn_status.setText(
                    raw.error or "确认接风前的不出后将自动继续推荐"
                )
            elif raw.withhold_reason == _TURN_RECOVERY_PENDING:
                self.live_status.setText("状态：正在补齐刚才的快速出牌")
                self.turn_status.setText(raw.error or "补齐完成后将自动继续推荐")
            elif raw.withhold_reason == "previous_action_reread_pending":
                self.live_status.setText("状态：正在复核上一手牌面")
                self.turn_status.setText(raw.error or "复核完成后将自动更新推荐")
            else:
                self.live_status.setText("状态：已确认牌局历史存在缺口")
                self.turn_status.setText(raw.error or "请补正缺失动作后再继续")
            return
        button_ready = bool(
            isinstance(raw, LiveAdvice) and raw.status == "ready" and raw.visible
            and getattr(raw.advice, "strategy", None) == "button_cannot_beat"
            and self._advice_matches_snapshot(raw, snapshot)
        )
        if cannot_beat_visible and not button_ready:
            self._show_fast_cannot_beat_status()
            return
        if not isinstance(raw, LiveAdvice):
            return
        if not self._advice_matches_snapshot(raw, snapshot):
            self.fabledan_debug_card.hide()
            return
        if raw.status == "ready" and raw.advice is not None:
            if button_ready:
                self.live_status.setText("状态：建议不出")
                self.turn_status.setText("按钮判定，无需模型计算；请点击不出")
            self._show_fabledan_decision(raw.advice)
            suggestion = (
                "建议：不出"
                if raw.advice.is_pass
                else f"建议：出牌 · {self._play_type_text(raw.advice.play_type)}"
            )
            self._append_advice_timeline_entry(
                raw,
                suggestion,
            )
        elif raw.status == "failed":
            self._append_advice_timeline_entry(
                raw,
                f"错误：{raw.error}",
            )

    def _show_fabledan_decision(self, advice: object) -> None:
        engine_input = getattr(advice, "engine_input", None)
        decision = engine_input.get("decision") if isinstance(engine_input, dict) else None
        if not isinstance(decision, dict):
            self.fabledan_debug_card.hide()
            self.fabledan_detail.hide()
            return

        has_full_diagnostics = engine_input.get("debug") is True
        self.fabledan_detail_button.setVisible(has_full_diagnostics)
        if not has_full_diagnostics:
            self.fabledan_detail.hide()
            self.fabledan_detail.clear()

        best_action_text = str(decision.get("best_action_text") or "-")
        self.fabledan_recommendation.setText(best_action_text)
        self.fabledan_q_value.setText(
            f"Q值：{self._format_q_value(decision.get('best_q'))}"
        )
        self.fabledan_q_gap.setText(
            f"Top1 - Top2：{self._format_q_value(decision.get('q_gap'))}"
        )
        self.fabledan_context.setText(
            f"当前需要压：{engine_input.get('lead_text') or '-'}　|　"
            f"当前级牌：{engine_input.get('level_text') or '-'}"
        )
        candidates = decision.get("candidates")
        lines: list[str] = []
        if isinstance(candidates, list):
            for candidate in candidates[:3]:
                if not isinstance(candidate, dict):
                    continue
                lines.append(
                    f"{candidate.get('rank', len(lines) + 1)}. "
                    f"{candidate.get('action_text') or '-'}    "
                    f"Q={self._format_q_value(candidate.get('q'))}"
                )
        self.fabledan_candidates.setText("\n".join(lines) or "无候选动作")
        if has_full_diagnostics:
            self.fabledan_detail.setPlainText(
                json.dumps(engine_input, ensure_ascii=False, indent=2, sort_keys=True)
            )
        self.fabledan_debug_card.show()

    def _toggle_fabledan_details(self) -> None:
        visible = not self.fabledan_detail.isVisible()
        self.fabledan_detail.setVisible(visible)
        self.fabledan_detail_button.setText(
            "收起模型诊断" if visible else "查看模型诊断"
        )

    @staticmethod
    def _format_q_value(value: object) -> str:
        return f"{float(value):.4f}" if isinstance(value, (int, float)) else "-"

    def _append_event_to_timeline(self, event: LiveEvent) -> None:
        if event.event_type in {
            "advice_requested",
            "advice_ready",
            "advice_visible",
            "advice_failed",
        }:
            return
        prefix = html.escape(event_prefix(event))
        accent, marker = self._event_accent(event)
        if event.event_type == "player_played":
            cards = tuple(str(card) for card in event.payload.get("cards", ()))
            action = (
                f"{html.escape(seat_text(event.actor, unknown='系统'))}出牌："
                f"{self._cards_html(cards, event.payload.get('suit_options', ()))}"
            )
        else:
            action = html.escape(event_action_text(event))
        self._append_timeline_html(
            "<div style='margin:3px 0 7px 0; padding-left:7px; "
            f"border-left:3px solid {accent};'>"
            f"<span style='color:#6b7280;'>{prefix}</span> "
            f"<span style='color:{accent}; font-weight:600;'>{marker}</span> "
            f"<span>{action}</span>"
            "</div>"
        )

    def _append_advice_timeline_entry(self, raw: LiveAdvice, detail: str) -> None:
        advice = raw.advice
        cards = tuple(advice.cards) if advice is not None else ()
        key = (
            raw.key.request_id,
            raw.status,
            cards,
            raw.error,
            raw.suit_uncertain,
            raw.variant_count,
            raw.advice_agrees_across_variants,
            raw.semantic_uncertain,
            raw.semantic_source_history_indices,
            raw.suit_variant_count,
            raw.suit_equivalence_class_count,
            raw.semantic_variant_count,
        )
        if key == self._last_advice_timeline_key:
            return
        self._last_advice_timeline_key = key
        cards_html = self._cards_html(cards) if cards else ""
        advisor_name = self._advisor_display_name()
        if raw.status == "failed":
            accent, background, title = (
                "#B91C1C",
                "#FEF2F2",
                f"{advisor_name} 计算失败",
            )
        else:
            accent, background, title = (
                "#0F766E",
                "#E7F6F2",
                f"{advisor_name} 建议",
            )
            if advice is not None:
                detail += f"　{advice.elapsed_ms:.0f} ms"
        if raw.suit_uncertain:
            agreement = "建议一致" if raw.advice_agrees_across_variants else "建议存在分歧"
            if raw.suit_equivalence_class_count < raw.suit_variant_count:
                detail += (
                    f"；花色遮挡：{raw.suit_variant_count} 个实体牌状态归并为 "
                    f"{raw.suit_equivalence_class_count} 个模型等价输入，"
                    f"实际评估 {raw.variant_count} 个完整状态，{agreement}"
                )
            else:
                detail += (
                    f"；花色遮挡：生成 {raw.suit_variant_count} 个花色分支，"
                    f"共评估 {raw.variant_count} 个完整状态，{agreement}"
                )
        if raw.semantic_uncertain:
            source_text = "、".join(
                str(index) for index in raw.semantic_source_history_indices
            )
            agreement = "建议一致" if raw.advice_agrees_across_variants else "建议存在分歧"
            detail += (
                f"；动作语义多解（历史第 {source_text} 条）："
                f"生成 {raw.semantic_variant_count} 个语义分支，"
                f"共评估 {raw.variant_count} 个完整状态，{agreement}"
            )
        self._append_timeline_html(
            "<div style='margin:5px 0 9px 0; padding:6px; "
            f"background:{background}; border-left:4px solid {accent};'>"
            f"<span style='color:{accent}; font-size:18px; font-weight:700;'>"
            f"{html.escape(title)} · {html.escape(detail)}</span>{cards_html}"
            "</div>"
        )

    @staticmethod
    def _event_accent(event: LiveEvent) -> tuple[str, str]:
        if event.event_type == "player_passed":
            return "#7C3AED", "⏭ 不出"
        if event.event_type == "player_finished":
            placement = str(event.payload.get("placement", ""))
            return {
                "head": ("#B45309", "★ 头游"),
                "second": ("#2563EB", "◆ 二游"),
                "third": ("#6B7280", "◇ 三游"),
                "last": ("#B91C1C", "● 末游"),
            }.get(placement, ("#B45309", "出完牌"))
        if event.event_type == "wind_caught":
            return "#7C3AED", "↪ 接风"
        if event.event_type == "suit_corrected":
            return "#B45309", "⟳ 花色修正"
        if event.event_type == "game_end_detected":
            return "#B45309", "■ 自动封存"
        if event.event_type == "recognition_retry":
            reason = str(event.payload.get("reason", ""))
            if reason == "conflicting_valid_candidates":
                return "#B45309", "↻ 继续识别"
            if reason == "action_timeout":
                return "#64748B", "⌛ 等待动作"
        if event.event_type in {"recognition_retry", "review_required"}:
            return "#B91C1C", "! 识别提示"
        if event.event_type == "turn_started":
            return "#2563EB", "▶ 回合"
        return "#0F766E", "● 对局"

    @staticmethod
    def _cards_html(
        cards: tuple[str, ...],
        suit_options: object = (),
    ) -> str:
        if not cards:
            return ""
        raw_options = (
            tuple(
                tuple(str(suit) for suit in choices)
                for choices in suit_options
            )
            if isinstance(suit_options, (tuple, list))
            else ()
        )
        cells: list[str] = []
        for index, card in enumerate(cards):
            rank, suit, color = CardBadge._display_parts(card)
            candidates = (
                "/".join(
                    dict.fromkeys(
                        {"S": "♠", "H": "♥", "C": "♣", "D": "♦"}.get(value, value)
                        for value in raw_options[index]
                    )
                )
                if card.endswith("?") and index < len(raw_options)
                else ""
            )
            title = card_code_to_text(card)
            if candidates:
                title += f"（候选：{candidates}）"
            cells.append(
                "<td title='{title}' style='background:#ffffff; border:1px solid #c8cdd3; "
                "border-radius:5px; min-width:36px; text-align:center; padding:3px 5px;'>"
                "<span style='color:{color}; font-size:21px; font-weight:700;'>{suit}</span>"
                "<span style='color:{color}; font-size:18px; font-weight:700;'> {rank}</span></td>".format(
                    title=html.escape(title),
                    color=color,
                    suit=html.escape(suit),
                    rank=html.escape(rank),
                )
            )
        return "<table cellspacing='2' cellpadding='0' style='display:inline-table; vertical-align:middle;'><tr>{}</tr></table>".format(
            "".join(cells)
        )

    @staticmethod
    def _play_type_text(play_type: str) -> str:
        return _PLAY_TYPE_LABELS.get(str(play_type), "推荐牌型")

    def _append_timeline_html(self, entry: str) -> None:
        cursor = self.timeline.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(entry)
        cursor.insertBlock()
        self.timeline.setTextCursor(cursor)
        QTimer.singleShot(0, self._scroll_timeline_to_bottom)

    def _scroll_timeline_to_bottom(self) -> None:
        scrollbar = self.timeline.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _event_text(event: LiveEvent) -> str:
        return f"{event_prefix(event)} {event_action_text(event)}"

    def show_review(self, review: ReviewRequest) -> None:
        while self.review_buttons_layout.count():
            item = self.review_buttons_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.review_candidate_buttons.clear()
        self.review_title.setText(f"请确认{_SEAT_LABELS[review.player]}本次动作")
        if review.reason.startswith("lead_player"):
            self.review_title.setText("请确认首发座位")
            self.review_reason.setText(
                f"原因：{reasons_text(review.reason)}。未能在限定时间内检测到首出标志。"
            )
            for seat in ("self", "right", "opposite", "left"):
                button = PrimaryPushButton(_SEAT_LABELS[seat])
                button.clicked.connect(
                    lambda _checked=False, value=seat: self._confirm_lead(value)
                )
                self.review_buttons_layout.addWidget(button)
            self.review_buttons_layout.addStretch(1)
            self.manual_editor.hide()
            self.review_bar.show()
            return
        self.review_reason.setText(
            f"原因：{reasons_text(review.reason)}。录像继续，但状态和建议模型已暂停推进。"
        )
        for candidate in review.candidates:
            text = "不出" if candidate.is_pass else "、".join(
                card_code_to_text(card) for card in candidate.cards
            )
            button = PrimaryPushButton(
                f"{text}（{candidate.votes}票 / {candidate.confidence:.0%}）"
            )
            button.setEnabled(candidate.valid)
            if not candidate.valid:
                button.setToolTip(
                    f"自动校验未通过：{candidate.rejected_reason or review.reason}"
                )
            button.clicked.connect(
                lambda _checked=False, value=candidate.candidate_id: self._confirm_candidate(value)
            )
            self.review_candidate_buttons.append(button)
            self.review_buttons_layout.addWidget(button)
        pass_button = PushButton("不出")
        pass_button.clicked.connect(self._confirm_pass)
        all_wrong = PushButton("都不对")
        all_wrong.clicked.connect(lambda: self.manual_editor.show())
        self.review_buttons_layout.addWidget(pass_button)
        self.review_buttons_layout.addWidget(all_wrong)
        self.review_buttons_layout.addStretch(1)
        self.manual_editor.hide()
        self.review_bar.show()

    def _confirm_candidate(self, candidate_id: str) -> None:
        self.runtime.confirm_candidate(candidate_id)
        self.review_bar.hide()

    def _confirm_lead(self, seat: str) -> None:
        self.runtime.confirm_lead_player(seat)
        self.review_bar.hide()

    def _confirm_pass(self) -> None:
        self.runtime.confirm_manual_action(cards=(), is_pass=True)
        self.review_bar.hide()

    def _confirm_manual(self) -> None:
        is_pass = self.manual_action_combo.currentData() == "pass"
        cards = () if is_pass else self._parse_cards(self.manual_cards_edit.text())
        self.runtime.confirm_manual_action(cards=cards, is_pass=is_pass)
        self.review_bar.hide()

    def show_error(self, message: str) -> None:
        text = str(message)
        self._geometry_terminal_error_visible = (
            "监听已停止，请打开完整助手" in text
        )
        self.error_status.setText(f"错误：{text}")

    def show_recording_status(self, value: object) -> None:
        if isinstance(value, dict) and value.get("reason") == "recording_capacity_reached":
            self.error_status.setText("提示：录像容量已达上限，已停止录像，识别和推荐继续")

    def show_danzero_warmup_status(self, message: str) -> None:
        self.danzero_warmup_status.setText(str(message))

    def show_listening_status(self, status: object) -> None:
        if not isinstance(status, dict):
            return
        state = str(status.get("state", "") or "")
        generation = status.get("generation")
        if isinstance(generation, int):
            if generation < getattr(self, "_listening_generation", -1):
                return
            self._listening_generation = generation
        if state == "opening" and getattr(self.runtime, "orchestrator", None) is not None:
            return
        message = str(status.get("message", "") or "")
        if message:
            self.initialization_status.setText(message)
        if state == "failed":
            reason = str(status.get("reason", "") or "牌桌窗口恢复失败")
            self._geometry_terminal_error_visible = True
            self.error_status.setText(f"错误：{reason}")
        elif state in {"listening", "recovering", "recovered", "opening"}:
            if self._geometry_terminal_error_visible:
                self.error_status.clear()
                self._geometry_terminal_error_visible = False

    def _session_finished(self, _value: object) -> None:
        self._session_active = False
        self._set_initial_fields_enabled(True)
        self._set_live_controls(False)
        self.review_bar.hide()
        self.initialization_status.setText(
            "本局已封存；持续监听页面会在识别到下一局的 27 张手牌后自动开始。"
        )

    def shutdown(self) -> None:
        self.runtime.shutdown()
