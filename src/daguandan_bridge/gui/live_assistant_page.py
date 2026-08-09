from __future__ import annotations

import html
import re
from typing import Any

import cv2
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QFormLayout,
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
    ComboBox,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    StrongBodyLabel,
    TitleLabel,
    isDarkTheme,
    qconfig,
)

from ..danzero.state import RANKS
from ..live.display_text import (
    event_action_text,
    event_prefix,
    live_status_text,
    reasons_text,
    seat_text,
)
from ..live.models import LiveEvent
from ..live.orchestrator import LiveAdvice, LiveUpdate, ReviewRequest
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

    def __init__(self, runtime: Any | None = None, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime or LiveAssistantController()
        self.review_candidate_buttons: list[PushButton] = []
        self._last_event_id = ""
        self._last_advice_timeline_key: tuple[object, ...] | None = None
        self._session_active = False
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
        root.addWidget(TitleLabel("实时 DanZero 助手"))
        subtitle = BodyLabel(
            "半自动模式：程序只观察与建议，不会点击游戏；开始前请确认 27 张手牌、级牌和首发座位。"
        )
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        self.initial_card = CardWidget()
        initial_layout = QVBoxLayout(self.initial_card)
        initial_layout.setContentsMargins(18, 16, 18, 16)
        initial_layout.setSpacing(10)
        initial_layout.addWidget(StrongBodyLabel("1. 单图识别与开局确认"))
        form = QFormLayout()
        self.round_level_combo = ComboBox()
        for rank in RANKS:
            self.round_level_combo.addItem(rank, userData=rank)
        self.lead_player_combo = ComboBox()
        self.lead_player_combo.addItem("待自动识别", userData="")
        for seat in ("self", "right", "opposite", "left"):
            self.lead_player_combo.addItem(_SEAT_LABELS[seat], userData=seat)
        self.recognition_strategy_combo = ComboBox()
        for value, label in RECOGNITION_STRATEGY_OPTIONS:
            self.recognition_strategy_combo.addItem(label, userData=value)
        default_strategy = self.recognition_strategy_combo.findData("two_valid_streak")
        if default_strategy >= 0:
            self.recognition_strategy_combo.setCurrentIndex(default_strategy)
        self.recognition_strategy_combo.setToolTip(
            "实时与“状态机管线”复测使用同一策略；可用同一录像横向比较。"
        )
        # Internal normalized codes remain here for the state machine; the
        # user edits the visible card strip through the existing picker.
        self.hand_edit = LineEdit(self)
        self.hand_edit.hide()
        self.initial_hand_badges: list[CardBadge] = []
        self.initial_hand_scroll = ScrollArea()
        self.initial_hand_scroll.setObjectName("liveInitialHandCards")
        self.initial_hand_scroll.setWidgetResizable(False)
        self.initial_hand_scroll.setFixedHeight(62)
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
        form.addRow("首发候选（实时会再次验证）", self.lead_player_combo)
        form.addRow("动作识别策略", self.recognition_strategy_combo)
        form.addRow("初始手牌（点击牌面可修改）", self.initial_hand_scroll)
        initial_layout.addLayout(form)
        initial_actions = QHBoxLayout()
        self.recognize_initial_button = PushButton("识别当前画面")
        self.start_session_button = PrimaryPushButton("开始实时对局")
        initial_actions.addWidget(self.recognize_initial_button)
        initial_actions.addStretch(1)
        initial_actions.addWidget(self.start_session_button)
        initial_layout.addLayout(initial_actions)
        self.initialization_status = CaptionLabel()
        self.initialization_status.setWordWrap(True)
        initial_layout.addWidget(self.initialization_status)
        self.danzero_warmup_status = CaptionLabel("DanZero 模型准备中")
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
        state_layout.addWidget(StrongBodyLabel("状态、对局时间线与 DanZero 建议"))
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
            "确认动作、DanZero 建议、纠错和异常会依次显示在这里；可直接框选并复制。"
        )
        state_layout.addWidget(self.timeline, 1)
        control_row = QHBoxLayout()
        self.pause_button = PushButton("暂停")
        self.resume_button = PushButton("继续")
        self.finish_button = PrimaryPushButton("结束并封存")
        control_row.addWidget(self.pause_button)
        control_row.addWidget(self.resume_button)
        control_row.addWidget(self.finish_button)
        state_layout.addLayout(control_row)
        columns.addWidget(state_card, 2)
        root.addLayout(columns)

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

        correction_card = CardWidget()
        correction_layout = QHBoxLayout(correction_card)
        correction_layout.setContentsMargins(16, 12, 16, 12)
        correction_layout.addWidget(StrongBodyLabel("最近动作快速纠错"))
        self.correction_cards_edit = LineEdit()
        self.correction_cards_edit.setPlaceholderText("正确牌面，例如黑桃7、红桃7")
        self.correct_cards_button = PushButton("改为该牌组")
        self.correct_pass_button = PushButton("改为不出")
        correction_layout.addWidget(self.correction_cards_edit, 1)
        correction_layout.addWidget(self.correct_cards_button)
        correction_layout.addWidget(self.correct_pass_button)
        root.addWidget(correction_card)

        self.error_status = CaptionLabel()
        self.error_status.setWordWrap(True)
        root.addWidget(self.error_status)
        root.addStretch(1)

        self.recognize_initial_button.clicked.connect(self.runtime.recognize_initial)
        self.start_session_button.clicked.connect(self._start_session)
        self.pause_button.clicked.connect(self.runtime.pause)
        self.resume_button.clicked.connect(self.runtime.resume)
        self.finish_button.clicked.connect(self.runtime.finish)
        self.manual_confirm_button.clicked.connect(self._confirm_manual)
        self.correct_cards_button.clicked.connect(self._correct_cards)
        self.correct_pass_button.clicked.connect(
            lambda: self.runtime.correct_latest(cards=(), is_pass=True)
        )
        self.hand_edit.textChanged.connect(self._refresh_initialization)
        self.initial_hand_cards.clicked.connect(self._edit_initial_hand)
        self.round_level_combo.currentIndexChanged.connect(self._refresh_initialization)
        self.lead_player_combo.currentIndexChanged.connect(self._refresh_initialization)
        self._set_live_controls(False)

    def _connect_runtime(self) -> None:
        for name, handler in (
            ("initial_recognized", self.apply_initial_recognition),
            ("update_ready", self.apply_update),
            ("frame_ready", self.show_frame),
            ("error", self.show_error),
            ("session_finished", self._session_finished),
            ("danzero_warmup_status", self.show_danzero_warmup_status),
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
        ready = len(cards) == 27 and not self._session_active
        self.start_session_button.setEnabled(ready)
        if self._session_active:
            self.initialization_status.setText("实时对局已开始；初始字段已锁定。")
        elif len(cards) == 27:
            self.initialization_status.setText("已确认 27 张初始手牌，可以开始。")
        else:
            self.initialization_status.setText(
                f"必须准确确认 27 张初始手牌；当前为 {len(cards)} 张。"
            )

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
        if "super_double" not in buttons:
            self._set_combo_data(self.lead_player_combo, lead)
        self.hand_edit.setText(" ".join(getattr(result, "my_hand", ())))
        self._render_initial_hand_cards(self._parse_cards(self.hand_edit.text()))
        if snapshot is not None:
            self.show_frame(snapshot)
        diagnostics = tuple(getattr(result, "diagnostics", ()))
        self.error_status.setText("；".join(diagnostics[:4]))
        self._refresh_initialization()

    @staticmethod
    def _set_combo_data(combo: ComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _start_session(self) -> None:
        cards = self._parse_cards(self.hand_edit.text())
        if len(cards) != 27:
            self._refresh_initialization()
            return
        # This is deliberately UI-only.  SessionStore is never touched here,
        # so prior recordings, timelines, and truth logs remain available.
        self._reset_transient_session_ui()
        started = self.runtime.start_session(
            round_level=str(self.round_level_combo.currentData()),
            hand=cards,
            # The combo is only a single-image candidate.  Starting with None
            # forces the same opening state machine in live play and replay.
            lead_player=None,
            recognition_strategy=str(self.recognition_strategy_combo.currentData()),
        )
        if started is False:
            return
        self._session_active = True
        self.hand_edit.setEnabled(False)
        self.initial_hand_scroll.setEnabled(False)
        self.round_level_combo.setEnabled(False)
        self.lead_player_combo.setEnabled(False)
        self.recognition_strategy_combo.setEnabled(False)
        self._set_live_controls(True)
        self._refresh_initialization()

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
            badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.initial_hand_badges.append(badge)
            self.initial_hand_cards_layout.addWidget(badge)
        self.initial_hand_cards_layout.addStretch(1)
        self.initial_hand_cards.setMinimumWidth(max(240, len(cards) * 40 + 12))
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
        self._last_advice_timeline_key = None
        self.timeline.clear()
        self.review_bar.hide()
        self.lead_player_combo.setCurrentIndex(0)

    def _set_live_controls(self, active: bool) -> None:
        self.pause_button.setEnabled(active)
        self.resume_button.setEnabled(active)
        self.finish_button.setEnabled(active)
        self.correct_cards_button.setEnabled(active)
        self.correct_pass_button.setEnabled(active)

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
            self.turn_status.setText(
                f"当前行动：{seat_text(player, unknown='等待确认')}"
                f"　|　第 {update.snapshot.turn_id} 手"
            )
        if update.event is not None and update.event.event_id != self._last_event_id:
            self._last_event_id = update.event.event_id
            self._append_event_to_timeline(update.event)
        if update.review is not None:
            self.show_review(update.review)
        else:
            self.review_bar.hide()
        self._show_advice(update.advice)
        if update.status == "sealed":
            self._session_finished(update)

    def _show_advice(self, raw: object | None) -> None:
        if not isinstance(raw, LiveAdvice):
            return
        if raw.status == "requested":
            self._append_advice_timeline_entry(
                raw,
                f"DanZero 正在计算建议（请求 {raw.key.request_id}）。",
            )
        elif raw.status == "ready" and not raw.visible:
            self._append_advice_timeline_entry(
                raw,
                "DanZero 建议已算好，等待我方回合旁证。",
            )
        elif raw.status == "ready" and raw.advice is not None:
            self._append_advice_timeline_entry(
                raw,
                f"{self._play_type_text(raw.advice.play_type)}；耗时 {raw.advice.elapsed_ms:.0f} ms；"
                f"请求 {raw.key.request_id}",
            )
        elif raw.status == "stale":
            self._append_advice_timeline_entry(raw, "DanZero 旧建议已丢弃。")
        elif raw.status == "failed":
            self._append_advice_timeline_entry(
                raw,
                f"DanZero 建议计算失败：{raw.error}",
            )

    def _append_event_to_timeline(self, event: LiveEvent) -> None:
        if event.event_type in {
            "advice_requested",
            "advice_ready",
            "advice_visible",
            "advice_failed",
        }:
            return
        prefix = html.escape(event_prefix(event))
        cards = tuple(str(card) for card in event.payload.get("cards", ()))
        if event.event_type == "player_played":
            action = html.escape(f"{seat_text(event.actor)}出牌：")
            cards_html = self._cards_html(cards)
        else:
            action = html.escape(event_action_text(event))
            cards_html = ""
        self._append_timeline_html(
            "<div style='margin:3px 0 7px 0;'>"
            f"<span style='color:#6b7280;'>{prefix}</span><br>"
            f"<span>{action}</span>{cards_html}"
            "</div>"
        )

    def _append_advice_timeline_entry(self, raw: LiveAdvice, detail: str) -> None:
        advice = raw.advice
        cards = tuple(advice.cards) if advice is not None and raw.visible else ()
        key = (
            raw.key.request_id,
            raw.status,
            raw.visible,
            cards,
            raw.error,
        )
        if key == self._last_advice_timeline_key:
            return
        self._last_advice_timeline_key = key
        cards_html = self._cards_html(cards) if cards else ""
        self._append_timeline_html(
            "<div style='margin:5px 0 9px 0; padding:6px; "
            "background:#e7f6f2; border-left:4px solid #0F766E;'>"
            "<span style='color:#0F766E; font-weight:600;'>DanZero 建议</span><br>"
            f"<span style='color:#0F766E;'>{html.escape(detail)}</span>{cards_html}"
            "</div>"
        )

    @staticmethod
    def _cards_html(cards: tuple[str, ...]) -> str:
        if not cards:
            return ""
        cells: list[str] = []
        for card in cards:
            rank, suit, color = CardBadge._display_parts(card)
            cells.append(
                "<td title='{title}' style='background:#ffffff; border:1px solid #c8cdd3; "
                "border-radius:4px; min-width:26px; text-align:center; padding:1px 3px;'>"
                "<span style='color:{color}; font-weight:700;'>{suit}</span><br>"
                "<span style='color:{color}; font-weight:700;'>{rank}</span></td>".format(
                    title=html.escape(card_code_to_text(card)),
                    color=color,
                    suit=html.escape(suit),
                    rank=html.escape(rank),
                )
            )
        return "<br><table cellspacing='2' cellpadding='0'><tr>{}</tr></table>".format(
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
            f"原因：{reasons_text(review.reason)}。录像继续，但状态和 DanZero 已暂停推进。"
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

    def _correct_cards(self) -> None:
        cards = self._parse_cards(self.correction_cards_edit.text())
        if cards:
            self.runtime.correct_latest(cards=cards, is_pass=False)

    def show_error(self, message: str) -> None:
        self.error_status.setText(f"错误：{message}")

    def show_danzero_warmup_status(self, message: str) -> None:
        self.danzero_warmup_status.setText(str(message))

    def _session_finished(self, _value: object) -> None:
        self._session_active = False
        self.hand_edit.setEnabled(True)
        self.initial_hand_scroll.setEnabled(True)
        self.round_level_combo.setEnabled(True)
        self.lead_player_combo.setEnabled(True)
        self.recognition_strategy_combo.setEnabled(True)
        self._set_live_controls(False)
        self.review_bar.hide()
        self._refresh_initialization()

    def shutdown(self) -> None:
        self.runtime.shutdown()
