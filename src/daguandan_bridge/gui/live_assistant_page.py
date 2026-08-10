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
        self._shown_event_ids: set[str] = set()
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
            "持续监听页面：程序只观察与建议，不会点击游戏；稳定识别两次相同的 27 张手牌后自动开始。"
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
        self.initial_hand_scroll.setFixedHeight(50)
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
        self.recognize_initial_button = PushButton("识别当前页面（持续监听）")
        initial_actions.addWidget(self.recognize_initial_button)
        initial_actions.addStretch(1)
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
            "动作、DanZero 建议与异常会依次显示在这里；可直接框选并复制。"
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

        self.error_status = CaptionLabel()
        self.error_status.setWordWrap(True)
        root.addWidget(self.error_status)
        root.addStretch(1)

        self.recognize_initial_button.clicked.connect(self._start_listening)
        self.pause_button.clicked.connect(self.runtime.pause)
        self.resume_button.clicked.connect(self.runtime.resume)
        self.finish_button.clicked.connect(self.runtime.finish)
        self.manual_confirm_button.clicked.connect(self._confirm_manual)
        self.hand_edit.textChanged.connect(self._refresh_initialization)
        self.initial_hand_cards.clicked.connect(self._edit_initial_hand)
        self.round_level_combo.currentIndexChanged.connect(self._refresh_initialization)
        self.lead_player_combo.currentIndexChanged.connect(self._refresh_initialization)
        self.recognition_strategy_combo.currentIndexChanged.connect(
            self._update_recognition_strategy
        )
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

    def _start_listening(self) -> None:
        setter = getattr(self.runtime, "set_recognition_strategy", None)
        if callable(setter):
            setter(str(self.recognition_strategy_combo.currentData()))
        start = getattr(self.runtime, "start_listening", None)
        if callable(start):
            start()
            self.initialization_status.setText(
                "正在持续监听页面；稳定识别两次相同的 27 张手牌后自动开始。"
            )
            return
        # Compatibility only for an older embedded controller.  The shipped
        # controller always exposes persistent listening.
        self.runtime.recognize_initial()

    def _update_recognition_strategy(self, *_args) -> None:
        setter = getattr(self.runtime, "set_recognition_strategy", None)
        if callable(setter):
            setter(str(self.recognition_strategy_combo.currentData()))

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
        self._refresh_initialization()

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
            badge.setFixedSize(27, 40)
            badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.initial_hand_badges.append(badge)
            self.initial_hand_cards_layout.addWidget(badge)
        self.initial_hand_cards_layout.addStretch(1)
        self.initial_hand_cards.setMinimumWidth(max(240, len(cards) * 30 + 12))
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
        elif raw.status == "ready" and raw.advice is not None and not raw.visible:
            suggestion = (
                "建议：不出"
                if raw.advice.is_pass
                else f"建议：{self._play_type_text(raw.advice.play_type)}"
            )
            self._append_advice_timeline_entry(
                raw,
                f"{suggestion}；耗时 {raw.advice.elapsed_ms:.0f} ms；"
                f"请求 {raw.key.request_id}；待画面确认。",
            )
        elif raw.status == "ready" and raw.advice is not None:
            suggestion = (
                "建议：不出"
                if raw.advice.is_pass
                else f"建议：{self._play_type_text(raw.advice.play_type)}"
            )
            self._append_advice_timeline_entry(
                raw,
                f"{suggestion}；耗时 {raw.advice.elapsed_ms:.0f} ms；"
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
            raw.visible,
            cards,
            raw.error,
            raw.suit_uncertain,
            raw.variant_count,
            raw.advice_agrees_across_variants,
        )
        if key == self._last_advice_timeline_key:
            return
        self._last_advice_timeline_key = key
        cards_html = self._cards_html(cards) if cards else ""
        confirmed = raw.status == "ready" and raw.visible
        accent = "#0F766E" if confirmed else "#B45309"
        background = "#e7f6f2" if confirmed else "#fff7ed"
        title = "DanZero 建议" if confirmed else "DanZero 待确认"
        if raw.suit_uncertain:
            agreement = "建议一致" if raw.advice_agrees_across_variants else "建议存在分歧"
            detail += f"；花色遮挡：已评估 {raw.variant_count} 个可行分支，{agreement}"
        self._append_timeline_html(
            "<div style='margin:5px 0 9px 0; padding:6px; "
            f"background:{background}; border-left:4px solid {accent};'>"
            f"<span style='color:{accent}; font-weight:600;'>{title}</span><br>"
            f"<span style='color:{accent};'>{html.escape(detail)}</span>{cards_html}"
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
                "border-radius:4px; min-width:26px; text-align:center; padding:1px 3px;'>"
                "<span style='color:{color}; font-weight:700;'>{suit}</span><br>"
                "<span style='color:{color}; font-weight:700;'>{rank}</span></td>".format(
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

    def show_error(self, message: str) -> None:
        self.error_status.setText(f"错误：{message}")

    def show_danzero_warmup_status(self, message: str) -> None:
        self.danzero_warmup_status.setText(str(message))

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
