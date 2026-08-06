from __future__ import annotations

import re
from typing import Any

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
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
    TextEdit,
    TitleLabel,
    isDarkTheme,
    qconfig,
)

from ..danzero.state import RANKS
from ..live.models import LiveEvent
from ..live.orchestrator import LiveAdvice, LiveUpdate, ReviewRequest
from .live_controller import LiveAssistantController


_SEAT_LABELS = {
    "self": "自己",
    "right": "右家",
    "opposite": "对家",
    "left": "左家",
}


class LiveAssistantPage(ScrollArea):
    """Semi-automatic live assistant; all game decisions stay in the orchestrator."""

    def __init__(self, runtime: Any | None = None, parent=None) -> None:
        super().__init__(parent)
        self.runtime = runtime or LiveAssistantController()
        self.review_candidate_buttons: list[PushButton] = []
        self._last_event_id = ""
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
        for seat in ("self", "right", "opposite", "left"):
            self.lead_player_combo.addItem(_SEAT_LABELS[seat], userData=seat)
        self.hand_edit = LineEdit()
        self.hand_edit.setPlaceholderText("识别结果会填入这里；可在开始前手动修正")
        form.addRow("当前级牌", self.round_level_combo)
        form.addRow("首发座位", self.lead_player_combo)
        form.addRow("初始手牌", self.hand_edit)
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
        state_layout.addWidget(StrongBodyLabel("状态与 DanZero 建议"))
        self.live_status = BodyLabel("状态：等待开局")
        self.turn_status = BodyLabel("当前回合：—")
        self.advice_status = StrongBodyLabel("建议：—")
        self.advice_detail = CaptionLabel("建议会提前计算，并在我方回合旁证命中后显示。")
        self.advice_detail.setWordWrap(True)
        state_layout.addWidget(self.live_status)
        state_layout.addWidget(self.turn_status)
        state_layout.addSpacing(8)
        state_layout.addWidget(self.advice_status)
        state_layout.addWidget(self.advice_detail)
        state_layout.addStretch(1)
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
        self.manual_cards_edit.setPlaceholderText("都不对时填写正确牌组，例如 7S 7H")
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
        self.correction_cards_edit.setPlaceholderText("正确牌组")
        self.correct_cards_button = PushButton("改为该牌组")
        self.correct_pass_button = PushButton("改为不出")
        correction_layout.addWidget(self.correction_cards_edit, 1)
        correction_layout.addWidget(self.correct_cards_button)
        correction_layout.addWidget(self.correct_pass_button)
        root.addWidget(correction_card)

        timeline_card = CardWidget()
        timeline_layout = QVBoxLayout(timeline_card)
        timeline_layout.setContentsMargins(16, 14, 16, 16)
        timeline_layout.addWidget(StrongBodyLabel("对局时间线（按局隔离并同步落盘）"))
        self.timeline = TextEdit()
        self.timeline.setReadOnly(True)
        self.timeline.setMinimumHeight(180)
        self.timeline.setPlaceholderText("确认动作、建议、纠错和异常会依次显示在这里。")
        timeline_layout.addWidget(self.timeline)
        root.addWidget(timeline_card)

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
        ):
            signal = getattr(self.runtime, name, None)
            if signal is not None:
                signal.connect(handler)

    @staticmethod
    def _parse_cards(raw: str) -> tuple[str, ...]:
        return tuple(value for value in re.split(r"[\s,，]+", raw.strip()) if value)

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
        self._set_combo_data(self.lead_player_combo, lead)
        self.hand_edit.setText(" ".join(getattr(result, "my_hand", ())))
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
        started = self.runtime.start_session(
            round_level=str(self.round_level_combo.currentData()),
            hand=cards,
            lead_player=str(self.lead_player_combo.currentData()),
        )
        if started is False:
            return
        self._session_active = True
        self.hand_edit.setEnabled(False)
        self.round_level_combo.setEnabled(False)
        self.lead_player_combo.setEnabled(False)
        self._set_live_controls(True)
        self._refresh_initialization()

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
        self.live_status.setText(f"状态：{update.status}")
        player = update.snapshot.current_player
        self.turn_status.setText(
            f"当前回合：{_SEAT_LABELS.get(player or '', player or '—')}"
            f"　|　墩 {update.snapshot.trick_id}　|　状态版本 {update.snapshot.revision}"
        )
        if update.event is not None and update.event.event_id != self._last_event_id:
            self._last_event_id = update.event.event_id
            self.timeline.append(self._event_text(update.event))
        if update.review is not None:
            self.show_review(update.review)
        self._show_advice(update.advice)
        if update.status == "sealed":
            self._session_finished(update)

    def _show_advice(self, raw: object | None) -> None:
        if not isinstance(raw, LiveAdvice):
            return
        if raw.status == "requested":
            self.advice_status.setText("建议：DanZero 计算中…")
            self.advice_detail.setText(f"请求 {raw.key.request_id} 已提前发起。")
        elif raw.status == "ready" and not raw.visible:
            self.advice_status.setText("建议：已算好，等待我方回合旁证")
            self.advice_detail.setText(f"请求 {raw.key.request_id}；暂不显示牌组，避免误导。")
        elif raw.status == "ready" and raw.advice is not None:
            cards = " ".join(raw.advice.cards) or "不出"
            self.advice_status.setText(f"DanZero 建议：{cards}")
            self.advice_detail.setText(
                f"{raw.advice.play_type}；耗时 {raw.advice.elapsed_ms:.0f} ms；请求 {raw.key.request_id}"
            )
        elif raw.status == "stale":
            self.advice_status.setText("建议：旧结果已丢弃")
        elif raw.status == "failed":
            self.advice_status.setText("建议：计算失败")
            self.advice_detail.setText(raw.error)

    @staticmethod
    def _event_text(event: LiveEvent) -> str:
        seat = _SEAT_LABELS.get(event.actor or "", event.actor or "系统")
        cards = " ".join(str(card) for card in event.payload.get("cards", ()))
        if event.event_type == "player_played":
            action = f"{seat}出牌 {cards}"
        elif event.event_type == "player_passed":
            action = f"{seat}不出"
        elif event.event_type == "event_correction":
            action = f"纠错：{event.payload}"
        else:
            action = f"{event.event_type}：{event.payload}"
        return f"[墩{event.trick_id}/回合{event.turn_id}] {action}"

    def show_review(self, review: ReviewRequest) -> None:
        while self.review_buttons_layout.count():
            item = self.review_buttons_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.review_candidate_buttons.clear()
        self.review_title.setText(f"请确认{_SEAT_LABELS[review.player]}本次动作")
        self.review_reason.setText(
            f"原因：{review.reason}。录像继续，但状态和 DanZero 已暂停推进。"
        )
        for candidate in review.candidates:
            text = "不出" if candidate.is_pass else " ".join(candidate.cards)
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

    def _session_finished(self, _value: object) -> None:
        self._session_active = False
        self.hand_edit.setEnabled(True)
        self.round_level_combo.setEnabled(True)
        self.lead_player_combo.setEnabled(True)
        self._set_live_controls(False)
        self.review_bar.hide()
        self._refresh_initialization()

    def shutdown(self) -> None:
        self.runtime.shutdown()
