from __future__ import annotations

from typing import Any
from time import monotonic_ns

from PySide6.QtCore import QPoint, QRect, Qt, Signal, QTimer
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    CardWidget,
    PushButton,
    StrongBodyLabel,
    TitleLabel,
    isDarkTheme,
    qconfig,
)

from ..advisor_strategy import ADVISOR_OPTIONS
from ..live.display_text import compact_cards_text, live_status_text, seat_text
from ..domain.live_runtime import LiveAdvice, LiveUpdate
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


class RecommendationFloatWindow(QWidget):
    """A read-only companion view over the shared live controller."""

    open_full_assistant_requested = Signal()

    def __init__(self, runtime: Any, parent=None) -> None:
        super().__init__(parent)
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
        self._fault_identity: tuple[str, int] | None = None
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
        actions.addWidget(self.capture_label, 1)
        self.open_button = PushButton("打开完整助手")
        self.open_button.clicked.connect(self.open_full_assistant_requested.emit)
        actions.addWidget(self.open_button)
        root.addLayout(actions)

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
        state = str(status.get("state", "") or "")
        if state == "listening":
            self._update_gate.begin_listening()
            self._latest_update = None
            self._render_view(CompactViewState("listening", "等待开局"))
        elif state == "opening":
            phase = str(status.get("phase", "") or "")
            title = "等待开局" if phase in {"unknown", "lobby", "settlement", "waiting_table"} else "确认开局中…"
            self._render_view(CompactViewState("opening", title))
        elif state == "recovering":
            self._render_view(CompactViewState("recovering", "重新连接中…"))
        elif state == "recovered":
            self._render_view(CompactViewState("listening", "等待开局"))
        elif state == "failed":
            self._render_view(CompactViewState("failed", "监听已停止", "请打开完整助手重新连接牌桌"))

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
        """Place outside the game client when the current screen has room."""

        center = QPoint(target.center().x(), target.center().y())
        screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
        if screen is None:
            return False
        available = screen.availableGeometry()
        width = max(self.minimumWidth(), min(self.width(), available.width()))
        height = max(self.minimumHeight(), min(self.height(), available.height()))
        candidates = (
            QRect(target.right() + 8, target.top(), width, height),
            QRect(target.left() - width - 8, target.top(), width, height),
            QRect(target.left(), target.bottom() + 8, width, height),
            QRect(target.left(), target.top() - height - 8, width, height),
        )
        for candidate in candidates:
            if available.contains(candidate) and not candidate.intersects(target):
                self.setGeometry(candidate)
                return True
        fallback = QRect(
            available.right() - width + 1,
            available.top(),
            width,
            height,
        )
        self.setGeometry(fallback)
        return not fallback.intersects(target)

    def closeEvent(self, event: QCloseEvent) -> None:
        event.ignore()
        self.hide()
        self.open_full_assistant_requested.emit()
