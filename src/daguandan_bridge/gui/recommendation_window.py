from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, QRect, Qt, Signal
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
from ..live.orchestrator import LiveAdvice, LiveUpdate
from .single_image_danzero_page import CardBadge


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
        self._backend = ""
        self._card_badges: list[CardBadge] = []
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

        self.detail_label = CaptionLabel("完整助手在后台运行")
        self.detail_label.setWordWrap(True)
        card_layout.addWidget(self.detail_label)
        root.addWidget(self.card, 1)

        actions = QHBoxLayout()
        self.capture_label = CaptionLabel("采集方式：等待连接")
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
        advisor_name = self._advisor_display_name()
        self.setWindowTitle(f"{advisor_name} 极简推荐")
        player = getattr(update.snapshot, "current_player", None)
        self.trick_strip.set_snapshot(update.snapshot, update.status)
        if update.status == "paused":
            self.suggestion_label.setText("识别已暂停")
            self.detail_label.setText("请排除窗口遮挡后，在完整助手中点击继续")
            self._render_cards(())
            return
        if update.status == "running" and player != "self":
            self.suggestion_label.setText("等待自己回合")
            self.detail_label.setText(f"当前轮到{seat_text(player, unknown='其他玩家')}")
            self._render_cards(())
            return
        raw = update.advice
        if not isinstance(raw, LiveAdvice):
            return
        if raw.status == "failed":
            self.suggestion_label.setText("建议计算失败")
            self.detail_label.setText(raw.error or "请打开完整助手查看原因")
            self._render_cards(())
            return
        if raw.status == "requested":
            self.suggestion_label.setText("正在计算建议")
            self.detail_label.setText(f"{advisor_name} 正在使用最新手牌")
            self._render_cards(())
            return
        if raw.status != "ready" or raw.advice is None:
            return
        if raw.key.request_id == self._last_request_id:
            return
        self._last_request_id = raw.key.request_id
        advice = raw.advice
        if advice.is_pass:
            self.suggestion_label.setText("不出")
        else:
            play_type = _PLAY_TYPE_LABELS.get(advice.play_type, "出牌")
            self.suggestion_label.setText(f"出牌 · {play_type}")
        self._render_cards(tuple(advice.cards))
        details = [f"耗时 {advice.elapsed_ms:.0f} ms"]
        engine_input = advice.engine_input
        decision = (
            engine_input.get("decision")
            if isinstance(engine_input, dict) and engine_input.get("debug") is True
            else None
        )
        if isinstance(decision, dict):
            best_q = decision.get("best_q")
            q_gap = decision.get("q_gap")
            if isinstance(best_q, (int, float)):
                details.append(f"Q值 {float(best_q):.4f}")
            if isinstance(q_gap, (int, float)):
                details.append(f"Q-gap {float(q_gap):.4f}")
        if raw.suit_uncertain:
            agreement = "各花色分支建议一致" if raw.advice_agrees_across_variants else "花色分支建议有差异"
            details.append(agreement)
        self.detail_label.setText(" · ".join(details))

    def _advisor_display_name(self) -> str:
        strategy = str(getattr(self.runtime, "advisor_strategy", "") or "")
        return dict(ADVISOR_OPTIONS).get(strategy, "建议模型")

    def apply_frame(self, snapshot: object) -> None:
        frame = getattr(snapshot, "frame", None)
        backend = str(getattr(frame, "backend", "") or "")
        if not backend or backend == self._backend:
            return
        self._backend = backend
        if backend == "printwindow":
            self.capture_label.setText("后台窗口采集 · 前台遮挡不入帧")
        elif backend in {"screen", "gdi_screen"}:
            self.capture_label.setText("屏幕采集 · 遮挡保护已启用")
        else:
            self.capture_label.setText(f"采集方式：{backend}")

    def show_error(self, message: str) -> None:
        text = str(message)
        if "遮挡" in text or "屏幕采集已暂停" in text:
            self.suggestion_label.setText("窗口遮挡，已暂停")
            self.detail_label.setText("移动浮窗或缩小游戏窗口，确保不覆盖牌桌后再继续")
            self._render_cards(())

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
