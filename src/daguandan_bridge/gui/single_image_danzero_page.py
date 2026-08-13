from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase, QGuiApplication, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget,
    CaptionLabel,
    LineEdit,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
    TableWidget,
)

from ..advisor_strategy import ADVISOR_OPTIONS, normalize_advisor_strategy
from ..danzero.state import GameStateError, GuanDanState, RANKS, SEATS
from ..image_io import read_image_unicode
from ..live.truth_log import card_code_to_text
from ..recognition_service import BUTTON_LABELS, RecognitionAnnotation, RecognitionResult


SEAT_LABELS = {
    "self": "我方",
    "left": "左家",
    "opposite": "对家",
    "right": "右家",
}
ACTION_LABELS = {"play": "出牌", "pass": "不出"}
RECOGNITION_FIELD_LABELS = {
    "round_level": "当前级牌",
    "wild_rank": "百搭牌级别",
    "current_player": "当前行动者",
    "lead_player": "本轮首出者",
    "my_hand": "我方手牌",
    "events": "出牌事件",
}
ANNOTATION_CATEGORY_LABELS = {
    "hand": "手牌",
    "play": "出牌",
    "button": "按钮",
    "level": "级牌",
    "timer": "计时器",
    "status": "状态",
}
ANNOTATION_LABELS = {
    "active": "行动中",
    "first_play": "首出标记",
    "passed": "不出",
}
SUIT_DISPLAY = {
    "S": ("♠", "#20252b"),
    "H": ("♥", "#d93025"),
    "C": ("♣", "#20252b"),
    "D": ("♦", "#d93025"),
}


class ScrollSafeComboBox(QComboBox):
    """Keep the page scroll gesture from changing recognition parameters."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class CopyableLabel(QLabel):
    """Copy the complete label text on double-click for quick handoff."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setToolTip("双击复制完整内容")

    def mouseDoubleClickEvent(self, event) -> None:
        QGuiApplication.clipboard().setText(self.text())
        if event is not None:
            event.accept()


class CopyableLineEdit(LineEdit):
    """Select and copy the complete value instead of only one word."""

    def mouseDoubleClickEvent(self, event) -> None:
        self.selectAll()
        self.copy()
        if event is not None:
            event.accept()


class CopyablePlainTextEdit(PlainTextEdit):
    """Select and copy all diagnostic/result text on double-click."""

    def mouseDoubleClickEvent(self, event) -> None:
        self.selectAll()
        self.copy()
        if event is not None:
            event.accept()


class CardBadge(CardWidget):
    """A compact, readable card with a colored suit symbol and rank."""

    def __init__(self, card_code: str, parent=None, *, compact: bool = False):
        super().__init__(parent)
        self.card_code = str(card_code)
        self.setObjectName("cardBadge")
        self.setToolTip(card_code_to_text(self.card_code))
        width, height = (36, 50) if compact else (52, 72)
        suit_size, rank_size = (16, 13) if compact else (22, 18)
        margin = 3 if compact else 7
        self.setMinimumSize(width, height)
        if compact:
            self.setMaximumSize(width, height)
        self.setStyleSheet(
            "CardWidget#cardBadge { background: #ffffff; border: 1px solid #c8cdd3; "
            "border-radius: 6px; }"
        )

        rank, suit, color = self._display_parts(self.card_code)
        layout = QVBoxLayout(self)
        vertical_margin = 3 if compact else 4
        layout.setContentsMargins(margin, vertical_margin, margin, vertical_margin)
        layout.setSpacing(0)
        self.suit_label = QLabel(suit)
        self.suit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.suit_label.setStyleSheet(
            f"color: {color}; font-size: {suit_size}px; font-weight: 700;"
        )
        self.rank_label = QLabel(rank)
        self.rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rank_label.setStyleSheet(
            f"color: {color}; font-size: {rank_size}px; font-weight: 700;"
        )
        layout.addWidget(self.suit_label)
        layout.addWidget(self.rank_label)

    @staticmethod
    def _display_parts(card_code: str) -> tuple[str, str, str]:
        if card_code == "small_joker":
            return "小王", "★", "#c47f00"
        if card_code == "big_joker":
            return "大王", "★", "#c47f00"
        if card_code.endswith("?"):
            return card_code[:-1], "？", "#b42318"
        rank, suit_code = card_code[:-1], card_code[-1:]
        suit, color = SUIT_DISPLAY.get(suit_code, (suit_code, "#20252b"))
        return rank, suit, color


class SingleImageDanzeroPage(QDialog):
    """Review one screenshot, correct recognition, and test DanZero."""

    test_requested = Signal(object)
    state_built = Signal(object)
    recognize_requested = Signal()
    advisor_strategy_changed = Signal(str)

    def __init__(
        self,
        image_path: Path | None = None,
        parent=None,
        *,
        advisor_strategy: str = "danzero",
    ):
        super().__init__(parent)
        self.advisor_strategy = normalize_advisor_strategy(advisor_strategy)
        self.image_path = Path(image_path) if image_path is not None else None
        self._events: list[tuple[str, str, tuple[str, ...]]] = []
        self._source_image: np.ndarray | None = None
        self._preview_image = QImage()
        self.preview_annotations: tuple[RecognitionAnnotation, ...] = ()
        self.recognition_elapsed_ms: float | None = None
        self.hand_card_widgets: list[CardBadge] = []
        self.setObjectName("singleImageDanzeroPage")
        self.setWindowTitle("单图标注 / 测试 DanZero")
        self.resize(980, 900)
        self.setMinimumSize(820, 720)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 18)
        self.content_scroll = ScrollArea()
        self.content_scroll.setObjectName("singleImageContentScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        content = QWidget()
        self.content_scroll.setWidget(content)
        outer.addWidget(self.content_scroll)
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(14)

        header_card = CardWidget(content)
        header = QHBoxLayout(header_card)
        header.setContentsMargins(20, 16, 20, 16)
        header.setSpacing(12)
        heading = QVBoxLayout()
        heading.setSpacing(3)
        heading.addWidget(SubtitleLabel("单图标注与 DanZero"))
        self.image_info_label = CopyableLabel()
        self.image_info_label.setWordWrap(True)
        heading.addWidget(self.image_info_label)
        header.addLayout(heading, 1)
        self.recognize_button = PushButton("重新识别")
        header.addWidget(self.recognize_button)
        root.addWidget(header_card)

        image_card = CardWidget(content)
        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(18, 16, 18, 16)
        image_layout.setSpacing(8)
        image_layout.addWidget(StrongBodyLabel("图片预览与识别标注"))
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(600, 360)
        self.image_preview.setMaximumHeight(480)
        self.image_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.image_preview.setStyleSheet(
            "background: rgba(0, 0, 0, 0.06); border: 1px dashed rgba(120, 120, 120, 0.45); border-radius: 10px;"
        )
        self.image_preview.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.image_preview.customContextMenuRequested.connect(
            self._show_image_context_menu
        )
        image_layout.addWidget(self.image_preview)
        self.annotation_legend = CopyableLabel("识别框会在图片上显示牌面和识别类别")
        self.annotation_legend.setWordWrap(True)
        image_layout.addWidget(self.annotation_legend)
        self.button_detection_label = CopyableLabel("按钮检测：尚未识别")
        self.button_detection_label.setWordWrap(True)
        image_layout.addWidget(self.button_detection_label)
        self.recognition_status = CopyableLabel("模板识别尚未运行")
        self.recognition_status.setWordWrap(True)
        image_layout.addWidget(self.recognition_status)
        root.addWidget(image_card)

        self.form_splitter = QSplitter(Qt.Orientation.Horizontal, content)
        self.form_splitter.setChildrenCollapsible(False)
        hand_card = CardWidget(self.form_splitter)
        hand_layout = QVBoxLayout(hand_card)
        hand_layout.setContentsMargins(18, 16, 18, 16)
        hand_layout.setSpacing(8)
        hand_layout.addWidget(StrongBodyLabel("我方手牌"))
        hand_layout.addWidget(CaptionLabel("上方是识别结果预览；DanZero 实际使用下方的标准牌面代码。"))
        self.hand_card_scroll = ScrollArea()
        self.hand_card_scroll.setWidgetResizable(True)
        self.hand_card_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.hand_card_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        hand_content = QWidget()
        self.hand_cards_layout = QHBoxLayout(hand_content)
        self.hand_cards_layout.setContentsMargins(4, 4, 4, 4)
        self.hand_cards_layout.setSpacing(6)
        self.hand_card_scroll.setWidget(hand_content)
        self.hand_card_scroll.setMinimumHeight(96)
        self.hand_card_scroll.setMaximumHeight(116)
        hand_layout.addWidget(self.hand_card_scroll)
        code_header = QHBoxLayout()
        code_header.addWidget(StrongBodyLabel("DanZero 手牌代码"))
        code_header.addStretch(1)
        self.copy_hand_code_button = PushButton("复制代码")
        self.copy_hand_code_button.setToolTip("复制当前 DanZero 手牌参数")
        code_header.addWidget(self.copy_hand_code_button)
        hand_layout.addLayout(code_header)
        self.my_hand_edit = CopyableLineEdit()
        self.my_hand_edit.setObjectName("handCodeEdit")
        self.my_hand_edit.setPlaceholderText("例如：3S 4H 5D small_joker（红桃 5 为 5H）")
        self.my_hand_edit.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        hand_layout.addWidget(self.my_hand_edit)
        self.hand_hint = CaptionLabel("代码规则：S=黑桃、H=红桃、C=梅花、D=方块；使用空格分隔。")
        self.hand_hint.setWordWrap(True)
        hand_layout.addWidget(self.hand_hint)

        parameter_card = CardWidget(self.form_splitter)
        parameter_layout = QVBoxLayout(parameter_card)
        parameter_layout.setContentsMargins(18, 16, 18, 16)
        parameter_layout.setSpacing(8)
        parameter_layout.addWidget(StrongBodyLabel("DanZero 参数"))
        parameter_layout.addWidget(CaptionLabel("模板识别会自动填入；请在运行前确认。"))
        context_form = QFormLayout()
        self.round_level_combo = self._rank_combo()
        self.wild_rank_combo = self._rank_combo()
        self.current_player_combo = self._seat_combo()
        self.lead_player_combo = self._seat_combo()
        self.advisor_strategy_combo = ScrollSafeComboBox()
        for value, label in ADVISOR_OPTIONS:
            self.advisor_strategy_combo.addItem(label, value)
        advisor_index = self.advisor_strategy_combo.findData(self.advisor_strategy)
        if advisor_index >= 0:
            self.advisor_strategy_combo.setCurrentIndex(advisor_index)
        context_form.addRow("当前级牌", self.round_level_combo)
        context_form.addRow("百搭牌级别", self.wild_rank_combo)
        context_form.addRow("当前行动者", self.current_player_combo)
        context_form.addRow("本轮首出者", self.lead_player_combo)
        context_form.addRow("建议模型", self.advisor_strategy_combo)
        parameter_layout.addLayout(context_form)
        self.incomplete_status = CopyableLabel()
        self.incomplete_status.setWordWrap(True)
        parameter_layout.addWidget(self.incomplete_status)
        parameter_layout.addStretch(1)
        self.form_splitter.addWidget(hand_card)
        self.form_splitter.addWidget(parameter_card)
        self.form_splitter.setStretchFactor(0, 1)
        self.form_splitter.setStretchFactor(1, 1)
        self.form_splitter.setSizes((470, 470))
        root.addWidget(self.form_splitter)

        event_card = CardWidget(content)
        event_layout = QVBoxLayout(event_card)
        event_layout.setContentsMargins(18, 16, 18, 16)
        event_layout.setSpacing(8)
        event_layout.addWidget(StrongBodyLabel("本轮出牌事件"))
        event_layout.addWidget(CaptionLabel(
            "可补充或删除事件；FableDan 必须从第一手开始填写完整标准出牌/不出历史。"
        ))
        event_form = QGridLayout()
        self.event_player_combo = self._seat_combo()
        self.event_action_combo = ScrollSafeComboBox()
        for value, label in ACTION_LABELS.items():
            self.event_action_combo.addItem(label, value)
        self.event_cards_edit = LineEdit()
        self.event_cards_edit.setPlaceholderText("出牌时填写，例如：5H 6H；不出时留空")
        self.add_event_button = PrimaryPushButton("添加事件")
        self.remove_event_button = PushButton("删除选中")
        event_form.addWidget(CaptionLabel("玩家"), 0, 0)
        event_form.addWidget(self.event_player_combo, 1, 0)
        event_form.addWidget(CaptionLabel("动作"), 0, 1)
        event_form.addWidget(self.event_action_combo, 1, 1)
        event_form.addWidget(CaptionLabel("牌面代码"), 0, 2)
        event_form.addWidget(self.event_cards_edit, 1, 2)
        event_form.addWidget(self.add_event_button, 1, 3)
        event_form.addWidget(self.remove_event_button, 1, 4)
        event_layout.addLayout(event_form)

        self.event_table = TableWidget()
        self.event_table.setColumnCount(3)
        self.event_table.setRowCount(0)
        self.event_table.setHorizontalHeaderLabels(("玩家", "动作", "牌面"))
        self.event_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.event_table.setMinimumHeight(110)
        self.event_table.horizontalHeader().setStretchLastSection(True)
        event_layout.addWidget(self.event_table)
        root.addWidget(event_card)

        diagnostic_card = CardWidget(content)
        diagnostic_layout = QVBoxLayout(diagnostic_card)
        diagnostic_layout.setContentsMargins(18, 16, 18, 16)
        diagnostic_layout.setSpacing(8)
        diagnostic_layout.addWidget(StrongBodyLabel("识别与参数状态"))
        self.status = CopyableLabel("请确认牌局状态")
        self.status.setWordWrap(True)
        diagnostic_layout.addWidget(self.status)
        self.state_summary = CopyablePlainTextEdit()
        self.state_summary.setReadOnly(True)
        self.state_summary.setPlaceholderText("识别诊断和构建后的状态摘要会显示在这里")
        self.state_summary.setMinimumHeight(70)
        self.state_summary.setMaximumHeight(110)
        diagnostic_layout.addWidget(self.state_summary)
        actions = QHBoxLayout()
        self.build_button = PushButton("构建参数")
        self.test_button = PrimaryPushButton("测试 DanZero")
        self.close_button = PushButton("关闭")
        actions.addWidget(self.build_button)
        actions.addWidget(self.test_button)
        actions.addStretch(1)
        actions.addWidget(self.close_button)
        diagnostic_layout.addLayout(actions)
        root.addWidget(diagnostic_card)

        result_card = CardWidget(content)
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(18, 16, 18, 16)
        result_layout.setSpacing(8)
        result_layout.addWidget(StrongBodyLabel("DanZero 返回值"))
        self.result_text = CopyablePlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("DanZero 返回值会显示在这里")
        self.result_text.setMinimumHeight(140)
        result_layout.addWidget(self.result_text)
        root.addWidget(result_card)

        self._refresh_image_info()
        self._refresh_image_preview()
        self._render_hand_cards(())
        self._update_completion_status()

        self.add_event_button.clicked.connect(self._add_event)
        self.remove_event_button.clicked.connect(self._remove_selected_event)
        self.copy_hand_code_button.clicked.connect(self._copy_hand_code)
        self.build_button.clicked.connect(self._build_and_report)
        self.test_button.clicked.connect(self._request_test)
        self.close_button.clicked.connect(self.close)
        self.recognize_button.clicked.connect(self.recognize_requested)
        self.advisor_strategy_combo.currentIndexChanged.connect(
            self._advisor_strategy_selected
        )
        for combo in (
            self.round_level_combo,
            self.wild_rank_combo,
            self.current_player_combo,
            self.lead_player_combo,
            self.event_player_combo,
            self.event_action_combo,
        ):
            combo.currentIndexChanged.connect(
                lambda *_args: self._update_completion_status()
            )
        self.my_hand_edit.textChanged.connect(self._update_completion_status)

    def _advisor_strategy_selected(self, _index: int = -1) -> None:
        self.advisor_strategy = normalize_advisor_strategy(
            self.advisor_strategy_combo.currentData()
        )
        self.advisor_strategy_changed.emit(self.advisor_strategy)

    def set_image_path(self, image_path: Path | None) -> None:
        self.image_path = Path(image_path) if image_path is not None else None
        self._events.clear()
        self._reset_recognition_inputs()
        self._refresh_events()
        self._refresh_image_info()
        self._refresh_image_preview()

    def set_frame_image(self, image_bgr: np.ndarray, info_text: str) -> None:
        """Show an in-memory frame (BGR) instead of a file on disk."""
        self.image_path = None
        self._source_image = image_bgr
        self.image_info_label.setText(info_text)
        self._events.clear()
        self._reset_recognition_inputs()
        self._refresh_events()

    def _refresh_image_info(self) -> None:
        if self.image_path is None:
            self.image_info_label.setText("当前没有截图，请先选择一张图片")
        else:
            self.image_info_label.setText(f"当前截图：{self.image_path.as_posix()}")

    def _build_image_context_menu(self) -> QMenu:
        menu = QMenu(self)
        copy_image_action = menu.addAction("复制图片")
        copy_image_action.setEnabled(self._source_image is not None)
        copy_image_action.triggered.connect(self._copy_image_to_clipboard)
        copy_path_action = menu.addAction("复制图片路径")
        copy_path_action.setEnabled(self.image_path is not None)
        copy_path_action.triggered.connect(self._copy_image_path_to_clipboard)
        return menu

    def _show_image_context_menu(self, position) -> None:
        menu = self._build_image_context_menu()
        menu.exec(self.image_preview.mapToGlobal(position))

    def _copy_image_to_clipboard(self) -> None:
        if self._source_image is None:
            return
        rgb = cv2.cvtColor(self._source_image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        QGuiApplication.clipboard().setImage(image)
        self.status.setText("图片已复制到剪贴板，可直接粘贴发送分析")

    def _copy_image_path_to_clipboard(self) -> None:
        if self.image_path is None:
            return
        QGuiApplication.clipboard().setText(str(self.image_path))
        self.status.setText("图片路径已复制到剪贴板")

    def _refresh_image_preview(self) -> None:
        self.preview_annotations = ()
        if self.image_path is None:
            self._source_image = None
            self._preview_image = QImage()
            self.image_preview.clear()
            self.image_preview.setText("暂无截图预览")
            self.annotation_legend.setText("识别框会在图片上显示牌面和识别类别")
            self.button_detection_label.setText("按钮检测：尚未识别")
            return
        try:
            self._source_image = read_image_unicode(self.image_path)
            self._update_preview_pixmap()
        except Exception as exc:
            self._source_image = None
            self._preview_image = QImage()
            self.image_preview.clear()
            self.image_preview.setText(f"截图预览失败：{exc}")

    def _update_preview_pixmap(self) -> None:
        if self._source_image is None:
            return
        image = self._source_image.copy()
        for annotation in self.preview_annotations:
            x, y, width, height = annotation.box
            x = max(0, min(image.shape[1] - 1, x))
            y = max(0, min(image.shape[0] - 1, y))
            right = max(x + 1, min(image.shape[1] - 1, x + width))
            bottom = max(y + 1, min(image.shape[0] - 1, y + height))
            color = {
                "hand": (55, 185, 90),
                "button": (210, 90, 210),
                "level": (30, 175, 235),
                "timer": (235, 155, 35),
                "status": (75, 105, 225),
            }.get(annotation.category, (40, 150, 245))
            cv2.rectangle(image, (x, y), (right, bottom), color, 2)
            label = (
                f"{ANNOTATION_LABELS.get(annotation.label, BUTTON_LABELS.get(annotation.label, annotation.label))} / "
                f"{ANNOTATION_CATEGORY_LABELS.get(annotation.category, annotation.category)}"
            )
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1,
            )
            text_top = max(text_height + baseline + 2, y)
            cv2.rectangle(
                image,
                (x, text_top - text_height - baseline - 4),
                (min(image.shape[1] - 1, x + text_width + 6), text_top),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x + 3, text_top - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        height, width, channels = image.shape
        self._preview_image = QImage(
            image.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_BGR888,
        ).copy()
        self._fit_image_preview()

    def _fit_image_preview(self) -> None:
        if self._preview_image.isNull():
            return
        target = self.image_preview.size()
        self.image_preview.setPixmap(
            QPixmap.fromImage(self._preview_image).scaled(
                max(1, target.width()),
                max(1, target.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "form_splitter"):
            compact = self.width() < 1100
            orientation = (
                Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
            )
            if self.form_splitter.orientation() != orientation:
                self.form_splitter.setOrientation(orientation)
                self.form_splitter.setSizes(
                    (420, 400) if compact else (500, 500)
                )
        self._fit_image_preview()

    def _copy_hand_code(self) -> None:
        hand_code = self.my_hand_edit.text().strip()
        if not hand_code:
            self.status.setText("暂无可复制的手牌代码")
            return
        QGuiApplication.clipboard().setText(hand_code)
        self.status.setText("DanZero 手牌代码已复制到剪贴板")

    @staticmethod
    def _rank_combo() -> ScrollSafeComboBox:
        combo = ScrollSafeComboBox()
        combo.addItem("未识别，请选择", None)
        for rank in RANKS:
            combo.addItem(rank, rank)
        return combo

    @staticmethod
    def _seat_combo() -> ScrollSafeComboBox:
        combo = ScrollSafeComboBox()
        combo.addItem("未识别，请选择", None)
        for seat in SEATS:
            combo.addItem(SEAT_LABELS[seat], seat)
        return combo

    def _reset_recognition_inputs(self) -> None:
        for combo in (
            self.round_level_combo,
            self.wild_rank_combo,
            self.current_player_combo,
            self.lead_player_combo,
        ):
            combo.setCurrentIndex(0)
        self.my_hand_edit.clear()
        self.preview_annotations = ()
        self.recognition_elapsed_ms = None
        self.recognition_status.setText("模板识别尚未运行")
        self.button_detection_label.setText("按钮检测：尚未识别")
        self.state_summary.clear()
        self._render_hand_cards(())
        self._update_completion_status()
        self._update_preview_pixmap()

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object | None) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    def apply_recognition(self, result: RecognitionResult) -> None:
        self.recognition_elapsed_ms = float(result.elapsed_ms)
        self._set_combo_data(self.round_level_combo, result.round_level)
        self._set_combo_data(self.wild_rank_combo, result.wild_rank)
        self._set_combo_data(self.current_player_combo, result.current_player)
        self._set_combo_data(self.lead_player_combo, result.lead_player)
        self.my_hand_edit.setText(" ".join(result.my_hand))
        self._render_hand_cards(result.my_hand)
        self._events = [
            (event.player, "pass" if event.is_pass else "play", event.cards)
            for event in result.events
        ]
        self._refresh_events()
        self.preview_annotations = result.annotations
        self._update_preview_pixmap()
        if result.buttons:
            button_text = "、".join(
                BUTTON_LABELS.get(value, value) for value in result.buttons
            )
            self.button_detection_label.setText(f"按钮检测结果：{button_text}")
        else:
            self.button_detection_label.setText("按钮检测结果：未识别到动作按钮")
        if result.annotations:
            legend = "；".join(
                f"{ANNOTATION_LABELS.get(item.label, BUTTON_LABELS.get(item.label, item.label))}（"
                f"{ANNOTATION_CATEGORY_LABELS.get(item.category, item.category)}，"
                f"{item.confidence:.0%}）"
                for item in result.annotations[:16]
            )
            self.annotation_legend.setText(f"图片标注框：{legend}")
        else:
            self.annotation_legend.setText("本次没有达到置信度阈值的识别框")
        if result.unresolved_fields:
            pending = "、".join(
                RECOGNITION_FIELD_LABELS.get(field, field)
                for field in result.unresolved_fields
            )
            self.recognition_status.setText(f"模板识别完成，待手动确认：{pending}")
        else:
            self.recognition_status.setText("模板识别完成，关键字段已自动填入")
        self.recognition_status.setText(
            f"{self.recognition_status.text()}；图片识别耗时："
            f"{self.recognition_elapsed_ms:.2f} ms"
        )
        details = list(result.diagnostics)
        details.insert(0, f"图片识别耗时：{self.recognition_elapsed_ms:.2f} ms")
        if result.buttons:
            details.insert(
                0,
                "按钮检测：" + "、".join(
                    BUTTON_LABELS.get(value, value) for value in result.buttons
                ),
            )
        if result.field_confidences:
            confidence = "；".join(
                f"{RECOGNITION_FIELD_LABELS.get(field, field)} {score:.0%}"
                for field, score in result.field_confidences.items()
            )
            details.insert(0, f"识别置信度：{confidence}")
        self.state_summary.setPlainText(
            "\n".join(details) or "模板库未发现需要人工补充的内容"
        )
        self._update_completion_status()

    def _render_hand_cards(self, cards: tuple[str, ...]) -> None:
        while self.hand_cards_layout.count():
            item = self.hand_cards_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.hand_card_widgets = []
        if not cards:
            empty = QLabel("暂未识别到手牌")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.hand_cards_layout.addWidget(empty)
            self.hand_cards_layout.addStretch(1)
            return
        for card in cards:
            badge = CardBadge(card)
            self.hand_card_widgets.append(badge)
            self.hand_cards_layout.addWidget(badge)
        self.hand_cards_layout.addStretch(1)

    @staticmethod
    def _card_strip(cards: tuple[str, ...]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        if not cards:
            label = QLabel("不出")
            layout.addWidget(label)
        else:
            for card in cards:
                layout.addWidget(CardBadge(card))
        layout.addStretch(1)
        return widget

    def show_recognition_error(self, message: str) -> None:
        self.recognition_status.setText(f"模板识别失败，请手动填写：{message}")
        self._update_completion_status()

    def set_recognition_busy(self, busy: bool) -> None:
        self.recognize_button.setEnabled(not busy)
        if busy:
            self.recognition_status.setText("正在使用区域库和模板库识别图片……")

    @staticmethod
    def _parse_cards(raw: str, *, required: bool = True) -> tuple[str, ...]:
        cards = tuple(
            value for value in re.split(r"[\s,，]+", str(raw).strip()) if value
        )
        if required and not cards:
            raise GameStateError("出牌时必须填写牌")
        return cards

    def _add_event(self) -> None:
        player = self.event_player_combo.currentData()
        action = self.event_action_combo.currentData()
        try:
            if player is None:
                raise GameStateError("请先选择事件玩家")
            action = str(action)
            cards = self._parse_cards(
                self.event_cards_edit.text(),
                required=action == "play",
            )
            if action == "pass" and cards:
                raise GameStateError("不出事件不能填写牌")
            self._events.append((str(player), action, cards))
        except GameStateError as exc:
            self.status.setText(f"事件无效：{exc}")
            return
        self.event_cards_edit.clear()
        self._refresh_events()
        self._update_completion_status()
        self.status.setText("已添加牌局事件")

    def _refresh_events(self) -> None:
        self.event_table.setRowCount(len(self._events))
        for row, (player, action, cards) in enumerate(self._events):
            self.event_table.setItem(row, 0, QTableWidgetItem(SEAT_LABELS[player]))
            self.event_table.setItem(row, 1, QTableWidgetItem(ACTION_LABELS[action]))
            self.event_table.setCellWidget(row, 2, self._card_strip(cards))
        self.event_table.resizeRowsToContents()

    def _remove_selected_event(self) -> None:
        rows = sorted(
            {index.row() for index in self.event_table.selectedIndexes()},
            reverse=True,
        )
        if not rows:
            self.status.setText("请先选择要删除的事件")
            return
        for row in rows:
            if 0 <= row < len(self._events):
                self._events.pop(row)
        self._refresh_events()
        self._update_completion_status()
        self.status.setText("已删除选中事件")

    def _update_completion_status(self, *_args) -> None:
        missing: list[str] = []
        for field, combo in (
            ("round_level", self.round_level_combo),
            ("wild_rank", self.wild_rank_combo),
            ("current_player", self.current_player_combo),
            ("lead_player", self.lead_player_combo),
        ):
            if combo.currentData() is None:
                missing.append(RECOGNITION_FIELD_LABELS[field])
        if not self.my_hand_edit.text().strip():
            missing.append(RECOGNITION_FIELD_LABELS["my_hand"])
        if missing:
            self.incomplete_status.setText(
                "⚠ 参数不完整：" + "、".join(missing) + "。请补充后再测试 DanZero。"
            )
            self.incomplete_status.setStyleSheet("color: #b42318; font-weight: 600;")
        elif self.current_player_combo.currentData() != "self":
            self.incomplete_status.setText(
                "⚠ 参数已填写，但当前行动者不是我方；DanZero 可能无法生成我方建议。"
            )
            self.incomplete_status.setStyleSheet("color: #9a6700; font-weight: 600;")
        else:
            self.incomplete_status.setText("✓ 参数完整，可以构建参数并测试 DanZero。")
            self.incomplete_status.setStyleSheet("color: #18794e; font-weight: 600;")

    def build_state(self) -> GuanDanState:
        try:
            state = GuanDanState()
            state.set_context(
                round_level=self._required_combo_value(
                    self.round_level_combo, "当前级牌"
                ),
                wild_rank=self._required_combo_value(
                    self.wild_rank_combo, "百搭牌级别"
                ),
                current_player=self._required_combo_value(
                    self.current_player_combo, "当前行动者"
                ),
                lead_player=self._required_combo_value(
                    self.lead_player_combo, "本轮首出者"
                ),
            )
            state.confirm_hand(self._parse_cards(self.my_hand_edit.text()))
            for player, action, cards in self._events:
                if action == "pass":
                    state.record_pass(player)
                else:
                    state.record_play(player, cards)
        except (GameStateError, ValueError) as exc:
            self.status.setText(f"参数无效：{exc}")
            self._update_completion_status()
            raise
        self.state_summary.setPlainText(state.summary())
        self.status.setText("参数已构建，可以测试 DanZero")
        self._update_completion_status()
        self.state_built.emit(state)
        return state

    @staticmethod
    def _required_combo_value(combo: QComboBox, label: str) -> str:
        value = combo.currentData()
        if value is None:
            raise GameStateError(f"请先确认{label}")
        return str(value)

    def _build_and_report(self) -> None:
        try:
            self.build_state()
        except (GameStateError, ValueError):
            return

    def _request_test(self) -> None:
        try:
            state = self.build_state()
        except (GameStateError, ValueError):
            return
        self.test_requested.emit(state)

    def set_test_busy(self, busy: bool) -> None:
        for widget in (
            self.build_button,
            self.test_button,
            self.add_event_button,
            self.remove_event_button,
        ):
            widget.setEnabled(not busy)
        if busy:
            self.status.setText("正在调用 DanZero……")

    def show_test_result(self, text: str) -> None:
        self.result_text.setPlainText(text)
        self.status.setText("DanZero 测试完成")

    def show_test_error(self, text: str) -> None:
        self.result_text.setPlainText(text)
        self.status.setText("DanZero 测试失败")
