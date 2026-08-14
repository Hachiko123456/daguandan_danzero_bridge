from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..annotation_service import AnnotationService
from ..application.model_evaluation import EvaluationRunResult, ModelEvaluationService
from ..application.placement_projection import (
    PlacementProjection,
    format_placement_summary,
    project_recorded_placements,
)
from ..application.timeline_truth_migration import TimelineTruthMigrationService
from ..danzero.state import RANKS, SEATS, SUITS
from ..live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    card_code_to_text,
    load_truth_log,
    save_truth_log,
)
from ..domain.truth import LabelProvenance, TruthEvidence
from ..live.turns import TURN_ORDER
from ..live.session_store import read_json_lines
from ..live.suit_correction import SuitCorrectionTracker
from ..recognition_service import ScreenshotRecognitionService
from ..template_service import TemplateService
from .single_image_danzero_page import CardBadge
from .model_evaluation_page import ModelEvaluationPanel

_SEAT_LABELS = {"self": "自己", "right": "右家", "opposite": "对家", "left": "左家"}
_RANK_LABELS = {rank: rank for rank in RANKS}
_SUIT_LABELS = {"S": "黑桃", "H": "红桃", "C": "梅花", "D": "方块"}
_STARTING_CARDS = 27
_PLAY_ROI_NAMES = {
    "self": "my_play",
    "right": "right_play",
    "opposite": "opposite_play",
    "left": "left_play",
}
_SUIT_ORDER = {"S": 0, "H": 1, "C": 2, "D": 3}
# 逆序展示时同点数按 黑桃 > 红桃 > 梅花 > 方块。
_SUIT_ORDER_DESC = {"S": 3, "H": 2, "C": 1, "D": 0}
_SUIT_CORRECTION_SCAN_AHEAD_FRAMES = 24
# 掼蛋牌力（逆序，大在前）：大王 > 小王 > 级牌 > A > K > … > 3 > 2。
# 级牌按对局轮次传入（如级牌 10，则 10 仅次于大小王）。
_JOKER_STRENGTH = {"big_joker": 18, "small_joker": 17}
_RANK_STRENGTH = {
    "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9,
    "9": 10, "10": 11, "J": 12, "Q": 13, "K": 14, "A": 15,
}


@dataclass(frozen=True)
class _FrameRecognitionCandidate:
    actor: str
    is_pass: bool
    cards: tuple[str, ...]
    confidence: float
    source: str
    roi_name: str


def _sort_hand_cards(
    cards: tuple[str, ...],
    level_rank: str | None = None,
) -> tuple[str, ...]:
    level_strength = _RANK_STRENGTH.get(str(level_rank), 16)

    def key(card: str) -> tuple[int, int]:
        if card in _JOKER_STRENGTH:
            return (_JOKER_STRENGTH[card], 9)
        rank, suit = card[:-1], card[-1:]
        strength = _RANK_STRENGTH.get(rank, 99)
        if str(rank) == str(level_rank):
            strength = 16
        return (strength, _SUIT_ORDER_DESC.get(suit, 9))

    return tuple(sorted(cards, key=key, reverse=True))


class CardPickerDialog(QDialog):
    def __init__(self, cards: tuple[str, ...] = (), parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择牌面")
        self._cards = list(cards)
        self._rank: str | None = None
        layout = QVBoxLayout(self)
        self.selected = QListWidget()
        layout.addWidget(QLabel("先选点数，再选花色；已选牌可双击删除"))
        rank_grid = QGridLayout()
        for index, rank in enumerate((*RANKS, "small_joker", "big_joker")):
            label = "小王" if rank == "small_joker" else "大王" if rank == "big_joker" else rank
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=rank: self._select_rank(value))
            rank_grid.addWidget(button, index // 8, index % 8)
        layout.addLayout(rank_grid)
        suit_grid = QGridLayout()
        for index, (suit, label) in enumerate(_SUIT_LABELS.items()):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=suit: self._select_suit(value))
            suit_grid.addWidget(button, 0, index)
        layout.addLayout(suit_grid)
        self.selected.itemDoubleClicked.connect(self._remove_selected)
        layout.addWidget(self.selected)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    def _select_rank(self, rank: str) -> None:
        self._rank = rank
        if rank in {"small_joker", "big_joker"}:
            self._append(rank)

    def _select_suit(self, suit: str) -> None:
        if self._rank is None or self._rank in {"small_joker", "big_joker"}:
            return
        self._append(f"{self._rank}{suit}")

    def _append(self, card: str) -> None:
        if self._cards.count(card) >= 2:
            return
        self._cards.append(card)
        self._refresh()

    def _remove_selected(self, item: QListWidgetItem) -> None:
        index = self.selected.row(item)
        if index >= 0:
            self._cards.pop(index)
            self._refresh()

    def _refresh(self) -> None:
        self.selected.clear()
        counts = Counter(self._cards)
        for card in self._cards:
            suffix = f" ×{counts[card]}" if counts[card] > 1 else ""
            self.selected.addItem(card_code_to_text(card) + suffix)

    def cards(self) -> tuple[str, ...]:
        return tuple(self._cards)


class ScrollSafeComboBox(QComboBox):
    """Let the surrounding log view consume wheel gestures safely."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class TruthLogEditor(QWidget):
    log_saved = Signal(object)
    draft_changed = Signal()

    def __init__(
        self,
        session: Path,
        truth_log: TruthLog,
        parent=None,
        *,
        frame_provider: Callable[[], tuple[int | None, np.ndarray | None]]
        | None = None,
        frame_scan_provider: Callable[[int], Iterable[tuple[int, np.ndarray]]]
        | None = None,
        recognition_service: ScreenshotRecognitionService | None = None,
        evaluation_service: ModelEvaluationService | None = None,
        repair_service: TimelineTruthMigrationService | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = Path(session)
        self.truth_log = truth_log
        self._frame_provider = frame_provider
        self._frame_scan_provider = frame_scan_provider
        self._recognition_service = recognition_service
        self._last_recognized_frame: int | None = None
        self._placements = self._load_placements(truth_log)
        self._placement_badges_by_turn = self._placement_badges(self._placements)
        self._suit_correction_tracker = SuitCorrectionTracker()
        self.setObjectName("truthLogEditor")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if truth_log.turns:
            resume_row = QHBoxLayout()
            self.resume_banner = QLabel(
                f"已载入保存的出牌记录 {len(truth_log.turns)} 条，可继续识别；"
                "如需重来请清空后从头开始"
            )
            self.resume_banner.setWordWrap(True)
            self.clear_button = QPushButton("清空记录，从头开始")
            resume_row.addWidget(self.resume_banner, 1)
            resume_row.addWidget(self.clear_button)
            layout.addLayout(resume_row)
            self.clear_button.clicked.connect(self._clear_all_rows)
        else:
            self.resume_banner = None
            self.clear_button = None
        self.placement_summary = QLabel()
        self.placement_summary.setObjectName("placementSummary")
        self.placement_summary.setWordWrap(True)
        self.placement_summary.setStyleSheet(
            "QLabel#placementSummary { color: #365c85; font-weight: 600; }"
        )
        self._refresh_placement_summary()
        layout.addWidget(self.placement_summary)
        lead_row = QHBoxLayout()
        lead_row.addWidget(QLabel("首出玩家"))
        self.lead_combo = ScrollSafeComboBox()
        for seat in SEATS:
            self.lead_combo.addItem(_SEAT_LABELS[seat], userData=seat)
        self.lead_combo.setCurrentIndex(max(0, self.lead_combo.findData(truth_log.initial_state.lead_player)))
        lead_row.addWidget(self.lead_combo)
        lead_row.addStretch(1)
        layout.addLayout(lead_row)
        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("当前级牌"))
        self.round_level_combo = ScrollSafeComboBox()
        for rank in RANKS:
            self.round_level_combo.addItem(rank, rank)
        self.round_level_combo.setCurrentIndex(
            max(0, self.round_level_combo.findData(truth_log.initial_state.round_level))
        )
        info_row.addWidget(self.round_level_combo)
        info_row.addSpacing(24)
        info_row.addWidget(QLabel("我方手牌"))
        self.hand_scroll = QScrollArea()
        self.hand_scroll.setWidgetResizable(True)
        self.hand_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.hand_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        hand_content = QWidget()
        self.hand_cards_layout = QHBoxLayout(hand_content)
        self.hand_cards_layout.setContentsMargins(2, 2, 2, 2)
        self.hand_cards_layout.setSpacing(4)
        self.hand_scroll.setWidget(hand_content)
        self.hand_scroll.setMinimumHeight(92)
        self.hand_scroll.setMaximumHeight(104)
        info_row.addWidget(self.hand_scroll, 1)
        self.edit_hand_button = QPushButton("编辑手牌")
        info_row.addWidget(self.edit_hand_button)
        self.recognize_hand_button = QPushButton("重新识别手牌")
        info_row.addWidget(self.recognize_hand_button)
        layout.addLayout(info_row)
        self._hand: tuple[str, ...] = tuple(truth_log.initial_state.my_hand)
        self._hand_badges: list[CardBadge] = []
        self._render_hand_badges()
        self.edit_hand_button.clicked.connect(self._edit_hand)
        self.recognize_hand_button.clicked.connect(self._recognize_hand)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ("序号", "玩家", "牌面", "牌墩", "模型推荐")
        )
        header = self.table.horizontalHeader()
        for column in (0, 1, 3):
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        for column in (2, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        self.add_button = QPushButton("新增一行")
        self.insert_button = QPushButton("插入行")
        self.remove_button = QPushButton("删除选中行")
        self.recognize_frame_button = QPushButton("识别当前画面")
        controls.addWidget(self.add_button)
        controls.addWidget(self.insert_button)
        controls.addWidget(self.remove_button)
        controls.addWidget(self.recognize_frame_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.recognition_hint = QLabel(
            "识别当前画面：选中一行可回填该行；清除选择后至多追加一行"
        )
        self.recognition_hint.setWordWrap(True)
        layout.addWidget(self.recognition_hint)
        buttons = QHBoxLayout()
        self.save_button = QPushButton("保存出牌日志")
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.end_status = QLabel("对局状态：进行中")
        self.end_status.setWordWrap(True)
        layout.addWidget(self.end_status)
        self.save_status = QLabel("未保存（保存后写入 truth_log.json）")
        self.save_status.setWordWrap(True)
        layout.addWidget(self.save_status)
        self.add_button.clicked.connect(self.add_row)
        self.insert_button.clicked.connect(self.insert_row)
        self.remove_button.clicked.connect(self.remove_row)
        self.recognize_frame_button.clicked.connect(self._recognize_frame)
        self.save_button.clicked.connect(self._save)
        self.table.cellChanged.connect(self._cell_changed)
        self._render()
        self.evaluation_panel = ModelEvaluationPanel(
            self.session,
            service=evaluation_service,
            repair_service=repair_service,
            parent=self,
        )
        self.evaluation_panel.result_ready.connect(self._show_evaluation_result)
        self.evaluation_panel.result_invalidated.connect(self._clear_recommendations)
        self.evaluation_panel.truth_repaired.connect(self._reload_repaired_truth)
        layout.addWidget(
            self.evaluation_panel,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        layout.setStretchFactor(self.table, 1)
        if not self._matches_saved_truth():
            self.evaluation_panel.invalidate_input()
        self.lead_combo.currentIndexChanged.connect(self._draft_mutated)
        self.round_level_combo.currentIndexChanged.connect(
            self._round_level_changed
        )

    def _render_hand_badges(self) -> None:
        while self.hand_cards_layout.count():
            item = self.hand_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._hand_badges = []
        cards = _sort_hand_cards(
            self._hand,
            self.round_level_combo.currentData(),
        )
        if not cards:
            empty = QLabel("手牌为空，点『编辑手牌』补充")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.hand_cards_layout.addWidget(empty)
            self.hand_cards_layout.addStretch(1)
            return
        for card in cards:
            badge = CardBadge(card)
            self._hand_badges.append(badge)
            self.hand_cards_layout.addWidget(badge)
        self.hand_cards_layout.addStretch(1)

    def _set_hand(self, cards: tuple[str, ...]) -> None:
        self._hand = tuple(cards)
        self._render_hand_badges()
        self._draft_mutated()

    def _round_level_changed(self, *_args) -> None:
        self._render_hand_badges()
        self._draft_mutated()

    def _edit_hand(self) -> None:
        dialog = CardPickerDialog(self._hand, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._set_hand(dialog.cards())

    def _recognize_hand(self) -> None:
        if self._frame_provider is None:
            self.recognition_hint.setText("识别手牌：当前编辑器未接入回放画面")
            return
        _frame_index, image = self._frame_provider()
        if image is None:
            self.recognition_hint.setText(
                "识别手牌：请先在回放页播放或点『下一帧』到开局画面"
            )
            return
        try:
            result = self._recognition().recognize(image)
        except Exception as exc:
            self.recognition_hint.setText(f"识别手牌失败：{exc}")
            return
        if not result.my_hand:
            self.recognition_hint.setText("识别手牌：画面中没有识别到手牌")
            return
        self._set_hand(tuple(result.my_hand))
        self.recognition_hint.setText(
            f"识别手牌：已更新为 {len(result.my_hand)} 张，可点『编辑手牌』微调"
        )

    def _clear_all_rows(self) -> None:
        self.table.setRowCount(0)
        self._resize_table()
        if self.clear_button is not None:
            self.clear_button.setEnabled(False)
        if self.resume_banner is not None:
            self.resume_banner.setText("已清空出牌记录，正在从头开始识别")
        self._draft_mutated()

    def _render(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for turn in self.truth_log.turns:
            self._append_row(turn, editable=True, notify=False)
        self.table.blockSignals(False)
        self._resize_table()

    def _matches_saved_truth(self) -> bool:
        try:
            saved = load_truth_log(
                self.session / "truth_log.json",
                session_id=self.session.name,
            )
        except Exception:
            return False
        return saved.to_dict() == self.truth_log.to_dict()

    def _load_placements(self, truth_log: TruthLog) -> tuple[PlacementProjection, ...]:
        """Read one rank-ordered, evidence-safe projection from the timeline."""

        timeline_path = self.session / "timeline.jsonl"
        if not timeline_path.is_file():
            return ()
        try:
            return project_recorded_placements(
                read_json_lines(timeline_path),
                truth_log.turns,
            )
        except Exception:
            return ()

    @staticmethod
    def _placement_badges(
        placements: tuple[PlacementProjection, ...],
    ) -> dict[int, str]:
        return {
            item.anchor_turn_id: item.label
            for item in placements
            if item.anchor_turn_id is not None
        }

    def _refresh_placement_summary(self) -> None:
        self.placement_summary.setText(
            format_placement_summary(self._placements, _SEAT_LABELS)
        )

    def _clear_placement_badges(self) -> None:
        """Keep timeline order visible, but remove stale row-level anchors."""

        if not self._placement_badges_by_turn:
            return
        self._placement_badges_by_turn = {}
        for row in range(self.table.rowCount()):
            self.table.setCellWidget(
                row,
                2,
                self._card_strip_widget(
                    self._cards_from_row(row),
                    is_pass=self._is_pass_from_row(row),
                ),
            )

    @staticmethod
    def _card_strip_widget(
        cards: tuple[str, ...],
        *,
        is_pass: bool = False,
        empty_label: str = "",
        placement: str = "",
    ) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        if not cards:
            label = QLabel(
                "不出" if is_pass else (empty_label or "等待画面识别…")
            )
            if is_pass:
                # A pass must remain as legible as a card strip when the model
                # recommends cards in the neighbouring column.  A fixed badge
                # also makes the table reserve a complete card-height row.
                label.setObjectName("passBadge")
                label.setMinimumSize(72, 72)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setStyleSheet(
                    "QLabel#passBadge { background: #f5f8fc; "
                    "border: 1px dashed #8ba2ba; border-radius: 6px; "
                    "color: #365c85; font-size: 14px; font-weight: 600; }"
                )
                label.setToolTip("不出")
            layout.addWidget(
                label,
                0,
                Qt.AlignmentFlag.AlignVCenter,
            )
        else:
            for card in cards:
                layout.addWidget(CardBadge(card))
        if placement:
            badge = QLabel(placement)
            badge.setObjectName("placementBadge")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                "QLabel#placementBadge { background: #fff3d6; "
                "border: 1px solid #e5b86e; border-radius: 6px; "
                "color: #8a5a12; font-weight: 600; padding: 4px 7px; }"
            )
            layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)
        return widget

    def _resize_table(self) -> None:
        self.table.resizeRowsToContents()
        self.table.resizeColumnToContents(3)
        self._update_end_status()

    def _update_end_status(self) -> None:
        out = self._out_players()
        if out >= {"self", "opposite"}:
            self.end_status.setText("对局已结束：己方（自己+对家）已全部出完")
            self.end_status.setStyleSheet("color: #18794e; font-weight: 600;")
        elif out >= {"right", "left"}:
            self.end_status.setText("对局已结束：对方（右家+左家）已全部出完")
            self.end_status.setStyleSheet("color: #b42318; font-weight: 600;")
        else:
            self.end_status.setText("对局状态：进行中")
            self.end_status.setStyleSheet("")

    def _append_row(
        self,
        turn: TruthTurn,
        row: int | None = None,
        *,
        editable: bool = False,
        status: str | None = None,
        notify: bool = True,
    ) -> None:
        if row is None:
            row = self.table.rowCount()
        self.table.insertRow(row)
        self._populate_row(
            row,
            turn,
            editable=editable,
            status=status,
        )
        self._renumber_rows()
        if notify:
            self._draft_mutated()

    def _populate_row(
        self,
        row: int,
        turn: TruthTurn,
        *,
        editable: bool,
        status: str | None = None,
    ) -> None:
        number_item = QTableWidgetItem(str(row + 1))
        if turn.frame_index is not None:
            number_item.setData(Qt.ItemDataRole.UserRole, turn.frame_index)
        number_item.setData(int(Qt.ItemDataRole.UserRole) + 1, turn)
        number_item.setFlags(
            number_item.flags() & ~Qt.ItemFlag.ItemIsEditable
        )
        self.table.setItem(row, 0, number_item)
        player = ScrollSafeComboBox()
        for seat in SEATS:
            player.addItem(_SEAT_LABELS[seat], userData=seat)
        player.setCurrentIndex(max(0, player.findData(turn.actor)))
        if not editable:
            player.setEnabled(False)
        player.currentIndexChanged.connect(
            lambda _index, combo=player: self._player_changed(
                self._row_for_widget(combo)
            )
        )
        player.activated.connect(
            lambda _index, combo=player: self._player_changed(
                self._row_for_widget(combo)
            )
        )
        self.table.setCellWidget(row, 1, player)
        self.table.setCellWidget(
            row,
            2,
            self._card_strip_widget(
                turn.cards,
                is_pass=turn.is_pass,
                placement=self._placement_badges_by_turn.get(turn.index, ""),
            ),
        )
        trick = QTableWidgetItem(str(turn.trick_id or "—"))
        trick.setFlags(trick.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 3, trick)
        if status:
            player.setToolTip(status)
        self.table.removeCellWidget(row, 4)

    def _replace_row(
        self,
        row: int,
        turn: TruthTurn,
        *,
        status: str,
    ) -> None:
        self.table.blockSignals(True)
        try:
            self._populate_row(row, turn, editable=True, status=status)
        finally:
            self.table.blockSignals(False)
        self._draft_mutated()

    def append_confirmed_turn(
        self,
        turn: TruthTurn,
        *,
        status: str = "扫描确认",
    ) -> int:
        """Append one scan-confirmed action to the unsaved in-memory draft."""

        row = self.table.rowCount()
        self._append_row(turn, editable=True, status=status)
        self._resize_table()
        return row

    def _renumber_rows(self) -> None:
        for index in range(self.table.rowCount()):
            item = self.table.item(index, 0)
            if item is not None:
                item.setText(str(index + 1))

    def add_row(self) -> None:
        scroll = self._table_scroll_position()
        row = self.table.rowCount()
        actor = self._expected_player_at(row) or "self"
        self._append_row(
            TruthTurn(row + 1, actor, False, ()),
            editable=True,
            status="待编辑",
        )
        self._resize_table()
        self._restore_table_scroll(scroll)

    def insert_row(self) -> None:
        """在当前选中行之前插入一行；未选中则追加到末尾。"""
        selected_rows = self._selected_rows()
        row = selected_rows[0] if len(selected_rows) == 1 else -1
        if row < 0:
            self.add_row()
            return
        scroll = self._table_scroll_position()
        before = self._expected_player_at(row)
        after_actor = self._player_at(row)
        after = (
            self._previous_active_player(
                after_actor,
                self._out_players_before(row),
            )
            if after_actor is not None
            else None
        )
        actor = before or after or "self"
        conflict = bool(before and after and before != after)
        status = "待确认：前后玩家候选冲突" if conflict else "待编辑"
        self._append_row(
            TruthTurn(self.table.rowCount() + 1, actor, False, ()),
            row=row,
            editable=True,
            status=status,
        )
        player = self.table.cellWidget(row, 1)
        if player is not None:
            player.setProperty("sequenceConflict", conflict)
        self._resize_table()
        self.table.selectRow(row)
        self._restore_table_scroll(scroll)
        if conflict:
            self.save_status.setText(
                "插入位置前后玩家候选冲突："
                f"前序推断为{_SEAT_LABELS[str(before)]}，"
                f"后序反推为{_SEAT_LABELS[str(after)]}；"
                "请人工选择玩家确认后再保存"
            )

    def remove_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self._renumber_rows()
            self._resize_table()
            if self.table.rowCount():
                self.table.selectRow(min(row, self.table.rowCount() - 1))
            self._draft_mutated()

    def _selected_rows(self) -> list[int]:
        selection = self.table.selectionModel()
        if selection is None:
            return []
        return sorted(index.row() for index in selection.selectedRows())

    def _table_scroll_position(self) -> tuple[int, int]:
        return (
            self.table.horizontalScrollBar().value(),
            self.table.verticalScrollBar().value(),
        )

    def _restore_table_scroll(self, position: tuple[int, int]) -> None:
        horizontal, vertical = position
        self.table.horizontalScrollBar().setValue(horizontal)
        self.table.verticalScrollBar().setValue(vertical)

    def _player_changed(self, row: int) -> None:
        if row < 0:
            return
        player = self.table.cellWidget(row, 1)
        if player is not None and bool(player.property("sequenceConflict")):
            player.setProperty("sequenceConflict", False)
            player.setToolTip("已人工确认玩家")
            self.save_status.setText("玩家顺序冲突已人工确认；出牌日志仍未保存")
        self._clear_recommendations()
        self._draft_mutated()

    def _row_for_widget(self, widget: QWidget) -> int:
        for row in range(self.table.rowCount()):
            if any(
                self.table.cellWidget(row, column) is widget
                for column in (1,)
            ):
                return row
        viewport_position = widget.mapTo(
            self.table.viewport(),
            widget.rect().center(),
        )
        return self.table.indexAt(viewport_position).row()

    def _cell_changed(self, _row: int, _column: int) -> None:
        return

    def _cards_from_row(self, row: int) -> tuple[str, ...]:
        widget = self.table.cellWidget(row, 2)
        if widget is None or widget.layout() is None:
            return ()
        cards: list[str] = []
        layout = widget.layout()
        for index in range(layout.count()):
            badge = layout.itemAt(index).widget()
            if isinstance(badge, CardBadge):
                cards.append(badge.card_code)
        return tuple(cards)

    def _is_pass_from_row(self, row: int) -> bool:
        item = self.table.item(row, 0)
        original = (
            item.data(int(Qt.ItemDataRole.UserRole) + 1)
            if item is not None
            else None
        )
        return bool(original.is_pass) if isinstance(original, TruthTurn) else False

    def _build_log(self) -> TruthLog:
        turns: list[TruthTurn] = []
        for row in range(self.table.rowCount()):
            player = self.table.cellWidget(row, 1)
            if player is not None and bool(player.property("sequenceConflict")):
                raise ValueError(
                    f"第 {row + 1} 条动作的前后玩家候选冲突，"
                    "请人工确认玩家后再保存"
                )
            actor = player.currentData()
            is_pass = self._is_pass_from_row(row)
            cards = () if is_pass else self._cards_from_row(row)
            if not is_pass and not cards:
                raise ValueError(f"第 {row + 1} 条动作还没有选择牌面")
            number_item = self.table.item(row, 0)
            frame_index = (
                number_item.data(Qt.ItemDataRole.UserRole)
                if number_item is not None
                else None
            )
            original = (
                number_item.data(int(Qt.ItemDataRole.UserRole) + 1)
                if number_item is not None
                else None
            )
            unchanged = bool(
                isinstance(original, TruthTurn)
                and original.actor == actor
                and original.is_pass == is_pass
                and original.cards == cards
            )
            turns.append(
                TruthTurn(
                    row + 1,
                    actor,
                    is_pass,
                    cards,
                    frame_index=frame_index if isinstance(frame_index, int) else None,
                    monotonic_ms=original.monotonic_ms if unchanged else None,
                    trick_id=original.trick_id if unchanged else None,
                    evidence=original.evidence if unchanged else None,
                    label_status=original.label_status if unchanged else "draft",
                    provenance=(
                        original.provenance
                        if unchanged
                        else LabelProvenance(source="human_editor")
                    ),
                    uncertainty=original.uncertainty if unchanged else (),
                )
            )
        lead = self.lead_combo.currentData()
        round_level = self.round_level_combo.currentData()
        hand = _sort_hand_cards(self._hand, round_level)
        if not hand:
            raise ValueError("我方手牌不能为空")
        return TruthLog(
            source_session_id=self.truth_log.source_session_id,
            initial_state=TruthInitialState(str(round_level), lead, hand),
            turns=tuple(turns),
            source_video=self.truth_log.source_video,
            frame_index_path=self.truth_log.frame_index_path,
            label_status=self.truth_log.label_status,
            provenance=self.truth_log.provenance,
            outcome=self.truth_log.outcome,
        )

    def _recognition(self) -> ScreenshotRecognitionService:
        if self._recognition_service is None:
            profile_root = self.session.parents[2]
            profile_name = self.session.parents[1].name
            self._recognition_service = ScreenshotRecognitionService(
                AnnotationService(profile_root, profile_name),
                TemplateService(profile_root, profile_name),
            )
        return self._recognition_service

    def _played_card_counts_before(self, stop_row: int) -> dict[str, int]:
        played: dict[str, int] = {}
        for row in range(max(0, min(stop_row, self.table.rowCount()))):
            player = self.table.cellWidget(row, 1)
            if player is None or self._is_pass_from_row(row):
                continue
            player_name = str(player.currentData())
            played[player_name] = played.get(player_name, 0) + len(
                self._cards_from_row(row)
            )
        return played

    def _finished_players(self, played: dict[str, int]) -> set[str]:
        out: set[str] = set()
        for player in TURN_ORDER:
            start = len(self._hand) if player == "self" else _STARTING_CARDS
            if played.get(player, 0) >= start:
                out.add(player)
        return out

    def _out_players_before(self, stop_row: int) -> set[str]:
        return self._finished_players(self._played_card_counts_before(stop_row))

    def _out_players(self) -> set[str]:
        """已出完（手牌打光）的玩家：累计出牌数达到起始手牌数。

        自己按编辑器里的初始手牌数计算，其他玩家按掼蛋初始 27 张计算。
        """
        return self._out_players_before(self.table.rowCount())

    def _player_at(self, row: int) -> str | None:
        if not 0 <= row < self.table.rowCount():
            return None
        player = self.table.cellWidget(row, 1)
        if player is None:
            return None
        actor = str(player.currentData())
        return actor if actor in TURN_ORDER else None

    @staticmethod
    def _next_active_player(actor: str, out: set[str]) -> str | None:
        if actor not in TURN_ORDER:
            return None
        index = TURN_ORDER.index(actor)
        for offset in range(1, len(TURN_ORDER) + 1):
            seat = TURN_ORDER[(index + offset) % len(TURN_ORDER)]
            if seat not in out:
                return seat
        return None

    @staticmethod
    def _previous_active_player(actor: str, out: set[str]) -> str | None:
        if actor not in TURN_ORDER:
            return None
        index = TURN_ORDER.index(actor)
        for offset in range(1, len(TURN_ORDER) + 1):
            seat = TURN_ORDER[(index - offset) % len(TURN_ORDER)]
            if seat not in out:
                return seat
        return None

    def _expected_player_at(self, row: int) -> str | None:
        row = max(0, min(row, self.table.rowCount()))
        if row == 0:
            lead = self.lead_combo.currentData()
            return str(lead) if lead in TURN_ORDER else None
        actor = self._player_at(row - 1)
        if actor is None:
            return None
        return self._next_active_player(actor, self._out_players_before(row))

    def _expected_next_player(self) -> str | None:
        return self._expected_player_at(self.table.rowCount())

    def _recognition_candidate(
        self,
        image: np.ndarray,
        expected: str,
    ) -> tuple[_FrameRecognitionCandidate | None, str]:
        recognition = self._recognition()
        region_result = recognition.recognize_play_region(
            image,
            expected,
            wild_rank=str(self.round_level_combo.currentData()),
            allow_unknown_suit=True,
        )
        if region_result is not None and (
            region_result.cards or region_result.is_pass
        ):
            return (
                _FrameRecognitionCandidate(
                    actor=str(region_result.player),
                    is_pass=bool(region_result.is_pass),
                    cards=tuple(region_result.cards),
                    confidence=float(region_result.confidence),
                    source=str(region_result.source),
                    roi_name=_PLAY_ROI_NAMES.get(str(region_result.player), ""),
                ),
                f"仅识别{_SEAT_LABELS.get(str(region_result.player), str(region_result.player))}区域",
            )

        try:
            result = recognition.recognize(image, allow_unknown_suit=True)
        except TypeError as exc:
            if "allow_unknown_suit" not in str(exc):
                raise
            result = recognition.recognize(image)
        if not result.events:
            return None, "当前画面没有识别到出牌动作"
        if len(result.events) != 1:
            raise ValueError(
                f"当前画面识别到 {len(result.events)} 个候选动作，无法安全写入"
            )
        event = result.events[0]
        return (
            _FrameRecognitionCandidate(
                actor=str(event.player),
                is_pass=bool(event.is_pass),
                cards=tuple(event.cards),
                confidence=float(event.confidence),
                source=str(event.source),
                roi_name=_PLAY_ROI_NAMES.get(str(event.player), ""),
            ),
            "整幅画面识别",
        )

    def _validate_recognition_context(
        self,
        candidate: _FrameRecognitionCandidate,
        *,
        target_row: int,
        replacing: bool,
    ) -> None:
        expected = self._expected_player_at(target_row)
        if expected is None:
            raise ValueError("无法从当前位置前的有效记录推断玩家")
        if candidate.actor not in TURN_ORDER:
            raise ValueError("识别结果包含无效玩家")
        if candidate.actor != expected:
            raise ValueError(
                "识别玩家与当前位置上下文冲突："
                f"应为{_SEAT_LABELS[expected]}，识别为{_SEAT_LABELS[candidate.actor]}"
            )
        if candidate.is_pass and candidate.cards:
            raise ValueError("识别结果同时包含不出和牌面")
        if not candidate.is_pass and not candidate.cards:
            raise ValueError("识别结果没有可写入的动作")
        if not 0.0 <= candidate.confidence <= 1.0:
            raise ValueError("识别置信度无效")
        for card in candidate.cards:
            card_code_to_text(card)

        if not replacing or target_row + 1 >= self.table.rowCount():
            return
        played = self._played_card_counts_before(target_row)
        if not candidate.is_pass:
            played[candidate.actor] = played.get(candidate.actor, 0) + len(
                candidate.cards
            )
        expected_after = self._next_active_player(
            candidate.actor,
            self._finished_players(played),
        )
        actual_after = self._player_at(target_row + 1)
        if expected_after != actual_after:
            expected_label = _SEAT_LABELS.get(str(expected_after), "无")
            actual_label = _SEAT_LABELS.get(str(actual_after), "无")
            raise ValueError(
                "识别结果与下一行上下文冲突："
                f"识别后应轮到{expected_label}，下一行为{actual_label}"
            )

    def _correct_unknown_suit_row(
        self,
        row: int,
        frame_index: int | None,
        image: np.ndarray,
    ) -> bool:
        """Apply the live listener's two-read suit correction to one old row."""

        number_item = self.table.item(row, 0)
        original = (
            number_item.data(int(Qt.ItemDataRole.UserRole) + 1)
            if number_item is not None
            else None
        )
        if (
            not isinstance(original, TruthTurn)
            or original.is_pass
            or not any(card.endswith("?") for card in original.cards)
        ):
            return False
        if self._frame_scan_provider is not None and frame_index is not None:
            return self._scan_unknown_suit_row(
                row, original, int(frame_index), image
            )
        try:
            result = self._recognition().recognize_play_region(
                image,
                original.actor,
                wild_rank=str(self.round_level_combo.currentData()),
                allow_unknown_suit=True,
            )
        except TypeError as exc:
            if "allow_unknown_suit" not in str(exc):
                raise
            result = self._recognition().recognize_play_region(
                image,
                original.actor,
                wild_rank=str(self.round_level_combo.currentData()),
            )
        if result is None or result.is_pass or result.player != original.actor:
            self.recognition_hint.setText(
                f"花色修正：第 {row + 1} 行未识别到可用的{_SEAT_LABELS[original.actor]}出牌；未修改"
            )
            return True
        target_id = f"row:{row}:{original.index}:{'|'.join(original.cards)}"
        observation = self._suit_correction_tracker.observe(
            target_id,
            original.cards,
            tuple(str(card) for card in result.cards),
        )
        if not observation.cards:
            self.recognition_hint.setText(
                f"花色修正：第 {row + 1} 行候选与原动作的张数或点数不一致；未修改"
            )
            return True
        if not observation.confirmed:
            self.recognition_hint.setText(
                f"花色修正：第 {row + 1} 行已得到候选牌面，请在下一帧再次识别确认；未修改"
            )
            return True
        evidence_indices = tuple(
            dict.fromkeys(
                (*original.evidence.frame_indices,)
                + ((frame_index,) if frame_index is not None else ())
            )
        )
        corrected = TruthTurn(
            original.index,
            original.actor,
            False,
            observation.cards,
            frame_index=frame_index,
            monotonic_ms=original.monotonic_ms,
            trick_id=original.trick_id,
            evidence=replace(
                original.evidence,
                frame_indices=evidence_indices,
                roi_name=original.evidence.roi_name
                or _PLAY_ROI_NAMES[original.actor],
            ),
            label_status=original.label_status,
            provenance=LabelProvenance(
                source=f"suit_correction:{result.source or 'frame_recognition'}",
                confidence=result.confidence,
            ),
            uncertainty=tuple(
                item for item in original.uncertainty if item != "unknown_suit"
            ),
        )
        self._suit_correction_tracker.clear(target_id)
        self._replace_row(row, corrected, status="花色修正已确认")
        self._resize_table()
        self._last_recognized_frame = frame_index
        self.recognition_hint.setText(
            f"花色修正：已回填第 {row + 1} 行；请保存出牌日志后重新评测"
        )
        return True

    def _scan_unknown_suit_row(
        self,
        row: int,
        original: TruthTurn,
        frame_index: int,
        image: np.ndarray,
    ) -> bool:
        """Confirm one unknown-suit row from nearby replay frames only."""

        assert self._frame_scan_provider is not None
        target_id = f"row:{row}:{original.index}:{'|'.join(original.cards)}"
        frames: list[tuple[int, np.ndarray]] = [(frame_index, image)]
        try:
            frames.extend(
                (int(candidate_index), candidate_image)
                for candidate_index, candidate_image in self._frame_scan_provider(
                    frame_index
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.recognition_hint.setText(
                f"花色修正：无法读取后续录像帧（{exc}）；未修改"
            )
            return True

        confirmed_result = None
        confirmed_frame_index: int | None = None
        candidate_frame_indices: list[int] = []
        for candidate_frame_index, candidate_image in frames[
            : 1 + _SUIT_CORRECTION_SCAN_AHEAD_FRAMES
        ]:
            try:
                result = self._recognition().recognize_play_region(
                    candidate_image,
                    original.actor,
                    wild_rank=str(self.round_level_combo.currentData()),
                    allow_unknown_suit=True,
                )
            except TypeError as exc:
                if "allow_unknown_suit" not in str(exc):
                    raise
                result = self._recognition().recognize_play_region(
                    candidate_image,
                    original.actor,
                    wild_rank=str(self.round_level_combo.currentData()),
                )
            if result is None or result.is_pass or result.player != original.actor:
                continue
            observation = self._suit_correction_tracker.observe(
                target_id,
                original.cards,
                tuple(str(card) for card in result.cards),
            )
            if not observation.cards:
                continue
            candidate_frame_indices.append(candidate_frame_index)
            if observation.confirmed:
                confirmed_result = result
                confirmed_frame_index = candidate_frame_index
                break

        if confirmed_result is None:
            self._suit_correction_tracker.clear(target_id)
            self.recognition_hint.setText(
                f"花色修正：已检查当前帧后的 {_SUIT_CORRECTION_SCAN_AHEAD_FRAMES} 帧，"
                "未得到两帧一致的完整花色；未修改"
            )
            return True

        evidence_indices = tuple(
            dict.fromkeys(
                (*original.evidence.frame_indices, *candidate_frame_indices)
            )
        )
        corrected = TruthTurn(
            original.index,
            original.actor,
            False,
            tuple(str(card) for card in confirmed_result.cards),
            frame_index=confirmed_frame_index,
            monotonic_ms=original.monotonic_ms,
            trick_id=original.trick_id,
            evidence=replace(
                original.evidence,
                frame_indices=evidence_indices,
                roi_name=original.evidence.roi_name
                or _PLAY_ROI_NAMES[original.actor],
            ),
            label_status=original.label_status,
            provenance=LabelProvenance(
                source=(
                    "suit_correction:"
                    f"{confirmed_result.source or 'frame_recognition'}"
                ),
                confidence=confirmed_result.confidence,
            ),
            uncertainty=tuple(
                item for item in original.uncertainty if item != "unknown_suit"
            ),
        )
        self._suit_correction_tracker.clear(target_id)
        self._replace_row(row, corrected, status="花色修正已确认")
        self._resize_table()
        self._last_recognized_frame = confirmed_frame_index
        self.recognition_hint.setText(
            f"花色修正：已在后续帧确认并回填第 {row + 1} 行"
            f"（确认帧 {confirmed_frame_index}）；请保存出牌日志后重新评测"
        )
        return True

    def _recognize_frame(self) -> None:
        if self._frame_provider is None:
            self.recognition_hint.setText(
                "识别当前画面：当前编辑器未接入回放画面"
            )
            return
        frame_index, image = self._frame_provider()
        if image is None:
            self.recognition_hint.setText(
                "识别当前画面：请先在回放页播放或点『下一帧』到目标画面"
            )
            return
        selected_rows = self._selected_rows()
        if len(selected_rows) > 1:
            self.recognition_hint.setText(
                "识别当前画面：选中了多行，无法确定唯一回填目标；未写入"
            )
            return
        replacing = len(selected_rows) == 1
        target_row = selected_rows[0] if replacing else self.table.rowCount()
        if replacing and self._correct_unknown_suit_row(target_row, frame_index, image):
            return
        expected = self._expected_player_at(target_row)
        if expected is None:
            self.recognition_hint.setText(
                "识别当前画面：无法从当前位置前的有效记录推断玩家；未写入"
            )
            return
        try:
            candidate, source_note = self._recognition_candidate(image, expected)
            if candidate is None:
                self.recognition_hint.setText(
                    f"识别当前画面：{source_note}；未写入"
                )
                return
            self._validate_recognition_context(
                candidate,
                target_row=target_row,
                replacing=replacing,
            )
        except Exception as exc:
            self.recognition_hint.setText(f"识别当前画面失败：{exc}；未写入")
            return

        signature = (candidate.actor, candidate.is_pass, candidate.cards)
        if not replacing and frame_index is not None:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item is None or item.data(Qt.ItemDataRole.UserRole) != frame_index:
                    continue
                player = self.table.cellWidget(row, 1)
                if player is None:
                    continue
                existing = (
                    str(player.currentData()),
                    self._is_pass_from_row(row),
                    self._cards_from_row(row),
                )
                if existing == signature:
                    self.recognition_hint.setText(
                        "识别当前画面：该动作已由当前画面记录；未重复写入"
                    )
                    return

        original = (
            self.table.item(target_row, 0).data(
                int(Qt.ItemDataRole.UserRole) + 1
            )
            if replacing and self.table.item(target_row, 0) is not None
            else None
        )
        trick_id = original.trick_id if isinstance(original, TruthTurn) else None
        turn = TruthTurn(
            target_row + 1,
            candidate.actor,
            candidate.is_pass,
            candidate.cards,
            frame_index=frame_index,
            trick_id=trick_id,
            evidence=TruthEvidence(
                frame_indices=(frame_index,) if frame_index is not None else (),
                roi_name=candidate.roi_name,
            ),
            label_status="draft",
            provenance=LabelProvenance(
                source=candidate.source or "frame_recognition",
                confidence=candidate.confidence,
            ),
        )
        scroll = self._table_scroll_position()
        if replacing:
            self._replace_row(target_row, turn, status="画面识别回填")
        else:
            self._append_row(turn, editable=True, status="画面识别追加")
        self._resize_table()
        self._restore_table_scroll(scroll)
        self._last_recognized_frame = frame_index
        if replacing:
            self.recognition_hint.setText(
                f"识别当前画面：已原子回填第 {target_row + 1} 行"
                f"（{source_note}），行数和相邻行未改变"
            )
        else:
            self.recognition_hint.setText(
                f"识别当前画面：已追加 1 条出牌记录（{source_note}）"
            )

    def _save(self) -> None:
        try:
            log = self._build_log()
            save_truth_log(self.session / "truth_log.json", log)
            self.truth_log = log
            self.save_status.setText(f"已保存 {len(log.turns)} 条，可继续编辑")
            self.log_saved.emit(log)
            self.evaluation_panel.saved_input_updated()
        except Exception as exc:
            self.save_status.setText(f"保存失败：{exc}")
            QLabel(str(exc), self).show()

    def _draft_mutated(self, *_args) -> None:
        self._clear_placement_badges()
        panel = getattr(self, "evaluation_panel", None)
        if panel is not None:
            panel.invalidate_input()
        self.draft_changed.emit()

    def _clear_recommendations(self, *, resize: bool = True) -> None:
        for row in range(self.table.rowCount()):
            self.table.removeCellWidget(row, 4)
        if resize:
            self._resize_table()

    @staticmethod
    def _recommendation_widget(value: object, *, error: str = "") -> QWidget:
        if error:
            return TruthLogEditor._card_strip_widget((), empty_label=error)
        if not isinstance(value, dict):
            return TruthLogEditor._card_strip_widget((), empty_label="—")
        is_pass = bool(value.get("is_pass", False))
        cards = tuple(str(card) for card in value.get("cards", ()))
        return TruthLogEditor._card_strip_widget(cards, is_pass=is_pass)

    def _show_evaluation_result(self, result: EvaluationRunResult) -> None:
        self._clear_recommendations(resize=False)
        for decision in result.decisions:
            turn_id = decision.get("turn_id")
            if not isinstance(turn_id, int):
                continue
            row = turn_id - 1
            if not 0 <= row < self.table.rowCount():
                continue
            player = self.table.cellWidget(row, 1)
            if player is None or player.currentData() != "self":
                continue
            predicted = decision.get("predicted_action")
            error = ""
            if decision.get("status") != "evaluated":
                error = f"错误：{decision.get('error_code') or 'unknown'}"
            self.table.setCellWidget(
                row,
                4,
                self._recommendation_widget(predicted, error=error),
            )
        # Recommendations are added after the original pass-only rows were
        # rendered.  Recompute once so model card badges can never be clipped.
        self._resize_table()

    def _reload_repaired_truth(self, _result: object) -> None:
        try:
            repaired = load_truth_log(
                self.session / "truth_log.json",
                session_id=self.session.name,
            )
        except Exception as exc:
            self.save_status.setText(f"修复后重新载入失败：{exc}")
            return
        self.truth_log = repaired
        self._placements = self._load_placements(repaired)
        self._placement_badges_by_turn = self._placement_badges(self._placements)
        self._refresh_placement_summary()
        self.lead_combo.setCurrentIndex(
            max(0, self.lead_combo.findData(repaired.initial_state.lead_player))
        )
        self.round_level_combo.setCurrentIndex(
            max(0, self.round_level_combo.findData(repaired.initial_state.round_level))
        )
        self._hand = tuple(repaired.initial_state.my_hand)
        self._render_hand_badges()
        self._render()
        self.evaluation_panel.saved_input_updated()
        self.save_status.setText("旧日志已安全修复并重新载入；原文件已备份")

    def shutdown(self) -> None:
        self.evaluation_panel.shutdown()
