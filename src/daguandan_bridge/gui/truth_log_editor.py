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
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import PrimaryPushButton, PushButton

from ..annotation_service import AnnotationService
from ..application.replay_turn_draft import (
    next_actor_after_prefix,
    validate_truth_log_with_live_reducer,
    validate_turn_actor_chain,
)
from ..application.placement_projection import (
    PlacementProjection,
    format_placement_summary,
    project_recorded_placements,
    project_truth_log_placements,
    derive_finish_order,
)
from ..application.truth_revision_store import TruthRevisionStore
from ..config import PROFILES_ROOT
from ..storage import atomic_write_json
from ..danzero.state import RANKS, SEATS, SUITS
from ..live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    TruthLogCardInventoryError,
    card_code_to_text,
    load_truth_log,
    save_truth_log,
    validate_truth_log_card_inventory,
)
from ..domain.truth import LabelProvenance, TruthEvidence, TruthOutcome
from ..live.turns import TURN_ORDER
from ..live.session_store import read_json_lines
from ..live.suit_correction import SuitCorrectionTracker
from ..recognition_service import ScreenshotRecognitionService
from ..template_service import TemplateService
from .single_image_danzero_page import CardBadge

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
    log_published = Signal(object)
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
        profiles_root: Path | None = None,
        profile_name: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = Path(session)
        self.truth_log = truth_log
        self._frame_provider = frame_provider
        self._frame_scan_provider = frame_scan_provider
        self._recognition_service = recognition_service
        self._profiles_root = (
            Path(profiles_root).expanduser().resolve()
            if profiles_root is not None
            else PROFILES_ROOT
        )
        self._profile_name = str(profile_name or "tencent_daguandan")
        self._revision_store = TruthRevisionStore(self.session)
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
            self.clear_button = QPushButton("清空")
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
        self.edit_hand_button = QPushButton("改手牌")
        info_row.addWidget(self.edit_hand_button)
        self.recognize_hand_button = QPushButton("识别手牌")
        info_row.addWidget(self.recognize_hand_button)
        layout.addLayout(info_row)
        self._hand: tuple[str, ...] = tuple(truth_log.initial_state.my_hand)
        self._hand_badges: list[CardBadge] = []
        self._render_hand_badges()
        self.edit_hand_button.clicked.connect(self._edit_hand)
        self.recognize_hand_button.clicked.connect(self._recognize_hand)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ("序号", "玩家", "牌面", "牌墩")
        )
        header = self.table.horizontalHeader()
        for column in (0, 1, 3):
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        for column in (2,):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        self.add_button = QPushButton("新增")
        self.insert_button = QPushButton("插入")
        self.remove_button = QPushButton("删除")
        self.recognize_frame_button = QPushButton("识别本帧")
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
        # One primary save flow: strict validation publishes a verified log;
        # a validation failure offers an explicit draft-only escape hatch in
        # the same dialog, so the main window never grows a second save button.
        self.save_button = PrimaryPushButton("保存 TruthLog")
        self.save_button.setToolTip(
            "默认严格校验并保存为可信 TruthLog；校验失败时可选择强制保存草稿"
        )
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        # Compatibility alias for integrations that used to inspect the old
        # publish button. It is intentionally not a second visible control.
        self.publish_button = self.save_button
        self.end_status = QLabel("对局状态：进行中")
        self.end_status.setWordWrap(True)
        layout.addWidget(self.end_status)
        self.save_status = QLabel(
            "未保存（默认严格校验后保存为可信 TruthLog；校验失败时可强制保存草稿）"
        )
        self.save_status.setWordWrap(True)
        layout.addWidget(self.save_status)
        self.add_button.clicked.connect(self.add_row)
        self.insert_button.clicked.connect(self.insert_row)
        self.remove_button.clicked.connect(self.remove_row)
        self.recognize_frame_button.clicked.connect(self._recognize_frame)
        self.save_button.clicked.connect(self._save_truth_log)
        self.table.cellChanged.connect(self._cell_changed)
        self._render()
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
        # Do not even inspect a legacy TruthLog while loading a pure AVI scan
        # draft. The draft must remain independent until the user explicitly
        # chooses to save or publish it.
        if self.truth_log.provenance.source == "video_scan":
            return False
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

        # Card-count placements are self-contained TruthLog evidence and work
        # for pure AVI scans too. Timeline badges are an optional secondary
        # source only when the card ledger cannot establish an order.
        derived = project_truth_log_placements(truth_log)
        if derived:
            return derived
        if truth_log.provenance.source == "video_scan":
            return ()
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

    def replace_confirmed_turn(
        self,
        turn: TruthTurn,
        *,
        status: str = "花色修正已回填",
    ) -> int:
        """Replace a scanned draft row without creating a second action."""

        row: int | None = None
        for index in range(self.table.rowCount()):
            number_item = self.table.item(index, 0)
            original = (
                number_item.data(int(Qt.ItemDataRole.UserRole) + 1)
                if number_item is not None
                else None
            )
            if isinstance(original, TruthTurn) and original.index == turn.index:
                row = index
                break
        if row is None:
            raise ValueError(f"找不到第 {turn.index} 条待回填的扫描草稿")
        self._replace_row(row, turn, status=status)
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
        # An inserted blank row has no action kind yet, so the following row
        # cannot reliably be reverse-derived.  Use the sole reliable fact:
        # the actor expected after the prefix before this row.  The complete
        # chain is still validated on Save once the row has been filled in.
        actor = self._expected_player_at(row) or "self"
        self._append_row(
            TruthTurn(self.table.rowCount() + 1, actor, False, ()),
            row=row,
            editable=True,
            status="待编辑",
        )
        self._resize_table()
        self.table.selectRow(row)
        self._restore_table_scroll(scroll)

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

    def _build_log(self, *, validate_actor_chain: bool = True) -> TruthLog:
        turns: list[TruthTurn] = []
        for row in range(self.table.rowCount()):
            player = self.table.cellWidget(row, 1)
            if (
                validate_actor_chain
                and player is not None
                and bool(player.property("sequenceConflict"))
            ):
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
                    move_semantics=original.move_semantics if unchanged else None,
                )
            )
        lead = self.lead_combo.currentData()
        round_level = self.round_level_combo.currentData()
        hand = _sort_hand_cards(self._hand, round_level)
        if not hand:
            raise ValueError("我方手牌不能为空")
        finish_order = derive_finish_order(
            turns,
            initial_hand_size=len(hand),
        )
        outcome = TruthOutcome(
            complete=False,
            finish_order=finish_order,
            team_result="unknown",
            reward=None,
            reward_scheme="",
        ) if finish_order else self.truth_log.outcome
        seat_hand_sizes = dict(self.truth_log.initial_state.seat_hand_sizes)
        if seat_hand_sizes:
            seat_hand_sizes["self"] = len(hand)
        log = TruthLog(
            source_session_id=self.truth_log.source_session_id,
            initial_state=TruthInitialState(
                str(round_level),
                lead,
                hand,
                tuple(sorted(seat_hand_sizes.items())),
            ),
            turns=tuple(turns),
            source_video=self.truth_log.source_video,
            frame_index_path=self.truth_log.frame_index_path,
            label_status=self.truth_log.label_status,
            provenance=self.truth_log.provenance,
            outcome=outcome,
        )
        # ``trick_id`` is display/provenance data in the editor.  Saving must
        # instead re-derive the complete action chain so a stale id cannot
        # conceal a direct seat jump (for example left -> right, skipping
        # self).  The validator also skips players whose recorded cards are
        # exhausted, matching the reducer's live turn ownership.
        if validate_actor_chain:
            validate_turn_actor_chain(log)
        return log

    def _recognition(self) -> ScreenshotRecognitionService:
        if self._recognition_service is None:
            profile_root = self._profiles_root
            profile_name = self._profile_name
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
            size_map = dict(self.truth_log.initial_state.seat_hand_sizes)
            start = len(self._hand) if player == "self" else int(size_map.get(player, _STARTING_CARDS))
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

    def _current_initial_state(self) -> TruthInitialState:
        sizes = dict(self.truth_log.initial_state.seat_hand_sizes)
        if sizes:
            sizes["self"] = len(self._hand)
        return TruthInitialState(
            str(self.round_level_combo.currentData()),
            str(self.lead_combo.currentData()),
            tuple(self._hand),
            tuple(sorted(sizes.items())),
        )

    def _prefix_turns(self, stop_row: int) -> tuple[TruthTurn, ...]:
        """Read the editable table prefix without trusting ``trick_id``."""

        turns: list[TruthTurn] = []
        for row in range(max(0, min(stop_row, self.table.rowCount()))):
            actor = self._player_at(row)
            if actor is None:
                raise ValueError(f"第 {row + 1} 条动作的玩家无效")
            is_pass = self._is_pass_from_row(row)
            turns.append(
                TruthTurn(
                    row + 1,
                    actor,
                    is_pass,
                    () if is_pass else self._cards_from_row(row),
                )
            )
        return tuple(turns)

    def _expected_player_at(self, row: int) -> str | None:
        row = max(0, min(row, self.table.rowCount()))
        try:
            return next_actor_after_prefix(
                self._current_initial_state(),
                self._prefix_turns(row),
            )
        except ValueError:
            return None

    def _expected_next_player(self) -> str | None:
        return self._expected_player_at(self.table.rowCount())

    def _recognition_candidate(
        self,
        image: np.ndarray,
        actor: str,
        *,
        allow_full_fallback: bool,
    ) -> tuple[_FrameRecognitionCandidate | None, str]:
        recognition = self._recognition()
        region_result = recognition.recognize_play_region(
            image,
            actor,
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

        if not allow_full_fallback:
            return (
                None,
                f"{_SEAT_LABELS.get(actor, actor)}区域未识别到动作",
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
        selected_actor: str | None = None,
    ) -> str:
        expected = self._expected_player_at(target_row)
        if selected_actor is None and expected is None:
            raise ValueError("无法从当前位置前的有效记录推断玩家")
        if candidate.actor not in TURN_ORDER:
            raise ValueError("识别结果包含无效玩家")
        if selected_actor is not None and candidate.actor != selected_actor:
            raise ValueError(
                "识别玩家与选中行玩家不一致："
                f"选中{_SEAT_LABELS[selected_actor]}，"
                f"识别为{_SEAT_LABELS.get(candidate.actor, candidate.actor)}"
            )
        if selected_actor is None and candidate.actor != expected:
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

        if selected_actor is not None and candidate.actor != expected:
            if expected is None:
                return "玩家链在本行之前无法推导；已按选中玩家写入，保存前请修正"
            return (
                f"玩家链冲突：第 {target_row + 1} 行应为{_SEAT_LABELS[expected]}，"
                f"已按选中{_SEAT_LABELS[selected_actor]}写入；保存前请修正"
            )
        if not replacing or target_row + 1 >= self.table.rowCount():
            return ""
        replacement = TruthTurn(
            target_row + 1,
            candidate.actor,
            candidate.is_pass,
            candidate.cards,
        )
        expected_after = next_actor_after_prefix(
            self._current_initial_state(),
            (*self._prefix_turns(target_row), replacement),
        )
        actual_after = self._player_at(target_row + 1)
        if expected_after != actual_after:
            expected_label = _SEAT_LABELS.get(str(expected_after), "无")
            actual_label = _SEAT_LABELS.get(str(actual_after), "无")
            return (
                f"下一行待修正：识别后应轮到{expected_label}，"
                f"第 {target_row + 2} 行当前为{actual_label}"
            )
        return ""

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
        expected = self._expected_player_at(target_row)
        selected_actor = self._player_at(target_row) if replacing else None
        if replacing and selected_actor is None:
            self.recognition_hint.setText(
                "识别当前画面：选中行的玩家无效；未写入"
            )
            return
        original = (
            self.table.item(target_row, 0).data(
                int(Qt.ItemDataRole.UserRole) + 1
            )
            if replacing and self.table.item(target_row, 0) is not None
            else None
        )
        if (
            replacing
            and isinstance(original, TruthTurn)
            and original.actor == selected_actor
            and self._correct_unknown_suit_row(target_row, frame_index, image)
        ):
            return
        actor = selected_actor if replacing else expected
        if actor is None:
            self.recognition_hint.setText(
                "识别当前画面：无法从当前位置前的有效记录推断玩家；未写入"
            )
            return
        try:
            candidate, source_note = self._recognition_candidate(
                image,
                actor,
                allow_full_fallback=not replacing,
            )
            if candidate is None:
                self.recognition_hint.setText(
                    f"识别当前画面：{source_note}；未写入"
                )
                return
            downstream_warning = self._validate_recognition_context(
                candidate,
                target_row=target_row,
                replacing=replacing,
                selected_actor=selected_actor,
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
                + (f"；{downstream_warning}" if downstream_warning else "")
            )
        else:
            self.recognition_hint.setText(
                f"识别当前画面：已追加 1 条出牌记录（{source_note}）"
            )

    def _save_truth_log(self) -> None:
        """Strictly save a verified log, or explicitly retain an invalid draft.

        The default remains intentionally strict.  A user may choose draft-only
        persistence only after seeing the validation error; this never changes
        an action to make the sequence appear valid.
        """
        try:
            # Build without the actor-chain gate first so a simultaneous turn
            # error cannot route an impossible card inventory into forced-draft
            # saving before the hard physical check has run.
            log = self._build_log(validate_actor_chain=False)
            # Physical double-deck inventory is a hard save invariant.  Unlike
            # an actor-chain draft error, it must not be bypassed by the
            # "强制保存草稿" escape hatch.
            validate_truth_log_card_inventory(log)
            validate_turn_actor_chain(log)
            validate_truth_log_with_live_reducer(log)
        except TruthLogCardInventoryError as exc:
            self.save_status.setText(f"保存失败：{exc}")
            QLabel(str(exc), self).show()
            return
        except ValueError as exc:
            self._offer_forced_draft_save(str(exc))
            return
        except Exception as exc:
            self.save_status.setText(f"保存失败：{exc}")
            QLabel(str(exc), self).show()
            return
        try:
            revision = self._revision_store.publish(
                log,
                author="truth_log_editor",
            )
            published = replace(log, label_status="verified")
            self.truth_log = published
            self.save_status.setText(
                f"已校验并保存可信 TruthLog：{revision.revision_id}（{len(published.turns)} 条）"
            )
            self.log_saved.emit(published)
            self.log_published.emit(published)
        except Exception as exc:
            self.save_status.setText(f"保存失败：{exc}")
            QLabel(str(exc), self).show()

    # Compatibility entrypoint retained for old callers/tests; it follows the
    # single-button save flow and does not create a separate draft button.
    def _save(self) -> None:
        self._save_truth_log()

    def _publish(self) -> None:
        self._save_truth_log()

    def _offer_forced_draft_save(self, validation_error: str) -> None:
        choice = QMessageBox.question(
            self,
            "TruthLog 严格校验未通过",
            "严格校验未通过，当前内容尚不能发布为可信 TruthLog。\n\n"
            f"{validation_error}\n\n"
            "校验包含玩家顺序和 LiveReducer 全量回放。\n"
            "是否强制保存为草稿？草稿会保留全部人工编辑和校验错误，"
            "但不会被标记为可信，也不能用于正式训练或发布。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice != QMessageBox.StandardButton.Yes:
            self.save_status.setText(f"未保存：{validation_error}")
            return
        try:
            log = self._build_log(validate_actor_chain=False)
            validation_kind = (
                "live_reducer_replay_failed"
                if "LiveReducer 全量回放失败" in validation_error
                else "actor_chain_validation_failed"
            )
            revision = self._revision_store.save_draft(
                log,
                author="truth_log_editor:forced_draft",
                changed_fields=(
                    "forced_draft",
                    validation_kind,
                    f"validation_error:{validation_error}",
                ),
            )
            audit_path = self.session / "truth_revisions" / f"{revision.revision_id}.validation.json"
            atomic_write_json(
                audit_path,
                {
                    "schema": "guandan.truth-draft-validation/1",
                    "revision_id": revision.revision_id,
                    "label_status": "draft",
                    "forced": True,
                    "validation_error": validation_error,
                },
            )
            saved = replace(log, label_status="draft")
            self.truth_log = saved
            self.save_status.setText(
                f"已强制保存草稿 {revision.revision_id}（{len(saved.turns)} 条）；"
                "动作链仍待修复，未发布为可信 TruthLog"
            )
            self.log_saved.emit(saved)
        except Exception as exc:
            self.save_status.setText(f"强制保存草稿失败：{exc}")
            QLabel(str(exc), self).show()

    def _draft_mutated(self, *_args) -> None:
        self._clear_placement_badges()
        self.draft_changed.emit()

    def shutdown(self) -> None:
        """Compatibility hook; the editor no longer owns evaluation workers."""
        return
