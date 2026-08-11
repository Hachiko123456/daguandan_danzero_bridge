from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
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
from ..danzero.state import RANKS, SEATS, SUITS
from ..live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    card_code_to_text,
    save_truth_log,
)
from ..domain.truth import LabelProvenance
from ..live.turns import TURN_ORDER
from ..recognition_service import ScreenshotRecognitionService
from ..template_service import TemplateService
from .single_image_danzero_page import CardBadge

_SEAT_LABELS = {"self": "自己", "right": "右家", "opposite": "对家", "left": "左家"}
_RANK_LABELS = {rank: rank for rank in RANKS}
_SUIT_LABELS = {"S": "黑桃", "H": "红桃", "C": "梅花", "D": "方块"}
_STARTING_CARDS = 27

_SUIT_ORDER = {"S": 0, "H": 1, "C": 2, "D": 3}
# 逆序展示时同点数按 黑桃 > 红桃 > 梅花 > 方块。
_SUIT_ORDER_DESC = {"S": 3, "H": 2, "C": 1, "D": 0}
# 掼蛋牌力（逆序，大在前）：大王 > 小王 > 级牌 > A > K > … > 3 > 2。
# 级牌按对局轮次传入（如级牌 10，则 10 仅次于大小王）。
_JOKER_STRENGTH = {"big_joker": 18, "small_joker": 17}
_RANK_STRENGTH = {
    "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9,
    "9": 10, "10": 11, "J": 12, "Q": 13, "K": 14, "A": 15,
}


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


class TruthLogEditor(QWidget):
    log_saved = Signal(object)

    def __init__(
        self,
        session: Path,
        truth_log: TruthLog,
        parent=None,
        *,
        frame_provider: Callable[[], tuple[int | None, np.ndarray | None]]
        | None = None,
        recognition_service: ScreenshotRecognitionService | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = Path(session)
        self.truth_log = truth_log
        self._frame_provider = frame_provider
        self._recognition_service = recognition_service
        self._last_recognized_frame: int | None = None
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
        lead_row = QHBoxLayout()
        lead_row.addWidget(QLabel("首出玩家"))
        self.lead_combo = QComboBox()
        for seat in SEATS:
            self.lead_combo.addItem(_SEAT_LABELS[seat], userData=seat)
        self.lead_combo.setCurrentIndex(max(0, self.lead_combo.findData(truth_log.initial_state.lead_player)))
        lead_row.addWidget(self.lead_combo)
        lead_row.addStretch(1)
        layout.addLayout(lead_row)
        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("当前级牌"))
        self.round_level_combo = QComboBox()
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
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("序号", "玩家", "动作", "牌面"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        self.add_button = QPushButton("新增一行")
        self.insert_button = QPushButton("插入行")
        self.remove_button = QPushButton("删除选中行")
        self.cards_button = QPushButton("选择牌面")
        self.recognize_frame_button = QPushButton("识别本帧")
        controls.addWidget(self.add_button)
        controls.addWidget(self.insert_button)
        controls.addWidget(self.remove_button)
        controls.addWidget(self.cards_button)
        controls.addWidget(self.recognize_frame_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.recognition_hint = QLabel("识别本帧：把回放停到目标画面后点此按钮")
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
        self.cards_button.clicked.connect(self.choose_cards)
        self.recognize_frame_button.clicked.connect(self._recognize_frame)
        self.save_button.clicked.connect(self._save)
        self.table.cellChanged.connect(self._cell_changed)
        self._render()

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
                "识别手牌：请先在回放页播放或单帧到开局画面"
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

    def _render(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for turn in self.truth_log.turns:
            self._append_row(turn)
        self.table.blockSignals(False)
        self._resize_table()

    @staticmethod
    def _card_strip_widget(cards: tuple[str, ...]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        if not cards:
            label = QLabel("不出")
            layout.addWidget(label)
        else:
            for card in cards:
                layout.addWidget(CardBadge(card))
        layout.addStretch(1)
        return widget

    def _resize_table(self) -> None:
        self.table.resizeRowsToContents()
        self.table.resizeColumnToContents(3)
        self.table.scrollToBottom()
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
    ) -> None:
        if row is None:
            row = self.table.rowCount()
        self.table.insertRow(row)
        number_item = QTableWidgetItem(str(row + 1))
        if turn.frame_index is not None:
            number_item.setData(Qt.ItemDataRole.UserRole, turn.frame_index)
        number_item.setData(int(Qt.ItemDataRole.UserRole) + 1, turn)
        self.table.setItem(row, 0, number_item)
        player = QComboBox()
        for seat in SEATS:
            player.addItem(_SEAT_LABELS[seat], userData=seat)
        player.setCurrentIndex(max(0, player.findData(turn.actor)))
        if not editable:
            player.setEnabled(False)
        self.table.setCellWidget(row, 1, player)
        action = QComboBox()
        action.addItems(("出牌", "不出"))
        action.setCurrentIndex(1 if turn.is_pass else 0)
        if not editable:
            action.setEnabled(False)
        action.currentIndexChanged.connect(lambda _index, r=row: self._action_changed(r))
        self.table.setCellWidget(row, 2, action)
        self.table.setCellWidget(row, 3, self._card_strip_widget(turn.cards))
        self._renumber_rows()

    def _renumber_rows(self) -> None:
        for index in range(self.table.rowCount()):
            item = self.table.item(index, 0)
            if item is not None:
                item.setText(str(index + 1))

    def add_row(self) -> None:
        self._append_row(
            TruthTurn(self.table.rowCount() + 1, "self", False, ()),
            editable=True,
        )
        self.table.selectRow(self.table.rowCount() - 1)
        self._resize_table()

    def insert_row(self) -> None:
        """在当前选中行之前插入一行；未选中则追加到末尾。"""
        row = self.table.currentRow()
        if row < 0:
            self.add_row()
            return
        self._append_row(
            TruthTurn(self.table.rowCount() + 1, "self", False, ()),
            row=row,
            editable=True,
        )
        self.table.selectRow(row)
        self._resize_table()

    def remove_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self._renumber_rows()
            self._resize_table()

    def _action_changed(self, row: int) -> None:
        combo = self.table.cellWidget(row, 2)
        if combo is not None and combo.currentIndex() == 1:
            self.table.setCellWidget(row, 3, self._card_strip_widget(()))
            self._resize_table()

    def _cell_changed(self, _row: int, _column: int) -> None:
        return

    def choose_cards(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        action = self.table.cellWidget(row, 2)
        if action is not None and action.currentIndex() == 1:
            return
        current = self._cards_from_row(row)
        dialog = CardPickerDialog(current, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.table.setCellWidget(row, 3, self._card_strip_widget(dialog.cards()))
            self._resize_table()

    def _cards_from_row(self, row: int) -> tuple[str, ...]:
        widget = self.table.cellWidget(row, 3)
        if widget is None or widget.layout() is None:
            return ()
        cards: list[str] = []
        layout = widget.layout()
        for index in range(layout.count()):
            badge = layout.itemAt(index).widget()
            if isinstance(badge, CardBadge):
                cards.append(badge.card_code)
        return tuple(cards)

    def _build_log(self) -> TruthLog:
        turns: list[TruthTurn] = []
        for row in range(self.table.rowCount()):
            player = self.table.cellWidget(row, 1)
            action = self.table.cellWidget(row, 2)
            actor = player.currentData()
            is_pass = action.currentIndex() == 1
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

    def _out_players(self) -> set[str]:
        """已出完（手牌打光）的玩家：累计出牌数达到起始手牌数。

        自己按编辑器里的初始手牌数计算，其他玩家按掼蛋初始 27 张计算。
        """
        played: dict[str, int] = {}
        for row in range(self.table.rowCount()):
            player = self.table.cellWidget(row, 1)
            action = self.table.cellWidget(row, 2)
            if player is None or action is None or action.currentIndex() == 1:
                continue
            player_name = str(player.currentData())
            played[player_name] = played.get(player_name, 0) + len(
                self._cards_from_row(row)
            )
        out: set[str] = set()
        for player in TURN_ORDER:
            start = len(self._hand) if player == "self" else _STARTING_CARDS
            if played.get(player, 0) >= start:
                out.add(player)
        return out

    def _expected_next_player(self) -> str | None:
        if self.table.rowCount() == 0:
            lead = self.lead_combo.currentData()
            return str(lead) if lead else None
        player = self.table.cellWidget(self.table.rowCount() - 1, 1)
        if player is None:
            return None
        actor = str(player.currentData())
        if actor not in TURN_ORDER:
            return None
        # 出完的玩家不再参与，轮转时跳过（如右家出完后：
        # 自己 -> 对家 -> 左家 -> 自己）。
        out = self._out_players()
        index = TURN_ORDER.index(actor)
        for offset in range(1, len(TURN_ORDER) + 1):
            seat = TURN_ORDER[(index + offset) % len(TURN_ORDER)]
            if seat not in out:
                return seat
        return None

    def _recognize_frame(self) -> None:
        if self._frame_provider is None:
            self.recognition_hint.setText(
                "识别本帧：当前编辑器未接入回放画面，请手动填写"
            )
            return
        frame_index, image = self._frame_provider()
        if image is None:
            self.recognition_hint.setText(
                "识别本帧：请先在回放页播放或单帧到目标画面"
            )
            return
        same_frame: set[tuple[str, bool, tuple[str, ...]]] = set()
        if frame_index is not None:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item is None or item.data(Qt.ItemDataRole.UserRole) != frame_index:
                    continue
                player = self.table.cellWidget(row, 1)
                action = self.table.cellWidget(row, 2)
                if player is None or action is None:
                    continue
                same_frame.add(
                    (
                        str(player.currentData()),
                        action.currentIndex() == 1,
                        self._cards_from_row(row),
                    )
                )
        pending: list[tuple[str, bool, tuple[str, ...]]] = []
        source_note = ""
        filtered_passes = 0
        try:
            recognition = self._recognition()
            expected = self._expected_next_player()
            if expected is not None:
                try:
                    region_result = recognition.recognize_play_region(
                        image,
                        expected,
                        wild_rank=self.truth_log.initial_state.round_level,
                        allow_unknown_suit=True,
                    )
                except Exception:
                    region_result = None
                if region_result is not None and (region_result.cards or region_result.is_pass):
                    pending.append(
                        (region_result.player, region_result.is_pass, region_result.cards)
                    )
                    source_note = f"（仅识别{_SEAT_LABELS[region_result.player]}区域）"
            if not pending:
                result = recognition.recognize(image)
                if result.lead_player in SEATS:
                    index = self.lead_combo.findData(result.lead_player)
                    if index >= 0:
                        self.lead_combo.setCurrentIndex(index)
                for event in result.events:
                    # "不出"标记会残留在画面上，只有轮到预期玩家时才算新动作；
                    # 其他玩家的不出是旧一轮的残留，直接丢弃。
                    if expected is not None and event.is_pass and event.player != expected:
                        filtered_passes += 1
                        continue
                    pending.append((event.player, event.is_pass, event.cards))
        except Exception as exc:
            self.recognition_hint.setText(f"识别本帧失败：{exc}")
            return
        self._last_recognized_frame = frame_index
        last_rows: dict[str, int] = {}
        for row in range(self.table.rowCount()):
            player = self.table.cellWidget(row, 1)
            if player is not None:
                last_rows[str(player.currentData())] = row
        added = 0
        skipped = 0
        for player, is_pass, cards in pending:
            signature = (player, is_pass, tuple(cards))
            if signature in same_frame:
                skipped += 1
                continue
            last_row = last_rows.get(player)
            if (
                not is_pass
                and last_row is not None
            ):
                action = self.table.cellWidget(last_row, 2)
                existing_pass = action is not None and action.currentIndex() == 1
                if existing_pass == is_pass and self._cards_from_row(last_row) == tuple(
                    cards
                ):
                    skipped += 1
                    continue
            self._append_row(
                TruthTurn(
                    self.table.rowCount() + 1,
                    player,
                    is_pass,
                    cards,
                    frame_index=frame_index,
                )
            )
            same_frame.add(signature)
            last_rows[player] = self.table.rowCount() - 1
            added += 1
        if added:
            self._resize_table()
            self.recognition_hint.setText(
                f"识别本帧：已追加 {added} 条出牌记录"
                + (
                    f"，忽略 {skipped + filtered_passes} 条已记录或残留的旧出牌"
                    if skipped + filtered_passes
                    else ""
                )
                + source_note
                + "，可手动修改"
            )
        else:
            self.recognition_hint.setText(
                "识别本帧：没有新的出牌记录"
                + (
                    f"（忽略 {skipped + filtered_passes} 条已记录或残留的旧出牌）"
                    if skipped + filtered_passes
                    else ""
                )
            )

    def _save(self) -> None:
        try:
            log = self._build_log()
            save_truth_log(self.session / "truth_log.json", log)
            self.truth_log = log
            self.save_status.setText(f"已保存 {len(log.turns)} 条，可继续编辑")
            self.log_saved.emit(log)
        except Exception as exc:
            QLabel(str(exc), self).show()
