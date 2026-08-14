from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.truth_log_editor import (
    CardPickerDialog,
    TruthLogEditor,
    _sort_hand_cards,
)
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn
from daguandan_bridge.recognition_service import (
    PlayRegionResult,
    RecognitionResult,
    RecognizedEvent,
)

HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _app():
    return QApplication.instance() or QApplication([])


def _log() -> TruthLog:
    return TruthLog("game", TruthInitialState("2", "self", HAND), ())


def _fake_result(**overrides) -> RecognitionResult:
    fields = dict(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="self",
        my_hand=("2S", "3H"),
        events=(),
        field_confidences={},
        sources={},
        unresolved_fields=(),
        diagnostics=(),
    )
    fields.update(overrides)
    return RecognitionResult(**fields)


class _FakeRecognition:
    def __init__(self, result, region_result=None):
        self.result = result
        self.region_result = region_result
        self.calls = []

    def recognize(self, image):
        self.calls.append(("recognize",))
        return self.result

    def recognize_play_region(self, image, seat, *, wild_rank, allow_unknown_suit=False):
        self.calls.append(("region", seat, wild_rank))
        if self.region_result is None:
            return None
        return self.region_result


def test_card_picker_uses_chinese_labels_and_double_deck_limit():
    _app()
    picker = CardPickerDialog()

    picker._select_rank("5")
    picker._select_suit("H")
    picker._select_suit("H")
    picker._select_suit("H")

    assert picker.cards() == ("5H", "5H")
    assert picker.selected.item(0).text().startswith("红桃5")
    assert "5H" not in picker.selected.item(0).text()


def test_editor_keeps_log_columns_compact_and_reorders_rows(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    assert [editor.table.horizontalHeaderItem(index).text() for index in range(4)] == [
        "序号", "玩家", "牌面", "牌墩"
    ]
    editor.add_row()
    editor.add_row()
    editor.table.selectRow(0)
    editor.remove_row()

    assert editor.table.rowCount() == 1
    assert editor.table.item(0, 0).text() == "1"


def test_editor_exposes_turn_metadata(tmp_path):
    _app()
    log = TruthLog(
        "game",
        TruthInitialState("2", "left", HAND),
        (
            TruthTurn(1, "left", False, ("3D",), frame_index=12, trick_id=1),
            TruthTurn(2, "self", False, ("4C",), frame_index=26, trick_id=1),
        ),
    )
    editor = TruthLogEditor(tmp_path, log)

    assert [
        editor.table.horizontalHeaderItem(index).text() for index in range(5)
    ] == ["序号", "玩家", "牌面", "牌墩", "模型推荐"]
    assert editor.table.item(1, 3).text() == "1"
    assert editor.table.item(1, 0).data(Qt.ItemDataRole.UserRole) == 26
    assert editor.table.cellWidget(0, 4) is None
    assert editor.table.cellWidget(1, 4) is None


def test_editor_shows_recorded_finish_rank_on_last_play(tmp_path):
    (tmp_path / "timeline.jsonl").write_text(
        json.dumps(
            {
                "event_type": "player_finished",
                "actor": "left",
                "turn_id": 2,
                "payload": {"placement": "head"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log = TruthLog(
        "game",
        TruthInitialState("2", "left", HAND),
        (
            TruthTurn(1, "left", False, ("3D",), trick_id=1),
            TruthTurn(2, "self", False, ("4C",), trick_id=1),
        ),
    )

    editor = TruthLogEditor(tmp_path, log)

    assert "1 左家·头游" in editor.placement_summary.text()
    assert editor._placement_badges_by_turn == {1: "头游"}
    editor.table.cellWidget(1, 1).setCurrentIndex(
        editor.table.cellWidget(1, 1).findData("right")
    )
    assert editor._placement_badges_by_turn == {}
    assert "1 左家·头游" in editor.placement_summary.text()


def test_append_confirmed_turn_stays_unsaved_without_status_column(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    row = editor.append_confirmed_turn(
        TruthTurn(1, "self", False, ("2S",), frame_index=18, trick_id=1)
    )

    assert row == 0
    assert editor.table.columnCount() == 5
    assert not (tmp_path / "truth_log.json").exists()


def test_insert_row_before_selection_and_renumbers(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "self", False, ("2S",)))
    editor._append_row(TruthTurn(2, "opposite", True, ()))

    editor.table.selectRow(1)
    editor.insert_row()
    editor._replace_row(
        1,
        TruthTurn(2, "right", True, ()),
        status="测试不出",
    )

    assert editor.table.rowCount() == 3
    assert [editor.table.item(r, 0).text() for r in range(3)] == ["1", "2", "3"]
    assert editor.table.cellWidget(1, 1).currentData() == "right"
    assert editor._cards_from_row(1) == ()
    assert editor._is_pass_from_row(1)

    log = editor._build_log()
    assert log.turns[0].cards == ("2S",)
    assert log.turns[1].index == 2
    assert log.turns[1].actor == "right"
    assert log.turns[1].is_pass is True
    assert log.turns[2].actor == "opposite"
    assert log.turns[2].is_pass is True


def test_editor_table_stretches_and_empty_play_prompts_for_cards(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    editor.add_row()

    table_index = editor.layout().indexOf(editor.table)
    placeholder = editor.table.cellWidget(0, 2).layout().itemAt(0).widget()
    assert editor.layout().stretch(table_index) == 1
    assert placeholder.text() == "等待画面识别…"
    assert not hasattr(editor, "action_combo")


def test_log_card_picker_control_is_removed_but_hand_picker_remains(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    assert not hasattr(editor, "cards_button")
    assert not hasattr(editor, "choose_cards")
    picker = CardPickerDialog(editor._hand)
    assert picker.cards() == editor._hand


def test_resize_table_preserves_scroll_position_and_widget_lookup(tmp_path):
    app = _app()
    editor = TruthLogEditor(tmp_path, _log())
    editor.table.setFixedHeight(120)
    for index in range(24):
        actor = ("self", "right", "opposite", "left")[index % 4]
        editor._append_row(TruthTurn(index + 1, actor, True, ()))
    editor.show()
    app.processEvents()
    bar = editor.table.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(0)

    editor._resize_table()

    assert bar.value() == 0
    target = editor.table.cellWidget(0, 2)
    assert editor._row_for_widget(target) == 0


def test_log_combo_wheel_does_not_change_selected_values(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())
    editor.add_row()
    player = editor.table.cellWidget(0, 1)
    editor.lead_combo.setCurrentIndex(editor.lead_combo.findData("self"))
    editor.round_level_combo.setCurrentIndex(editor.round_level_combo.findData("2"))
    player.setCurrentIndex(player.findData("self"))
    event = QWheelEvent(
        QPointF(4, 4),
        QPointF(4, 4),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    for combo in (editor.lead_combo, editor.round_level_combo, player):
        combo.wheelEvent(event)

    assert editor.lead_combo.currentData() == "self"
    assert editor.round_level_combo.currentData() == "2"
    assert player.currentData() == "self"


def test_selected_row_recognition_atomically_replaces_only_that_row(tmp_path):
    app = _app()
    log = TruthLog(
        "game",
        TruthInitialState("2", "self", HAND),
        (
            TruthTurn(1, "self", False, ("3S",), frame_index=1),
            TruthTurn(2, "right", True, (), frame_index=2),
            TruthTurn(3, "opposite", False, ("4H",), frame_index=3),
        ),
    )
    region = PlayRegionResult(
        player="right",
        cards=("5?", "5H"),
        is_pass=False,
        confidence=0.87,
        diagnostics=(),
        annotations=(),
        source="template:right-play",
    )
    editor = TruthLogEditor(
        tmp_path,
        log,
        frame_provider=lambda: (44, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(_fake_result(), region_result=region),
    )
    editor.table.setFixedHeight(80)
    editor.show()
    app.processEvents()
    editor.table.selectRow(1)
    bar = editor.table.verticalScrollBar()
    bar.setValue(bar.maximum())
    scroll_before = bar.value()

    editor._recognize_frame()

    assert editor.table.rowCount() == 3
    assert editor.table.cellWidget(0, 1).currentData() == "self"
    assert editor._cards_from_row(0) == ("3S",)
    assert editor.table.cellWidget(1, 1).currentData() == "right"
    assert not editor._is_pass_from_row(1)
    assert editor._cards_from_row(1) == ("5?", "5H")
    assert editor.table.item(1, 0).data(Qt.ItemDataRole.UserRole) == 44
    assert editor.table.cellWidget(2, 1).currentData() == "opposite"
    assert editor._cards_from_row(2) == ("4H",)
    assert bar.value() == scroll_before
    assert "原子回填" in editor.recognition_hint.text()

    built = editor._build_log()
    replaced = built.turns[1]
    assert replaced.evidence.frame_indices == (44,)
    assert replaced.evidence.roi_name == "right_play"
    assert replaced.provenance.source == "template:right-play"
    assert replaced.provenance.confidence == 0.87
    assert replaced.uncertainty == ("unknown_suit",)


def test_cleared_selection_appends_even_when_current_row_remains(tmp_path):
    _app()
    editor = TruthLogEditor(
        tmp_path,
        TruthLog(
            "game",
            TruthInitialState("2", "self", HAND),
            (TruthTurn(1, "self", True, (), frame_index=1),),
        ),
        frame_provider=lambda: (9, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(
            _fake_result(),
            region_result=PlayRegionResult(
                player="right",
                cards=(),
                is_pass=True,
                confidence=0.91,
                diagnostics=(),
                annotations=(),
                source="template:right-pass",
            ),
        ),
    )
    editor.table.selectRow(0)
    editor.table.clearSelection()
    assert editor.table.currentRow() == 0
    assert editor._selected_rows() == []

    editor._recognize_frame()

    assert editor.table.rowCount() == 2
    assert editor.table.cellWidget(0, 1).currentData() == "self"
    assert editor.table.cellWidget(1, 1).currentData() == "right"
    assert editor.table.item(1, 0).data(Qt.ItemDataRole.UserRole) == 9
    assert "已追加 1 条" in editor.recognition_hint.text()


def test_editor_rejects_multiple_recognized_plays_from_frame(tmp_path):
    _app()
    result = _fake_result(
        lead_player="left",
        events=(
            RecognizedEvent("left", ("3H", "3H"), False, 0.95, "test"),
            RecognizedEvent("self", (), True, 0.9, "test"),
        ),
    )
    editor = TruthLogEditor(
        tmp_path,
        _log(),
        frame_provider=lambda: (7, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(result),
    )

    editor._recognize_frame()

    assert editor.table.rowCount() == 0
    assert editor.lead_combo.currentData() == "self"
    assert "2 个候选" in editor.recognition_hint.text()
    assert "未写入" in editor.recognition_hint.text()
    assert editor._last_recognized_frame is None

    editor._recognize_frame()
    assert editor.table.rowCount() == 0


def test_editor_recognize_hand_updates_hand(tmp_path):
    _app()
    editor = TruthLogEditor(
        tmp_path,
        _log(),
        frame_provider=lambda: (1, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(
            _fake_result(my_hand=("small_joker", "2S", "3H"))
        ),
    )

    editor._recognize_hand()

    assert editor._hand == ("small_joker", "2S", "3H")
    assert [b.card_code for b in editor._hand_badges] == ["small_joker", "2S", "3H"]
    assert "已更新" in editor.recognition_hint.text()

    editor._frame_provider = lambda: (None, None)
    editor._recognize_hand()
    assert "播放或点『下一帧』" in editor.recognition_hint.text()


def test_editor_recognize_requires_frame(tmp_path):
    _app()
    editor = TruthLogEditor(
        tmp_path, _log(), frame_provider=lambda: (None, None)
    )

    editor._recognize_frame()

    assert "播放" in editor.recognition_hint.text()
    assert editor.table.rowCount() == 0


def test_editor_cards_column_uses_badges_and_round_trips(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    editor._append_row(
        TruthTurn(1, "left", False, ("3H", "3H", "2S", "10D", "big_joker"))
    )
    editor._append_row(TruthTurn(2, "self", True, ()))

    assert editor.table.cellWidget(0, 2) is not None
    assert editor.table.item(0, 2) is None
    assert editor._cards_from_row(0) == ("3H", "3H", "2S", "10D", "big_joker")
    assert editor._cards_from_row(1) == ()

    editor._replace_row(0, TruthTurn(1, "left", True, ()), status="测试不出")
    assert editor._cards_from_row(0) == ()


def test_recognize_frame_rejects_persisting_multiple_plays(tmp_path):
    _app()
    self_play = RecognizedEvent("self", ("4H", "4H"), False, 0.95, "test")
    right_play = RecognizedEvent("right", ("5S", "5S", "5S"), False, 0.9, "test")
    frame_one = _fake_result(events=(self_play,))
    frame_two = _fake_result(events=(self_play, right_play))
    editor = TruthLogEditor(
        tmp_path,
        _log(),
        frame_provider=lambda: (1, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(frame_one),
    )

    editor._recognize_frame()
    assert editor.table.rowCount() == 1

    editor._frame_provider = lambda: (2, np.zeros((32, 64, 3), np.uint8))
    editor._recognition_service = _FakeRecognition(frame_two)
    editor._recognize_frame()

    assert editor.table.rowCount() == 1
    assert editor._cards_from_row(0) == ("4H", "4H")
    assert "2 个候选" in editor.recognition_hint.text()
    assert "未写入" in editor.recognition_hint.text()


def test_recognize_frame_rejects_new_and_stale_pass_candidates_together(tmp_path):
    _app()
    # 用户场景：左家出完（rows=[left]）-> 轮到自己，点识别追加"自己不出"；
    # 换帧后再点（expected=右家），画面残留的"自己不出"被丢弃，
    # 右家新的"不出"正常追加。
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "left", False, ("4H", "4H")))
    self_pass = RecognizedEvent("self", (), True, 0.9, "test")
    right_pass = RecognizedEvent("right", (), True, 0.95, "test")

    editor._frame_provider = lambda: (1, np.zeros((32, 64, 3), np.uint8))
    editor._recognition_service = _FakeRecognition(
        _fake_result(),
        region_result=PlayRegionResult(
            player="self",
            cards=(),
            is_pass=True,
            confidence=0.9,
            diagnostics=(),
            annotations=(),
            source="region",
        ),
    )
    editor._recognize_frame()
    assert editor.table.rowCount() == 2

    editor._frame_provider = lambda: (2, np.zeros((32, 64, 3), np.uint8))
    editor._recognition_service = _FakeRecognition(
        _fake_result(events=(self_pass, right_pass))
    )
    editor._recognize_frame()

    assert editor.table.rowCount() == 2
    assert "2 个候选" in editor.recognition_hint.text()
    assert "未写入" in editor.recognition_hint.text()


def test_recognize_frame_drops_stale_pass_of_non_expected_player(tmp_path):
    _app()
    # 用户场景：左家出完 -> 自己不出一条 -> 画面里"自己不出"残留，
    # 换帧后再识别（expected=右家）不应重复追加自己不出。
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "left", False, ("4H", "4H")))
    editor._append_row(TruthTurn(2, "self", True, (), frame_index=1))
    self_pass = RecognizedEvent("self", (), True, 0.9, "test")
    editor._frame_provider = lambda: (2, np.zeros((32, 64, 3), np.uint8))
    editor._recognition_service = _FakeRecognition(
        _fake_result(events=(self_pass,))
    )

    editor._recognize_frame()

    assert editor.table.rowCount() == 2
    assert "上下文冲突" in editor.recognition_hint.text()
    assert "未写入" in editor.recognition_hint.text()


@pytest.mark.parametrize("case", ["failure", "empty", "multiple", "conflict"])
def test_recognition_non_committable_results_leave_all_editor_state_untouched(
    tmp_path,
    case,
):
    _app()

    class FailingRecognition:
        def recognize_play_region(self, *args, **kwargs):
            raise RuntimeError("matcher failed")

    result = _fake_result(events=())
    recognition = _FakeRecognition(result)
    if case == "failure":
        recognition = FailingRecognition()
    elif case == "multiple":
        recognition = _FakeRecognition(
            _fake_result(
                lead_player="left",
                events=(
                    RecognizedEvent("right", (), True, 0.9, "one"),
                    RecognizedEvent("opposite", ("5S",), False, 0.8, "two"),
                ),
            )
        )
    elif case == "conflict":
        recognition = _FakeRecognition(
            _fake_result(
                lead_player="left",
                events=(RecognizedEvent("left", (), True, 0.9, "stale"),),
            )
        )

    editor = TruthLogEditor(
        tmp_path,
        TruthLog(
            "game",
            TruthInitialState("2", "self", HAND),
            (TruthTurn(1, "self", True, (), frame_index=3),),
        ),
        frame_provider=lambda: (12, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=recognition,
    )
    before = editor._build_log().to_dict()
    lead_before = editor.lead_combo.currentData()
    scroll_before = editor._table_scroll_position()

    editor._recognize_frame()

    assert editor._build_log().to_dict() == before
    assert editor.lead_combo.currentData() == lead_before
    assert editor._table_scroll_position() == scroll_before
    assert editor._last_recognized_frame is None
    assert "未写入" in editor.recognition_hint.text()


def test_recognize_frame_targets_only_next_player_region(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "right", False, ("4H", "4H")))
    region_result = PlayRegionResult(
        player="opposite",
        cards=("5S", "5S", "5S"),
        is_pass=False,
        confidence=0.9,
        diagnostics=(),
        annotations=(),
        source="region",
    )
    fake = _FakeRecognition(_fake_result(), region_result=region_result)
    editor._recognition_service = fake
    editor._frame_provider = lambda: (2, np.zeros((32, 64, 3), np.uint8))

    editor._recognize_frame()

    assert editor.table.rowCount() == 2
    assert editor._cards_from_row(1) == ("5S", "5S", "5S")
    assert fake.calls == [("region", "opposite", "2")]
    assert "仅识别" in editor.recognition_hint.text()


def test_recognize_frame_falls_back_to_full_image_when_region_empty(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "right", False, ("4H", "4H")))
    full = _fake_result(
        events=(RecognizedEvent("opposite", ("6D", "6D"), False, 0.9, "test"),)
    )
    fake = _FakeRecognition(full)
    editor._recognition_service = fake
    editor._frame_provider = lambda: (2, np.zeros((32, 64, 3), np.uint8))

    editor._recognize_frame()

    assert editor.table.rowCount() == 2
    assert editor._cards_from_row(1) == ("6D", "6D")
    assert fake.calls == [("region", "opposite", "2"), ("recognize",)]


def test_expected_next_player_follows_turn_order(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    assert editor._expected_next_player() == "self"


def test_add_row_infers_player_from_tail_and_skips_finished_player(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    editor.add_row()
    assert editor.table.cellWidget(0, 1).currentData() == "self"
    editor._replace_row(0, TruthTurn(1, "self", True, ()), status="测试不出")
    editor._append_row(TruthTurn(2, "right", False, ("3S",) * 27))
    editor._append_row(TruthTurn(3, "opposite", True, ()))
    editor._append_row(TruthTurn(4, "left", True, ()))

    editor.add_row()

    assert editor.table.cellWidget(4, 1).currentData() == "self"


def test_insert_row_uses_only_prior_history_and_warns_on_conflict(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "self", True, ()))
    editor._append_row(TruthTurn(2, "opposite", True, ()))
    editor._append_row(TruthTurn(3, "left", True, ()))
    editor._append_row(TruthTurn(4, "self", True, ()))
    editor._append_row(TruthTurn(5, "right", False, ("3S",) * 27))

    editor.table.selectRow(1)
    editor.insert_row()

    inserted_player = editor.table.cellWidget(1, 1)
    assert inserted_player.currentData() == "right"
    assert not bool(inserted_player.property("sequenceConflict"))

    conflict_editor = TruthLogEditor(tmp_path, _log())
    conflict_editor._append_row(TruthTurn(1, "self", True, ()))
    conflict_editor._append_row(TruthTurn(2, "right", True, ()))
    conflict_editor.table.selectRow(1)
    conflict_editor.insert_row()

    conflict_player = conflict_editor.table.cellWidget(1, 1)
    assert bool(conflict_player.property("sequenceConflict"))
    assert "冲突" in conflict_editor.save_status.text()
    with pytest.raises(ValueError, match="冲突"):
        conflict_editor._build_log()

    conflict_player.setCurrentIndex(conflict_player.findData("opposite"))
    conflict_editor._replace_row(
        1,
        TruthTurn(2, "opposite", True, ()),
        status="人工确认",
    )
    assert not bool(conflict_editor.table.cellWidget(1, 1).property("sequenceConflict"))
    conflict_editor._build_log()
    for seat in ("self", "right", "opposite", "left"):
        editor._append_row(TruthTurn(editor.table.rowCount() + 1, seat, False, ()))
    assert editor._expected_next_player() == "self"


def test_out_player_is_skipped_in_turn_order(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    editor._append_row(TruthTurn(1, "self", False, ("2S", "2S")))
    editor._append_row(TruthTurn(2, "right", False, ("2S",) * 27))
    assert "right" in editor._out_players()

    # 对家出完前：左家之后轮到右家（但右家已出完 -> 跳到自己）
    editor._append_row(TruthTurn(3, "opposite", False, ("3S",)))
    assert editor._expected_next_player() == "left"
    editor._append_row(TruthTurn(4, "left", False, ("3S",)))
    assert editor._expected_next_player() == "self"

    # 右家出完后循环顺序：自己 -> 对家 -> 左家 -> 自己（跳过右家）
    editor._append_row(TruthTurn(5, "self", False, ("4S",)))
    assert editor._expected_next_player() == "opposite"
    editor._append_row(TruthTurn(6, "opposite", False, ("4S",)))
    assert editor._expected_next_player() == "left"
    editor._append_row(TruthTurn(7, "left", False, ("4S",)))
    assert editor._expected_next_player() == "self"


def test_out_player_skip_applies_to_region_recognition(tmp_path):
    _app()
    # 右家出完后，识别本帧应只查对家/左家区域，绝不查右家。
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "self", False, ("2S", "2S")))
    editor._append_row(TruthTurn(2, "right", False, ("2S",) * 27))
    editor._append_row(TruthTurn(3, "opposite", False, ("3S",)))
    editor._append_row(TruthTurn(4, "left", False, ("3S",)))
    editor._frame_provider = lambda: (5, np.zeros((32, 64, 3), np.uint8))
    fake = _FakeRecognition(_fake_result())
    editor._recognition_service = fake

    editor._recognize_frame()

    seats = [seat for call in fake.calls if call[0] == "region" for seat in [call[1]]]
    assert seats == ["self"]
    assert "right" not in seats


def test_end_status_marks_game_over_when_a_team_finishes(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())
    assert "进行中" in editor.end_status.text()

    editor._append_row(TruthTurn(1, "right", False, ("2S",) * 27))
    editor._append_row(TruthTurn(2, "left", False, ("3S",) * 27))
    editor._resize_table()
    assert "对方" in editor.end_status.text()
    assert "已结束" in editor.end_status.text()

    editor2 = TruthLogEditor(tmp_path, _log())
    editor2._append_row(TruthTurn(1, "self", False, ("2S",) * 27))
    editor2._append_row(TruthTurn(2, "opposite", False, ("3S",) * 27))
    editor2._resize_table()
    assert "己方" in editor2.end_status.text()
    assert "已结束" in editor2.end_status.text()

    editor3 = TruthLogEditor(tmp_path, _log())
    editor3._append_row(TruthTurn(1, "right", False, ("2S",) * 27))
    editor3._resize_table()
    assert "进行中" in editor3.end_status.text()


def test_unknown_suit_card_round_trips(tmp_path):
    _app()
    region_result = PlayRegionResult(
        player="right",
        cards=("5?", "7H"),
        is_pass=False,
        confidence=0.8,
        diagnostics=(),
        annotations=(),
        source="region",
    )
    editor = TruthLogEditor(
        tmp_path,
        _log(),
        frame_provider=lambda: (1, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(
            _fake_result(), region_result=region_result
        ),
    )
    editor._append_row(TruthTurn(1, "self", False, ("2S",)))

    editor._recognize_frame()

    assert editor.table.rowCount() == 2
    assert editor._cards_from_row(1) == ("5?", "7H")
    log = editor._build_log()
    assert log.turns[1].cards == ("5?", "7H")
    assert log.turns[1].evidence.frame_indices == (1,)
    assert log.turns[1].evidence.roi_name == "right_play"
    assert log.turns[1].provenance.source == "region"
    assert log.turns[1].provenance.confidence == 0.8
    assert log.turns[1].uncertainty == ("unknown_suit",)
    assert log.turns[1].label_status == "draft"
    editor._save()
    from daguandan_bridge.live.truth_log import load_truth_log

    loaded = load_truth_log(tmp_path / "truth_log.json", session_id="game")
    assert loaded.turns[1].cards == ("5?", "7H")
    assert loaded.turns[1].evidence.roi_name == "right_play"
    assert loaded.turns[1].provenance.confidence == 0.8
    assert loaded.turns[1].uncertainty == ("unknown_suit",)


def test_recognize_current_frame_corrects_selected_unknown_suit_after_two_reads(tmp_path):
    _app()
    frame_index = [40]
    region_result = PlayRegionResult(
        player="left",
        cards=("4C", "4H", "5C"),
        is_pass=False,
        confidence=0.91,
        diagnostics=(),
        annotations=(),
        source="left-play",
    )
    log = TruthLog(
        "game",
        TruthInitialState("2", "left", HAND),
        (
            TruthTurn(
                1,
                "left",
                False,
                ("4?", "4H", "5?"),
                frame_index=10,
                trick_id=1,
                uncertainty=("unknown_suit",),
            ),
            TruthTurn(2, "self", True, (), trick_id=1),
        ),
    )
    editor = TruthLogEditor(
        tmp_path,
        log,
        frame_provider=lambda: (frame_index[0], np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(_fake_result(), region_result=region_result),
    )
    editor.table.selectRow(0)

    editor._recognize_frame()

    assert editor._cards_from_row(0) == ("4?", "4H", "5?")
    frame_index[0] = 41
    editor._recognize_frame()

    assert editor._cards_from_row(0) == ("4C", "4H", "5C")
    corrected = editor._build_log().turns[0]
    assert corrected.uncertainty == ()
    assert corrected.provenance.source == "suit_correction:left-play"


def test_recognize_current_frame_scans_nearby_replay_frames_for_suit_correction(tmp_path):
    _app()
    frames = {
        40: np.full((4, 4, 3), 40, np.uint8),
        41: np.full((4, 4, 3), 41, np.uint8),
        42: np.full((4, 4, 3), 42, np.uint8),
    }

    class _SequenceRecognition:
        def recognize_play_region(self, image, seat, *, wild_rank, allow_unknown_suit=False):
            assert seat == "left"
            value = int(image[0, 0, 0])
            cards = (
                ("4H", "4H", "5?")
                if value == 40
                else ("4C", "4H", "5C")
            )
            return PlayRegionResult(
                player="left",
                cards=cards,
                is_pass=False,
                confidence=0.91,
                diagnostics=(),
                annotations=(),
                source="replay-scan",
            )

    scans: list[int] = []
    log = TruthLog(
        "game",
        TruthInitialState("2", "left", HAND),
        (
            TruthTurn(
                1,
                "left",
                False,
                ("4?", "4H", "5?"),
                frame_index=10,
                trick_id=1,
                uncertainty=("unknown_suit",),
            ),
            TruthTurn(2, "self", True, (), trick_id=1),
        ),
    )
    editor = TruthLogEditor(
        tmp_path,
        log,
        frame_provider=lambda: (40, frames[40]),
        frame_scan_provider=lambda start: (
            scans.append(start) or ((41, frames[41]), (42, frames[42]))
        ),
        recognition_service=_SequenceRecognition(),
    )
    editor.table.selectRow(0)

    editor._recognize_frame()

    assert scans == [40]
    assert editor._cards_from_row(0) == ("4C", "4H", "5C")
    corrected = editor._build_log().turns[0]
    assert corrected.evidence.frame_indices == (10, 41, 42)
    assert corrected.uncertainty == ()
    assert corrected.provenance.source == "suit_correction:replay-scan"
    assert "后续帧确认" in editor.recognition_hint.text()


def test_card_text_code_round_trips_unknown_suit():
    from daguandan_bridge.live.truth_log import (
        card_code_to_text,
        card_text_to_code,
    )

    assert card_text_to_code("5?") == "5?"
    assert card_text_to_code("5？") == "5?"
    assert card_text_to_code(card_code_to_text("5?")) == "5?"


def test_editor_shows_and_saves_my_hand(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    assert editor.round_level_combo.currentData() == "2"
    assert editor._hand == HAND
    assert [badge.card_code for badge in editor._hand_badges] == list(
        _sort_hand_cards(HAND, "2")
    )

    editor._set_hand(("3H", "2S", "10D"))
    log = editor._build_log()
    # 级牌为 2：2S（级牌）> 10D > 3H
    assert log.initial_state.my_hand == ("2S", "10D", "3H")
    assert [badge.card_code for badge in editor._hand_badges] == ["2S", "10D", "3H"]

    editor._set_hand(())
    with pytest.raises(ValueError):
        editor._build_log()


def test_hand_sorted_by_guandan_strength_with_level_first():
    from daguandan_bridge.gui.truth_log_editor import _sort_hand_cards

    hand = ("2S", "10C", "AH", "big_joker", "KH", "3D", "small_joker", "10D")
    # 级牌 10：大王 > 小王 > 级牌(10S/10H/10C/10D) > A > K > 3 > 2
    sorted_hand = _sort_hand_cards(hand, "10")
    assert sorted_hand == (
        "big_joker",
        "small_joker",
        "10C",
        "10D",
        "AH",
        "KH",
        "3D",
        "2S",
    )
    # 不传级牌时保持默认（无级牌提升）
    assert _sort_hand_cards(("10C", "AH"), None) == ("AH", "10C")


def test_hand_sorted_same_rank_suit_order_spade_first():
    from daguandan_bridge.gui.truth_log_editor import _sort_hand_cards

    assert _sort_hand_cards(("10D", "10H", "10S", "10C"), "10") == (
        "10S",
        "10H",
        "10C",
        "10D",
    )


def test_rerun_same_frame_appends_missed_event_only(tmp_path):
    _app()
    self_joker = RecognizedEvent("self", ("small_joker",), False, 0.9, "test")
    editor = TruthLogEditor(
        tmp_path,
        _log(),
        frame_provider=lambda: (5, np.zeros((32, 64, 3), np.uint8)),
        recognition_service=_FakeRecognition(_fake_result(events=())),
    )

    editor._recognize_frame()
    assert editor.table.rowCount() == 0

    editor._recognition_service = _FakeRecognition(
        _fake_result(events=(self_joker,))
    )
    editor._recognize_frame()

    assert editor.table.rowCount() == 1
    assert editor._cards_from_row(0) == ("small_joker",)
    assert editor._last_recognized_frame == 5

    editor._recognize_frame()
    assert editor.table.rowCount() == 1


def test_editor_shows_resume_banner_and_can_clear(tmp_path):
    _app()
    log = TruthLog(
        "game",
        TruthInitialState("2", "self", HAND),
        (TruthTurn(1, "self", False, ("2S", "2S")),),
    )
    editor = TruthLogEditor(tmp_path, log)

    assert editor.resume_banner is not None
    assert editor.table.rowCount() == 1
    editor._clear_all_rows()
    assert editor.table.rowCount() == 0
    assert not editor.clear_button.isEnabled()

    fresh = TruthLogEditor(tmp_path, _log())
    assert fresh.resume_banner is None
    assert fresh.clear_button is None
