from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
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


def test_editor_has_only_four_action_columns_and_reorders_rows(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())

    assert [editor.table.horizontalHeaderItem(index).text() for index in range(4)] == [
        "序号", "玩家", "动作", "牌面"
    ]
    editor.add_row()
    editor.add_row()
    editor.table.selectRow(0)
    editor.remove_row()

    assert editor.table.rowCount() == 1
    assert editor.table.item(0, 0).text() == "1"


def test_insert_row_before_selection_and_renumbers(tmp_path):
    _app()
    editor = TruthLogEditor(tmp_path, _log())
    editor._append_row(TruthTurn(1, "right", False, ("2S",)))
    editor._append_row(TruthTurn(2, "opposite", True, ()))

    editor.table.selectRow(1)
    editor.insert_row()
    editor.table.selectRow(1)
    editor.table.cellWidget(1, 1).setCurrentIndex(
        editor.table.cellWidget(1, 1).findData("self")
    )
    editor.table.cellWidget(1, 2).setCurrentIndex(1)

    assert editor.table.rowCount() == 3
    assert [editor.table.item(r, 0).text() for r in range(3)] == ["1", "2", "3"]
    assert editor._cards_from_row(1) == ()
    assert editor.table.cellWidget(1, 2).currentIndex() == 1

    log = editor._build_log()
    assert log.turns[0].cards == ("2S",)
    assert log.turns[1].index == 2
    assert log.turns[1].actor == "self"
    assert log.turns[1].is_pass is True
    assert log.turns[2].actor == "opposite"
    assert log.turns[2].is_pass is True


def test_editor_appends_recognized_plays_from_frame(tmp_path):
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

    assert editor.table.rowCount() == 2
    assert editor.lead_combo.currentData() == "left"
    assert "2" in editor.recognition_hint.text()
    assert editor._last_recognized_frame == 7

    editor._recognize_frame()
    assert editor.table.rowCount() == 2


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
    assert "播放或单帧" in editor.recognition_hint.text()


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

    assert editor.table.cellWidget(0, 3) is not None
    assert editor.table.item(0, 3) is None
    assert editor._cards_from_row(0) == ("3H", "3H", "2S", "10D", "big_joker")
    assert editor._cards_from_row(1) == ()

    editor.table.cellWidget(0, 2).setCurrentIndex(1)
    assert editor._cards_from_row(0) == ()


def test_recognize_frame_skips_persisting_previous_plays(tmp_path):
    _app()
    right_play = RecognizedEvent("right", ("4H", "4H"), False, 0.95, "test")
    opposite_play = RecognizedEvent("opposite", ("5S", "5S", "5S"), False, 0.9, "test")
    frame_one = _fake_result(events=(right_play,))
    frame_two = _fake_result(events=(right_play, opposite_play))
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

    assert editor.table.rowCount() == 2
    assert editor._cards_from_row(1) == ("5S", "5S", "5S")
    assert "忽略 1" in editor.recognition_hint.text()


def test_recognize_frame_keeps_new_pass_and_drops_stale_ones(tmp_path):
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

    assert editor.table.rowCount() == 3
    assert editor._cards_from_row(2) == ()
    assert "忽略 1" in editor.recognition_hint.text()


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
    assert "忽略" in editor.recognition_hint.text()


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
        player="left",
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
    editor._save()
    from daguandan_bridge.live.truth_log import load_truth_log

    loaded = load_truth_log(tmp_path / "truth_log.json", session_id="game")
    assert loaded.turns[1].cards == ("5?", "7H")


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
