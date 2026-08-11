from __future__ import annotations

import pytest
from copy import deepcopy

from daguandan_bridge.domain.truth import LabelProvenance, TruthEvidence

from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    card_code_to_text,
    card_text_to_code,
    load_truth_log,
    save_truth_log,
    truth_log_from_dict,
)

HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _log() -> TruthLog:
    return TruthLog(
        source_session_id="game-test",
        initial_state=TruthInitialState("2", "self", HAND),
        turns=(
            TruthTurn(1, "self", False, ("5H", "5S", "5C")),
            TruthTurn(2, "right", True, ()),
            TruthTurn(3, "opposite", True, ()),
            TruthTurn(4, "self", False, ("6D",)),
        ),
    )


def test_truth_log_round_trips_arbitrary_action_chain(tmp_path):
    path = tmp_path / "truth_log.json"
    save_truth_log(path, _log())

    loaded = load_truth_log(path, session_id="game-test")

    assert loaded == _log()
    assert loaded.to_dict()["schema_version"] == 3
    assert loaded.to_dict()["schema"] == "guandan.truth/3"
    assert [turn.actor for turn in loaded.turns] == ["self", "right", "opposite", "self"]


def test_truth_log_uses_chinese_card_names_and_limits_duplicates():
    assert card_code_to_text("5H") == "红桃5"
    assert card_code_to_text("big_joker") == "大王"
    assert card_text_to_code("黑桃5") == "5S"
    assert card_text_to_code("小王") == "small_joker"

    raw = _log().to_dict()
    raw["turns"][0]["cards"] = ["红桃5", "红桃5", "红桃5"]  # type: ignore[index]
    with pytest.raises(ValueError, match="不能超过两张"):
        truth_log_from_dict(raw)


def test_truth_log_rejects_cards_on_pass_and_wrong_session(tmp_path):
    path = tmp_path / "truth_log.json"
    save_truth_log(path, _log())
    with pytest.raises(ValueError, match="不属于"):
        load_truth_log(path, session_id="another-game")

    raw = _log().to_dict()
    raw["turns"][1]["cards"] = ["红桃5"]  # type: ignore[index]
    with pytest.raises(ValueError, match="不出动作"):
        truth_log_from_dict(raw)


def test_truth_log_migrates_schema_one_turns():
    raw = {
        "schema_version": 1,
        "source_session_id": "legacy",
        "source_video": {"path": "video/game.avi", "frame_index_path": "video/frame_index.jsonl"},
        "initial_state": {"round_level": "2", "lead_player": "self", "my_hand": list(HAND)},
        "turns": [
            {"turn_id": 1, "trick_id": 1, "actor": "self", "is_pass": False, "cards": ["5H"], "monotonic_ms": 100},
        ],
    }

    log = truth_log_from_dict(raw)

    assert log.turns[0].index == 1
    assert log.to_dict()["schema_version"] == 3


def test_schema_two_read_is_non_mutating_and_infers_real_trick_ids():
    raw = {
        "schema_version": 2,
        "source_session_id": "legacy-v2",
        "source_video": {"path": "video/game.avi", "frame_index_path": "video/frame_index.jsonl"},
        "initial_state": {"round_level": "2", "lead_player": "self", "my_hand": list(HAND)},
        "turns": [
            {"index": 1, "actor": "self", "is_pass": False, "cards": ["5H"]},
            {"index": 2, "actor": "right", "is_pass": True, "cards": []},
            {"index": 3, "actor": "opposite", "is_pass": True, "cards": []},
            {"index": 4, "actor": "left", "is_pass": True, "cards": []},
            {"index": 5, "actor": "self", "is_pass": False, "cards": ["6H"]},
        ],
    }
    original = deepcopy(raw)
    log = truth_log_from_dict(raw)
    assert raw == original
    assert [turn.trick_id for turn in log.turns] == [1, 1, 1, 1, 2]
    assert [event.trick_id for event in log.to_events()[1:]] == [1, 1, 1, 1, 2]


def test_truth_v3_round_trips_label_provenance_evidence_and_uncertainty():
    turn = TruthTurn(
        1,
        "left",
        False,
        ("5?",),
        trick_id=2,
        evidence=TruthEvidence((10, 11), 500, "left_play"),
        label_status="verified",
        provenance=LabelProvenance(source="human_review", annotator="a", confidence=1),
    )
    log = TruthLog("v3", TruthInitialState("2", "left", HAND), (turn,))
    loaded = truth_log_from_dict(log.to_dict())
    assert loaded.turns[0].trick_id == 2
    assert loaded.turns[0].evidence.frame_indices == (10, 11)
    assert loaded.turns[0].label_status == "verified"
    assert loaded.turns[0].uncertainty == ("unknown_suit",)
