from __future__ import annotations

from copy import deepcopy
import json

import pytest

from daguandan_bridge.domain.truth import LabelProvenance, TruthEvidence

from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthLogCardInventoryError,
    TruthTurn,
    card_code_to_text,
    card_text_to_code,
    load_truth_log,
    save_truth_log,
    truth_log_from_dict,
    validate_truth_log_card_inventory,
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
    assert loaded.to_dict()["schema_version"] == 4
    assert loaded.to_dict()["schema"] == "guandan.truth/4"
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


@pytest.mark.parametrize("schema_version", (1, 2, 3, 4))
def test_context_session_id_loads_legacy_log_without_rewriting_file(
    tmp_path, schema_version
):
    raw = _log().to_dict()
    raw.pop("source_session_id")
    if schema_version < 3:
        raw.pop("schema")
        raw["schema_version"] = schema_version
    elif schema_version == 3:
        raw["schema"] = "guandan.truth/3"
        raw["schema_version"] = 3
    path = tmp_path / "truth_log.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    original_bytes = path.read_bytes()

    loaded = load_truth_log(path, session_id="game-test")

    assert loaded.source_session_id == "game-test"
    assert path.read_bytes() == original_bytes


def test_legacy_log_without_source_id_requires_session_context(tmp_path):
    raw = _log().to_dict()
    raw.pop("source_session_id")
    path = tmp_path / "truth_log.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="缺少源对局 ID"):
        truth_log_from_dict(raw)
    with pytest.raises(ValueError, match="缺少源对局 ID"):
        load_truth_log(path)


def test_context_session_id_does_not_override_explicit_source_id(tmp_path):
    raw = _log().to_dict()
    raw["source_session_id"] = "another-game"
    path = tmp_path / "truth_log.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="不属于"):
        load_truth_log(path, session_id="game-test")


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
    assert log.to_dict()["schema_version"] == 4


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


def test_truth_v4_round_trips_label_provenance_evidence_and_uncertainty():
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


def test_truth_v4_round_trips_explicit_wildcard_semantics_into_live_events():
    semantics = {
        "move_type": "STRAIGHT",
        "key": 6,
        "claim_ranks": ["6", "7", "8", "9", "10"],
        "wildcard_assignments": [
            {"physical_card": "6H", "as_rank": "8"}
        ],
        "ambiguity": True,
        "candidate_interpretations": [
            {"move_type": "STRAIGHT", "key": 6},
            {"move_type": "SFLUSH", "key": 6},
        ],
        "selected_interpretation": {
            "move_type": "STRAIGHT",
            "key": 6,
            "claim_ranks": ["6", "7", "8", "9", "10"],
        },
        "selection_source": "exact_engine_state",
    }
    turn = TruthTurn(
        1,
        "self",
        False,
        ("10C", "6C", "6H", "7C", "9C"),
        move_semantics=semantics,
    )
    log = TruthLog("semantic", TruthInitialState("6", "self", HAND), (turn,))

    loaded = truth_log_from_dict(log.to_dict())
    event = loaded.to_events()[1]

    assert loaded.turns[0].move_semantics == semantics
    assert event.payload["physical_cards"] == list(turn.cards)
    assert event.payload["move_semantics"] == semantics



def test_card_inventory_rejects_third_exact_card_across_hand_and_opponents():
    log = TruthLog(
        "inventory-exact",
        TruthInitialState("2", "right", HAND),
        (
            TruthTurn(1, "right", False, ("3D",)),
            TruthTurn(2, "opposite", False, ("3D",)),
        ),
    )

    with pytest.raises(TruthLogCardInventoryError) as captured:
        validate_truth_log_card_inventory(log)

    message = str(captured.value)
    assert "方块3（3D）共 3 张" in message
    assert "双副牌最多 2 张" in message
    assert "第 1、2 条动作" in message


def test_card_inventory_checks_rank_and_suit_double_deck_limits():
    rank_overflow = TruthLog(
        "inventory-rank",
        TruthInitialState(
            "2",
            "right",
            ("3S", "3S", "3H", "3H", "3C", "3C", "3D", "3D"),
        ),
        (TruthTurn(1, "right", False, ("3?",)),),
    )
    with pytest.raises(TruthLogCardInventoryError, match="点数 3 共 9 张"):
        validate_truth_log_card_inventory(rank_overflow)

    full_spade_suit = tuple(
        card for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
        for card in (f"{rank}S", f"{rank}S")
    )
    suit_overflow = TruthLog(
        "inventory-suit",
        TruthInitialState("2", "right", full_spade_suit),
        (TruthTurn(1, "right", False, ("AS",)),),
    )
    with pytest.raises(TruthLogCardInventoryError, match="花色 黑桃 共 27 张"):
        validate_truth_log_card_inventory(suit_overflow)


def test_card_inventory_rejects_self_card_not_present_in_initial_hand():
    log = TruthLog(
        "inventory-self",
        TruthInitialState("2", "self", HAND),
        (TruthTurn(1, "self", False, ("AS",)),),
    )

    with pytest.raises(TruthLogCardInventoryError, match="自己累计打出点数 A 1 张"):
        validate_truth_log_card_inventory(log)


def test_save_truth_log_never_persists_impossible_verified_inventory(tmp_path):
    path = tmp_path / "truth_log.json"
    log = TruthLog(
        "inventory-save",
        TruthInitialState("2", "right", HAND),
        (
            TruthTurn(1, "right", False, ("3D",)),
            TruthTurn(2, "opposite", False, ("3D",)),
        ),
        label_status="verified",
    )

    with pytest.raises(TruthLogCardInventoryError):
        save_truth_log(path, log)

    assert not path.exists()
