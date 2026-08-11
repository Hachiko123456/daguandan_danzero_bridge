import pytest

from daguandan_bridge.danzero import DanzeroAdvisor, GameStateError, GuanDanState
from daguandan_bridge.danzero.rules import (
    action_for_cards,
    actions_for_cards,
    infer_best_action,
)


def test_danzero_public_api_can_be_imported_without_qt():
    assert DanzeroAdvisor.__name__ == "DanzeroAdvisor"
    assert GuanDanState.__name__ == "GuanDanState"


def test_incomplete_manual_state_is_rejected_before_model_execution():
    state = GuanDanState(round_level="2", wild_rank="2")

    with pytest.raises(GameStateError):
        DanzeroAdvisor().recommend(state)


def test_danzero_returns_advice_for_a_confirmed_local_hand():
    state = GuanDanState()
    state.set_context(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="self",
    )
    state.confirm_hand(("3S", "4H", "5D"))

    advice = DanzeroAdvisor().recommend(state)

    assert advice.strategy == "danzero"
    assert advice.cards in {("3S",), ("4H",), ("5D",)}
    assert advice.engine_input is not None
    assert advice.engine_input["feature_schema"] == "danzero-567/v1"
    features = advice.engine_input["features_567"]
    assert len(features) == advice.engine_input["legal_action_count"]
    assert all(len(row) == 567 for row in features)


def test_wildcard_full_house_uses_the_interpretation_that_beats_the_table():
    cards = ("JC", "JD", "9H", "2D", "2H")
    table = ("10C", "10H", "10S", "KH", "KS")

    actions = actions_for_cards(cards, "9")
    inference = infer_best_action(
        cards,
        table,
        "9",
        preferred_play_type="ThreeWithTwo",
    )

    assert {tuple(action[:2]) for action in actions} >= {
        ("ThreeWithTwo", "2"),
        ("ThreeWithTwo", "J"),
    }
    assert inference.action is not None
    assert inference.action[:2] == ["ThreeWithTwo", "J"]
    assert inference.beats_table
    assert inference.ambiguous
    assert inference.logical_label == "JJJ22"
    assert inference.wildcard_substitutions == (("9H", "J"),)
    assert action_for_cards(cards, "9")[:2] == ["ThreeWithTwo", "J"]
