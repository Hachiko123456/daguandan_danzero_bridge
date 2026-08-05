import pytest

from daguandan_bridge.danzero import DanzeroAdvisor, GameStateError, GuanDanState


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
