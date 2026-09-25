from __future__ import annotations

from daguandan_bridge.live.card_uncertainty import (
    feasible_action_variants,
    feasible_self_hand_variants,
    state_variants_for_unknown_suits,
    state_variants_for_unknown_suits_detailed,
)
from daguandan_bridge.live.consensus import canonical_candidate
from daguandan_bridge.danzero.state import GuanDanState


def test_unknown_black_eight_never_becomes_a_third_spade():
    variants = feasible_action_variants(
        cards=("8?",),
        suit_options=(("S", "C"),),
        known_cards=("8S", "8S"),
    )

    assert variants == (("8C",),)


def test_historical_unknown_suit_cannot_block_a_later_clear_card():
    variants = feasible_action_variants(
        cards=("6H",),
        suit_options=(("H",),),
        known_cards=("6D", "6D", "6H", "6S", "6C", "6?"),
        known_suit_options=(
            ("D",), ("D",), ("H",), ("S",), ("C",), ("H", "D"),
        ),
    )

    # An old effect-covered 6? cannot force the later clear heart into a
    # false third-copy rejection.
    assert variants == (("6H",),)


def test_advisor_variants_relax_only_impossible_historical_suit_constraints():
    state = GuanDanState(
        round_level="10",
        wild_rank="10",
        current_player="self",
        lead_player="left",
        my_hand=("5C",),
    )
    # Both H/D candidates are exhausted.  This reproduces an old effect read
    # whose visual suit candidates are no longer compatible with later clear
    # cards; the advisor must keep the game playable through S/C branches.
    for card in ("6D", "6D", "6H", "6H", "6S"):
        state.record_play("left", (card,), suit_options=((card[-1],),))
    state.record_play("right", ("6?",), suit_options=(("H", "D"),))

    variants = state_variants_for_unknown_suits(state)

    assert variants
    assert {variant.play_history[-1].cards for variant in variants} == {
        ("6S",), ("6C",),
    }


def test_unknown_self_suit_resolves_only_to_confirmed_hand_cards():
    variants = feasible_self_hand_variants(
        cards=("8?", "8?"),
        suit_options=(("S", "C"), ("S", "C")),
        known_hand=("8C", "8C", "8S"),
    )

    assert set(variants) == {
        ("8C", "8C"),
        ("8C", "8S"),
        ("8S", "8C"),
    }


def test_unknown_self_suit_is_expanded_without_double_counting_the_hand():
    variants = feasible_action_variants(
        cards=("8?",),
        suit_options=(("S", "C"),),
        known_cards=("8S", "8C"),
        candidate_already_known=True,
    )

    assert set(variants) == {("8S",), ("8C",)}


def test_unknown_suit_is_kept_as_multiple_temporary_advice_variants():
    state = GuanDanState(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="left",
        my_hand=("3S",),
    )
    state.record_play("left", ("8?",), suit_options=(("S", "C"),))

    variants = state_variants_for_unknown_suits(state)

    assert {variant.play_history[0].cards for variant in variants} == {
        ("8S",),
        ("8C",),
    }


def test_unknown_suit_variant_cap_is_reported_instead_of_silent_truncation():
    state = GuanDanState(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="left",
        my_hand=("3S",),
    )
    state.record_play(
        "left",
        ("8?", "9?"),
        suit_options=(("S", "H", "C", "D"), ("S", "H", "C", "D")),
    )

    result = state_variants_for_unknown_suits_detailed(state, limit=2)

    assert len(result.states) == 2
    assert result.truncated is True
    assert result.limit == 2


def test_sorting_cards_keeps_their_suit_options_attached():
    cards, options = canonical_candidate(
        ("5H", "5?"),
        (("H",), ("S", "C")),
    )

    assert cards == ("5?", "5H")
    assert options == (("S", "C"), ("H",))


def test_state_history_keeps_suit_options_attached_after_card_normalization():
    state = GuanDanState()

    event = state.record_play(
        "left",
        ("8H", "8?"),
        suit_options=(("H",), ("S", "C")),
    )

    assert event.cards == ("8?", "8H")
    assert event.suit_options == (("S", "C"), ("H",))
from datetime import datetime, timedelta, timezone

from daguandan_bridge.live.card_uncertainty import (
    assess_occluded_action,
    confirm_occluded_action,
)
from daguandan_bridge.live.occlusion_evidence import OcclusionEvidence, mark_stale


def test_five_card_occlusion_blocks_strategy_when_candidates_disagree():
    evidence = assess_occluded_action(
        cards=("8?", "8?", "9?", "10?", "J?"),
        suit_options=(
            ("S", "H"), ("S", "H"), ("S", "H"), ("S", "H"), ("S", "H"),
        ),
        action_id="left-5",
    )

    assert evidence.state == "occluded"
    assert evidence.strategy_blocked is True
    assert evidence.is_actionable is False
    assert len(evidence.candidates) > 1


def test_double_deck_limits_reject_impossible_five_card_candidate():
    evidence = assess_occluded_action(
        cards=("8?", "8?", "8?", "8?", "8?"),
        suit_options=(("S", "H", "C", "D"),) * 5,
        known_cards=("8S", "8S", "8H", "8H", "8C", "8C", "8D", "8D"),
        action_id="left-impossible",
    )

    assert evidence.state == "rejected"
    assert evidence.strategy_blocked is True
    assert "double-deck" in evidence.reason


def test_occlusion_evidence_is_serializable_and_delayed_confirmation_is_explicit():
    first = assess_occluded_action(
        cards=("7?", "7C", "7D", "7D"),
        suit_options=(("H", "D"), ("C",), ("D",), ("D",)),
        action_id="left-bomb",
    )
    assert first.state == "confirmed"
    assert first.confirmations == 1
    assert first.delayed is True

    second = confirm_occluded_action(
        first,
        cards=("7?", "7C", "7D", "7D"),
        suit_options=(("H", "D"), ("C",), ("D",), ("D",)),
    )
    assert second.state == "confirmed"
    assert second.confirmations == 2
    assert second.is_actionable is True
    assert OcclusionEvidence.from_dict(second.to_dict()) == second


def test_empty_recognition_is_rejected_and_expired_evidence_is_stale():
    empty = assess_occluded_action(cards=(), action_id="left-empty")
    assert empty.state == "rejected"
    assert empty.strategy_blocked is True
    assert empty.is_actionable is False

    expires = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    evidence = OcclusionEvidence(
        state="occluded", cards=("8?",), expires_at=expires,
    )
    stale = mark_stale(evidence)
    assert stale.state == "stale"
    assert stale.strategy_blocked is True
