from __future__ import annotations

from daguandan_bridge.application.truth_log_semantic_validation import (
    validate_truth_log_semantics,
)
from daguandan_bridge.domain.truth import TruthEvidence, TruthOutcome
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn


def _log(*turns: TruthTurn, hand=("3S", "5S"), lead="self", sizes=None, outcome=None):
    size_rows = tuple((seat, count) for seat, count in (sizes or {}).items())
    return TruthLog(
        "semantic-test",
        TruthInitialState("2", lead, tuple(hand), size_rows),
        tuple(turns),
        outcome=outcome or TruthOutcome(),
    )


def _codes(report):
    return {item.code for item in report.findings}


def test_valid_lead_prefix_uses_production_card_classifier():
    report = validate_truth_log_semantics(
        _log(TruthTurn(1, "self", False, ("3S",), trick_id=1))
    )
    assert report.valid
    assert report.next_actor == "right"


def test_rejects_play_that_cannot_beat_current_table():
    report = validate_truth_log_semantics(_log(
        TruthTurn(1, "self", False, ("5S",), trick_id=1),
        TruthTurn(2, "right", False, ("4S",), trick_id=1),
    ))
    assert "TL-DOES-NOT-BEAT" in _codes(report)


def test_rejects_cards_that_do_not_form_a_legal_action():
    report = validate_truth_log_semantics(_log(
        TruthTurn(1, "self", False, ("3S", "5S"), trick_id=1),
    ))
    assert "TL-ILLEGAL-PLAY" in _codes(report)


def test_tencent_finished_leader_hands_wind_to_partner_without_partner_pass():
    report = validate_truth_log_semantics(_log(
        TruthTurn(1, "left", False, ("3H",), trick_id=1),
        TruthTurn(2, "self", True, (), trick_id=1),
        TruthTurn(3, "right", True, (), trick_id=1),
        TruthTurn(4, "opposite", True, (), trick_id=1),
        TruthTurn(5, "right", False, ("4H",), trick_id=2),
        TruthTurn(6, "opposite", True, (), trick_id=2),
        TruthTurn(7, "self", False, ("5H",), trick_id=2),
        TruthTurn(8, "right", True, (), trick_id=2),
        TruthTurn(9, "opposite", False, ("6H",), trick_id=3),
        hand=("5H",), lead="left",
        sizes={"self": 1, "right": 2, "opposite": 1, "left": 1},
    ))
    assert report.valid
    assert report.wind_catches == (
        (4, "left", "right"),
        (8, "self", "opposite"),
    )
    assert report.finish_order == ("left", "self", "opposite")
    assert report.next_actor is None

def test_wind_receiver_pass_is_not_required_once_all_opponents_responded():
    report = validate_truth_log_semantics(_log(
        TruthTurn(1, "left", False, ("3H",), trick_id=1),
        TruthTurn(2, "self", True, (), trick_id=1),
        TruthTurn(3, "right", True, (), trick_id=1),
        TruthTurn(4, "opposite", True, (), trick_id=1),
        TruthTurn(5, "right", False, ("4H",), trick_id=2),
        TruthTurn(6, "opposite", True, (), trick_id=2),
        TruthTurn(7, "self", False, ("5H",), trick_id=2),
        TruthTurn(8, "right", True, (), trick_id=2),
        hand=("5H",), lead="left",
        sizes={"self": 1, "right": 2, "opposite": 1, "left": 1},
    ))
    assert report.valid
    assert report.next_actor == "opposite"
    assert report.wind_catches[-1] == (8, "self", "opposite")

def test_rejects_action_after_round_is_decided():
    report = validate_truth_log_semantics(_log(
        TruthTurn(1, "left", False, ("3H",), trick_id=1),
        TruthTurn(2, "self", True, (), trick_id=1),
        TruthTurn(3, "right", True, (), trick_id=1),
        TruthTurn(4, "opposite", True, (), trick_id=1),
        TruthTurn(5, "right", False, ("4H",), trick_id=2),
        TruthTurn(6, "opposite", True, (), trick_id=2),
        TruthTurn(7, "self", False, ("5H",), trick_id=2),
        TruthTurn(8, "right", True, (), trick_id=2),
        TruthTurn(9, "opposite", False, ("6H",), trick_id=3),
        TruthTurn(10, "right", False, ("7H",), trick_id=3),
        hand=("5H",), lead="left",
        sizes={"self": 1, "right": 2, "opposite": 1, "left": 1},
    ))
    assert "TL-POST-TERMINAL-ACTION" in _codes(report)

def test_publish_mode_requires_verified_resolved_evidence():
    report = validate_truth_log_semantics(_log(
        TruthTurn(
            1, "self", False, ("3S",), trick_id=1,
            evidence=TruthEvidence((10,), 100),
            uncertainty=("button_occluded",),
        ),
    ), mode="publish")
    assert {"TL-LOG-NOT-VERIFIED", "TL-TURN-NOT-VERIFIED", "TL-UNCERTAINTY"}.issubset(_codes(report))


def test_publish_mode_detects_evidence_time_regression():
    log = _log(
        TruthTurn(1, "self", False, ("3S",), trick_id=1,
                  evidence=TruthEvidence((10,), 200), label_status="verified"),
        TruthTurn(2, "right", True, (), trick_id=1,
                  evidence=TruthEvidence((11,), 100), label_status="verified"),
    )
    log = TruthLog(
        log.source_session_id, log.initial_state, log.turns,
        label_status="verified", outcome=log.outcome,
    )
    report = validate_truth_log_semantics(log, mode="publish")
    assert "TL-EVIDENCE-TIME-REGRESSION" in _codes(report)


def test_inventory_error_and_illegal_beat_are_reported_together():
    hand = ("6D", "AH", "AD", "AC", "9C", "9S")
    report = validate_truth_log_semantics(_log(
        TruthTurn(1, "self", False, ("AH", "AD", "AC", "9C", "9S"), trick_id=1),
        TruthTurn(2, "right", True, (), trick_id=1),
        TruthTurn(3, "opposite", True, (), trick_id=1),
        TruthTurn(4, "left", False, ("2C", "3C", "4C", "5D", "6D"), trick_id=1),
        hand=hand,
    ))
    assert "TL-DOES-NOT-BEAT" in _codes(report)



def test_repaired_wind_catch_regression_truth_log_is_semantically_valid():
    from pathlib import Path
    from daguandan_bridge.live.truth_log import load_truth_log

    session = (
        Path(__file__).parents[1]
        / "data" / "profiles" / "tencent_daguandan" / "sessions"
        / "game_20260816_193531_f94cbf"
    )
    log = load_truth_log(session / "truth_log.json", session_id=session.name)
    report = validate_truth_log_semantics(log)

    assert report.valid, report.format_errors()
    assert report.wind_catches[-1] == (69, "self", "opposite")



def test_standard_playing_rejects_non_27_opponent_starting_count():
    report = validate_truth_log_semantics(
        _log(
            TruthTurn(1, "self", False, ("3S",), trick_id=1),
            hand=("3S", "5S"),
            sizes={"opposite": 28},
        ),
        standard_playing=True,
    )

    assert "TL-STARTING-HAND-SIZE" in _codes(report)


def test_normalize_trick_ids_rewrites_the_complete_valid_chain():
    from daguandan_bridge.application.truth_log_semantic_validation import (
        normalize_truth_log_trick_ids,
    )

    log = _log(
        TruthTurn(1, "self", False, ("3S",), trick_id=7),
        TruthTurn(2, "right", True, (), trick_id=7),
        TruthTurn(3, "opposite", True, (), trick_id=7),
        TruthTurn(4, "left", True, (), trick_id=7),
        TruthTurn(5, "self", False, ("5S",), trick_id=9),
        hand=("3S", "5S"),
    )

    normalized = normalize_truth_log_trick_ids(log)

    assert [turn.trick_id for turn in normalized.turns] == [1, 1, 1, 1, 2]
    assert [turn.cards for turn in normalized.turns] == [
        ("3S",), (), (), (), ("5S",),
    ]
