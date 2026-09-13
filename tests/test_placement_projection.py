from __future__ import annotations

from daguandan_bridge.application.placement_projection import (
    derive_finish_order,
    project_truth_log_placements,
    format_placement_summary,
    project_recorded_placements,
)
from daguandan_bridge.live.truth_log import TruthTurn


SEAT_LABELS = {
    "self": "自己",
    "right": "右家",
    "opposite": "对家",
    "left": "左家",
}


def test_late_visual_finish_badges_remain_rank_ordered_without_old_row_anchor():
    turns = (
        TruthTurn(15, "opposite", False, ("big_joker",), trick_id=3),
        TruthTurn(20, "left", False, ("3H", "3H", "4?"), trick_id=4),
    )
    events = (
        {
            "event_id": "FINISH-HEAD",
            "event_type": "player_finished",
            "actor": "left",
            "turn_id": 26,
            "payload": {"placement": "head"},
        },
        {
            "event_id": "FINISH-SECOND",
            "event_type": "player_finished",
            "actor": "opposite",
            "turn_id": 45,
            "payload": {"placement": "second"},
        },
    )

    placements = project_recorded_placements(events, turns)

    assert [(item.placement, item.actor, item.anchor_turn_id) for item in placements] == [
        ("head", "left", None),
        ("second", "opposite", None),
    ]
    assert format_placement_summary(placements, SEAT_LABELS) == (
        "出完顺序：1 左家·头游  →  2 对家·二游"
    )


def test_exact_action_reference_and_legacy_immediate_boundary_can_show_row_badges():
    turns = (
        TruthTurn(1, "left", False, ("3D",), trick_id=1),
        TruthTurn(2, "self", False, ("4C",), trick_id=1),
    )
    events = (
        {
            "event_id": "PLAY-LEFT",
            "event_type": "player_played",
            "actor": "left",
            "turn_id": 1,
            "payload": {"cards": ["3D"]},
        },
        {
            "event_id": "FINISH-HEAD",
            "event_type": "player_finished",
            "actor": "left",
            "turn_id": 2,
            "payload": {
                "placement": "head",
                "trigger_action_event_id": "PLAY-LEFT",
            },
        },
    )

    placements = project_recorded_placements(events, turns)

    assert placements[0].anchor_turn_id == 1


def test_card_count_finish_order_infers_last_place_after_three_zero_crossings():
    from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn

    hand = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")
    log = TruthLog(
        "game",
        TruthInitialState("6", "opposite", hand),
        (
            TruthTurn(1, "opposite", False, ("3S",) * 27),
            TruthTurn(2, "right", False, ("4S",) * 27),
            TruthTurn(3, "left", False, ("5S",) * 27),
        ),
    )

    assert derive_finish_order(log.turns, initial_hand_size=len(hand)) == (
        "opposite", "right", "left", "self"
    )
    placements = project_truth_log_placements(log)
    assert [(item.placement, item.actor, item.anchor_turn_id) for item in placements] == [
        ("head", "opposite", 1),
        ("second", "right", 2),
        ("third", "left", 3),
        ("last", "self", None),
    ]
