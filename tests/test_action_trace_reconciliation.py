from copy import deepcopy

from daguandan_bridge.application.action_trace_reconciliation import reconcile_action_trace


def row(frame, actor, cards=(), *, is_pass=False, confidence=0.9, current_player=None, buttons=()):
    regions = {actor: {"cards": list(cards), "is_pass": is_pass, "confidence": confidence}}
    return {"frame_index": frame, "timestamp_ms": frame * 10, "decode_ok": True,
            "regions": regions, "current_player": current_player, "buttons": list(buttons)}


def action(action_id, actor, cards, start, end, *, variants=(), uncertainty=(), is_pass=False):
    return {"action_id": action_id, "actor": actor, "is_pass": is_pass, "cards": list(cards),
            "frame_start": start, "frame_end": end, "evidence_frames": list(range(start, end + 1)),
            "observed_variants": list(variants), "uncertainty": list(uncertainty)}


def test_brief_missing_frame_duplicate_is_merged_and_ids_are_continuous():
    actions = [action(8, "self", ("2C", "2D"), 10, 12), action(99, "self", ("2C", "2D"), 14, 16)]
    observations = [row(frame, "self", ("2C", "2D")) for frame in (10, 11, 12, 14, 15, 16)]
    result = reconcile_action_trace(actions, observations)
    assert len(result) == 1
    assert result[0]["action_id"] == 1
    assert result[0]["cards"] == ["2C", "2D"]
    assert result[0]["reconciliation"]["merged_action_ids"] == [99]
    assert result[0]["frame_end"] == 16


def test_progressive_unknown_suit_and_suit_variant_choose_complete_hand():
    actions = [action(3, "left", ("A?", "K?"), 20, 21, uncertainty=("unknown_suit",)),
               action(4, "left", ("AC", "KS", "QD"), 22, 23)]
    observations = [row(20, "left", ("A?", "K?")), row(21, "left", ("A?", "K?")),
                    row(22, "left", ("AC", "KS", "QD")), row(23, "left", ("AC", "KS", "QD"))]
    result = reconcile_action_trace(actions, observations)
    assert len(result) == 1
    assert result[0]["cards"] == ["AC", "KS", "QD"]
    assert result[0]["repair_status"] == "resolved"
    assert result[0]["uncertainty"] == []
    assert result[0]["reconciliation"]["source_action_ids"] == [3, 4]


def test_joker_flicker_does_not_create_two_actions_and_uses_supported_label():
    actions = [action(1, "right", ("small_joker",), 30, 30),
               action(2, "right", ("big_joker",), 31, 33)]
    observations = [row(30, "right", ("small_joker",), confidence=.7),
                    row(31, "right", ("big_joker",), confidence=.8),
                    row(32, "right", ("big_joker",), confidence=.9),
                    row(33, "right", ("big_joker",), confidence=.9)]
    result = reconcile_action_trace(actions, observations)
    assert len(result) == 1
    assert result[0]["cards"] == ["big_joker"]
    assert result[0]["reconciliation"]["events"][0]["reason"] == "joker_flicker"


def test_one_frame_unknown_suit_noise_is_dropped_with_audit():
    actions = [action(4, "self", ("AC",), 40, 42), action(5, "self", ("A?",), 43, 43)]
    observations = [row(40, "self", ("AC",)), row(41, "self", ("AC",)),
                    row(42, "self", ("AC",)), row(43, "self", ("A?",)), row(44, "self", ("AC",))]
    result = reconcile_action_trace(actions, observations)
    assert len(result) == 1
    assert result[0]["action_id"] == 1
    assert result[0]["cards"] == ["AC"]
    assert result[0]["reconciliation"]["dropped_action_ids"] == [5]


def test_unknown_noise_after_gap_is_merged_without_current_player_or_buttons():
    actions = [action(10, "opposite", ("Q?",), 50, 50, uncertainty=("unknown_suit",)),
               action(11, "opposite", ("QD",), 52, 53)]
    observations = [row(50, "opposite", ("Q?",)), row(52, "opposite", ("QD",)), row(53, "opposite", ("QD",))]
    original = deepcopy(observations)
    result = reconcile_action_trace(actions, observations)
    assert result[0]["cards"] == ["QD"]
    assert observations == original
    assert result[0]["action_id"] == 1


def test_nested_opening_current_player_signal_can_audit_stale_pass_without_being_required():
    actions = [action(20, "opposite", (), 10, 10, is_pass=True),
               action(21, "self", (), 12, 12, is_pass=True)]
    observations = [
        {"frame_index": 10, "timestamp_ms": 100, "decode_ok": True,
         "opening": {"current_player_signal": "self"}, "buttons": [],
         "regions": {"opposite": {"cards": [], "is_pass": True, "confidence": .9}}},
        {"frame_index": 12, "timestamp_ms": 120, "decode_ok": True,
         "opening": {"current_player_signal": "self"}, "buttons": [],
         "regions": {"self": {"cards": [], "is_pass": True, "confidence": .9}}},
    ]
    result = reconcile_action_trace(actions, observations)
    assert [item["actor"] for item in result] == ["self"]
    assert result[0]["reconciliation"]["dropped_action_ids"] == [20]
    assert result[0]["reconciliation"]["drop_events"][0]["reason"] == "out_of_turn_pass_display"


def test_lone_one_frame_unknown_suit_is_preserved_for_review_without_complete_support():
    actions = [action(30, "self", ("A?",), 60, 60, uncertainty=("unknown_suit",))]
    observations = [row(60, "self", ("A?",))]

    result = reconcile_action_trace(actions, observations)

    assert len(result) == 1
    assert result[0]["action_id"] == 1
    assert result[0]["cards"] == ["A?"]
    assert result[0]["review_status"] == "needs_review"
    assert result[0]["uncertainty"] == ["unknown_suit"]
    assert result[0]["reconciliation"]["dropped_action_ids"] == []


def test_terminal_drop_does_not_revive_raw_action_when_everything_is_filtered():
    actions = [action(40, "left", ("7H", "7S", "8H"), 100, 101)]
    observations = [
        row(100, "left", ("7H", "7S", "8H")),
        row(101, "left", ("7H", "7S", "8H"), buttons=("continue_game",)),
    ]

    assert reconcile_action_trace(actions, observations) == []


def test_output_order_is_frame_start_then_source_position_and_ids_remain_continuous():
    actions = [
        action(50, "self", ("AC",), 30, 31),
        action(51, "right", ("KD",), 10, 11),
        action(52, "left", ("QH",), 30, 31),
    ]
    observations = [
        row(10, "right", ("KD",)), row(11, "right", ("KD",)),
        row(30, "self", ("AC",)), row(31, "self", ("AC",)),
        row(30, "left", ("QH",)), row(31, "left", ("QH",)),
    ]

    result = reconcile_action_trace(actions, observations)

    assert [item["actor"] for item in result] == ["right", "self", "left"]
    assert [item["action_id"] for item in result] == [1, 2, 3]


def test_complete_card_order_uses_repeated_observed_variant_not_single_projected_summary():
    actions = [action(70, "opposite", ("6H", "JD", "JH", "JS"), 80, 82)]
    observations = [
        row(frame, "opposite", ("JH", "JD", "JS", "6H"))
        for frame in (80, 81, 82)
    ]

    result = reconcile_action_trace(actions, observations)

    assert result[0]["cards"] == ["JH", "JD", "JS", "6H"]


def test_pass_after_signal_advance_is_recovered_in_same_turn_window():
    actions = [
        action(1, "right", ("9S", "6H", "JS", "QS", "KS"), 10, 15),
        action(2, "opposite", ("J?",), 14, 16, uncertainty=("unknown_suit",)),
        action(3, "opposite", (), 17, 21, is_pass=True),
    ]
    observations = [
        {"frame_index": 10, "timestamp_ms": 100, "decode_ok": True, "opening": {"current_player_signal": "opposite"}, "regions": {"right": {"cards": ["9S", "6H", "JS", "QS", "KS"], "is_pass": False}}},
        {"frame_index": 14, "timestamp_ms": 140, "decode_ok": True, "opening": {"current_player_signal": "opposite"}, "regions": {"opposite": {"cards": ["J?"], "is_pass": False}}},
        {"frame_index": 17, "timestamp_ms": 170, "decode_ok": True, "opening": {"current_player_signal": "left"}, "regions": {"opposite": {"cards": [], "is_pass": True, "confidence": .98}}},
        {"frame_index": 18, "timestamp_ms": 180, "decode_ok": True, "opening": {"current_player_signal": "left"}, "regions": {"opposite": {"cards": [], "is_pass": True, "confidence": .98}}},
    ]

    result = reconcile_action_trace(actions, observations)

    assert [(item["actor"], item["is_pass"], item["cards"]) for item in result] == [
        ("right", False, ["9S", "6H", "JS", "QS", "KS"]),
        ("opposite", True, []),
    ]
    assert result[0]["reconciliation"]["drop_events"][0]["reason"] == "one_frame_unknown_suit_noise"


def test_terminal_final_play_is_preserved_when_last_current_actor_has_cards():
    actions = [
        action(1, "self", (), 80, 84, is_pass=True),
        action(2, "left", ("KH", "KC", "6H", "JH", "JC"), 85, 90),
        action(3, "self", ("QS", "10D", "10C", "10S"), 86, 90),
    ]
    observations = [
        {"frame_index": 85, "timestamp_ms": 850, "decode_ok": True, "opening": {"current_player_signal": "left"}, "regions": {"left": {"cards": ["KH", "KC", "6H", "JH", "JC"], "is_pass": False}}},
        {"frame_index": 89, "timestamp_ms": 890, "decode_ok": True, "opening": {"current_player_signal": "left"}, "regions": {"left": {"cards": ["KH", "KC", "6H", "JH", "JC"], "is_pass": False}, "self": {"cards": ["QS", "10D", "10C", "10S"], "is_pass": False}}},
        {"frame_index": 91, "timestamp_ms": 910, "decode_ok": True, "opening": {"current_player_signal": None}, "buttons": ["continue_game"], "regions": {}},
    ]

    result = reconcile_action_trace(actions, observations)

    assert [(item["actor"], item["cards"]) for item in result] == [("left", ["KH", "KC", "6H", "JH", "JC"])]
    assert result[0]["reconciliation"]["events"][-1]["type"] == "terminal_last_turn_preserved"


def test_initial_table_context_keeps_only_passes_between_visible_leader_and_current_player():
    actions = [
        action(1, "opposite", ("5C", "6C", "7C", "8C", "9C"), 0, 9, uncertainty=("display_present_at_scan_start",)),
        action(2, "left", (), 0, 12, is_pass=True),
        action(3, "right", (), 14, 16, is_pass=True),
        action(4, "self", (), 10, 15, is_pass=True),
    ]
    observations = [
        {"frame_index": 0, "timestamp_ms": 0, "decode_ok": True, "opening": {"current_player_signal": "self"}, "regions": {
            "opposite": {"cards": ["5C", "6C", "7C", "8C", "9C"], "is_pass": False},
            "left": {"cards": [], "is_pass": True},
            "right": {"cards": [], "is_pass": True},
        }},
        {"frame_index": 10, "timestamp_ms": 100, "decode_ok": True, "opening": {"current_player_signal": "right"}, "regions": {"self": {"cards": [], "is_pass": True}}},
    ]

    result = reconcile_action_trace(actions, observations)

    assert [(item["actor"], item["is_pass"]) for item in result] == [
        ("opposite", False), ("left", True), ("self", True), ("right", True),
    ]


def test_same_surface_reappearance_without_new_actor_turn_merges_across_long_gap():
    actions = [
        action(1, "self", ("KH", "KD", "KC", "9H"), 100, 110),
        action(2, "right", ("10C", "JC", "QC", "9H", "AC"), 111, 120),
        action(3, "self", ("KH", "KD", "KC", "9H"), 121, 140),
        action(4, "opposite", (), 141, 145, is_pass=True),
    ]
    observations = [
        {"frame_index": 99, "timestamp_ms": 990, "decode_ok": True, "opening": {"current_player_signal": "self"}, "regions": {}},
        {"frame_index": 111, "timestamp_ms": 1110, "decode_ok": True, "opening": {"current_player_signal": "right"}, "regions": {"self": {"cards": ["KH", "KD", "KC", "9H"], "is_pass": False}}},
        {"frame_index": 121, "timestamp_ms": 1210, "decode_ok": True, "opening": {"current_player_signal": "opposite"}, "regions": {"self": {"cards": ["KH", "KD", "KC", "9H"], "is_pass": False}}},
        {"frame_index": 141, "timestamp_ms": 1410, "decode_ok": True, "opening": {"current_player_signal": "self"}, "regions": {"opposite": {"cards": [], "is_pass": True}}},
    ]
    result = reconcile_action_trace(actions, observations)
    self_plays = [item for item in result if item["actor"] == "self" and not item["is_pass"]]
    assert len(self_plays) == 1
    assert self_plays[0]["frame_end"] == 140
    assert self_plays[0]["reconciliation"]["events"][-1]["reason"] == "stale_surface_reappearance_without_new_turn"


def test_same_cards_after_actor_receives_new_turn_stay_separate():
    actions = [
        action(1, "right", ("AH",), 10, 20),
        action(2, "left", (), 21, 24, is_pass=True),
        action(3, "self", (), 25, 28, is_pass=True),
        action(4, "right", ("AH",), 30, 40),
    ]
    observations = [
        {"frame_index": 9, "timestamp_ms": 90, "decode_ok": True, "opening": {"current_player_signal": "right"}, "regions": {}},
        {"frame_index": 21, "timestamp_ms": 210, "decode_ok": True, "opening": {"current_player_signal": "left"}, "regions": {}},
        {"frame_index": 25, "timestamp_ms": 250, "decode_ok": True, "opening": {"current_player_signal": "self"}, "regions": {}},
        {"frame_index": 29, "timestamp_ms": 290, "decode_ok": True, "opening": {"current_player_signal": "right"}, "regions": {}},
    ]
    result = reconcile_action_trace(actions, observations)
    assert len([item for item in result if item["actor"] == "right" and not item["is_pass"]]) == 2
