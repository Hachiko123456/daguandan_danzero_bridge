from __future__ import annotations

from daguandan_bridge.application.replay_turn_draft import (
    ReplayTurnDraftAssembler,
    compare_truth_scan_draft,
    next_actor_after_prefix,
)
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _log(turns=()):
    return TruthLog("game", TruthInitialState("2", "left", HAND), tuple(turns))


def test_streaming_assembler_appends_each_confirmed_turn_in_memory():
    assembler = ReplayTurnDraftAssembler(_log())

    first = assembler.append(
        {
            "turn_id": 8,
            "trick_id": 1,
            "frame_index": 140,
            "actor": "left",
            "recognized_pass": False,
            "recognized_cards": ["3D"],
        }
    )
    second = assembler.append(
        {
            "turn_id": 9,
            "trick_id": 1,
            "frame_index": 155,
            "actor": "self",
            "recognized_pass": True,
            "recognized_cards": [],
        }
    )

    assert first.accepted and second.accepted
    assert first.status == "扫描确认"
    assert [(turn.index, turn.actor, turn.frame_index) for turn in second.truth_log.turns] == [
        (1, "left", 140),
        (2, "self", 155),
    ]
    assert not assembler.append({"turn_id": 9, "actor": "self", "is_pass": True}).accepted
    malformed = assembler.append(
        {"turn_id": "bad", "frame_index": "bad", "actor": "self", "is_pass": True}
    )
    assert not malformed.accepted
    assert malformed.truth_log == second.truth_log


def test_streaming_assembler_rejects_a_direct_actor_jump_without_shifting_draft():
    assembler = ReplayTurnDraftAssembler(_log())
    first = assembler.append(
        {
            "turn_id": 1,
            "actor": "left",
            "recognized_pass": False,
            "recognized_cards": ["3D"],
        }
    )
    skipped = assembler.append(
        {
            "turn_id": 2,
            "actor": "right",
            "recognized_pass": True,
        }
    )

    assert first.accepted
    assert not skipped.accepted
    assert "应为 self" in skipped.reason
    assert [(turn.index, turn.actor) for turn in skipped.truth_log.turns] == [
        (1, "left"),
    ]


def test_streaming_assembler_rejects_source_turn_gap_without_reindexing_rows():
    assembler = ReplayTurnDraftAssembler(_log())
    assert assembler.append(
        {
            "turn_id": 8,
            "actor": "left",
            "recognized_pass": False,
            "recognized_cards": ["3D"],
        }
    ).accepted

    skipped = assembler.append(
        {
            "turn_id": 10,
            "actor": "self",
            "recognized_pass": True,
        }
    )

    assert not skipped.accepted
    assert "来源 turn_id 不连续：应为 9，实际为 10" in skipped.reason
    assert [(turn.index, turn.actor) for turn in skipped.truth_log.turns] == [
        (1, "left"),
    ]


def test_suit_correction_replaces_its_existing_draft_row_without_adding_turn():
    assembler = ReplayTurnDraftAssembler(_log())
    appended = assembler.append(
        {
            "turn_id": 1,
            "actor": "left",
            "recognized_pass": False,
            "recognized_cards": ["J?"],
        }
    )
    corrected = assembler.apply_suit_correction(
        {
            "kind": "suit_corrected",
            "target_turn_id": 1,
            "actor": "left",
            "recognized_cards": ["JD"],
        }
    )

    assert appended.accepted and corrected.accepted
    assert corrected.status == "花色修正已回填"
    assert [(turn.index, turn.actor, turn.cards) for turn in corrected.truth_log.turns] == [
        (1, "left", ("JD",)),
    ]


def test_event_correction_replaces_effective_action_without_adding_a_turn():
    assembler = ReplayTurnDraftAssembler(_log())
    assert assembler.append(
        {
            "turn_id": 1,
            "actor": "left",
            "recognized_pass": False,
            "recognized_cards": ["A?", "K?"],
        }
    ).accepted

    corrected = assembler.append(
        {
            "kind": "event_correction",
            "target_turn_id": 1,
            "actor": "left",
            "recognized_pass": False,
            "recognized_cards": ["AC", "AS", "KC", "KS", "QD", "QS"],
        }
    )

    assert corrected.accepted
    assert corrected.status == "动作修正已回填"
    assert [(turn.index, turn.actor, turn.cards) for turn in corrected.truth_log.turns] == [
        (1, "left", ("AC", "AS", "KC", "KS", "QD", "QS")),
    ]


def test_actor_chain_returns_to_left_then_self_after_a_completed_pass_cycle():
    assembler = ReplayTurnDraftAssembler(_log())
    records = (
        ("left", False, ("3D",)),
        ("self", True, ()),
        ("right", True, ()),
        ("opposite", True, ()),
        ("left", True, ()),
        ("self", False, ("2D",)),
    )

    results = [
        assembler.append(
            {
                "turn_id": index,
                "actor": actor,
                "recognized_pass": is_pass,
                "recognized_cards": list(cards),
            }
        )
        for index, (actor, is_pass, cards) in enumerate(records, start=1)
    ]

    assert all(result.accepted for result in results)
    assert [turn.actor for turn in results[-1].truth_log.turns[-2:]] == [
        "left",
        "self",
    ]
    assert results[-1].truth_log.turns[-1].cards == ("2D",)


def test_prefix_derivation_returns_wind_receiver_after_every_active_pass():
    initial = TruthInitialState("2", "right", HAND)
    prefix = (
        TruthTurn(1, "right", False, ("3S",) * 27, trick_id=99),
        TruthTurn(2, "opposite", True, (), trick_id=3),
        TruthTurn(3, "left", True, (), trick_id=3),
        TruthTurn(4, "self", True, (), trick_id=3),
    )

    assert next_actor_after_prefix(initial, prefix) == "left"


def test_scan_comparison_keeps_card_multisets_and_turns_50_to_53_explicit():
    initial = TruthInitialState("2", "right", HAND)
    canonical = TruthLog(
        "game",
        initial,
        (
            TruthTurn(50, "right", False, ("small_joker",)),
            TruthTurn(51, "opposite", True, ()),
            TruthTurn(52, "left", True, ()),
            TruthTurn(53, "self", False, ("3S", "3H")),
        ),
    )
    draft = TruthLog(
        "game",
        initial,
        (
            TruthTurn(50, "right", False, ("small_joker",)),
            TruthTurn(51, "opposite", True, ()),
            TruthTurn(52, "left", True, ()),
            TruthTurn(53, "self", False, ("3H", "3S")),
        ),
    )
    events = (
        {
            "event_type": "player_finished",
            "actor": "right",
            "turn_id": 51,
            "payload": {"placement": "head"},
        },
    )

    comparison = compare_truth_scan_draft(
        canonical,
        draft,
        recorded_events=events,
    )

    rows = comparison["action_semantics"]["rows"]
    assert [row["turn_id"] for row in rows] == [50, 51, 52, 53]
    assert all(row["status"] == "identical" for row in rows)
    assert comparison["ranking_projection"] == {
        "read_only_source": "timeline.jsonl",
        "canonical": [{"placement": "head", "actor": "right", "anchor_turn_id": 50}],
        "scan_draft": [{"placement": "head", "actor": "right", "anchor_turn_id": 50}],
        "identical": True,
    }



def test_prefix_can_use_verified_finish_anchor_to_skip_bad_hand_count():
    initial = TruthInitialState(
        "3", "right", HAND,
        (("opposite", 28),),
    )
    prefix = (
        TruthTurn(1, "right", False, ("3S",) * 27, trick_id=1),
        TruthTurn(2, "opposite", False, ("4S",) * 27, trick_id=1),
        TruthTurn(3, "left", False, ("5S",), trick_id=1),
        TruthTurn(4, "self", True, (), trick_id=1),
    )

    # Pure card counts still think opposite has one card and would expect it.
    assert next_actor_after_prefix(initial, prefix) == "opposite"
    # A verified second-place/finish anchor makes the downstream turn skip it.
    assert next_actor_after_prefix(
        initial,
        prefix,
        forced_finished_after_turn={"right": 1, "opposite": 2},
    ) == "left"
