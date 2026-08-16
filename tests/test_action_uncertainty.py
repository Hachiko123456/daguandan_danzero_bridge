from __future__ import annotations

from datetime import datetime

from daguandan_bridge.danzero.state import GuanDanState, PlayEvent
from daguandan_bridge.live.action_uncertainty import (
    state_variants_for_action_semantics,
)


def _ambiguous_event() -> PlayEvent:
    return PlayEvent(
        player="left",
        cards=("2D", "2H", "9H", "JC", "JD"),
        is_pass=False,
        observed_at=datetime.now().astimezone(),
        action_metadata={
            "interpretation_ambiguous": True,
            "candidate_interpretations": [
                {
                    "move_type": "ThreeWithTwo",
                    "key": "J",
                    "wildcard_assignments": [
                        {"physical_card": "9H", "as_rank": "J"}
                    ],
                },
                {
                    "move_type": "ThreeWithTwo",
                    "key": "2",
                    "wildcard_assignments": [
                        {"physical_card": "9H", "as_rank": "2"}
                    ],
                },
            ],
            "selected_interpretation": None,
            "selection_source": "unresolved",
        },
    )


def _state(events: list[PlayEvent]) -> GuanDanState:
    return GuanDanState(
        round_level="9",
        wild_rank="9",
        current_player="self",
        lead_player="left",
        my_hand=("3S",),
        trick_plays=list(events),
        play_history=list(events),
    )


def test_action_semantic_variants_are_temporary_and_auditable():
    event = _ambiguous_event()
    state = _state([event])

    result = state_variants_for_action_semantics(state)

    assert len(result.states) == 2
    assert result.source_history_indices == (1,)
    assert result.candidate_counts == ((1, 2),)
    assert result.error == ""
    assert {
        branch.play_history[0].action_metadata["selected_interpretation"]["key"]
        for branch in result.states
    } == {"J", "2"}
    assert state.play_history[0].action_metadata["selected_interpretation"] is None


def test_action_semantic_variant_limit_blocks_before_partial_evaluation():
    events = [_ambiguous_event() for _ in range(6)]

    result = state_variants_for_action_semantics(_state(events), limit=32)

    assert result.states == ()
    assert result.total_variant_count == 64
    assert result.source_history_indices == (1, 2, 3, 4, 5, 6)
    assert "超过安全上限 32" in result.error
    assert "未执行不完整的模型评估" in result.error


def test_unrecoverable_semantics_report_history_cards_level_and_reason():
    event = PlayEvent(
        player="left",
        cards=("3S", "4D", "9H"),
        is_pass=False,
        observed_at=datetime.now().astimezone(),
        action_metadata={
            "interpretation_ambiguous": True,
            "candidate_interpretations": [],
            "selected_interpretation": None,
            "selection_source": "unresolved",
        },
    )

    result = state_variants_for_action_semantics(_state([event]))

    assert result.states == ()
    assert "第 1 条历史动作语义无法恢复" in result.error
    assert "实体牌 3S 4D 9H" in result.error
    assert "级牌/逢人配点数 9" in result.error
    assert "无法从实体牌恢复任何合法解释" in result.error
