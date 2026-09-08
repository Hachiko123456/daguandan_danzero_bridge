from __future__ import annotations

import pytest

from daguandan_bridge.live_v2.candidates import ActionKind
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat
from daguandan_bridge.live_v2.simple_advice_gate import (
    AdviceGateDecision,
    SimpleAdviceGate,
)
from daguandan_bridge.live_v2.turn_core import (
    CommittedAction,
    HistoryIntegrity,
    PendingAction,
    SimpleSeatRules,
    TurnState,
)


def _frame(index: int) -> FrameIdentity:
    return FrameIdentity("advice-gate", 1, index, 5_000 + index, "roi", "capture")


def _state() -> TurnState:
    opening = PendingAction(
        Seat.LEFT, 0, ActionKind.PLAY, ("2S",), (("S",),), _frame(1), _frame(2), 1.0
    )
    return TurnState.opening().confirm_opening(
        CommittedAction.from_pending(opening, action_id="opening", turn_index=0),
        rules=SimpleSeatRules(),
    )


def test_only_trusted_self_turn_with_digest_is_allowed() -> None:
    gate = SimpleAdviceGate()
    result = gate.evaluate(_state(), history_digest="h1")
    assert result.decision is AdviceGateDecision.ALLOW
    assert result.reason == "trusted_self_turn"


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda state: state.mark_desync(), "desync"),
        (lambda state: state.begin_action(_pending_self()), "pending_action"),
    ],
)
def test_gate_blocks_non_advisable_state(mutator, reason: str) -> None:
    result = SimpleAdviceGate().evaluate(mutator(_state()), history_digest="h1")
    assert result.decision is AdviceGateDecision.BLOCK
    assert result.reason == reason


def _pending_self() -> PendingAction:
    return PendingAction(
        Seat.SELF, 1, ActionKind.PLAY, ("3S",), (("S",),), _frame(3), _frame(4), 1.0
    )


def test_gate_deduplicates_same_turn_and_history_digest() -> None:
    gate = SimpleAdviceGate()
    first = gate.evaluate(_state(), history_digest="h1")
    gate.record_submitted(first)
    duplicate = gate.evaluate(_state(), history_digest="h1")
    assert duplicate.decision is AdviceGateDecision.BLOCK
    assert duplicate.reason == "duplicate_submission"
    different_history = gate.evaluate(_state(), history_digest="h2")
    assert different_history.allowed


def test_partial_suits_require_consistent_semantic_key() -> None:
    from dataclasses import replace

    state = replace(_state(), integrity=HistoryIntegrity.PARTIAL_SUITS)
    gate = SimpleAdviceGate()
    blocked = gate.evaluate(state, history_digest="h1")
    assert blocked.reason == "partial_suits_ambiguous"
    allowed = gate.evaluate(
        state,
        history_digest="h1",
        partial_suits_safe=True,
        semantic_key="straight:6",
    )
    assert allowed.allowed


def test_missing_digest_is_blocked() -> None:
    result = SimpleAdviceGate().evaluate(_state(), history_digest=" ")
    assert result.reason == "history_digest_missing"


def test_record_submitted_rejects_blocked_result() -> None:
    gate = SimpleAdviceGate()
    with pytest.raises(ValueError, match="allowed"):
        gate.record_submitted(gate.evaluate(_state(), history_digest=""))
