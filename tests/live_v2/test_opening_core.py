from __future__ import annotations

import pytest

from daguandan_bridge.live_v2.candidates import ActionKind
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat
from daguandan_bridge.live_v2.opening_core import (
    HandLevelEvidence,
    LeadEvidence,
    OpeningPhase,
    OpeningState,
)
from daguandan_bridge.live_v2.turn_core import PendingAction


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")


def _frame(index: int) -> FrameIdentity:
    return FrameIdentity("opening", 1, index, 1_000 + index, "roi", "capture")


def _lead(frame: int, winner: Seat, *, score: float = 0.95) -> LeadEvidence:
    values = tuple((seat, score if seat is winner else 0.2) for seat in Seat)
    return LeadEvidence(_frame(frame), values)


def _ready_for_lead() -> OpeningState:
    state = OpeningState().table_stable(_frame(0))
    state = state.observe_hand_level(HandLevelEvidence(_frame(1), HAND, "6"))
    return state.observe_hand_level(HandLevelEvidence(_frame(2), HAND, "6"))


def test_opening_requires_two_visual_hand_level_reads() -> None:
    state = OpeningState().table_stable(_frame(0))
    once = state.observe_hand_level(HandLevelEvidence(_frame(1), HAND, "6"))
    assert once.phase is OpeningPhase.CONFIRMING_HAND_LEVEL
    confirmed = once.observe_hand_level(HandLevelEvidence(_frame(2), HAND, "6"))
    assert confirmed.phase is OpeningPhase.CONFIRMING_LEAD
    assert confirmed.hand == HAND
    assert confirmed.round_level == "6"


def test_hand_or_level_change_resets_confirmation_streak() -> None:
    state = OpeningState().table_stable(_frame(0))
    state = state.observe_hand_level(HandLevelEvidence(_frame(1), HAND, "6"))
    changed = state.observe_hand_level(HandLevelEvidence(_frame(2), HAND, "7"))
    assert changed.phase is OpeningPhase.CONFIRMING_HAND_LEVEL
    assert len(changed.hand_level_votes) == 1


def test_lead_requires_unique_two_frame_visual_winner() -> None:
    state = _ready_for_lead()
    once = state.observe_lead(_lead(3, Seat.LEFT))
    assert once.phase is OpeningPhase.CONFIRMING_LEAD
    changed = once.observe_lead(_lead(4, Seat.RIGHT))
    assert len(changed.lead_votes) == 1
    confirmed = changed.observe_lead(_lead(5, Seat.RIGHT))
    assert confirmed.phase is OpeningPhase.CONFIRMING_OPENING_PLAY
    assert confirmed.lead_seat is Seat.RIGHT


def test_nonvisual_lead_seed_cannot_masquerade_as_confirmation() -> None:
    with pytest.raises(ValueError, match="visual evidence"):
        LeadEvidence(
            _frame(3),
            tuple((seat, 1.0 if seat is Seat.LEFT else 0.0) for seat in Seat),
            visual=False,
        )


def test_ambiguous_lead_scores_do_not_advance() -> None:
    state = _ready_for_lead()
    ambiguous = LeadEvidence(
        _frame(3),
        ((Seat.SELF, 0.85), (Seat.RIGHT, 0.82), (Seat.OPPOSITE, 0.1), (Seat.LEFT, 0.1)),
    )
    state = state.observe_lead(ambiguous)
    assert state.phase is OpeningPhase.CONFIRMING_LEAD
    assert state.lead_votes == ()


def test_only_confirmed_lead_can_complete_opening_play() -> None:
    state = _ready_for_lead().observe_lead(_lead(3, Seat.LEFT)).observe_lead(_lead(4, Seat.LEFT))
    foreign = PendingAction(Seat.SELF, 0, ActionKind.PLAY, ("3S",), (("S",),), _frame(5), _frame(6), 0.9)
    with pytest.raises(ValueError, match="visually confirmed lead"):
        state.complete_opening(foreign, action_id="bad")

    opening = PendingAction(Seat.LEFT, 0, ActionKind.PLAY, ("2S",), (("S",),), _frame(5), _frame(6), 0.9)
    completed = state.complete_opening(opening, action_id="opening-action")
    assert completed.phase is OpeningPhase.COMPLETED
    assert completed.result is not None
    assert completed.result.lead_seat is Seat.LEFT
    assert completed.result.opening_action.cards == ("2S",)
    assert completed.result.reset_observer_baselines


def test_opening_play_evidence_must_follow_lead_confirmation() -> None:
    state = _ready_for_lead().observe_lead(_lead(3, Seat.LEFT)).observe_lead(_lead(4, Seat.LEFT))
    stale = PendingAction(Seat.LEFT, 0, ActionKind.PLAY, ("2S",), (("S",),), _frame(3), _frame(5), 0.9)
    with pytest.raises(ValueError, match="follow lead confirmation"):
        state.complete_opening(stale, action_id="stale")
