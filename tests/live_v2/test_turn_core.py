from __future__ import annotations

from dataclasses import replace

import pytest

from daguandan_bridge.live_v2.candidates import ActionKind
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat
from daguandan_bridge.live_v2.turn_core import (
    CommittedAction,
    HistoryIntegrity,
    PendingAction,
    RuleAdvance,
    SimpleSeatRules,
    TurnCursor,
    TurnPhase,
    TurnState,
)


def _frame(sequence: int) -> FrameIdentity:
    return FrameIdentity("turn-core", 1, sequence, 10_000 + sequence, "roi-v1", "capture")


def _play(
    seat: Seat,
    token: int,
    first: int,
    last: int,
    *,
    cards: tuple[str, ...] = ("3S",),
    suits: tuple[tuple[str, ...], ...] = (("S",),),
) -> PendingAction:
    return PendingAction(
        seat=seat,
        turn_token=token,
        kind=ActionKind.PLAY,
        cards=cards,
        suit_options=suits,
        first_frame=_frame(first),
        last_frame=_frame(last),
        confidence=0.9,
    )


def _passed(seat: Seat, token: int, first: int, last: int) -> PendingAction:
    return PendingAction(
        seat=seat,
        turn_token=token,
        kind=ActionKind.PASS,
        cards=(),
        suit_options=(),
        first_frame=_frame(first),
        last_frame=_frame(last),
        confidence=0.99,
    )


def _opening(
    *,
    seat: Seat = Seat.LEFT,
    cards: tuple[str, ...] = ("2S",),
    suits: tuple[tuple[str, ...], ...] = (("S",),),
) -> CommittedAction:
    return CommittedAction.from_pending(
        _play(seat, 0, 1, 2, cards=cards, suits=suits),
        action_id="opening-action",
        turn_index=0,
    )


def _started(*, opening: CommittedAction | None = None) -> TurnState:
    return TurnState.opening().confirm_opening(
        opening or _opening(), rules=SimpleSeatRules()
    )


def test_opening_is_a_hard_barrier_and_binds_one_next_seat() -> None:
    opening = TurnState.opening()
    with pytest.raises(ValueError, match="waiting for expected seat"):
        opening.begin_action(_play(Seat.LEFT, 0, 1, 2))

    state = opening.confirm_opening(_opening(), rules=SimpleSeatRules())

    assert state.phase is TurnPhase.WAIT_EXPECTED
    assert state.opening_seat is Seat.LEFT
    assert state.cursor is not None
    assert state.cursor.current_seat is Seat.SELF
    assert state.cursor.lead_seat is Seat.LEFT
    assert state.cursor.turn_token == 1
    assert state.advice_allowed


def test_foreign_seat_never_becomes_pending_or_blocks_current_turn() -> None:
    state = _started()

    with pytest.raises(ValueError, match="foreign seat"):
        state.begin_action(_play(Seat.RIGHT, 1, 3, 4))

    assert state.pending is None
    assert state.cursor is not None and state.cursor.current_seat is Seat.SELF


def test_formal_actions_require_strictly_new_frames_and_commit_one_at_a_time() -> None:
    state = _started()
    with pytest.raises(ValueError, match="follow last formal action"):
        state.begin_action(_play(Seat.SELF, 1, 2, 3))

    confirming = state.begin_action(_play(Seat.SELF, 1, 3, 4))
    assert confirming.phase is TurnPhase.CONFIRMING_ACTION
    assert confirming.pending is not None
    with pytest.raises(ValueError, match="waiting for expected seat"):
        confirming.begin_action(_play(Seat.SELF, 1, 5, 6))

    advanced, action = confirming.commit_pending(
        action_id="self-action", rules=SimpleSeatRules()
    )
    assert action.seat is Seat.SELF
    assert len(advanced.committed) == 2
    assert advanced.pending is None
    assert advanced.cursor is not None
    assert advanced.cursor.current_seat is Seat.RIGHT


def test_three_passes_start_a_new_trick_at_last_non_pass_leader() -> None:
    state = _started()
    sequence = (
        (Seat.SELF, 1, 3, 4),
        (Seat.RIGHT, 2, 5, 6),
        (Seat.OPPOSITE, 3, 7, 8),
    )
    for index, (seat, token, first, last) in enumerate(sequence, start=1):
        state, _ = state.begin_action(
            _passed(seat, token, first, last)
        ).commit_pending(action_id=f"pass-{index}", rules=SimpleSeatRules())

    assert state.cursor is not None
    assert state.cursor.current_seat is Seat.LEFT
    assert state.cursor.lead_seat is Seat.LEFT
    assert state.cursor.last_non_pass_seat is Seat.LEFT
    assert state.cursor.passes_in_trick == 0
    assert state.cursor.trick_index == 1


def test_partial_suit_repair_updates_same_action_without_advancing_turn() -> None:
    state = _started(
        opening=_opening(
            cards=("2?", "2S"),
            suits=(("C", "D"), ("S",)),
        )
    )
    assert state.integrity is HistoryIntegrity.PARTIAL_SUITS
    assert not state.advice_allowed

    state, _ = state.begin_action(_passed(Seat.SELF, 1, 3, 4)).commit_pending(
        action_id="self-pass", rules=SimpleSeatRules()
    )
    cursor_before = state.cursor
    turn_count_before = len(state.committed)

    repaired, event = state.repair_action(
        repair_id="repair-opening",
        target_action_id="opening-action",
        suit_options=(("C",), ("S",)),
        evidence_frame=_frame(5),
        repaired_ms=10_005,
    )

    assert event.target_action_id == "opening-action"
    assert repaired.committed[0].action_id == "opening-action"
    assert repaired.committed[0].suit_options == (("C",), ("S",))
    assert len(repaired.committed) == turn_count_before
    assert repaired.cursor == cursor_before
    assert repaired.integrity is HistoryIntegrity.TRUSTED


def test_repair_only_narrows_partial_suits_and_never_rewrites_complete_action() -> None:
    partial = _started(
        opening=_opening(cards=("2?",), suits=(("C", "D"),))
    )
    with pytest.raises(ValueError, match="only narrow"):
        partial.repair_action(
            repair_id="expand",
            target_action_id="opening-action",
            suit_options=(("C", "D", "H"),),
            evidence_frame=_frame(3),
            repaired_ms=10_003,
        )

    complete = _started()
    with pytest.raises(ValueError, match="does not contain partial"):
        complete.repair_action(
            repair_id="rewrite",
            target_action_id="opening-action",
            suit_options=(("S",),),
            evidence_frame=_frame(3),
            repaired_ms=10_003,
        )


def test_desync_requires_explicit_integrity_confirmation_before_advice() -> None:
    state = _started().mark_desync()
    assert state.phase is TurnPhase.DESYNC
    assert state.integrity is HistoryIntegrity.DESYNC

    resynced = state.resync(current_seat=Seat.SELF, lead_seat=Seat.LEFT)
    assert resynced.phase is TurnPhase.WAIT_EXPECTED
    assert resynced.integrity is HistoryIntegrity.UNTRUSTED
    assert not resynced.advice_allowed

    trusted = resynced.confirm_resync_integrity()
    assert trusted.integrity is HistoryIntegrity.TRUSTED
    assert trusted.advice_allowed


class _TerminalRules(SimpleSeatRules):
    def after_action(
        self,
        action: CommittedAction,
        history: tuple[CommittedAction, ...],
        cursor: TurnCursor,
    ) -> RuleAdvance:
        return RuleAdvance(
            next_seat=None,
            lead_seat=cursor.lead_seat,
            trick_index=cursor.trick_index,
            finished=(Seat.LEFT, Seat.SELF),
            terminal=True,
        )


def test_terminal_repair_keeps_terminal_phase_and_does_not_create_cursor() -> None:
    state = _started(
        opening=_opening(cards=("2?",), suits=(("C", "D"),))
    )
    state, _ = state.begin_action(_passed(Seat.SELF, 1, 3, 4)).commit_pending(
        action_id="terminal-pass", rules=_TerminalRules()
    )
    assert state.phase is TurnPhase.FINISHED and state.cursor is None

    repaired, _ = state.repair_action(
        repair_id="terminal-repair",
        target_action_id="opening-action",
        suit_options=(("C",),),
        evidence_frame=_frame(5),
        repaired_ms=10_005,
    )

    assert repaired.phase is TurnPhase.FINISHED
    assert repaired.cursor is None
    assert len(repaired.committed) == 2


class _WindCatchRules(SimpleSeatRules):
    def after_action(
        self,
        action: CommittedAction,
        history: tuple[CommittedAction, ...],
        cursor: TurnCursor,
    ) -> RuleAdvance:
        return RuleAdvance(
            next_seat=Seat.OPPOSITE,
            lead_seat=Seat.OPPOSITE,
            trick_index=cursor.trick_index + 1,
            last_non_pass_seat=Seat.OPPOSITE,
            finished=(Seat.RIGHT,),
            wind_catch=Seat.OPPOSITE,
        )


def test_rule_port_can_skip_finished_player_and_bind_wind_catch_receiver() -> None:
    state = _started()
    state, _ = state.begin_action(_play(Seat.SELF, 1, 3, 4)).commit_pending(
        action_id="self-finishes", rules=_WindCatchRules()
    )

    assert state.finished == (Seat.RIGHT,)
    assert state.cursor is not None
    assert state.cursor.current_seat is Seat.OPPOSITE
    assert state.cursor.lead_seat is Seat.OPPOSITE
    assert state.cursor.trick_index == 1


def test_state_rejects_overlapping_committed_evidence_even_if_history_is_legal() -> None:
    opening = _opening()
    overlapping = CommittedAction.from_pending(
        _play(Seat.SELF, 1, 2, 3),
        action_id="overlap",
        turn_index=1,
    )
    cursor = TurnCursor(0, Seat.LEFT, Seat.RIGHT, 2, _frame(3))

    with pytest.raises(ValueError, match="strictly ordered"):
        TurnState(
            TurnPhase.WAIT_EXPECTED,
            cursor,
            committed=(opening, overlapping),
        )


def test_pending_evidence_requires_distinct_increasing_frames() -> None:
    with pytest.raises(ValueError, match="strictly increase"):
        _play(Seat.LEFT, 0, 1, 1)
    with pytest.raises(ValueError, match="same stream|strictly increase"):
        pending = _play(Seat.LEFT, 0, 1, 2)
        replace(
            pending,
            last_frame=FrameIdentity("other", 1, 3, 10_003, "roi-v1", "capture"),
        )
