from __future__ import annotations

from types import SimpleNamespace

from daguandan_bridge.application.live_v2_candidate_gate import gate_visual_candidates
from daguandan_bridge.live_v2.candidates import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    EvidenceOrigin,
)
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, VersionIdentity


def _candidate(name: str, seat: Seat, cards: tuple[str, ...], frame: int) -> ActionCandidate:
    first = FrameIdentity("gate", 1, frame, 1_000 + frame, "roi", "capture")
    last = FrameIdentity("gate", 1, frame + 1, 1_001 + frame, "roi", "capture")
    return ActionCandidate(
        name,
        VersionIdentity("gate", 1, 1, 0, 0),
        seat,
        ActionKind.PLAY,
        cards,
        tuple((card,) for card in cards),
        (f"e-{name}-1", f"e-{name}-2"),
        0,
        first,
        last,
        1_100 + frame,
        0.9,
        CandidateReason.STABLE_PLAY,
        EvidenceOrigin.VISUAL,
    )


def _snapshot(*, current: Seat | None, lead: Seat | None, history=()):
    return SimpleNamespace(
        current_seat=current,
        lead_seat=lead,
        play_history=tuple(history),
    )


def test_opening_multi_seat_conflict_never_guesses_a_lead() -> None:
    left = _candidate("left", Seat.LEFT, ("2S",), 2)
    self_play = _candidate("self", Seat.SELF, ("4S",), 3)
    result = gate_visual_candidates(_snapshot(current=None, lead=None), (left, self_play))
    assert result.selected == ()
    assert result.reason == "eligible_candidate_conflict"


def test_opening_known_lead_ignores_foreign_static_cards() -> None:
    left = _candidate("left", Seat.LEFT, ("2S",), 2)
    self_play = _candidate("self", Seat.SELF, ("4S",), 3)
    result = gate_visual_candidates(_snapshot(current=None, lead=Seat.LEFT), (self_play, left))
    assert result.selected == (left,)
    assert result.ignored_ids == ("self",)


def test_normal_turn_only_selects_authoritative_current_seat() -> None:
    snapshot = _snapshot(current=Seat.SELF, lead=Seat.LEFT, history=(object(),))
    self_play = _candidate("self", Seat.SELF, ("4S",), 4)
    right = _candidate("right", Seat.RIGHT, ("5S",), 4)
    result = gate_visual_candidates(snapshot, (self_play, right))
    assert result.selected == (self_play,)
    assert result.ignored_ids == ("right",)


def test_conflicting_current_seat_reads_are_not_queued_or_guessed() -> None:
    snapshot = _snapshot(current=Seat.SELF, lead=Seat.LEFT, history=(object(),))
    first = _candidate("a", Seat.SELF, ("4S",), 4)
    second = _candidate("b", Seat.SELF, ("5S",), 5)
    result = gate_visual_candidates(snapshot, (first, second))
    assert result.selected == ()
    assert result.reason == "eligible_candidate_conflict"


def test_duplicate_same_action_keeps_latest_evidence_only() -> None:
    snapshot = _snapshot(current=Seat.SELF, lead=Seat.LEFT, history=(object(),))
    first = _candidate("a", Seat.SELF, ("4S",), 4)
    second = _candidate("b", Seat.SELF, ("4S",), 5)
    result = gate_visual_candidates(snapshot, (first, second))
    assert result.selected == (second,)
    assert result.ignored_ids == ("a",)
