from __future__ import annotations

"""Pure opening-state validation shared by live startup and support replay."""

from dataclasses import dataclass
from typing import Mapping, Sequence

from .danzero.state import GuanDanState, RANKS, Seat
from .live.turns import TURN_ORDER, next_active_seat


DEFAULT_TABLE_ANCHOR_THRESHOLD = 0.85
_SETTLEMENT_BUTTONS = frozenset({"change_table", "continue_game"})


@dataclass(frozen=True)
class OpeningActionSeed:
    actor: Seat
    cards: tuple[str, ...]
    next_player: Seat
    confidence: float
    source: str


@dataclass(frozen=True)
class OpeningSessionSeed:
    round_level: str
    hand: tuple[str, ...]
    lead_player: Seat | None
    opening_action: OpeningActionSeed | None = None


@dataclass(frozen=True)
class OpeningGateEvaluation:
    ready: bool
    reason: str
    seed: OpeningSessionSeed | None
    normalized_hand: tuple[str, ...] | None


def evaluate_opening_gate(
    result: object,
    *,
    anchor_score: float | None,
    anchor_required: float = DEFAULT_TABLE_ANCHOR_THRESHOLD,
) -> OpeningGateEvaluation:
    """Apply the same fail-closed opening rules used by the live controller."""

    buttons = {str(item) for item in tuple(getattr(result, "buttons", ()) or ())}
    if buttons & _SETTLEMENT_BUTTONS:
        return OpeningGateEvaluation(False, "settlement_screen", None, None)
    if anchor_score is None or float(anchor_score) < float(anchor_required):
        return OpeningGateEvaluation(False, "table_anchor_unresolved", None, None)
    level = str(getattr(result, "round_level", "") or "")
    if level not in RANKS:
        return OpeningGateEvaluation(False, "round_level_unresolved", None, None)
    hand = tuple(str(card) for card in tuple(getattr(result, "my_hand", ()) or ()))
    if len(hand) != 27:
        return OpeningGateEvaluation(False, "hand_count_mismatch", None, None)
    try:
        state = GuanDanState()
        state.confirm_hand(hand)
    except Exception:
        return OpeningGateEvaluation(False, "hand_invalid", None, None)
    normalized = tuple(state.my_hand)
    seed = build_opening_seed(result, round_level=level, hand=normalized)
    if seed is None:
        return OpeningGateEvaluation(False, "opening_seed_invalid", None, normalized)
    return OpeningGateEvaluation(True, "ready", seed, normalized)


def build_opening_seed(
    result: object,
    *,
    round_level: str,
    hand: tuple[str, ...],
) -> OpeningSessionSeed | None:
    lead_player = getattr(result, "lead_player", None)
    current_player = getattr(result, "current_player", None)
    events = tuple(getattr(result, "events", ()) or ())
    if not events:
        if lead_player is None and current_player is None:
            return OpeningSessionSeed(round_level, hand, None)
        if lead_player in TURN_ORDER and current_player in {None, lead_player}:
            return OpeningSessionSeed(round_level, hand, lead_player)
        return None
    if len(events) != 1 or lead_player not in TURN_ORDER:
        return None
    event = events[0]
    actor = getattr(event, "player", None)
    cards = tuple(str(card) for card in getattr(event, "cards", ()) or ())
    if (
        actor != lead_player
        or actor not in TURN_ORDER
        or bool(getattr(event, "is_pass", False))
        or not cards
        or any("?" in card for card in cards)
        or current_player != next_active_seat(actor, frozenset())
    ):
        return None
    return OpeningSessionSeed(
        round_level,
        hand,
        lead_player,
        OpeningActionSeed(
            actor=actor,
            cards=cards,
            next_player=current_player,
            confidence=float(getattr(event, "confidence", 0.0)),
            source=str(getattr(event, "source", "visual_opening_anchor")),
        ),
    )


def serialized_result(
    *,
    round_level: str | None,
    hand: Sequence[str],
    lead_player: object = None,
    current_player: object = None,
    buttons: Sequence[str] = (),
    events: Sequence[Mapping[str, object]] = (),
) -> object:
    """Build an attribute object for replay without importing Qt/controller code."""

    from types import SimpleNamespace

    event_values = tuple(SimpleNamespace(**dict(item)) for item in events)
    return SimpleNamespace(
        round_level=round_level,
        my_hand=tuple(hand),
        lead_player=lead_player,
        current_player=current_player,
        buttons=tuple(buttons),
        events=event_values,
    )


__all__ = [
    "DEFAULT_TABLE_ANCHOR_THRESHOLD",
    "OpeningActionSeed",
    "OpeningGateEvaluation",
    "OpeningSessionSeed",
    "build_opening_seed",
    "evaluate_opening_gate",
    "serialized_result",
]
