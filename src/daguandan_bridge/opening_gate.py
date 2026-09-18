from __future__ import annotations

"""Pure opening-state validation shared by live startup and support replay."""

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Mapping, Sequence

from .danzero.state import GuanDanState, RANKS, Seat
from .live.turns import TURN_ORDER, next_active_seat


DEFAULT_TABLE_ANCHOR_THRESHOLD = 0.85
_SETTLEMENT_BUTTONS = frozenset({"change_table", "continue_game"})


@dataclass(frozen=True)
class ListeningPageSignal:
    """Cheap page evidence; unknown pages never authorize media writes."""

    stage: str
    anchor_score: float
    buttons: tuple[str, ...] = ()
    table_anchor_1_score: float | None = None
    table_anchor_2_score: float | None = None
    game_logo_anchor_score: float | None = None

    @property
    def allows_media(self) -> bool:
        return self.stage == "table"


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


def opening_semantic_key(seed: OpeningSessionSeed) -> tuple[object, ...]:
    """Compare game meaning; sorting tuples preserves duplicate deck cards."""
    action = seed.opening_action
    return (
        seed.round_level, tuple(sorted(seed.hand)), seed.lead_player,
        None if action is None else
        (action.actor, tuple(sorted(action.cards)), action.next_player),
    )


class OpeningTracker:
    """Bounded, generation-local hand and opening confirmation.

    Missing observations are not votes. Explicit changes invalidate their
    stage. A reduced hand never combines with an old cache to invent a deal.
    """

    def __init__(self, *, max_age_ms: int = 8_000) -> None:
        self.max_age_ms = int(max_age_ms)
        self.reset()

    def reset(self) -> None:
        """Start an explicitly new table phase (scene boundary/generation)."""
        self.generation: object = None
        self.started_ms: int | None = None
        self.hand_key: tuple[object, ...] | None = None
        self.hand_count = 0
        self.lead: Seat | None = None
        self.lead_count = 0
        self.candidate: OpeningSessionSeed | None = None
        self.candidate_count = 0
        self.last_observation: object = None
        self.last_ms: int | None = None
        self.completed = False
        self.reason = "waiting_table"
        self.saw_action = False
        self.evidence_level: str | None = None

    def discard_candidates(self) -> None:
        """Expire observations without erasing facts about this table phase."""
        generation, completed, saw_action = self.generation, self.completed, self.saw_action
        self.reset()
        self.generation = generation
        self.completed = completed
        self.saw_action = saw_action

    def _has_reliable_action_evidence(self, result: object) -> bool:
        """An invalid/weak one-frame glyph is not a permanent history fact.

        A legible first action must fit the confirmed lead and immediate next
        player. The conservative sticky guard is distinct from session votes;
        fresh complete first-action observations can still establish a seed.
        """
        effective = result
        if getattr(result, "lead_player", None) is None and self.lead_count >= 2:
            effective = SimpleNamespace(
                lead_player=self.lead,
                current_player=getattr(result, "current_player", None),
                events=tuple(getattr(result, "events", ()) or ()),
            )
        seed = build_opening_seed(effective, round_level="", hand=())
        return bool(seed and seed.opening_action and seed.opening_action.confidence >= .80)

    def observe(
        self, result: object, *, anchor_score: float | None,
        generation: object, monotonic_ms: int, observation_id: object = None,
    ) -> OpeningGateEvaluation:
        now = int(monotonic_ms)
        if self.generation != generation:
            self.reset()
            self.generation = generation
        buttons = set(getattr(result, "buttons", ()) or ())
        if buttons & _SETTLEMENT_BUTTONS:
            self.reset()
            self.generation = generation
            self.reason = "settlement_screen"
            return OpeningGateEvaluation(False, self.reason, None, None)
        if self.completed:
            return OpeningGateEvaluation(False, "already_started", None, None)
        if (
            (self.started_ms is not None and now - self.started_ms > self.max_age_ms)
            or (self.last_ms is not None and now < self.last_ms)
        ):
            self.discard_candidates()
        self.last_ms = now
        if observation_id is not None and observation_id == self.last_observation:
            return OpeningGateEvaluation(False, "duplicate_frame", None, None)
        self.last_observation = observation_id
        evaluation = evaluate_opening_gate(result, anchor_score=anchor_score)
        hand = tuple(getattr(result, "my_hand", ()) or ())
        level = str(getattr(result, "round_level", "") or "")
        if self.evidence_level is not None and level in RANKS and level != self.evidence_level:
            self.discard_candidates()
        if not hand and self.candidate is not None:
            current = getattr(result, "current_player", None)
            action = self.candidate.opening_action
            expected = action.next_player if action else self.candidate.lead_player
            if current in TURN_ORDER and expected in TURN_ORDER and current != expected:
                self.discard_candidates()
        if level in RANKS:
            self.evidence_level = level
        self.last_observation = observation_id
        self.last_ms = now
        if evaluation.reason == "table_anchor_unresolved":
            self.discard_candidates()
            self.reason = evaluation.reason
            return evaluation
        if self._has_reliable_action_evidence(result):
            self.saw_action = True
            if self.started_ms is None:
                self.started_ms = now
        if (
            0 < len(hand) != 27
            or evaluation.reason in {"hand_invalid", "hand_unresolved"}
        ):
            self.discard_candidates()
            self.reason = (
                "missed_opening"
                if 0 < len(hand) < 27
                else evaluation.reason
            )
            return OpeningGateEvaluation(False, self.reason, None, None)
        if evaluation.normalized_hand is None:
            # Explicit seat/first-action evidence may arrive while the hand
            # ROI is temporarily unreadable. Keep the marker stage separate.
            marker = getattr(result, "lead_player", None)
            if marker in TURN_ORDER:
                if marker != self.lead:
                    self.lead, self.lead_count = marker, 0
                self.lead_count += 1
                if self.started_ms is None:
                    self.started_ms = now
            self.reason = evaluation.reason
            return evaluation
        normalized = evaluation.normalized_hand
        key = (level, tuple(sorted(normalized)))
        if key != self.hand_key:
            prior_lead, prior_lead_count = self.lead, self.lead_count
            prior_action_seen = self.saw_action
            prior_started_ms = self.started_ms
            first_hand = self.hand_key is None
            self.discard_candidates()
            self.started_ms = now
            self.hand_key = key
            self.evidence_level = level
            if first_hand:
                self.lead, self.lead_count = prior_lead, prior_lead_count
                self.saw_action = prior_action_seen
                self.started_ms = prior_started_ms if prior_started_ms is not None else now
        self.last_observation = observation_id
        self.last_ms = now
        self.hand_count += 1
        lead = getattr(result, "lead_player", None)
        if lead in TURN_ORDER:
            if lead != self.lead:
                self.lead = lead
                self.lead_count = 0
                self.candidate = None
                self.candidate_count = 0
            self.lead_count += 1
        effective = result
        events = tuple(getattr(result, "events", ()) or ())
        if not events and self.saw_action:
            self.candidate = None
            self.candidate_count = 0
            self.reason = "opening_seed_invalid"
            return OpeningGateEvaluation(False, self.reason, None, normalized)
        # A confirmed marker can disappear during the first play. The action
        # must still prove that actor and its immediate next player.
        if lead is None and self.lead_count >= 2 and events:
            effective = SimpleNamespace(
                lead_player=self.lead,
                current_player=getattr(result, "current_player", None),
                events=events,
            )
        candidate = build_opening_seed(effective, round_level=level, hand=normalized)
        if candidate is None:
            self.candidate = None
            self.candidate_count = 0
            self.reason = "opening_seed_invalid"
            return OpeningGateEvaluation(False, self.reason, None, normalized)
        if self.candidate is None or opening_semantic_key(candidate) != opening_semantic_key(self.candidate):
            self.candidate_count = 0
        self.candidate = candidate
        self.candidate_count += 1
        if self.hand_count < 2 or self.candidate_count < 2:
            self.reason = "confirming_hand" if self.hand_count < 2 else "confirming_opening"
            return OpeningGateEvaluation(False, self.reason, None, normalized)
        self.completed = True
        self.reason = "ready"
        return OpeningGateEvaluation(True, self.reason, candidate, normalized)


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
    try:
        anchor_ready = (
            anchor_score is not None
            and float(anchor_score) >= float(anchor_required)
        )
    except (TypeError, ValueError, OverflowError):
        anchor_ready = False
    if not anchor_ready:
        return OpeningGateEvaluation(False, "table_anchor_unresolved", None, None)
    level = str(getattr(result, "round_level", "") or "")
    if level not in RANKS:
        return OpeningGateEvaluation(False, "round_level_unresolved", None, None)
    hand = tuple(str(card) for card in tuple(getattr(result, "my_hand", ()) or ()))
    if len(hand) != 27:
        return OpeningGateEvaluation(False, "hand_count_mismatch", None, None)
    if any(card.endswith("?") for card in hand):
        return OpeningGateEvaluation(False, "hand_unresolved", None, None)
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
    try:
        confidence = float(getattr(event, "confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return OpeningSessionSeed(
        round_level,
        hand,
        lead_player,
        OpeningActionSeed(
            actor=actor,
            cards=cards,
            next_player=current_player,
            confidence=confidence,
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
    "OpeningTracker",
    "ListeningPageSignal",
    "opening_semantic_key",
    "build_opening_seed",
    "evaluate_opening_gate",
    "serialized_result",
]
