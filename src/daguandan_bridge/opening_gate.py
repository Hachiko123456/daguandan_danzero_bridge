from __future__ import annotations

"""Pure opening-state validation shared by live startup and support replay."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Mapping, Sequence

from .danzero.state import GuanDanState, RANKS, Seat
from .live.lead_evidence import LeadEvidence
from .live.turns import TURN_ORDER, next_active_seat


DEFAULT_TABLE_ANCHOR_THRESHOLD = 0.85
_SETTLEMENT_BUTTONS = frozenset({"change_table", "continue_game"})
MIN_OPENING_ACTION_CONFIDENCE = 0.80

# Canonical opening evidence states.  These are deliberately strings rather
# than an Enum so JSON/replay callers can consume them without an adapter.
NOT_READY = "NOT_READY"
READY_WAITING_FIRST_ACTION = "READY_WAITING_FIRST_ACTION"
READY_ACTION_CONFIRMED = "READY_ACTION_CONFIRMED"
BLOCKED = "BLOCKED"
CONFLICT = "CONFLICT"
OPENING_EVIDENCE_STATUSES = frozenset({
    NOT_READY,
    READY_WAITING_FIRST_ACTION,
    READY_ACTION_CONFIRMED,
    BLOCKED,
    CONFLICT,
})

# Compatibility mapping for persisted/legacy reason strings.  New callers
# should use ``evaluation.status``; old callers can continue to inspect
# ``evaluation.reason`` unchanged.
LEGACY_REASON_STATUS = {
    "ready": READY_ACTION_CONFIRMED,
    "ready_waiting_first_action": READY_WAITING_FIRST_ACTION,
    "confirming_hand": NOT_READY,
    "confirming_opening": NOT_READY,
    "duplicate_frame": NOT_READY,
    "waiting_table": NOT_READY,
    "already_started": READY_ACTION_CONFIRMED,
    "candidate_conflict": CONFLICT,
    "opening_seed_conflict": CONFLICT,
    "settlement_screen": BLOCKED,
    "table_anchor_unresolved": NOT_READY,
    "round_level_unresolved": NOT_READY,
    "hand_count_mismatch": NOT_READY,
    "hand_unresolved": BLOCKED,
    "hand_invalid": BLOCKED,
    "opening_seed_invalid": BLOCKED,
    "action_unresolved": NOT_READY,
    "missed_opening": BLOCKED,
}


def opening_status_for_reason(reason: str) -> str:
    """Map a legacy reason to the canonical opening evidence status."""

    if reason in LEGACY_REASON_STATUS:
        return LEGACY_REASON_STATUS[reason]
    if "conflict" in reason:
        return CONFLICT
    if reason.startswith(("hand_", "opening_", "action_", "missed_")):
        return BLOCKED
    return NOT_READY


def legacy_reason_for_status(status: str) -> str:
    """Return the stable legacy reason for a canonical status.

    This is intentionally one-way and conservative: detailed failure reasons
    remain available on ``OpeningGateEvaluation.reason``.
    """

    return {
        NOT_READY: "confirming_opening",
        READY_WAITING_FIRST_ACTION: "ready_waiting_first_action",
        READY_ACTION_CONFIRMED: "ready",
        BLOCKED: "opening_seed_invalid",
        CONFLICT: "candidate_conflict",
    }.get(status, "confirming_opening")


def _seat_value(value: object) -> str | None:
    """Normalize legacy strings and live-v2 Seat values at the opening boundary."""
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        return None
    if raw.startswith("Seat."):
        raw = raw.split(".", 1)[1].lower()
    return raw if raw in TURN_ORDER else None


def _rank_only(card: object) -> str:
    value = str(card)
    return value[:-1] if value.endswith("?") or value[-1:] in "SHCD" else value


def _opening_suit_options(cards: tuple[str, ...], raw_options: object) -> tuple[tuple[str, ...], ...]:
    supplied = tuple(tuple(str(item) for item in choices) for choices in (raw_options or ()))
    result: list[tuple[str, ...]] = []
    for index, card in enumerate(cards):
        rank = card[:-1] if card.endswith("?") or card[-1:] in "SHCD" else card
        choices = supplied[index] if index < len(supplied) else ()
        if choices:
            result.append(tuple(dict.fromkeys(
                f"{rank}{choice}" if choice in "SHCD" and rank not in {"small_joker", "big_joker"}
                else choice
                for choice in choices
            )))
        elif card.endswith("?"):
            result.append(tuple(f"{rank}{suit}" for suit in "SHCD"))
        elif card not in {"small_joker", "big_joker"} and card[-1:] in "SHCD":
            result.append((card,))
        else:
            result.append((card,))
    return tuple(result)


def _valid_action_cards(cards: tuple[str, ...]) -> bool:
    """Validate an observed play without inventing rank/suit information."""

    if not cards or any(card.endswith("?") for card in cards):
        return False
    try:
        state = GuanDanState()
        state.confirm_hand(cards, source="opening_action")
    except Exception:
        return False
    return True


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
    suit_options: tuple[tuple[str, ...], ...] = ()


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
    status: str = ""

    def __post_init__(self) -> None:
        status = self.status or opening_status_for_reason(self.reason)
        if status not in OPENING_EVIDENCE_STATUSES:
            status = opening_status_for_reason(self.reason)
        object.__setattr__(self, "status", status)

    @property
    def session_ready(self) -> bool:
        """Whether a session seed is safe to establish (action may be absent)."""

        return self.status in {READY_WAITING_FIRST_ACTION, READY_ACTION_CONFIRMED}

    @property
    def action_confirmed(self) -> bool:
        return self.status == READY_ACTION_CONFIRMED


def opening_semantic_key(seed: OpeningSessionSeed) -> tuple[object, ...]:
    """Compare game meaning; sorting tuples preserves duplicate deck cards."""
    action = seed.opening_action
    return (
        seed.round_level, tuple(sorted(seed.hand)), _seat_value(seed.lead_player),
        None if action is None else
        (
            _seat_value(action.actor),
            tuple(sorted(_rank_only(card) for card in action.cards)),
            _seat_value(action.next_player),
        ),
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
        # Raw lead candidates survive until an explicit scope/settlement
        # boundary, even when no formal OpeningSessionSeed exists yet.
        self.candidate_evidence: tuple[LeadEvidence, ...] = ()
        self.candidate_history: tuple[LeadEvidence, ...] = ()
        self.last_observation: object = None
        self.last_ms: int | None = None
        self.completed = False
        self.reason = "waiting_table"
        self.status = NOT_READY
        self.waiting_for_action = False
        self.saw_action = False
        self.evidence_level: str | None = None

    def discard_candidates(self) -> None:
        """Expire observations without inventing a session from stale data."""

        generation = self.generation
        history = (self.candidate_history + self.candidate_evidence)[-16:]
        saw_action = self.saw_action
        self.reset()
        self.generation = generation
        self.saw_action = saw_action
        self.candidate_history = history


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
        return bool(
            seed
            and seed.opening_action
            and seed.opening_action.confidence >= MIN_OPENING_ACTION_CONFIDENCE
        )

    def observe(
        self, result: object, *, anchor_score: float | None,
        generation: object, monotonic_ms: int, observation_id: object = None,
    ) -> OpeningGateEvaluation:
        """Accumulate the same opening semantics used by single-frame replay.

        Two complete semantic observations establish a session seed.  The seed
        may intentionally have no opening action yet; an action is confirmed
        separately after its actor/cards/successor/confidence pass validation.
        """

        now = int(monotonic_ms)
        if self.generation != generation:
            self.reset()
            self.generation = generation
        buttons = set(getattr(result, "buttons", ()) or ())
        if buttons & _SETTLEMENT_BUTTONS:
            self.reset()
            self.generation = generation
            self.reason = "settlement_screen"
            self.status = BLOCKED
            return OpeningGateEvaluation(False, self.reason, None, None, self.status)

        # A fully action-confirmed tracker is terminal until an explicit
        # generation/settlement boundary.  A waiting-first-action tracker is
        # deliberately *not* terminal: it must continue observing the first
        # real play and detect reductions/conflicts.
        if self.completed and self.status == READY_ACTION_CONFIRMED:
            return OpeningGateEvaluation(
                False, "already_started", None, None, READY_ACTION_CONFIRMED
            )
        if (
            (self.started_ms is not None and now - self.started_ms > self.max_age_ms)
            or (self.last_ms is not None and now < self.last_ms)
        ):
            self.discard_candidates()
        if observation_id is not None and observation_id == self.last_observation:
            return OpeningGateEvaluation(False, "duplicate_frame", None, None, NOT_READY)
        self.last_observation = observation_id
        self.last_ms = now

        inferred_lead = False
        lead_evidence = tuple(
            getattr(result, "lead_evidence", ())
            or getattr(getattr(result, "opening_signal", None), "lead_evidence", ())
            or getattr(result, "candidates", ())
            or ()
        )
        if lead_evidence:
            self.candidate_evidence = lead_evidence
            conflicts = tuple(item for item in lead_evidence if item.status == "conflict")
            preferred = next(
                (item for item in lead_evidence
                 if item.status in {"pending_confirmation", "confirmed"}),
                None,
            )
            explicit_lead = _seat_value(getattr(result, "lead_player", None))
            if conflicts:
                self.candidate_history = (self.candidate_history + conflicts)[-16:]
                self.candidate = None
                self.candidate_count = 0
                self.completed = False
                self.status = CONFLICT
                self.reason = "candidate_conflict"
                return OpeningGateEvaluation(False, self.reason, None, None, CONFLICT)
            if preferred is not None and explicit_lead is None:
                inferred_lead = True
                result = SimpleNamespace(
                    round_level=getattr(result, "round_level", None),
                    my_hand=getattr(result, "my_hand", ()),
                    lead_player=preferred.candidate_seat,
                    current_player=getattr(result, "current_player", None),
                    events=tuple(getattr(result, "events", ()) or ()),
                    buttons=tuple(getattr(result, "buttons", ()) or ()),
                )
            elif preferred is not None and explicit_lead != preferred.candidate_seat:
                self.candidate_history = (
                    self.candidate_history
                    + (replace(preferred, status="rejected", rejection_reason="lead_conflict"),)
                )[-16:]
                self.candidate = None
                self.candidate_count = 0
                self.completed = False
                self.status = CONFLICT
                self.reason = "candidate_conflict"
                return OpeningGateEvaluation(False, self.reason, None, None, CONFLICT)

        evaluation = evaluate_opening_gate(result, anchor_score=anchor_score)
        hand = tuple(getattr(result, "my_hand", ()) or ())
        level = str(getattr(result, "round_level", "") or "")
        waiting = self.waiting_for_action and self.candidate is not None
        if self.evidence_level is not None and level in RANKS and level != self.evidence_level:
            self.discard_candidates()
            waiting = False
        if not hand and self.candidate is not None:
            current = _seat_value(getattr(result, "current_player", None))
            action = self.candidate.opening_action
            expected = action.next_player if action else self.candidate.lead_player
            if current in TURN_ORDER and expected in TURN_ORDER and current != expected:
                self.discard_candidates()
                waiting = False
        if level in RANKS:
            self.evidence_level = level

        if evaluation.reason == "table_anchor_unresolved":
            self.discard_candidates()
            self.reason = evaluation.reason
            self.status = NOT_READY
            return OpeningGateEvaluation(False, self.reason, None, None, NOT_READY)

        events = tuple(getattr(result, "events", ()) or ())
        # A reduced hand is invalid while establishing the opening hand.  Once
        # the waiting-first-action seed exists, a reduced hand is acceptable
        # only when the same frame contains a validated first-play candidate;
        # the seed retains the pre-play 27-card hand for auditability.
        reduced_action_candidate: OpeningSessionSeed | None = None
        if 0 < len(hand) != 27:
            if waiting and events and level == self.candidate.round_level:
                effective = result
                if _seat_value(getattr(result, "lead_player", None)) is None and self.lead_count >= 2:
                    effective = SimpleNamespace(
                        lead_player=self.lead,
                        current_player=getattr(result, "current_player", None),
                        events=events,
                    )
                reduced_action_candidate = build_opening_seed(
                    effective,
                    round_level=self.candidate.round_level,
                    hand=self.candidate.hand,
                )
                if reduced_action_candidate is None:
                    self.reason = "action_unresolved"
                    self.status = NOT_READY
                    return OpeningGateEvaluation(False, self.reason, None, self.candidate.hand, NOT_READY)
            else:
                self.discard_candidates()
                self.reason = "missed_opening"
                self.status = BLOCKED
                return OpeningGateEvaluation(False, self.reason, None, None, BLOCKED)

        if evaluation.normalized_hand is None and reduced_action_candidate is None:
            marker = _seat_value(getattr(result, "lead_player", None))
            if marker in TURN_ORDER:
                if marker != self.lead:
                    self.lead, self.lead_count = marker, 0
                self.lead_count += 1
                if self.started_ms is None:
                    self.started_ms = now
            self.reason = evaluation.reason
            self.status = opening_status_for_reason(self.reason)
            return OpeningGateEvaluation(False, self.reason, None, None, self.status)

        normalized = (
            self.candidate.hand
            if reduced_action_candidate is not None and self.candidate is not None
            else evaluation.normalized_hand
        )
        if normalized is None:
            self.reason = evaluation.reason
            self.status = opening_status_for_reason(self.reason)
            return OpeningGateEvaluation(False, self.reason, None, None, self.status)
        normalized = tuple(normalized)
        key = (level or (self.candidate.round_level if self.candidate else ""), tuple(sorted(normalized)))
        if key != self.hand_key:
            prior_lead, prior_lead_count = self.lead, self.lead_count
            prior_action_seen = self.saw_action
            prior_started_ms = self.started_ms
            first_hand = self.hand_key is None
            self.discard_candidates()
            self.started_ms = now
            self.hand_key = key
            self.evidence_level = key[0]
            if first_hand:
                self.lead, self.lead_count = prior_lead, prior_lead_count
                self.saw_action = prior_action_seen
                self.started_ms = prior_started_ms if prior_started_ms is not None else now
            waiting = False

        self.last_observation = observation_id
        self.last_ms = now
        if key[0] in RANKS:
            self.evidence_level = key[0]
        self.hand_count += 1
        lead = _seat_value(getattr(result, "lead_player", None))
        if lead in TURN_ORDER:
            if lead != self.lead:
                self.lead = lead
                self.lead_count = 0
                self.candidate = None
                self.candidate_count = 0
            self.lead_count += 1

        effective = result
        if lead is None and self.lead_count >= 2 and events:
            effective = SimpleNamespace(
                lead_player=self.lead,
                current_player=getattr(result, "current_player", None),
                events=events,
            )
        candidate = reduced_action_candidate or build_opening_seed(
            effective, round_level=key[0], hand=normalized
        )
        if candidate is None:
            # A previously established waiting seed remains usable while a
            # single frame is inconclusive, but no action is fabricated.
            if self.waiting_for_action and self.candidate is not None and not events:
                self.reason = "ready_waiting_first_action"
                return OpeningGateEvaluation(True, self.reason, self.candidate, normalized, READY_WAITING_FIRST_ACTION)
            self.candidate = None
            self.candidate_count = 0
            self.completed = False
            self.reason = "opening_seed_invalid"
            self.status = (
                evaluation.status
                if evaluation.status in {BLOCKED, CONFLICT}
                else BLOCKED
            )
            return OpeningGateEvaluation(
                False, self.reason, None, normalized, self.status
            )

        if self.candidate is None or opening_semantic_key(candidate) != opening_semantic_key(self.candidate):
            self.candidate_count = 0
            if self.candidate_evidence:
                self.candidate_history = (
                    self.candidate_history
                    + tuple(
                        replace(item, status="rejected", rejection_reason="candidate_changed")
                        for item in self.candidate_evidence
                        if item.status not in {"rejected", "conflict"}
                    )
                )[-16:]
        self.candidate = candidate
        self.candidate_count += 1
        if self.hand_count < 2 or self.candidate_count < 2:
            self.completed = False
            self.status = NOT_READY
            self.reason = "confirming_hand" if self.hand_count < 2 else "confirming_opening"
            return OpeningGateEvaluation(False, self.reason, None, normalized, NOT_READY)

        if candidate.opening_action is None:
            # A ranked LeadEvidence candidate without an explicit lead marker
            # is useful for continuity, but is not enough to establish a
            # waiting session.  An actual validated action may still confirm
            # the inferred actor below.
            if inferred_lead:
                self.completed = False
                self.waiting_for_action = False
                self.status = NOT_READY
                self.reason = "confirming_opening"
                return OpeningGateEvaluation(False, self.reason, None, normalized, NOT_READY)
            self.completed = True
            self.waiting_for_action = True
            self.reason = "ready_waiting_first_action"
            self.status = READY_WAITING_FIRST_ACTION
            return OpeningGateEvaluation(True, self.reason, candidate, normalized, self.status)

        self.completed = True
        self.waiting_for_action = False
        self.saw_action = True
        self.reason = "ready"
        self.status = READY_ACTION_CONFIRMED
        if self.candidate_evidence:
            self.candidate_evidence = tuple(
                replace(
                    item,
                    status=("confirmed" if item.candidate_seat == candidate.lead_player else item.status),
                    rejection_reason=None if item.candidate_seat == candidate.lead_player else item.rejection_reason,
                )
                for item in self.candidate_evidence
            )
        return OpeningGateEvaluation(True, self.reason, candidate, normalized, self.status)


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
        lead = _seat_value(getattr(result, "lead_player", None))
        current = _seat_value(getattr(result, "current_player", None))
        events = tuple(getattr(result, "events", ()) or ())
        conflict = (
            lead in TURN_ORDER
            and current in TURN_ORDER
            and (
                (not events and current != lead)
                or (events and current != next_active_seat(lead, frozenset()))
            )
        )
        return OpeningGateEvaluation(
            False,
            "opening_seed_invalid",
            None,
            normalized,
            CONFLICT if conflict else BLOCKED,
        )
    if seed.opening_action is None:
        return OpeningGateEvaluation(
            True, "ready_waiting_first_action", seed, normalized,
            READY_WAITING_FIRST_ACTION,
        )
    return OpeningGateEvaluation(
        True, "ready", seed, normalized, READY_ACTION_CONFIRMED
    )


def build_opening_seed(
    result: object,
    *,
    round_level: str,
    hand: tuple[str, ...],
) -> OpeningSessionSeed | None:
    lead_player = _seat_value(getattr(result, "lead_player", None))
    current_player = _seat_value(getattr(result, "current_player", None))
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
    actor = _seat_value(getattr(event, "player", None))
    cards = tuple(str(card) for card in getattr(event, "cards", ()) or ())
    next_player = _seat_value(current_player)
    if (
        actor != lead_player
        or actor not in TURN_ORDER
        or bool(getattr(event, "is_pass", False))
        or not _valid_action_cards(cards)
        or next_player != next_active_seat(actor, frozenset())
    ):
        return None
    try:
        confidence = float(getattr(event, "confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(confidence)
        or not MIN_OPENING_ACTION_CONFIDENCE <= confidence <= 1.0
    ):
        return None
    return OpeningSessionSeed(
        round_level,
        hand,
        lead_player,
        OpeningActionSeed(
            actor=actor,
            cards=cards,
            next_player=next_player,
            confidence=confidence,
            source=str(getattr(event, "source", "visual_opening_anchor")),
            suit_options=_opening_suit_options(cards, getattr(event, "suit_options", ())),
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
    lead_evidence: Sequence[LeadEvidence] = (),
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
        lead_evidence=tuple(lead_evidence),
    )


__all__ = [
    "DEFAULT_TABLE_ANCHOR_THRESHOLD",
    "MIN_OPENING_ACTION_CONFIDENCE",
    "NOT_READY",
    "READY_WAITING_FIRST_ACTION",
    "READY_ACTION_CONFIRMED",
    "BLOCKED",
    "CONFLICT",
    "OPENING_EVIDENCE_STATUSES",
    "LEGACY_REASON_STATUS",
    "opening_status_for_reason",
    "legacy_reason_for_status",
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
