"""Keep visually occluded suits explicit until an advisor needs concrete cards.

The live record stores ``8?`` rather than permanently guessing one of the
four suits.  Concrete assignments are only created in short-lived validation
or advice branches, so a later visual correction cannot corrupt the canonical
history or double-deck accounting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable

from ..danzero.state import GuanDanState, PlayEvent
from .occlusion_evidence import OcclusionEvidence, utc_now_iso


SUITS: tuple[str, ...] = ("S", "H", "C", "D")
_SUIT_SET = frozenset(SUITS)


@dataclass(frozen=True)
class SuitStateVariants:
    states: tuple[GuanDanState, ...]
    truncated: bool = False
    limit: int = 32
    used_relaxed_suits: bool = False


def is_unknown_suit_card(card: str) -> bool:
    return len(str(card)) >= 2 and str(card).endswith("?")


def normalized_suit_options(
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return one deterministic suit candidate list for every card.

    A concrete card carries its own exact suit.  Unknown ranks fall back to
    all four suits when an old recording has no colour metadata.
    """

    card_values = tuple(str(card) for card in cards)
    supplied = tuple(tuple(str(suit) for suit in choices) for choices in suit_options)
    result: list[tuple[str, ...]] = []
    for index, card in enumerate(card_values):
        if len(card) >= 2 and card[-1] in _SUIT_SET:
            result.append((card[-1],))
            continue
        if not is_unknown_suit_card(card):
            result.append(())
            continue
        choices = supplied[index] if index < len(supplied) else ()
        normalized = tuple(dict.fromkeys(suit for suit in choices if suit in _SUIT_SET))
        result.append(normalized or SUITS)
    return tuple(result)


def feasible_action_variants(
    *,
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
    known_cards: Iterable[str] = (),
    known_suit_options: Iterable[Iterable[str]] = (),
    candidate_already_known: bool = False,
    limit: int = 64,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate candidate suits that can exist in a double deck.

    Unknown historical cards are allocated together with the candidate before
    applying the two-copies-per-exact-card bound.  This prevents a black 8
    from being silently treated as a third ``8S`` while still allowing the
    action to proceed through the live state machine.  If those *historical*
    soft observations cannot be assigned consistently, retry without letting
    them consume an exact physical card.  A transient effect must never turn
    an old ``8?`` into a permanent blocker for a later clear ``8S``.
    """

    candidate = tuple(str(card) for card in cards)
    candidate_options = tuple(tuple(str(suit) for suit in options) for options in suit_options)
    known = tuple(str(card) for card in known_cards)
    known_options = tuple(tuple(str(suit) for suit in options) for options in known_suit_options)
    strict = _enumerate_action_variants(
        cards=candidate,
        suit_options=candidate_options,
        known_cards=known,
        known_suit_options=known_options,
        candidate_already_known=candidate_already_known,
        limit=limit,
        include_historical_unknowns=True,
    )
    if strict:
        return strict
    return _enumerate_action_variants(
        cards=candidate,
        suit_options=candidate_options,
        known_cards=known,
        known_suit_options=known_options,
        candidate_already_known=candidate_already_known,
        limit=limit,
        include_historical_unknowns=False,
    )


def historical_suit_constraints_relaxed(
    *,
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
    known_cards: Iterable[str] = (),
    known_suit_options: Iterable[Iterable[str]] = (),
    candidate_already_known: bool = False,
    limit: int = 64,
) -> bool:
    """Return whether an action only passed after relaxing old ``?`` suits.

    This is diagnostic metadata, not a new fact about either player's cards.
    Exact historical cards and the current candidate always keep their normal
    double-deck limit; only an earlier visually uncertain card is softened.
    """

    candidate = tuple(str(card) for card in cards)
    candidate_options = tuple(tuple(str(suit) for suit in options) for options in suit_options)
    known = tuple(str(card) for card in known_cards)
    known_options = tuple(tuple(str(suit) for suit in options) for options in known_suit_options)
    if not any(is_unknown_suit_card(card) for card in known):
        return False
    strict = _enumerate_action_variants(
        cards=candidate,
        suit_options=candidate_options,
        known_cards=known,
        known_suit_options=known_options,
        candidate_already_known=candidate_already_known,
        limit=limit,
        include_historical_unknowns=True,
    )
    if strict:
        return False
    return bool(
        _enumerate_action_variants(
            cards=candidate,
            suit_options=candidate_options,
            known_cards=known,
            known_suit_options=known_options,
            candidate_already_known=candidate_already_known,
            limit=limit,
            include_historical_unknowns=False,
        )
    )


def _enumerate_action_variants(
    *,
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
    known_cards: Iterable[str] = (),
    known_suit_options: Iterable[Iterable[str]] = (),
    candidate_already_known: bool = False,
    limit: int = 64,
    include_historical_unknowns: bool,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate one bounded validation branch for an action candidate."""

    candidate = tuple(str(card) for card in cards)
    known = tuple(str(card) for card in known_cards)
    known_options = normalized_suit_options(known, known_suit_options)
    candidate_options = normalized_suit_options(candidate, suit_options)
    entries: list[tuple[bool, int, tuple[str, ...]]] = []

    def append_entries(values: tuple[str, ...], options: tuple[tuple[str, ...], ...], *, is_candidate: bool) -> None:
        for index, value in enumerate(values):
            if is_unknown_suit_card(value):
                if not is_candidate and not include_historical_unknowns:
                    continue
                entries.append((is_candidate, index, tuple(f"{value[:-1]}{suit}" for suit in options[index])))
            else:
                entries.append((is_candidate, index, (value,)))

    append_entries(known, known_options, is_candidate=False)
    # A self action is already represented in ``known`` through the current
    # hand, so candidate entries must not increase deck counts a second time.
    # They still need to be expanded, however: ``8?`` is not itself a
    # physical card and must become one of its concrete suit candidates.
    append_entries(candidate, candidate_options, is_candidate=True)

    chosen = list(candidate)
    counts: Counter[str] = Counter()
    variants: list[tuple[str, ...]] = []

    def visit(index: int) -> None:
        if len(variants) >= max(1, int(limit)):
            return
        if index == len(entries):
            resolved = tuple(chosen)
            if resolved not in variants:
                variants.append(resolved)
            return
        is_candidate, card_index, options = entries[index]
        for concrete in options:
            if candidate_already_known and is_candidate:
                # The self hand has already participated in the known list.
                # Resolve an uncertain observed suit, but do not count that
                # physical card twice against the double-deck limit.
                pass
            if not (candidate_already_known and is_candidate) and counts[concrete] >= 2:
                continue
            if not (candidate_already_known and is_candidate):
                counts[concrete] += 1
            if is_candidate:
                chosen[card_index] = concrete
            visit(index + 1)
            if not (candidate_already_known and is_candidate):
                counts[concrete] -= 1

    visit(0)
    return tuple(variants)


def feasible_self_hand_variants(
    *,
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
    known_hand: Iterable[str],
    limit: int = 64,
) -> tuple[tuple[str, ...], ...]:
    """Resolve an observed self action to real cards from the known hand.

    The live recognizer may legitimately return ``8?`` when the suit glyph is
    unclear.  A self play cannot be committed with that placeholder because
    the reducer must remove a real card from ``my_hand``.  This helper keeps
    the visual uncertainty bounded while allowing only one-to-one assignments
    to the already confirmed physical cards in the hand.
    """

    candidate = tuple(str(card) for card in cards)
    options = normalized_suit_options(candidate, suit_options)
    available = Counter(str(card) for card in known_hand)
    choices_by_index: list[tuple[str, ...]] = []
    for index, card in enumerate(candidate):
        if not is_unknown_suit_card(card):
            choices_by_index.append((card,))
            continue
        rank = card[:-1]
        choices_by_index.append(
            tuple(f"{rank}{suit}" for suit in options[index])
        )

    chosen = list(candidate)
    variants: list[tuple[str, ...]] = []

    def visit(index: int) -> None:
        if len(variants) >= max(1, int(limit)):
            return
        if index == len(candidate):
            resolved = tuple(chosen)
            if resolved not in variants:
                variants.append(resolved)
            return
        for concrete in choices_by_index[index]:
            if available[concrete] <= 0:
                continue
            available[concrete] -= 1
            chosen[index] = concrete
            visit(index + 1)
            available[concrete] += 1

    visit(0)
    return tuple(variants)


def state_variants_for_unknown_suits(
    state: GuanDanState,
    *,
    limit: int = 32,
) -> tuple[GuanDanState, ...]:
    """Build bounded advisor snapshots without changing canonical history.

    Prefer every stored suit option.  If a past animation has made that set
    globally impossible, widen only those historical unknowns for this
    short-lived advisor branch.  The canonical event continues to contain its
    original ``?`` and candidate suits for later audit/correction.
    """

    return state_variants_for_unknown_suits_detailed(state, limit=limit).states


def state_variants_for_unknown_suits_detailed(
    state: GuanDanState,
    *,
    limit: int = 32,
) -> SuitStateVariants:
    """Return temporary states plus an explicit cap/truncation audit."""

    max_variants = max(1, int(limit))
    hand = tuple(state.my_hand)
    history = tuple(state.play_history)
    unknown_hand_positions = [
        (card_index, card, options)
        for card_index, card in enumerate(hand)
        if is_unknown_suit_card(card)
        for options in (normalized_suit_options(hand)[card_index],)
    ]
    unknown_history_positions = [
        (event_index, card_index, card, options)
        for event_index, event in enumerate(history)
        for card_index, card in enumerate(event.cards)
        if is_unknown_suit_card(card)
        for options in (normalized_suit_options(event.cards, event.suit_options)[card_index],)
    ]
    if not unknown_hand_positions and not unknown_history_positions:
        return SuitStateVariants(states=(state,), limit=max_variants)

    counts: Counter[str] = Counter(
        card for card in state.my_hand if not is_unknown_suit_card(card)
    )
    for event in history:
        counts.update(card for card in event.cards if not is_unknown_suit_card(card))
    if any(value > 2 for value in counts.values()):
        return SuitStateVariants(states=(), limit=max_variants)

    def build_variants(
        *,
        relax_unknown_suits: bool,
    ) -> tuple[tuple[GuanDanState, ...], bool]:
        resolved_hand = list(hand)
        resolved = [list(event.cards) for event in history]
        branch_counts = counts.copy()
        variants: list[GuanDanState] = []
        seen: set[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = set()
        collection_limit = max_variants + 1

        def visit(index: int) -> None:
            if len(variants) >= collection_limit:
                return
            if index == len(unknown_hand_positions) + len(unknown_history_positions):
                normalized_hand = tuple(sorted(resolved_hand))
                signature = (normalized_hand, tuple(tuple(cards) for cards in resolved))
                if signature in seen:
                    return
                seen.add(signature)
                concrete_history = [
                    replace(event, cards=tuple(resolved[event_index]), suit_options=())
                    for event_index, event in enumerate(history)
                ]
                trick_count = len(state.trick_plays)
                concrete_trick = concrete_history[-trick_count:] if trick_count else []
                variants.append(
                    replace(
                        state,
                        my_hand=normalized_hand,
                        play_history=concrete_history,
                        trick_plays=concrete_trick,
                    )
                )
                return
            if index < len(unknown_hand_positions):
                card_index, card, options = unknown_hand_positions[index]
                choices = SUITS if relax_unknown_suits else options
                for suit in choices:
                    concrete = f"{card[:-1]}{suit}"
                    if branch_counts[concrete] >= 2:
                        continue
                    branch_counts[concrete] += 1
                    resolved_hand[card_index] = concrete
                    visit(index + 1)
                    branch_counts[concrete] -= 1
                return
            event_index, card_index, card, options = unknown_history_positions[
                index - len(unknown_hand_positions)
            ]
            choices = SUITS if relax_unknown_suits else options
            for suit in choices:
                concrete = f"{card[:-1]}{suit}"
                if branch_counts[concrete] >= 2:
                    continue
                branch_counts[concrete] += 1
                resolved[event_index][card_index] = concrete
                visit(index + 1)
                branch_counts[concrete] -= 1

        visit(0)
        return tuple(variants[:max_variants]), len(variants) > max_variants

    strict, strict_truncated = build_variants(relax_unknown_suits=False)
    if strict:
        return SuitStateVariants(
            states=strict,
            truncated=strict_truncated,
            limit=max_variants,
        )
    relaxed, relaxed_truncated = build_variants(relax_unknown_suits=True)
    return SuitStateVariants(
        states=relaxed,
        truncated=relaxed_truncated,
        limit=max_variants,
        used_relaxed_suits=True,
    )

# --- Explicit, delayed occlusion decisions ---------------------------------


def assess_occluded_action(
    *,
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
    known_cards: Iterable[str] = (),
    known_suit_options: Iterable[Iterable[str]] = (),
    candidate_already_known: bool = False,
    limit: int = 64,
    strategy_sensitive: bool | None = None,
    action_id: str = "",
    observed_at: str | None = None,
) -> OcclusionEvidence:
    """Assess an action without guessing an unknown suit.

    A five-card (or larger) action is strategy-sensitive by default because a
    wrong interpretation can change the legal response class.  If more than
    one concrete branch survives the double-deck constraints, no branch is
    selected; callers receive ``occluded`` and ``strategy_blocked`` instead.
    """

    observed = tuple(str(card) for card in cards)
    supplied_options = tuple(tuple(str(suit) for suit in choices) for choices in suit_options)
    if not observed:
        return OcclusionEvidence(
            state="rejected", action_id=action_id, suit_options=supplied_options,
            strategy_blocked=True, reason="empty recognition", observed_at=observed_at,
        )

    if not any(is_unknown_suit_card(card) for card in observed):
        return OcclusionEvidence(
            state="known", action_id=action_id, cards=observed,
            suit_options=supplied_options, candidates=(tuple(sorted(observed)),),
            confirmations=1, required_confirmations=1, observed_at=observed_at,
        )

    variants = feasible_action_variants(
        cards=observed,
        suit_options=supplied_options,
        known_cards=known_cards,
        known_suit_options=known_suit_options,
        candidate_already_known=candidate_already_known,
        limit=limit,
    )
    if not variants:
        return OcclusionEvidence(
            state="rejected", action_id=action_id, cards=observed,
            suit_options=supplied_options, strategy_blocked=True,
            reason="no double-deck-consistent candidate", observed_at=observed_at,
        )

    candidates = tuple(dict.fromkeys(tuple(sorted(variant)) for variant in variants))
    if len(candidates) == 1:
        return OcclusionEvidence(
            state="confirmed", action_id=action_id, cards=observed,
            suit_options=supplied_options, candidates=candidates,
            confirmations=1, required_confirmations=2,
            reason="unique physical candidate; delayed confirmation still required",
            observed_at=observed_at,
        )

    sensitive = len(observed) >= 5 if strategy_sensitive is None else strategy_sensitive
    return OcclusionEvidence(
        state="occluded" if supplied_options else "unknown",
        action_id=action_id, cards=observed, suit_options=supplied_options,
        candidates=candidates, strategy_blocked=bool(sensitive),
        reason=("candidate disagreement blocks strategy" if sensitive
                else "multiple physical candidates remain"),
        observed_at=observed_at,
    )


def confirm_occluded_action(
    previous: OcclusionEvidence,
    *,
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
    known_cards: Iterable[str] = (),
    known_suit_options: Iterable[Iterable[str]] = (),
    candidate_already_known: bool = False,
    strategy_sensitive: bool | None = None,
    now: str | None = None,
) -> OcclusionEvidence:
    """Perform one delayed reread and require two matching safe observations."""

    current = assess_occluded_action(
        cards=cards, suit_options=suit_options, known_cards=known_cards,
        known_suit_options=known_suit_options,
        candidate_already_known=candidate_already_known,
        strategy_sensitive=strategy_sensitive, action_id=previous.action_id,
        observed_at=now or utc_now_iso(),
    )
    if current.state in {"rejected", "unknown", "occluded"}:
        return current
    if previous.candidates != current.candidates or previous.cards != current.cards:
        return current
    confirmations = previous.confirmations + 1
    if current.state == "known":
        return current
    return OcclusionEvidence(
        **{**current.__dict__, "confirmations": confirmations,
           "state": "confirmed" if confirmations >= current.required_confirmations else current.state,
           "reason": "matching delayed confirmation" if confirmations >= current.required_confirmations
                     else "awaiting matching delayed confirmation"}
    )

# Friendly names for callers that prefer decision/evaluation terminology.
evaluate_occlusion = assess_occluded_action
confirm_occlusion = confirm_occluded_action
