"""Shared, conservative confirmation for a previously unknown card suit.

The live listener, replay editor, and timeline importer must agree on when an
``8?`` observation may become a concrete physical card.  This module contains
only that decision rule; callers remain responsible for where evidence comes
from and how a confirmed correction is persisted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .card_uncertainty import is_unknown_suit_card, normalized_suit_options


def _rank_counts(cards: tuple[str, ...]) -> Counter[str]:
    """Compare an action shape without treating a suit as part of its rank."""

    return Counter(
        card
        if card in {"small_joker", "big_joker"}
        else card[:-1]
        if len(card) >= 2
        else card
        for card in cards
    )


def validate_suit_correction(
    target_cards: tuple[str, ...],
    observed_cards: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Return a normalized correction only when it proves the same action.

    A correction is deliberately narrower than a new recognition: the target
    must contain an unknown suit, while the new observation must make every
    suit concrete without changing the card count or rank multiset.
    """

    target = tuple(str(card) for card in target_cards)
    corrected = tuple(sorted(str(card) for card in observed_cards))
    if (
        not target
        or not any(is_unknown_suit_card(card) for card in target)
        or not corrected
        or any(is_unknown_suit_card(card) for card in corrected)
        or len(corrected) != len(target)
        or _rank_counts(corrected) != _rank_counts(target)
    ):
        return None
    return corrected


def validate_visual_action_correction(
    target_cards: tuple[str, ...],
    observed_cards: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Validate a conservative reread of the immediately preceding action.

    The live reader can capture only the first visible card while an action
    animation is still expanding.  A later reread is allowed to *add* cards,
    but never to remove cards.  Unknown suits retain the old, exact-rank suit
    correction rule.  This helper intentionally does not decide whether the
    result beats the table; the reducer/recognition path remains responsible
    for that semantic validation.
    """

    target = tuple(str(card) for card in target_cards)
    observed = tuple(sorted(str(card) for card in observed_cards))
    if not target or not observed or any(is_unknown_suit_card(card) for card in observed):
        return None

    target_ranks = _rank_counts(target)
    observed_ranks = _rank_counts(observed)
    # A same-size reread is safe only when it resolves an unknown suit.  For
    # known cards it is merely confirmation, not a correction event.
    if len(observed) == len(target):
        if not any(is_unknown_suit_card(card) for card in target):
            return None
        if observed_ranks != target_ranks:
            return None
        return observed

    # Expansion is the only count change permitted: every previously seen
    # rank must still be present, while newly visible cards may be appended.
    if len(observed) < len(target):
        return None
    if any(observed_ranks[rank] < count for rank, count in target_ranks.items()):
        return None
    return observed


def validate_visual_action_compatibility(
    target_cards: tuple[str, ...],
    observed_cards: tuple[str, ...],
    observed_suit_options: tuple[tuple[str, ...], ...] = (),
) -> tuple[str, ...] | None:
    """Match an occluded reread to an already-exact action without rewriting it.

    ``suit_options`` is positional recognition evidence, so it must be aligned
    before any card sorting takes place.  Concrete rereads consume their exact
    physical cards first; the remaining unknown cards are matched by a small
    backtracking search.  This avoids a greedy duplicate-suit assignment and
    retains the two-copies-per-exact-card limit of a double deck.
    """

    target = tuple(str(card) for card in target_cards)
    observed = tuple(str(card) for card in observed_cards)
    if (
        not target
        or len(target) != len(observed)
        or any(is_unknown_suit_card(card) for card in target)
        or not any(is_unknown_suit_card(card) for card in observed)
        or _rank_counts(target) != _rank_counts(observed)
    ):
        return None

    remaining = Counter(target)
    if any(count > 2 for count in remaining.values()):
        return None

    supplied_options = tuple(
        tuple(str(suit) for suit in choices)
        for choices in observed_suit_options
    )
    # Compatibility is an evidence-based relaxation, not a license to guess
    # from all four suits when an old/partial result omitted candidate data.
    if any(
        is_unknown_suit_card(card)
        and (index >= len(supplied_options) or not supplied_options[index])
        for index, card in enumerate(observed)
    ):
        return None
    options = normalized_suit_options(observed, supplied_options)
    unknown_choices: list[tuple[str, ...]] = []
    for index, card in enumerate(observed):
        if not is_unknown_suit_card(card):
            if remaining[card] <= 0:
                return None
            remaining[card] -= 1
            continue
        rank = card[:-1]
        choices = tuple(
            f"{rank}{suit}"
            for suit in options[index]
            if remaining[f"{rank}{suit}"] > 0
        )
        if not choices:
            return None
        unknown_choices.append(choices)

    # Fewest choices first keeps the bounded search deterministic while still
    # accepting duplicate-rank actions independently of recognition order.
    unknown_choices.sort(key=lambda choices: (len(choices), choices))

    def matches(index: int) -> bool:
        if index == len(unknown_choices):
            return not any(remaining.values())
        for concrete in unknown_choices[index]:
            if remaining[concrete] <= 0:
                continue
            remaining[concrete] -= 1
            if matches(index + 1):
                return True
            remaining[concrete] += 1
        return False

    return tuple(sorted(target)) if matches(0) else None


@dataclass(frozen=True)
class SuitCorrectionObservation:
    """One observation result, including whether two matching reads exist."""

    cards: tuple[str, ...] = ()
    confirmations: int = 0
    evidence_kind: str = ""

    @property
    def confirmed(self) -> bool:
        return bool(self.cards) and self.confirmations >= 2


class SuitCorrectionTracker:
    """Require two identical safe reads before a caller changes any record."""

    def __init__(self) -> None:
        self._streaks: dict[str, SuitCorrectionObservation] = {}

    def observe(
        self,
        target_id: str,
        target_cards: tuple[str, ...],
        observed_cards: tuple[str, ...],
    ) -> SuitCorrectionObservation:
        corrected = validate_suit_correction(target_cards, observed_cards)
        if corrected is None:
            self._streaks.pop(str(target_id), None)
            return SuitCorrectionObservation()
        previous = self._streaks.get(str(target_id))
        observation = SuitCorrectionObservation(
            cards=corrected,
            confirmations=(previous.confirmations + 1)
            if previous is not None and previous.cards == corrected
            else 1,
            evidence_kind="correction",
        )
        self._streaks[str(target_id)] = observation
        return observation

    def observe_visual_action(
        self,
        target_id: str,
        target_cards: tuple[str, ...],
        observed_cards: tuple[str, ...],
        observed_suit_options: tuple[tuple[str, ...], ...] = (),
    ) -> SuitCorrectionObservation:
        """Track a generic previous-action reread using the same two-frame rule."""

        target = tuple(sorted(str(card) for card in target_cards))
        observed_aligned = tuple(str(card) for card in observed_cards)
        observed = tuple(sorted(observed_aligned))
        # Identical physical cards are confirmation evidence but do not need
        # a reducer correction event.
        if target and observed == target:
            corrected = observed
            evidence_kind = "exact"
        else:
            corrected = validate_visual_action_compatibility(
                target,
                observed_aligned,
                observed_suit_options,
            )
            evidence_kind = "compatible"
            if corrected is None:
                corrected = validate_visual_action_correction(target, observed)
                evidence_kind = "correction"
        if corrected is None:
            self._streaks.pop(str(target_id), None)
            return SuitCorrectionObservation()
        previous = self._streaks.get(str(target_id))
        confirmations = (
            previous.confirmations + 1
            if previous is not None and previous.cards == corrected
            else 1
        )
        if (
            previous is not None
            and previous.cards == corrected
            and "compatible" in {previous.evidence_kind, evidence_kind}
            and {previous.evidence_kind, evidence_kind} <= {"exact", "compatible"}
        ):
            evidence_kind = "compatible"
        observation = SuitCorrectionObservation(
            cards=corrected,
            confirmations=confirmations,
            evidence_kind=evidence_kind,
        )
        self._streaks[str(target_id)] = observation
        return observation

    def clear(self, target_id: str) -> None:
        self._streaks.pop(str(target_id), None)
