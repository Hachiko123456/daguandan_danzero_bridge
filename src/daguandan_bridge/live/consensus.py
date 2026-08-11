from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from ..danzero.rules import actions_for_cards, infer_best_action
from .card_uncertainty import (
    feasible_action_variants,
    feasible_self_hand_variants,
    historical_suit_constraints_relaxed,
    is_unknown_suit_card,
    normalized_suit_options,
)


ConsensusStatus = Literal["confirmed", "needs_confirmation", "review_required"]


@dataclass(frozen=True)
class RecognitionSample:
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    source: str
    evidence_ref: str = ""
    # Candidate suits are aligned with ``cards``.  Exact cards use one suit;
    # rank-only cards preserve a colour-constrained candidate set.
    suit_options: tuple[tuple[str, ...], ...] = ()
    # Kept as a compatibility field for old logs.  It is no longer used to
    # confirm or reject an action.
    post_hand: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsensusContext:
    level_rank: str
    remaining_cards: int
    allow_pass: bool
    known_hand: tuple[str, ...] = ()
    table_cards: tuple[str, ...] = ()
    # Known physical cards include the current self hand plus every confirmed
    # non-pass action.  Opponents' hidden hands intentionally stay unknown.
    known_cards: tuple[str, ...] = ()
    known_suit_options: tuple[tuple[str, ...], ...] = ()
    table_suit_options: tuple[tuple[str, ...], ...] = ()
    # A self action is selected from ``known_hand``.  Its cards must therefore
    # not be added a second time when checking the global double-deck bound.
    candidate_already_known: bool = False
    region_empty: bool = False
    next_turn_evidence: bool = False
    # The live path keeps card-in-hand and play-rule checks enabled.
    validate_rules: bool = True


@dataclass(frozen=True)
class ConsensusCandidate:
    cards: tuple[str, ...]
    is_pass: bool
    votes: int
    mean_confidence: float
    valid: bool
    rejected_reason: str = ""


@dataclass(frozen=True)
class ConsensusResult:
    status: ConsensusStatus
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    source: str
    vote_count: int
    candidates: tuple[ConsensusCandidate, ...]
    # ``cards`` keeps the normalized visual evidence.  A self action with an
    # unknown suit also carries the exact physical cards to remove from the
    # confirmed hand when it is committed.
    resolved_cards: tuple[str, ...] = ()
    rejected_reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    suit_options: tuple[tuple[str, ...], ...] = ()
    # Audit-only warnings which never alter canonical cards or turn flow.
    integrity_warnings: tuple[str, ...] = ()


class BurstConsensus:
    """Vote on stable reads, then apply card/deck constraints."""

    def __init__(self, *, min_votes: int = 3) -> None:
        if min_votes <= 0:
            raise ValueError("min_votes must be positive")
        self.min_votes = int(min_votes)

    def decide(
        self,
        samples: Iterable[RecognitionSample],
        *,
        context: ConsensusContext,
    ) -> ConsensusResult:
        items = tuple(samples)
        if not items:
            return self._review((), ("no_recognition_samples",))

        grouped: dict[tuple[bool, tuple[str, ...]], list[RecognitionSample]] = defaultdict(list)
        for sample in items:
            is_pass = bool(sample.is_pass)
            cards, suit_options = canonical_candidate(
                sample.cards,
                sample.suit_options,
                is_pass=is_pass,
            )
            grouped[(is_pass, cards)].append(
                RecognitionSample(
                    cards=cards,
                    is_pass=is_pass,
                    confidence=sample.confidence,
                    source=sample.source,
                    evidence_ref=sample.evidence_ref,
                    suit_options=suit_options,
                    post_hand=sample.post_hand,
                )
            )

        candidates: list[ConsensusCandidate] = []
        evidence_by_key: dict[tuple[bool, tuple[str, ...]], tuple[str, ...]] = {}
        suit_options_by_key: dict[tuple[bool, tuple[str, ...]], tuple[tuple[str, ...], ...]] = {}
        resolved_by_key: dict[tuple[bool, tuple[str, ...]], tuple[str, ...]] = {}
        warnings_by_key: dict[tuple[bool, tuple[str, ...]], tuple[str, ...]] = {}
        for (is_pass, cards), votes in grouped.items():
            rejected_reason = self.validate_candidate(
                is_pass,
                cards,
                context,
                suit_options=votes[-1].suit_options,
            )
            candidates.append(
                ConsensusCandidate(
                    cards=cards,
                    is_pass=is_pass,
                    votes=len(votes),
                    mean_confidence=sum(float(item.confidence) for item in votes)
                    / len(votes),
                    valid=not rejected_reason,
                    rejected_reason=rejected_reason,
                )
            )
            evidence_by_key[(is_pass, cards)] = tuple(
                item.evidence_ref for item in votes if item.evidence_ref
            )
            suit_options_by_key[(is_pass, cards)] = votes[-1].suit_options
            if not rejected_reason:
                resolved_by_key[(is_pass, cards)] = self.resolve_commit_cards(
                    is_pass,
                    cards,
                    context,
                    suit_options=votes[-1].suit_options,
                )
                warnings_by_key[(is_pass, cards)] = self.integrity_warnings(
                    is_pass,
                    cards,
                    context,
                    suit_options=votes[-1].suit_options,
                )
        candidates.sort(
            key=lambda item: (item.votes, item.mean_confidence), reverse=True
        )
        accepted = [
            item
            for item in candidates
            if item.valid and item.votes >= self.min_votes
        ]
        rejected = tuple(
            dict.fromkeys(
                item.rejected_reason for item in candidates if item.rejected_reason
            )
        )
        if accepted:
            # A stable visible play is an observed event, not a proposal for
            # the rule engine to approve.  Prefer any non-pass candidate over
            # a later pass marker and resolve remaining conflicts by evidence
            # strength so the live turn can never stall here.
            winner = max(
                accepted,
                key=lambda item: (
                    not item.is_pass,
                    item.votes,
                    item.mean_confidence,
                ),
            )
            return ConsensusResult(
                status="confirmed",
                cards=winner.cards,
                is_pass=winner.is_pass,
                confidence=winner.mean_confidence,
                source="multi_frame_consensus",
                vote_count=winner.votes,
                resolved_cards=resolved_by_key.get(
                    (winner.is_pass, winner.cards), winner.cards
                ),
                candidates=tuple(candidates),
                rejected_reasons=rejected,
                evidence_refs=evidence_by_key[(winner.is_pass, winner.cards)],
                suit_options=suit_options_by_key[(winner.is_pass, winner.cards)],
                integrity_warnings=warnings_by_key.get(
                    (winner.is_pass, winner.cards), ()
                ),
            )
        reasons = list(rejected)
        reasons.append("candidate_conflict" if len(accepted) > 1 else "insufficient_consensus")
        return self._review(tuple(candidates), tuple(dict.fromkeys(reasons)))

    @staticmethod
    def validate_candidate(
        is_pass: bool,
        cards: tuple[str, ...],
        context: ConsensusContext,
        *,
        suit_options: tuple[tuple[str, ...], ...] = (),
    ) -> str:
        if is_pass:
            return "" if context.allow_pass else "pass_not_allowed"
        if not cards:
            return "empty_play"
        if len(cards) > context.remaining_cards:
            return "exceeds_remaining_cards"
        if any(
            count > 2
            for card, count in Counter(cards).items()
            if not is_unknown_suit_card(card)
        ):
            return "exceeds_double_deck_limit"
        variants = feasible_action_variants(
            cards=cards,
            suit_options=suit_options,
            known_cards=context.known_cards,
            known_suit_options=context.known_suit_options,
            candidate_already_known=context.candidate_already_known,
        )
        if not variants:
            return "exceeds_double_deck_limit"
        if context.known_hand:
            # ``A?`` is visual evidence, not a physical card code.  Resolve
            # it against the confirmed self hand before applying membership
            # and rule checks; a raw Counter comparison would reject every
            # legitimate unknown-suit self card.
            variants = feasible_self_hand_variants(
                cards=cards,
                suit_options=suit_options,
                known_hand=context.known_hand,
            )
            if not variants:
                return "cards_not_in_known_hand"
        if not context.validate_rules:
            return ""
        try:
            if not any(
                actions_for_cards(variant, context.level_rank)
                for variant in variants
            ):
                # This is a structural OCR guard only.  A legal action that
                # does not beat the reconstructed table is still accepted;
                # ``does_not_beat_table`` is never a rejection reason.
                return "illegal_pattern"
        except (ImportError, ModuleNotFoundError, ValueError):
            return "illegal_pattern"
        # Rule interpretation is deliberately not a commit gate.  Once the
        # game UI has shown a stable non-empty action, the action happened.
        # Whether its reconstructed cards form a known pattern or beat the
        # reconstructed table is captured by ``integrity_warnings`` below and
        # must never turn the action into PASS or block the turn.
        return ""

    @staticmethod
    def resolve_commit_cards(
        is_pass: bool,
        cards: tuple[str, ...],
        context: ConsensusContext,
        *,
        suit_options: tuple[tuple[str, ...], ...] = (),
    ) -> tuple[str, ...]:
        """Return the concrete self-hand variant that will reach reducer."""

        if is_pass or not context.known_hand:
            return cards
        variants = feasible_self_hand_variants(
            cards=cards,
            suit_options=suit_options,
            known_hand=context.known_hand,
        )
        ranked: list[tuple[int, int, tuple[str, ...]]] = []
        for variant in variants:
            try:
                inference = infer_best_action(
                    variant,
                    context.table_cards,
                    context.level_rank,
                )
                ranked.append(
                    (
                        int(inference.beats_table),
                        int(inference.action is not None),
                        variant,
                    )
                )
            except (ImportError, ModuleNotFoundError, ValueError):
                ranked.append((0, 0, variant))
        return max(ranked, default=(0, 0, cards))[-1]

    _validate_candidate = validate_candidate

    @staticmethod
    def integrity_warnings(
        is_pass: bool,
        cards: tuple[str, ...],
        context: ConsensusContext,
        *,
        suit_options: tuple[tuple[str, ...], ...] = (),
    ) -> tuple[str, ...]:
        """Return audit warnings for a candidate that has already validated."""

        if is_pass:
            return ()
        warnings: list[str] = []
        if historical_suit_constraints_relaxed(
            cards=cards,
            suit_options=suit_options,
            known_cards=context.known_cards,
            known_suit_options=context.known_suit_options,
            candidate_already_known=context.candidate_already_known,
        ):
            warnings.append("historical_suit_constraints_relaxed")

        variants = feasible_action_variants(
            cards=cards,
            suit_options=suit_options,
            known_cards=context.known_cards,
            known_suit_options=context.known_suit_options,
            candidate_already_known=context.candidate_already_known,
        )
        if context.known_hand:
            variants = feasible_self_hand_variants(
                cards=cards,
                suit_options=suit_options,
                known_hand=context.known_hand,
            )
        try:
            inferences = tuple(
                infer_best_action(
                    variant,
                    context.table_cards,
                    context.level_rank,
                )
                for variant in variants
            )
        except (ImportError, ModuleNotFoundError, ValueError):
            inferences = ()
        if not any(inference.action is not None for inference in inferences):
            warnings.append("observed_pattern_unresolved")
        elif (
            context.table_cards
            and not any(is_unknown_suit_card(card) for card in context.table_cards)
            and not any(inference.beats_table for inference in inferences)
        ):
            warnings.append("observed_table_mismatch")
        if any(inference.ambiguous for inference in inferences):
            warnings.append("wildcard_interpretation_ambiguous")
        return tuple(dict.fromkeys(warnings))

    @staticmethod
    def _review(
        candidates: tuple[ConsensusCandidate, ...],
        reasons: tuple[str, ...],
    ) -> ConsensusResult:
        best = candidates[0] if candidates else None
        return ConsensusResult(
            status="review_required",
            cards=best.cards if best else (),
            is_pass=best.is_pass if best else False,
            confidence=best.mean_confidence if best else 0.0,
            source="multi_frame_consensus",
            vote_count=best.votes if best else 0,
            candidates=candidates,
            rejected_reasons=reasons,
        )


def canonical_candidate(
    cards: Iterable[str],
    suit_options: Iterable[Iterable[str]] = (),
    *,
    is_pass: bool = False,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Sort cards and their suit metadata as inseparable pairs."""

    if is_pass:
        return (), ()
    values = tuple(str(card) for card in cards)
    options = normalized_suit_options(values, suit_options)
    pairs = sorted(zip(values, options), key=lambda item: item[0])
    return (
        tuple(card for card, _options in pairs),
        tuple(options for _card, options in pairs),
    )
