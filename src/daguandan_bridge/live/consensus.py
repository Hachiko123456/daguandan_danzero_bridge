from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from ..danzero.rules import action_for_cards


ConsensusStatus = Literal["confirmed", "needs_confirmation", "review_required"]


@dataclass(frozen=True)
class RecognitionSample:
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    source: str
    evidence_ref: str = ""


@dataclass(frozen=True)
class ConsensusContext:
    level_rank: str
    remaining_cards: int
    allow_pass: bool
    known_hand: tuple[str, ...] = ()
    region_empty: bool = False
    next_turn_evidence: bool = False


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
    rejected_reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class BurstConsensus:
    """Vote on 3–5 stable reads, then apply card/deck constraints."""

    def __init__(self, *, min_votes: int = 3) -> None:
        if min_votes <= 0:
            raise ValueError("min_votes 必须为正数")
        self.min_votes = int(min_votes)

    def decide(
        self,
        samples: Iterable[RecognitionSample],
        *,
        context: ConsensusContext,
    ) -> ConsensusResult:
        items = tuple(samples)
        if not items:
            if context.region_empty and context.next_turn_evidence and context.allow_pass:
                return ConsensusResult(
                    status="needs_confirmation",
                    cards=(),
                    is_pass=True,
                    confidence=0.0,
                    source="inferred_pass",
                    vote_count=0,
                    candidates=(),
                    rejected_reasons=("pass_template_missing",),
                )
            return self._review((), ("no_recognition_samples",))

        grouped: dict[tuple[bool, tuple[str, ...]], list[RecognitionSample]] = defaultdict(list)
        for sample in items:
            is_pass = bool(sample.is_pass)
            cards = () if is_pass else tuple(sorted(str(card) for card in sample.cards))
            grouped[(is_pass, cards)].append(sample)

        candidates: list[ConsensusCandidate] = []
        evidence_by_key: dict[tuple[bool, tuple[str, ...]], tuple[str, ...]] = {}
        for (is_pass, cards), votes in grouped.items():
            rejected_reason = self._validate_candidate(is_pass, cards, context)
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
        if len(accepted) == 1:
            winner = accepted[0]
            return ConsensusResult(
                status="confirmed",
                cards=winner.cards,
                is_pass=winner.is_pass,
                confidence=winner.mean_confidence,
                source="multi_frame_consensus",
                vote_count=winner.votes,
                candidates=tuple(candidates),
                rejected_reasons=rejected,
                evidence_refs=evidence_by_key[(winner.is_pass, winner.cards)],
            )
        reasons = list(rejected)
        reasons.append("candidate_conflict" if len(accepted) > 1 else "insufficient_consensus")
        return self._review(tuple(candidates), tuple(dict.fromkeys(reasons)))

    @staticmethod
    def _validate_candidate(
        is_pass: bool,
        cards: tuple[str, ...],
        context: ConsensusContext,
    ) -> str:
        if is_pass:
            return "" if context.allow_pass else "pass_not_allowed"
        if not cards:
            return "empty_play"
        if len(cards) > context.remaining_cards:
            return "exceeds_remaining_cards"
        if any(count > 2 for count in Counter(cards).values()):
            return "exceeds_double_deck_limit"
        if context.known_hand and Counter(cards) - Counter(context.known_hand):
            return "cards_not_in_known_hand"
        try:
            if action_for_cards(cards, context.level_rank) is None:
                return "illegal_pattern"
        except (ImportError, ModuleNotFoundError, ValueError):
            return "illegal_pattern"
        return ""

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
