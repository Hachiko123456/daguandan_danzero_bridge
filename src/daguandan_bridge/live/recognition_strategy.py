from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from .consensus import (
    BurstConsensus,
    canonical_candidate,
    ConsensusCandidate,
    ConsensusContext,
    ConsensusResult,
    RecognitionSample,
)
from .card_uncertainty import is_unknown_suit_card


class RecognitionStrategy(StrEnum):
    """How one already-open action region is turned into a game action."""

    REFERENCE_SINGLE_SHOT = "reference_single_shot"
    TWO_VALID_STREAK = "two_valid_streak"
    STABLE_SINGLE_SHOT = "stable_single_shot"
    VALID_CANDIDATE_VOTE = "valid_candidate_vote"


@dataclass(frozen=True)
class RecognitionStrategySpec:
    value: RecognitionStrategy
    label: str
    description: str
    settle_ms: int
    stable_ms: int


_SPECS: tuple[RecognitionStrategySpec, ...] = (
    RecognitionStrategySpec(
        RecognitionStrategy.REFERENCE_SINGLE_SHOT,
        "参考项目：等待 1000ms 后单帧",
        "区域出现动作后等待 1000ms，只识别一张截图；特效未结束会重新计时。",
        settle_ms=1_000,
        stable_ms=0,
    ),
    RecognitionStrategySpec(
        RecognitionStrategy.TWO_VALID_STREAK,
        "两次有效牌型一致（推荐）",
        "忽略空结果和非法牌型；两次相同的有效结果即确认。含未知花色时多取一帧，避免特效未退场时过早固化花色。",
        settle_ms=0,
        stable_ms=0,
    ),
    RecognitionStrategySpec(
        RecognitionStrategy.STABLE_SINGLE_SHOT,
        "画面稳定后单帧",
        "动作区域静止 250ms 后读取一帧，适合动画结束明确的界面。",
        settle_ms=400,
        stable_ms=250,
    ),
    RecognitionStrategySpec(
        RecognitionStrategy.VALID_CANDIDATE_VOTE,
        "有效候选累计两票",
        "忽略无效识别；同一有效候选在采样窗口累计两票即确认。",
        settle_ms=0,
        stable_ms=0,
    ),
)

RECOGNITION_STRATEGY_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (spec.value.value, spec.label) for spec in _SPECS
)


def coerce_recognition_strategy(value: str | RecognitionStrategy | None) -> RecognitionStrategy:
    try:
        return RecognitionStrategy(value or RecognitionStrategy.TWO_VALID_STREAK)
    except ValueError:
        return RecognitionStrategy.TWO_VALID_STREAK


def strategy_spec(value: str | RecognitionStrategy | None) -> RecognitionStrategySpec:
    strategy = coerce_recognition_strategy(value)
    return next(spec for spec in _SPECS if spec.value == strategy)


def decide_recognition_strategy(
    strategy: str | RecognitionStrategy | None,
    samples: Iterable[RecognitionSample],
    *,
    context: ConsensusContext,
) -> ConsensusResult | None:
    """Return a confirmed action once this strategy has sufficient evidence.

    Invalid reads are deliberately ignored. They are diagnostic evidence, not a
    vote against a visible action; this prevents one animation frame from
    repeatedly resetting a real turn.
    """

    items = tuple(samples)
    valid: list[RecognitionSample] = []
    rejected: list[str] = []
    for sample in items:
        cards, suit_options = canonical_candidate(
            sample.cards,
            sample.suit_options,
            is_pass=sample.is_pass,
        )
        reason = BurstConsensus.validate_candidate(
            sample.is_pass,
            cards,
            context,
            suit_options=suit_options,
        )
        if reason:
            rejected.append(reason)
        else:
            valid.append(
                RecognitionSample(
                    cards=cards,
                    is_pass=bool(sample.is_pass),
                    confidence=sample.confidence,
                    source=sample.source,
                    evidence_ref=sample.evidence_ref,
                    suit_options=suit_options,
                    post_hand=sample.post_hand,
                )
            )
    if not valid:
        return None

    selected = coerce_recognition_strategy(strategy)
    winner: list[RecognitionSample] | None = None
    source = selected.value
    if selected in {
        RecognitionStrategy.REFERENCE_SINGLE_SHOT,
        RecognitionStrategy.STABLE_SINGLE_SHOT,
    }:
        winner = [valid[-1]]
    elif selected == RecognitionStrategy.TWO_VALID_STREAK:
        if len(valid) >= 2 and _key(valid[-1]) == _key(valid[-2]):
            # 花色字样最容易被按钮、特效或动画边缘短暂遮住。此时仍然
            # 保留牌的点数以保证流程不断，但比纯花色明确的牌多等一帧：
            # 遮挡消退后通常会立即读到完整花色；若遮挡持续，也会在第三
            # 个一致结果后照常提交 ``?``，不会无限等待。
            has_unknown_suit = any(
                is_unknown_suit_card(card) for card in valid[-1].cards
            )
            if not has_unknown_suit:
                winner = [valid[-2], valid[-1]]
        # An occluded suit can change from ``?`` to a temporary clear glyph
        # between adjacent frames, so this deliberately does not require two
        # byte-for-byte equal candidates.  Three rank-equivalent legal reads
        # are enough to preserve the play and its uncertainty.
        if (
            winner is None
            and len(valid) >= 3
            and any(is_unknown_suit_card(card) for card in valid[-1].cards)
            and all(
                _rank_key(sample) == _rank_key(valid[-1])
                for sample in valid[-3:]
            )
        ):
            winner = valid[-3:]
    else:
        grouped: dict[tuple[bool, tuple[str, ...]], list[RecognitionSample]] = defaultdict(list)
        for sample in valid:
            grouped[_key(sample)].append(sample)
        eligible = [votes for votes in grouped.values() if len(votes) >= 2]
        if len(eligible) == 1:
            winner = eligible[0]

    if winner is None:
        return None
    sample = _conservative_unknown_sample(winner)
    candidate = ConsensusCandidate(
        cards=sample.cards,
        is_pass=sample.is_pass,
        votes=len(winner),
        mean_confidence=sum(item.confidence for item in winner) / len(winner),
        valid=True,
    )
    return ConsensusResult(
        status="confirmed",
        cards=sample.cards,
        is_pass=sample.is_pass,
        confidence=candidate.mean_confidence,
        source=source,
        vote_count=candidate.votes,
        candidates=(candidate,),
        resolved_cards=BurstConsensus.resolve_commit_cards(
            sample.is_pass,
            sample.cards,
            context,
            suit_options=sample.suit_options,
        ),
        rejected_reasons=tuple(dict.fromkeys(rejected)),
        evidence_refs=tuple(item.evidence_ref for item in winner if item.evidence_ref),
        suit_options=sample.suit_options,
        integrity_warnings=BurstConsensus.integrity_warnings(
            sample.is_pass,
            sample.cards,
            context,
            suit_options=sample.suit_options,
        ),
    )


def decide_best_effort_candidate(
    samples: Iterable[RecognitionSample],
    *,
    context: ConsensusContext,
) -> ConsensusResult | None:
    """Resolve an exhausted burst without pausing the live game.

    Exact consecutive agreement remains the normal path.  When animation or
    suit flicker exhausts the bounded burst, select the strongest observed
    physical candidate instead of emitting ``conflicting_valid_candidates``.
    A non-pass candidate always outranks a pass marker in the same turn.
    """

    grouped: dict[tuple[bool, tuple[str, ...]], list[tuple[int, RecognitionSample]]] = (
        defaultdict(list)
    )
    unresolved: dict[tuple[bool, tuple[str, ...]], list[tuple[int, RecognitionSample]]] = (
        defaultdict(list)
    )
    rejected: list[str] = []
    for index, sample in enumerate(samples):
        cards, suit_options = canonical_candidate(
            sample.cards,
            sample.suit_options,
            is_pass=sample.is_pass,
        )
        reason = BurstConsensus.validate_candidate(
            sample.is_pass,
            cards,
            context,
            suit_options=suit_options,
        )
        if reason:
            rejected.append(reason)
            if (
                reason == "illegal_pattern"
                and context.next_turn_evidence
                and not sample.is_pass
                and cards
            ):
                unresolved[(False, cards)].append(
                    (
                        index,
                        RecognitionSample(
                            cards=cards,
                            is_pass=False,
                            confidence=sample.confidence,
                            source=sample.source,
                            evidence_ref=sample.evidence_ref,
                            suit_options=suit_options,
                            post_hand=sample.post_hand,
                        ),
                    )
                )
            continue
        normalized = RecognitionSample(
            cards=cards,
            is_pass=bool(sample.is_pass),
            confidence=sample.confidence,
            source=sample.source,
            evidence_ref=sample.evidence_ref,
            suit_options=suit_options,
            post_hand=sample.post_hand,
        )
        grouped[_key(normalized)].append((index, normalized))
    if not grouped and unresolved:
        grouped = unresolved
    if not grouped:
        return None

    winner_items = max(
        grouped.values(),
        key=lambda votes: (
            not votes[-1][1].is_pass,
            len(votes),
            sum(item.confidence for _index, item in votes) / len(votes),
            votes[-1][0],
        ),
    )
    winner = [item for _index, item in winner_items]
    sample = _conservative_unknown_sample(winner)
    confidence = sum(item.confidence for item in winner) / len(winner)
    warnings = list(
        BurstConsensus.integrity_warnings(
            sample.is_pass,
            sample.cards,
            context,
            suit_options=sample.suit_options,
        )
    )
    if len(grouped) > 1:
        warnings.append("candidate_conflict_resolved_best_effort")
    if unresolved and grouped is unresolved:
        warnings.append("observed_pattern_unresolved")
    candidate = ConsensusCandidate(
        cards=sample.cards,
        is_pass=sample.is_pass,
        votes=len(winner),
        mean_confidence=confidence,
        valid=True,
    )
    return ConsensusResult(
        status="confirmed",
        cards=sample.cards,
        is_pass=sample.is_pass,
        confidence=confidence,
        source="best_effort_burst",
        vote_count=len(winner),
        candidates=(candidate,),
        resolved_cards=BurstConsensus.resolve_commit_cards(
            sample.is_pass,
            sample.cards,
            context,
            suit_options=sample.suit_options,
        ),
        rejected_reasons=tuple(dict.fromkeys(rejected)),
        evidence_refs=tuple(item.evidence_ref for item in winner if item.evidence_ref),
        suit_options=sample.suit_options,
        integrity_warnings=tuple(dict.fromkeys(warnings)),
    )


def has_exhausted_valid_candidates(
    samples: Iterable[RecognitionSample],
    *,
    context: ConsensusContext,
    limit: int,
) -> bool:
    """Only conflicting *valid* reads can exhaust an action window early."""

    valid: list[RecognitionSample] = []
    for sample in samples:
        cards, suit_options = canonical_candidate(
            sample.cards,
            sample.suit_options,
            is_pass=sample.is_pass,
        )
        if not BurstConsensus.validate_candidate(
            sample.is_pass,
            cards,
            context,
            suit_options=suit_options,
        ):
            valid.append(
                RecognitionSample(
                    cards=cards,
                    suit_options=suit_options,
                    is_pass=sample.is_pass,
                    confidence=sample.confidence,
                    source=sample.source,
                    evidence_ref=sample.evidence_ref,
                    post_hand=sample.post_hand,
                )
            )
    # Varying only between ``10?`` and ``10♦`` is an in-flight suit read,
    # not two conflicting actions.  A real conflict must disagree on the
    # pass/play choice or on the rank multiset, otherwise the valid burst can
    # wait for the rank-consensus path above without emitting a false alarm.
    return len(valid) >= limit and len({_rank_key(sample) for sample in valid}) > 1


def has_no_valid_candidates(
    samples: Iterable[RecognitionSample],
    *,
    context: ConsensusContext,
    limit: int,
) -> bool:
    """Retry a visible region after repeated blanks/illegal transient reads."""

    items = tuple(samples)
    if len(items) < limit:
        return False
    for sample in items:
        cards, suit_options = canonical_candidate(
            sample.cards,
            sample.suit_options,
            is_pass=sample.is_pass,
        )
        if not BurstConsensus.validate_candidate(
            sample.is_pass,
            cards,
            context,
            suit_options=suit_options,
        ):
            return False
    return True


def _key(sample: RecognitionSample) -> tuple[bool, tuple[str, ...]]:
    return bool(sample.is_pass), tuple(sample.cards)


def _rank_key(sample: RecognitionSample) -> tuple[bool, tuple[str, ...]]:
    """Compare action shape without turning an uncertain suit into a fact."""

    ranks = tuple(_card_rank(card) for card in sample.cards)
    return bool(sample.is_pass), tuple(sorted(ranks))


def _card_rank(card: str) -> str:
    if card in {"small_joker", "big_joker"}:
        return card
    return card[:-1] if len(card) >= 2 else card


def _conservative_unknown_sample(
    samples: list[RecognitionSample],
) -> RecognitionSample:
    """Keep rank consensus usable while retaining any unresolved suit as ``?``.

    One transient clear glyph does not prove a suit.  Choose the sample with
    the fewest concrete suits, then merge all observed candidates for each
    unknown position.  This is deliberately more conservative than choosing
    the last frame and lets the reducer advance on a stable five-card shape
    without inventing a physical card.
    """

    if not any(is_unknown_suit_card(card) for sample in samples for card in sample.cards):
        return samples[-1]
    base = min(
        samples,
        key=lambda sample: sum(
            not is_unknown_suit_card(card) for card in sample.cards
        ),
    )
    unknown_occurrences: dict[str, int] = defaultdict(int)
    merged_options: list[tuple[str, ...]] = []
    for index, card in enumerate(base.cards):
        if not is_unknown_suit_card(card):
            merged_options.append(base.suit_options[index])
            continue
        rank = card[:-1]
        occurrence = unknown_occurrences[rank]
        unknown_occurrences[rank] += 1
        candidates: list[str] = []
        for sample in samples:
            same_rank = [
                position
                for position, other in enumerate(sample.cards)
                if _card_rank(other) == rank
            ]
            if occurrence >= len(same_rank):
                continue
            matched_index = same_rank[occurrence]
            candidates.extend(sample.suit_options[matched_index])
        merged_options.append(tuple(dict.fromkeys(candidates)) or base.suit_options[index])
    return RecognitionSample(
        cards=base.cards,
        is_pass=base.is_pass,
        confidence=sum(sample.confidence for sample in samples) / len(samples),
        source=base.source,
        evidence_ref=base.evidence_ref,
        suit_options=tuple(merged_options),
        post_hand=base.post_hand,
    )
