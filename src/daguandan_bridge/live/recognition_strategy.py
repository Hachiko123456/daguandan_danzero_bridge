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
        "忽略空结果和非法牌型；两次相同的有效结果即确认。",
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
            winner = [valid[-2], valid[-1]]
    else:
        grouped: dict[tuple[bool, tuple[str, ...]], list[RecognitionSample]] = defaultdict(list)
        for sample in valid:
            grouped[_key(sample)].append(sample)
        eligible = [votes for votes in grouped.values() if len(votes) >= 2]
        if len(eligible) == 1:
            winner = eligible[0]

    if winner is None:
        return None
    sample = winner[-1]
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


def has_exhausted_valid_candidates(
    samples: Iterable[RecognitionSample],
    *,
    context: ConsensusContext,
    limit: int,
) -> bool:
    """Only conflicting *valid* reads can exhaust an action window early."""

    count = 0
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
            count += 1
    return count >= limit


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
