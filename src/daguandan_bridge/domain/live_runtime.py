"""Stable records shared by the live runtime and its presentation adapters.

This module deliberately contains data only.  The legacy live orchestrator
re-exports these names for compatibility, while new consumers can depend on
the domain contract without importing the concrete runtime implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..danzero.state import Seat
from .advice import LocalAdvice
from .live import LiveEvent, LiveSnapshot
from .recognition import FastSignalResult


LiveStatus = Literal[
    "initializing",
    "waiting_lead",
    "running",
    "review_required",
    "paused",
    "finalizing",
    "sealed",
]


@dataclass(frozen=True)
class ReviewCandidate:
    candidate_id: str
    cards: tuple[str, ...]
    is_pass: bool
    votes: int
    confidence: float
    valid: bool
    rejected_reason: str = ""


@dataclass(frozen=True)
class ReviewRequest:
    reason: str
    player: Seat
    candidates: tuple[ReviewCandidate, ...]
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdviceRequestKey:
    session_id: str
    turn_id: int
    state_revision: int

    @property
    def request_id(self) -> str:
        return f"ADV-{self.turn_id:04d}-{self.state_revision:04d}"

    @property
    def decision_id(self) -> str:
        return f"{self.session_id}:turn_{self.turn_id}:revision_{self.state_revision}"


@dataclass(frozen=True)
class LiveAdvice:
    key: AdviceRequestKey
    status: Literal["requested", "ready", "stale", "failed", "withheld"]
    advice: LocalAdvice | None = None
    visible: bool = False
    error: str = ""
    withhold_reason: str = ""
    suit_uncertain: bool = False
    variant_count: int = 1
    advice_agrees_across_variants: bool = True
    semantic_uncertain: bool = False
    semantic_source_history_indices: tuple[int, ...] = ()
    suit_variant_count: int = 1
    suit_equivalence_class_count: int = 1
    semantic_variant_count: int = 1


@dataclass(frozen=True)
class LiveUpdate:
    status: LiveStatus
    snapshot: LiveSnapshot
    event: LiveEvent | None = None
    events: tuple[LiveEvent, ...] = ()
    advice: object | None = None
    review: ReviewRequest | None = None
    fast_signals: FastSignalResult | None = None
    # Kept presentation-neutral so the domain contract does not import the
    # legacy visual tracker implementation.
    local_rule_hint: object | None = None
    capture_generation: int = 0
    update_sequence: int = 0
    block_reason: str = ""
    missing_player: Seat | None = None
    missing_action_kind: str = ""


__all__ = [
    "AdviceRequestKey",
    "LiveAdvice",
    "LiveStatus",
    "LiveUpdate",
    "ReviewCandidate",
    "ReviewRequest",
]
