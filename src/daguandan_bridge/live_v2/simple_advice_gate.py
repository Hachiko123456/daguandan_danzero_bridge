"""Small, explicit gate between the deterministic turn core and Advice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .turn_core import HistoryIntegrity, TurnPhase, TurnState


class AdviceGateDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class AdviceGateResult:
    decision: AdviceGateDecision
    reason: str
    submission_key: tuple[int, str] | None = None

    @property
    def allowed(self) -> bool:
        return self.decision is AdviceGateDecision.ALLOW


class SimpleAdviceGate:
    """Prevent Advice from consuming incomplete or duplicated turn states."""

    def __init__(self) -> None:
        self._submitted: set[tuple[int, str]] = set()

    def evaluate(
        self,
        state: TurnState,
        *,
        history_digest: str,
        partial_suits_safe: bool = False,
        semantic_key: str | None = None,
    ) -> AdviceGateResult:
        if state.pending is not None:
            return AdviceGateResult(AdviceGateDecision.BLOCK, "pending_action")
        if state.integrity is HistoryIntegrity.DESYNC:
            return AdviceGateResult(AdviceGateDecision.BLOCK, "desync")
        if state.integrity is HistoryIntegrity.UNTRUSTED:
            return AdviceGateResult(AdviceGateDecision.BLOCK, "untrusted_history")
        if state.phase is not TurnPhase.WAIT_EXPECTED:
            return AdviceGateResult(AdviceGateDecision.BLOCK, "phase_not_waiting")
        if state.cursor is None:
            return AdviceGateResult(AdviceGateDecision.BLOCK, "cursor_missing")
        if state.cursor.current_seat.value != "self":
            return AdviceGateResult(AdviceGateDecision.BLOCK, "not_self_turn")
        if state.integrity is HistoryIntegrity.PARTIAL_SUITS and not (
            partial_suits_safe and semantic_key and semantic_key.strip()
        ):
            return AdviceGateResult(AdviceGateDecision.BLOCK, "partial_suits_ambiguous")
        digest = str(history_digest).strip()
        if not digest:
            return AdviceGateResult(AdviceGateDecision.BLOCK, "history_digest_missing")
        key = (state.cursor.turn_token, digest)
        if key in self._submitted:
            return AdviceGateResult(AdviceGateDecision.BLOCK, "duplicate_submission", key)
        return AdviceGateResult(AdviceGateDecision.ALLOW, "trusted_self_turn", key)

    def record_submitted(self, result: AdviceGateResult) -> None:
        if not result.allowed or result.submission_key is None:
            raise ValueError("only an allowed gate result can be submitted")
        self._submitted.add(result.submission_key)

    def clear_turn(self, turn_token: int) -> None:
        self._submitted = {
            key for key in self._submitted if key[0] != int(turn_token)
        }


__all__ = ["AdviceGateDecision", "AdviceGateResult", "SimpleAdviceGate"]
