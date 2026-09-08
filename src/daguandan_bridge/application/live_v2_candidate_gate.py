"""Select at most one turn-eligible visual candidate for production."""

from __future__ import annotations

from dataclasses import dataclass

from ..live_v2.candidates import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    EvidenceOrigin,
)
from ..live_v2.game_state import TrustedGameSnapshot


@dataclass(frozen=True, slots=True)
class CandidateGateResult:
    selected: tuple[ActionCandidate, ...]
    ignored_ids: tuple[str, ...]
    reason: str


def gate_visual_candidates(
    snapshot: TrustedGameSnapshot,
    candidates: tuple[ActionCandidate, ...],
) -> CandidateGateResult:
    """Return zero or one candidate without attempting multi-seat ordering."""

    if not candidates:
        return CandidateGateResult((), (), "no_candidate")
    opening = not snapshot.play_history
    if opening:
        eligible = tuple(
            item
            for item in candidates
            if item.kind is ActionKind.PLAY
            and item.reason is CandidateReason.STABLE_PLAY
            and item.evidence_origin is EvidenceOrigin.VISUAL
            and (
                snapshot.lead_seat is None
                or item.seat is snapshot.lead_seat
            )
        )
        reason = "opening_candidate"
    else:
        if snapshot.current_seat is None:
            return CandidateGateResult(
                (), tuple(item.candidate_id for item in candidates), "no_current_seat"
            )
        eligible = tuple(
            item for item in candidates if item.seat is snapshot.current_seat
        )
        reason = "current_seat_candidate"
    if not eligible:
        return CandidateGateResult(
            (), tuple(item.candidate_id for item in candidates), "foreign_candidates_ignored"
        )
    signatures = {
        (item.seat, item.kind, item.cards, item.suit_options)
        for item in eligible
    }
    if len(signatures) != 1:
        return CandidateGateResult(
            (), tuple(item.candidate_id for item in candidates), "eligible_candidate_conflict"
        )
    selected = max(
        eligible,
        key=lambda item: (
            item.last_frame.frame_sequence,
            item.last_captured_ms,
            item.confidence,
        ),
    )
    ignored = tuple(
        item.candidate_id for item in candidates if item is not selected
    )
    return CandidateGateResult((selected,), ignored, reason)


__all__ = ["CandidateGateResult", "gate_visual_candidates"]
