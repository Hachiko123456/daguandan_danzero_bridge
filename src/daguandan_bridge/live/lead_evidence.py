"""Small, dependency-free helpers for the auditable opening candidate chain."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ..domain.recognition import LeadEvidence
from ..danzero.state import Seat


LEAD_EVIDENCE_THRESHOLD = 0.75
LEAD_EVIDENCE_MARGIN = 0.08


def rank_lead_evidence(
    evidence: Iterable[LeadEvidence],
    *,
    threshold: float = LEAD_EVIDENCE_THRESHOLD,
    margin: float = LEAD_EVIDENCE_MARGIN,
) -> tuple[LeadEvidence, ...]:
    """Classify a frame's candidates without committing a session.

    A strong card-action score is sufficient to keep a candidate alive even
    when the first-play template is weak.  A close second candidate turns the
    frame into an explicit conflict instead of silently selecting one seat.
    """

    ordered = sorted(
        tuple(evidence),
        key=lambda item: (item.candidate_score, item.card_action_score, item.candidate_seat),
        reverse=True,
    )
    if not ordered:
        return ()
    best = ordered[0]
    second_score = ordered[1].candidate_score if len(ordered) > 1 else 0.0
    if best.candidate_score < threshold:
        return tuple(
            replace(item, status="rejected", rejection_reason="insufficient_evidence")
            for item in ordered
        )
    if len(ordered) > 1 and best.candidate_score - second_score < margin:
        return tuple(
            replace(item, status="conflict", rejection_reason="candidate_conflict")
            for item in ordered
        )
    return tuple(
        replace(
            item,
            status=("pending_confirmation" if item is best else "rejected"),
            rejection_reason=(None if item is best else "lower_score"),
        )
        for item in ordered
    )


def evidence_for_seat(
    evidence: Iterable[LeadEvidence], seat: Seat
) -> LeadEvidence | None:
    """Return the latest evidence for one seat, if present."""

    return next((item for item in evidence if item.candidate_seat == seat), None)


__all__ = [
    "LEAD_EVIDENCE_MARGIN",
    "LEAD_EVIDENCE_THRESHOLD",
    "LeadEvidence",
    "evidence_for_seat",
    "rank_lead_evidence",
]
