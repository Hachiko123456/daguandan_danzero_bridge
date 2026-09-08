"""Pure candidate identity, evidence, and capture-order planning."""

from __future__ import annotations

from itertools import permutations
from typing import TypeAlias

from .types import ActionCandidate, FrameIdentity, ProjectionReason, VersionIdentity


CandidateFingerprint: TypeAlias = tuple[
    object, int, FrameIdentity, FrameIdentity
]


def candidate_fingerprint(candidate: ActionCandidate) -> CandidateFingerprint:
    """Identity of one seat action-cycle over one complete evidence interval."""

    return (
        candidate.seat,
        candidate.action_epoch,
        candidate.first_frame,
        candidate.last_frame,
    )


def candidate_orders(
    *,
    base_version: VersionIdentity,
    candidates: tuple[ActionCandidate, ...],
    consumed: frozenset[CandidateFingerprint],
    consumed_evidence: frozenset[str],
    max_candidates: int,
) -> tuple[tuple[tuple[ActionCandidate, ...], ...], ProjectionReason | None]:
    """Return every non-empty capture-consistent subchain and ordering."""

    if not candidates:
        return (), ProjectionReason.NEED_MORE_EVIDENCE
    if len(candidates) > max_candidates:
        return (), ProjectionReason.NEED_MORE_EVIDENCE
    if any(candidate.version != base_version for candidate in candidates):
        return (), ProjectionReason.VERSION_MISMATCH
    fingerprints = tuple(candidate_fingerprint(item) for item in candidates)
    candidate_ids = tuple(item.candidate_id for item in candidates)
    evidence_ids = tuple(
        evidence for item in candidates for evidence in item.all_evidence_ids
    )
    if len(fingerprints) != len(set(fingerprints)):
        return (), ProjectionReason.OUT_OF_ORDER
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or len(evidence_ids) != len(set(evidence_ids))
        or bool(set(fingerprints) & consumed)
        or bool(set(evidence_ids) & consumed_evidence)
    ):
        return (), ProjectionReason.RULE_REJECTED
    if _has_overlapping_alternatives(candidates):
        return (), ProjectionReason.OUT_OF_ORDER

    orders = tuple(
        order
        for size in range(1, len(candidates) + 1)
        for order in permutations(candidates, size)
        if _one_capture_stream(order)
        and _capture_clocks_agree(order)
        and _respects_capture_order(order)
    )
    return orders, None if orders else ProjectionReason.OUT_OF_ORDER


def _frame_stream(frame: FrameIdentity) -> tuple[str, int, str, str]:
    return (
        frame.session_id,
        frame.capture_generation,
        frame.roi_version,
        frame.source_id,
    )


def _one_capture_stream(candidates: tuple[ActionCandidate, ...]) -> bool:
    anchor = _frame_stream(candidates[0].first_frame)
    return all(_frame_stream(item.first_frame) == anchor for item in candidates)


def _strictly_before(left: ActionCandidate, right: ActionCandidate) -> bool:
    return (
        left.last_frame.frame_sequence < right.first_frame.frame_sequence
        and left.last_frame.captured_ms < right.first_frame.captured_ms
    )


def _capture_clocks_agree(candidates: tuple[ActionCandidate, ...]) -> bool:
    frames = tuple(
        frame
        for candidate in candidates
        for frame in (candidate.first_frame, candidate.last_frame)
    )
    return all(
        (left.frame_sequence == right.frame_sequence)
        == (left.captured_ms == right.captured_ms)
        and (left.frame_sequence < right.frame_sequence)
        == (left.captured_ms < right.captured_ms)
        for index, left in enumerate(frames)
        for right in frames[index + 1 :]
    )


def _has_overlapping_alternatives(
    candidates: tuple[ActionCandidate, ...]
) -> bool:
    return any(
        left.seat is right.seat
        and left.action_epoch == right.action_epoch
        and not _strictly_before(left, right)
        and not _strictly_before(right, left)
        for index, left in enumerate(candidates)
        for right in candidates[index + 1 :]
    )


def _respects_capture_order(order: tuple[ActionCandidate, ...]) -> bool:
    positions = {
        candidate_fingerprint(item): index for index, item in enumerate(order)
    }
    return all(
        not _strictly_before(right, left)
        or positions[candidate_fingerprint(right)]
        < positions[candidate_fingerprint(left)]
        for left in order
        for right in order
        if left is not right
    )
