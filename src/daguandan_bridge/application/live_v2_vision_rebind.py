"""Project completed vision work onto the current formal rule state."""

from __future__ import annotations

from dataclasses import replace

from ..live_v2.identity import FrameIdentity, VersionIdentity
from .live_v2_frame_types import FramePipelineResult


def salvage_vision_result(
    result: FramePipelineResult,
    *,
    source_version: VersionIdentity,
    current_version: VersionIdentity,
    formal_action_boundary: FrameIdentity | None,
) -> FramePipelineResult | None:
    """Keep same-state evidence or post-boundary candidates from an ancestor."""

    if not _same_stream(source_version, current_version):
        return None
    same_state = (
        source_version.state_revision,
        source_version.turn_index,
    ) == (
        current_version.state_revision,
        current_version.turn_index,
    )
    ancestor = (
        source_version.state_revision < current_version.state_revision
        and source_version.turn_index <= current_version.turn_index
    )
    if not same_state and not ancestor:
        return None
    candidates = tuple(
        replace(candidate, version=current_version)
        for candidate in result.candidates
        if same_state or _candidate_crosses_boundary(
            candidate.first_frame, candidate.last_frame, formal_action_boundary
        )
    )
    return replace(
        result,
        observations=result.observations if same_state else (),
        candidates=candidates,
    )


def _same_stream(left: VersionIdentity, right: VersionIdentity) -> bool:
    return (
        left.session_id,
        left.capture_generation,
    ) == (
        right.session_id,
        right.capture_generation,
    )


def _candidate_crosses_boundary(
    first: FrameIdentity,
    last: FrameIdentity,
    boundary: FrameIdentity | None,
) -> bool:
    """Salvage a two-frame candidate that starts at, then crosses, a boundary."""

    if boundary is None:
        return True
    return (
        first.session_id == boundary.session_id
        and first.capture_generation == boundary.capture_generation
        and first.frame_sequence >= boundary.frame_sequence
        and first.captured_ms >= boundary.captured_ms
        and _strictly_after(last, boundary)
    )


def _strictly_after(frame: FrameIdentity, boundary: FrameIdentity | None) -> bool:
    if boundary is None:
        return True
    return (
        frame.session_id == boundary.session_id
        and frame.capture_generation == boundary.capture_generation
        and frame.frame_sequence > boundary.frame_sequence
        and frame.captured_ms > boundary.captured_ms
    )


__all__ = ["salvage_vision_result"]
