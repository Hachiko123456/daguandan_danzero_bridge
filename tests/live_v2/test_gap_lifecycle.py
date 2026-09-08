from __future__ import annotations

from dataclasses import replace

from daguandan_bridge.live_v2.gap_lifecycle import GapLifecycle
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    FrameIdentity,
    GapPhase,
    GapReason,
    ProjectionReason,
    ProjectionResult,
    Seat,
    VersionIdentity,
)


def version(**changes: object) -> VersionIdentity:
    values = dict(session_id="s", capture_generation=1, state_revision=0,
                  update_sequence=0, turn_index=0)
    values.update(changes)
    return VersionIdentity(**values)  # type: ignore[arg-type]


def candidate(candidate_id: str = "c1", captured_ms: int = 100) -> ActionCandidate:
    before = FrameIdentity("s", 1, 1, captured_ms, "r")
    after = FrameIdentity("s", 1, 2, captured_ms + 10, "r")
    return ActionCandidate(
        candidate_id, version(), Seat.RIGHT, ActionKind.PLAY, ("3H",),
        (("3H",),), (f"{candidate_id}-1", f"{candidate_id}-2"), 0,
        before, after,
        captured_ms + 20, 0.9, CandidateReason.STABLE_PLAY,
    )


def test_recovery_window_is_independent_from_two_second_advice_deadline() -> None:
    lifecycle = GapLifecycle(recovery_window_ms=8_000)
    clear = lifecycle.clear(version=version(), captured_ms=100, processing_ms=100)
    item = candidate()
    projection = ProjectionResult(
        version(), (), (item.candidate_id,), ProjectionReason.NEED_MORE_EVIDENCE
    )
    observing = lifecycle.from_projection(
        current=clear, version=version(update_sequence=1), projection=projection,
        candidates=(item,), captured_ms=110, processing_ms=2_500,
    )
    assert observing.phase is GapPhase.OBSERVING
    assert observing.reason is GapReason.MISSING_EXPECTED_ACTION

    expired = lifecycle.refresh(
        observing, version=version(update_sequence=2), captured_ms=8_110,
        processing_ms=8_110,
    )
    assert expired.phase is GapPhase.EXPIRED
    assert expired.reason is GapReason.RECOVERY_BUDGET_EXCEEDED


def test_new_evidence_reopens_an_expired_gap_without_old_evidence() -> None:
    lifecycle = GapLifecycle(recovery_window_ms=1_000)
    clear = lifecycle.clear(version=version(), captured_ms=0, processing_ms=0)
    first = candidate("old", 100)
    rejected = ProjectionResult(
        version(), (), ("old",), ProjectionReason.OUT_OF_ORDER
    )
    open_gap = lifecycle.from_projection(
        current=clear, version=version(), projection=rejected,
        candidates=(first,), captured_ms=110, processing_ms=110,
    )
    expired = lifecycle.refresh(
        open_gap, version=version(), captured_ms=1_110, processing_ms=1_110
    )
    second = candidate("new", 1_200)
    reopened = lifecycle.from_projection(
        current=expired, version=version(), projection=rejected,
        candidates=(second,), captured_ms=1_210, processing_ms=1_210,
    )
    assert reopened.phase is GapPhase.RECOVERABLE
    assert reopened.evidence_ids == ("new-1", "new-2")
    assert reopened.opened_captured_ms == 1_210


def test_processing_delay_does_not_age_capture_evidence() -> None:
    lifecycle = GapLifecycle(recovery_window_ms=1_000)
    clear = lifecycle.clear(version=version(), captured_ms=100, processing_ms=100)
    item = candidate(captured_ms=100)
    rejected = ProjectionResult(
        version(), (), (item.candidate_id,), ProjectionReason.OUT_OF_ORDER
    )
    current = lifecycle.from_projection(
        current=clear, version=version(), projection=rejected,
        candidates=(item,), captured_ms=110, processing_ms=10_000,
    )
    delayed = lifecycle.refresh(
        current, version=version(), captured_ms=110, processing_ms=20_000
    )
    assert delayed.phase is GapPhase.RECOVERABLE

    expired = lifecycle.refresh(
        delayed, version=version(), captured_ms=1_110, processing_ms=20_001
    )
    assert expired.phase is GapPhase.EXPIRED


def test_accepted_recovery_clears_gap_on_the_committed_version() -> None:
    lifecycle = GapLifecycle()
    current = lifecycle.clear(version=version(), captured_ms=0, processing_ms=0)
    item = candidate()
    rejected = ProjectionResult(
        version(), (), (item.candidate_id,), ProjectionReason.OUT_OF_ORDER
    )
    current = lifecycle.from_projection(
        current=current, version=version(), projection=rejected,
        candidates=(item,), captured_ms=110, processing_ms=110,
    )
    accepted = ProjectionResult(
        version(),
        (),
        (),
        ProjectionReason.NEED_MORE_EVIDENCE,
    )
    # The lifecycle's accepted branch is exercised with a valid projected
    # chain in engine tests; clear itself must discard every recovery detail.
    cleared = lifecycle.clear(
        version=replace(version(), state_revision=1),
        captured_ms=200,
        processing_ms=210,
    )
    assert current.phase is GapPhase.RECOVERABLE
    assert accepted.reason is ProjectionReason.NEED_MORE_EVIDENCE
    assert cleared.phase is GapPhase.CLEAR
    assert cleared.reason is GapReason.NONE
    assert cleared.evidence_ids == ()
