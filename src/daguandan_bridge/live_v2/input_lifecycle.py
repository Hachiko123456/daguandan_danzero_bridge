"""Input freshness, idempotency, version-sync, and update classification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .types import (
    ActionCandidate,
    ConfirmedAction,
    EngineUpdateReason,
    FrameIdentity,
    GapState,
    Seat,
    SeatObservation,
    VersionIdentity,
)


class InputKind(str, Enum):
    OBSERVATION = "observation"
    CANDIDATE = "candidate"
    SNAPSHOT = "snapshot"
    REBIND = "rebind"


class InputRejectionReason(str, Enum):
    STREAM_MISMATCH = "stream_mismatch"
    STALE_REVISION = "stale_revision"
    FUTURE_REVISION = "future_revision"
    EXPIRED_EVIDENCE = "expired_evidence"
    OUT_OF_ORDER_OBSERVATION = "out_of_order_observation"
    DUPLICATE_CANDIDATE = "duplicate_candidate"


@dataclass(frozen=True, slots=True)
class InputRejection:
    kind: InputKind
    reason: InputRejectionReason
    input_id: str


@dataclass(frozen=True, slots=True)
class EngineInput:
    observations: tuple[SeatObservation, ...] = ()
    candidates: tuple[ActionCandidate, ...] = ()
    rebind_version: VersionIdentity | None = None
    captured_watermark_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")
        if self.captured_watermark_ms is not None and (
            isinstance(self.captured_watermark_ms, bool)
            or not isinstance(self.captured_watermark_ms, int)
            or self.captured_watermark_ms < 0
        ):
            raise ValueError("captured_watermark_ms must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CandidateReceipt:
    candidate_id: str
    seat: Seat
    action_epoch: int
    first_frame: FrameIdentity
    last_frame: FrameIdentity
    captured_ms: int

    @classmethod
    def from_candidate(cls, candidate: ActionCandidate) -> CandidateReceipt:
        return cls(
            candidate.candidate_id,
            candidate.seat,
            candidate.action_epoch,
            candidate.first_frame,
            candidate.last_frame,
            candidate.last_captured_ms,
        )

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.candidate_id,
            self.seat,
            self.action_epoch,
            self.first_frame,
            self.last_frame,
        )


def _candidate_key(candidate: ActionCandidate) -> tuple[object, ...]:
    return (
        candidate.candidate_id,
        candidate.seat,
        candidate.action_epoch,
        candidate.first_frame,
        candidate.last_frame,
    )


@dataclass(frozen=True, slots=True)
class EvidenceLifecycle:
    """Generation-aware freshness and idempotency policy for extracted facts."""

    max_age_ms: int = 1_500
    receipt_capacity: int = 256

    def __post_init__(self) -> None:
        for name, value in (
            ("max_age_ms", self.max_age_ms),
            ("receipt_capacity", self.receipt_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def observations(
        self,
        items: tuple[SeatObservation, ...],
        current: tuple[SeatObservation | None, ...],
        version: VersionIdentity,
        captured_now: int,
    ) -> tuple[tuple[SeatObservation, ...], tuple[InputRejection, ...]]:
        accepted: list[SeatObservation] = []
        rejected: list[InputRejection] = []
        newest = list(current)
        seats = tuple(Seat)
        for item in items:
            reason = None
            if not version.belongs_to(item.frame):
                reason = InputRejectionReason.STREAM_MISMATCH
            elif captured_now - item.frame.captured_ms > self.max_age_ms:
                reason = InputRejectionReason.EXPIRED_EVIDENCE
            else:
                prior = newest[seats.index(item.seat)]
                if prior is not None and (
                    item.frame.frame_sequence <= prior.frame.frame_sequence
                ):
                    reason = InputRejectionReason.OUT_OF_ORDER_OBSERVATION
            if reason is not None:
                rejected.append(InputRejection(
                    InputKind.OBSERVATION,
                    reason,
                    item.observation_id,
                ))
            else:
                accepted.append(item)
                newest[seats.index(item.seat)] = item
        return tuple(accepted), tuple(rejected)

    @staticmethod
    def merge_observations(
        current: tuple[SeatObservation | None, ...],
        items: tuple[SeatObservation, ...],
    ) -> tuple[SeatObservation | None, ...]:
        merged = list(current)
        seats = tuple(Seat)
        for item in items:
            merged[seats.index(item.seat)] = item
        return tuple(merged)

    def candidates(
        self,
        items: tuple[ActionCandidate, ...],
        pending: tuple[ActionCandidate, ...],
        receipts: tuple[CandidateReceipt, ...],
        version: VersionIdentity,
        captured_now: int,
    ) -> tuple[tuple[ActionCandidate, ...], tuple[InputRejection, ...]]:
        accepted: list[ActionCandidate] = []
        rejected: list[InputRejection] = []
        known = {_candidate_key(item) for item in pending}
        known.update(item.key for item in receipts)
        stream = (version.session_id, version.capture_generation)
        for item in items:
            item_stream = (item.version.session_id, item.version.capture_generation)
            if item_stream != stream:
                reason = InputRejectionReason.STREAM_MISMATCH
            elif _candidate_key(item) in known:
                reason = InputRejectionReason.DUPLICATE_CANDIDATE
            elif item.version.state_version != version.state_version:
                item_position = (
                    item.version.state_revision,
                    item.version.turn_index,
                )
                current_position = (version.state_revision, version.turn_index)
                reason = (
                    InputRejectionReason.STALE_REVISION
                    if item_position < current_position
                    else InputRejectionReason.FUTURE_REVISION
                )
            elif captured_now - item.last_captured_ms > self.max_age_ms:
                reason = InputRejectionReason.EXPIRED_EVIDENCE
            else:
                known.add(_candidate_key(item))
                accepted.append(item)
                continue
            rejected.append(InputRejection(
                InputKind.CANDIDATE,
                reason,
                item.candidate_id,
            ))
        return tuple(accepted), tuple(rejected)

    def pending(
        self,
        items: tuple[ActionCandidate, ...],
        version: VersionIdentity,
        captured_now: int,
    ) -> tuple[ActionCandidate, ...]:
        return tuple(
            item for item in items
            if captured_now - item.last_captured_ms <= self.max_age_ms
            and item.version.state_version == version.state_version
        )

    def receipts(
        self,
        items: tuple[CandidateReceipt, ...],
        captured_now: int,
        committed: tuple[ActionCandidate, ...] = (),
    ) -> tuple[CandidateReceipt, ...]:
        kept = tuple(
            item for item in items
            if captured_now - item.captured_ms <= self.max_age_ms
        )
        added = tuple(CandidateReceipt.from_candidate(item) for item in committed)
        return (kept + added)[-self.receipt_capacity :]

    @staticmethod
    def watermark(
        current: int,
        version: VersionIdentity,
        observations: tuple[SeatObservation, ...],
        candidates: tuple[ActionCandidate, ...],
        snapshot_version: VersionIdentity | None,
        snapshot_captured_ms: int | None,
        explicit: int | None,
    ) -> int:
        values = [current]
        stream = (version.session_id, version.capture_generation)
        if explicit is not None:
            values.append(explicit)
        values.extend(
            item.frame.captured_ms for item in observations
            if (item.frame.session_id, item.frame.capture_generation) == stream
        )
        values.extend(
            item.last_captured_ms for item in candidates
            if (item.version.session_id, item.version.capture_generation) == stream
            and item.version.state_version == version.state_version
        )
        if snapshot_version is not None and (
            snapshot_version.session_id,
            snapshot_version.capture_generation,
        ) == stream and (
            snapshot_version.state_revision == version.state_revision
            and snapshot_version.turn_index == version.turn_index
            and snapshot_captured_ms is not None
        ):
            values.append(snapshot_captured_ms)
        return max(values)


def validate_version_sync(
    current: VersionIdentity,
    requested: VersionIdentity | None,
) -> InputRejection | None:
    if requested is None:
        return None
    if (requested.session_id, requested.capture_generation) != (
        current.session_id, current.capture_generation
    ):
        return InputRejection(
            InputKind.REBIND,
            InputRejectionReason.STREAM_MISMATCH,
            "rebind_version",
        )
    same_state = (
        requested.state_revision == current.state_revision
        and requested.turn_index == current.turn_index
    )
    if same_state and requested.update_sequence >= current.update_sequence:
        return None
    reason = (
        InputRejectionReason.STALE_REVISION
        if requested.state_revision <= current.state_revision
        else InputRejectionReason.FUTURE_REVISION
    )
    return InputRejection(InputKind.REBIND, reason, "rebind_version")


def choose_update_reason(
    *,
    observations: tuple[SeatObservation, ...],
    candidates: tuple[ActionCandidate, ...],
    committed: tuple[ConfirmedAction, ...],
    old_gap: GapState,
    gap: GapState,
    opportunity_changed: bool,
) -> EngineUpdateReason:
    if committed:
        return EngineUpdateReason.ACTION_COMMITTED
    if candidates:
        return EngineUpdateReason.CANDIDATE_CREATED
    if (old_gap.phase, old_gap.reason) != (gap.phase, gap.reason):
        return EngineUpdateReason.GAP_CHANGED
    if opportunity_changed:
        return EngineUpdateReason.OPPORTUNITY_CHANGED
    return (
        EngineUpdateReason.OBSERVATION_RECEIVED
        if observations else EngineUpdateReason.OPPORTUNITY_CHANGED
    )


def processing_watermark(
    clock_ms: int,
    previous_processing_ms: int,
    observations: tuple[SeatObservation, ...],
    candidates: tuple[ActionCandidate, ...],
) -> int:
    values = [clock_ms, previous_processing_ms]
    values.extend(item.processing_ms for item in observations)
    values.extend(item.processing_ms for item in candidates)
    return max(values)


def next_flow_version(
    committed_view: VersionIdentity,
    previous_update_sequence: int,
) -> VersionIdentity:
    return replace(
        committed_view,
        update_sequence=max(
            committed_view.update_sequence,
            previous_update_sequence + 1,
        ),
    )
