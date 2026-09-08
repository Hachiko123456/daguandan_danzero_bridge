"""Seat-local observation lifecycle for the second-generation live engine.

The tracker deliberately knows nothing about turn order or GuanDan rules. It
answers one smaller question: has one seat produced a stable, *new* visual
action? All four seats use the same class; expected-turn information belongs
to scheduling and event resolution, never to observation acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    FrameIdentity,
    ObservationKind,
    Seat,
    SeatObservation,
    VersionIdentity,
)


_ACTION_KINDS = frozenset({ObservationKind.PASS, ObservationKind.PLAY})


def _frame_key(frame: FrameIdentity) -> tuple[object, ...]:
    return (
        frame.session_id,
        frame.capture_generation,
        frame.frame_sequence,
        frame.captured_ms,
        frame.roi_version,
        frame.source_id,
    )


def _signature(observation: SeatObservation) -> tuple[object, ...] | None:
    if observation.kind is ObservationKind.PASS:
        return (ObservationKind.PASS,)
    if observation.kind is ObservationKind.PLAY:
        return (
            ObservationKind.PLAY,
            observation.cards,
            observation.suit_options,
        )
    return None


@dataclass(frozen=True)
class SeatTrackerSnapshot:
    """Small, immutable diagnostic view of one seat tracker."""

    seat: Seat
    action_epoch: int
    emitted_signature: tuple[object, ...] | None
    pending_signature: tuple[object, ...] | None
    pending_count: int
    empty_count: int
    seen_frames: int
    duplicate_frames: int


class SeatTracker:
    """Turn frame observations into at-most-once seat action candidates.

    A candidate requires ``confirmations`` distinct frames. Once emitted, the
    same visual action stays latched and cannot be emitted again until either:

    * ``empty_confirmations`` distinct EMPTY frames prove that the old visual
      surface was cleared; or
    * a different actionable signature is stable for ``confirmations`` frames.

    UNKNOWN and ANIMATING are intentionally non-semantic. They break a
    confirmation streak but cannot clear an emitted action or imply PASS.
    """

    def __init__(
        self,
        seat: Seat,
        *,
        confirmations: int = 2,
        empty_confirmations: int = 2,
        seen_frame_capacity: int = 256,
    ) -> None:
        if confirmations < 2:
            raise ValueError("confirmations must require at least two observations")
        if empty_confirmations < 1:
            raise ValueError("empty_confirmations must be positive")
        if seen_frame_capacity < confirmations + empty_confirmations:
            raise ValueError("seen_frame_capacity is too small")
        self.seat = Seat(seat)
        self.confirmations = confirmations
        self.empty_confirmations = empty_confirmations
        self.seen_frame_capacity = seen_frame_capacity

        self._action_epoch = 0
        self._emitted_signature: tuple[object, ...] | None = None
        self._pending_signature: tuple[object, ...] | None = None
        self._pending: list[SeatObservation] = []
        self._empty_count = 0
        self._seen_order: list[tuple[object, ...]] = []
        self._seen: set[tuple[object, ...]] = set()
        self._duplicate_frames = 0
        self._stream: tuple[str, int, str, str] | None = None
        self._last_frame: FrameIdentity | None = None
        self._latest_version: VersionIdentity | None = None
        self._pending_turn: tuple[str, int, int, int] | None = None
        self._emitted_turn: tuple[str, int, int, int] | None = None
        self._armed_pass_turn: tuple[str, int, int, int] | None = None

    @property
    def action_epoch(self) -> int:
        return self._action_epoch

    def snapshot(self) -> SeatTrackerSnapshot:
        return SeatTrackerSnapshot(
            seat=self.seat,
            action_epoch=self._action_epoch,
            emitted_signature=self._emitted_signature,
            pending_signature=self._pending_signature,
            pending_count=len(self._pending),
            empty_count=self._empty_count,
            seen_frames=len(self._seen),
            duplicate_frames=self._duplicate_frames,
        )

    def reset(self) -> None:
        """Clear lifecycle state, normally at session/generation boundaries."""

        self._action_epoch = 0
        self._emitted_signature = None
        self._pending_signature = None
        self._pending.clear()
        self._empty_count = 0
        self._seen_order.clear()
        self._seen.clear()
        self._duplicate_frames = 0
        self._stream = None
        self._last_frame = None
        self._latest_version = None
        self._pending_turn = None
        self._emitted_turn = None
        self._armed_pass_turn = None

    def arm_persistent_pass(
        self, version: VersionIdentity, *, turnover_confirmed: bool
    ) -> bool:
        """Rearm a latched PASS only for a crossed, newer formal turn."""

        if not turnover_confirmed or not self._version_is_current(version):
            return False
        self._latest_version = version
        turn = self._turn_key(version)
        if (
            turn == self._armed_pass_turn
            or self._emitted_turn is not None and turn == self._emitted_turn
        ):
            return False
        if self._emitted_signature == (ObservationKind.PASS,):
            self._open_next_epoch()
        else:
            self._pending_signature = None
            self._pending.clear()
            self._pending_turn = None
        self._armed_pass_turn = turn
        return True

    def ingest(
        self,
        observation: SeatObservation,
        *,
        version: VersionIdentity | None = None,
    ) -> ActionCandidate | None:
        """Consume one observation and return a newly confirmed candidate."""

        if observation.seat != self.seat:
            raise ValueError(
                f"observation seat {observation.seat!r} does not match tracker "
                f"seat {self.seat!r}"
            )
        if version is not None and not version.belongs_to(observation.frame):
            raise ValueError("version and observation must belong to the same stream")
        if version is not None and not self._version_is_current(version):
            return None
        stream = (
            observation.frame.session_id,
            observation.frame.capture_generation,
            observation.frame.roi_version,
            observation.frame.source_id,
        )
        if self._stream is not None and stream != self._stream:
            old_session, old_generation, _, _ = self._stream
            if (
                stream[0] == old_session
                and stream[1] < old_generation
            ):
                return None
            self.reset()
        self._stream = stream
        turn = self._turn_key(version) if version is not None else None
        if turn is not None:
            self._latest_version = version
            if self._pending_turn is not None and self._pending_turn != turn:
                self._pending_signature = None
                self._pending.clear()
            self._pending_turn = turn
        key = _frame_key(observation.frame)
        if key in self._seen:
            self._duplicate_frames += 1
            return None
        if self._last_frame is not None and not self._is_later(
            observation.frame, self._last_frame
        ):
            return None
        self._remember_frame(key)
        self._last_frame = observation.frame

        if observation.kind is ObservationKind.EMPTY:
            self._pending_signature = None
            self._pending.clear()
            self._empty_count += 1
            if (
                self._emitted_signature is not None
                and self._empty_count >= self.empty_confirmations
            ):
                self._open_next_epoch()
            return None

        self._empty_count = 0
        if observation.kind not in _ACTION_KINDS:
            self._pending_signature = None
            self._pending.clear()
            return None

        signature = _signature(observation)
        if signature is None:
            return None
        if signature == self._emitted_signature:
            self._pending_signature = None
            self._pending.clear()
            return None

        if signature != self._pending_signature:
            self._pending_signature = signature
            self._pending = [observation]
        else:
            self._pending.append(observation)

        if len(self._pending) < self.confirmations:
            return None

        # A stable different action surface is itself a new action-cycle edge
        # when a short empty/animation transition was not sampled.
        if self._emitted_signature is not None:
            self._action_epoch += 1
        if version is None:
            raise ValueError("version is required when an action becomes confirmed")
        candidate = self._build_candidate(self._pending, version=version)
        self._emitted_signature = signature
        self._emitted_turn = self._turn_key(version)
        self._pending_signature = None
        self._pending = []
        self._pending_turn = None
        return candidate

    def observe(
        self,
        observation: SeatObservation,
        *,
        version: VersionIdentity | None = None,
    ) -> ActionCandidate | None:
        """Readable integration alias for :meth:`ingest`."""

        return self.ingest(observation, version=version)

    def _open_next_epoch(self) -> None:
        self._action_epoch += 1
        self._emitted_signature = None
        self._pending_signature = None
        self._pending.clear()
        self._empty_count = 0
        self._pending_turn = None
        self._emitted_turn = None
        self._armed_pass_turn = None

    def _remember_frame(self, key: tuple[object, ...]) -> None:
        self._seen.add(key)
        self._seen_order.append(key)
        overflow = len(self._seen_order) - self.seen_frame_capacity
        if overflow <= 0:
            return
        for old in self._seen_order[:overflow]:
            self._seen.discard(old)
        del self._seen_order[:overflow]

    def _build_candidate(
        self,
        observations: list[SeatObservation],
        *,
        version: VersionIdentity,
    ) -> ActionCandidate:
        first = observations[0]
        last = observations[-1]
        confidence = min(item.confidence for item in observations)
        kind = ActionKind.PASS if first.kind is ObservationKind.PASS else ActionKind.PLAY
        reason = (
            CandidateReason.FRESH_PASS_EDGE
            if kind is ActionKind.PASS
            else CandidateReason.STABLE_PLAY
        )
        diagnostics = tuple(dict.fromkeys(
            detail
            for observation in observations
            for detail in observation.diagnostics
        )) + (f"candidate_confidence={confidence:.3f}",)
        return ActionCandidate(
            candidate_id=(
                f"candidate:{first.frame.session_id}:"
                f"{first.frame.capture_generation}:{first.seat.value}:"
                f"{self._action_epoch}:"
                f"{first.frame.frame_sequence}-{last.frame.frame_sequence}:"
                f"{first.frame.roi_version}:{first.frame.source_id}"
            ),
            version=version,
            seat=first.seat,
            kind=kind,
            cards=first.cards,
            suit_options=first.suit_options,
            evidence_ids=tuple(item.observation_id for item in observations),
            action_epoch=self._action_epoch,
            first_frame=first.frame,
            last_frame=last.frame,
            processing_ms=max(item.processing_ms for item in observations),
            confidence=confidence,
            reason=reason,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _is_later(frame: FrameIdentity, previous: FrameIdentity) -> bool:
        return (
            frame.frame_sequence > previous.frame_sequence
            and frame.captured_ms > previous.captured_ms
        )

    @staticmethod
    def _turn_key(
        version: VersionIdentity | None,
    ) -> tuple[str, int, int, int] | None:
        if version is None:
            return None
        return (
            version.session_id, version.capture_generation,
            version.state_revision, version.turn_index,
        )

    def _version_is_current(self, version: VersionIdentity) -> bool:
        previous = self._latest_version
        if previous is None:
            return True
        if version.session_id != previous.session_id:
            return False
        if version.capture_generation != previous.capture_generation:
            return version.capture_generation > previous.capture_generation
        return (version.state_revision, version.turn_index) >= (
            previous.state_revision, previous.turn_index
        )
