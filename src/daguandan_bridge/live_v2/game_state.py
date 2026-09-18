"""Complete immutable game state handed from live-v2 to an adviser."""
from __future__ import annotations
from dataclasses import dataclass
from .action_semantics import ActionSemantics
from .corrections import ConfirmedCorrection
from .events import ActionKind, ConfirmedAction
from .identity import (
    FrameIdentity,
    Seat,
    StateVersion,
    VersionIdentity,
    require_enum,
    require_instance,
    require_non_negative_int,
    require_probability,
    require_text,
    require_tuple,
    require_unique_text_tuple,
)
from .observations import require_cards, require_suit_options
INITIAL_CARDS_PER_SEAT = 27
_SEATS = tuple(Seat)
@dataclass(frozen=True, slots=True)
class GameAction:
    """One rule-confirmed action with its physical-card evidence intact."""
    action_id: str
    version_before: StateVersion
    version_after: StateVersion
    seat: Seat
    kind: ActionKind
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    action_epoch: int
    evidence_ids: tuple[str, ...]
    first_frame: FrameIdentity
    last_frame: FrameIdentity
    confidence: float
    captured_ms: int
    correction_id: str | None = None
    semantics: ActionSemantics | None = None

    @classmethod
    def from_confirmed(
        cls,
        action: ConfirmedAction,
        correction: ConfirmedCorrection | None = None,
    ) -> GameAction:
        """Copy a committed event without collapsing cards or evidence."""
        require_instance(action, ConfirmedAction, "action")
        return cls(
            action_id=action.action_id,
            version_before=action.version_before,
            version_after=action.version_after,
            seat=action.seat,
            kind=action.kind if correction is None else correction.corrected_kind,
            cards=action.cards if correction is None else correction.corrected_cards,
            suit_options=(
                action.suit_options
                if correction is None
                else correction.corrected_suit_options
            ),
            action_epoch=action.action_epoch,
            evidence_ids=action.evidence_ids,
            first_frame=action.first_frame,
            last_frame=action.last_frame,
            confidence=(
                action.source_candidate.confidence
                if correction is None
                else correction.confidence
            ),
            captured_ms=action.captured_ms,
            correction_id=None if correction is None else correction.correction_id,
            semantics=(
                action.semantics
                if correction is None
                else correction.corrected_semantics
            ),
        )

    def __post_init__(self) -> None:
        require_text(self.action_id, "action_id")
        require_instance(self.version_before, StateVersion, "version_before")
        require_instance(self.version_after, StateVersion, "version_after")
        require_enum(self.seat, Seat, "seat")
        require_enum(self.kind, ActionKind, "kind")
        require_non_negative_int(self.action_epoch, "action_epoch")
        require_unique_text_tuple(self.evidence_ids, "evidence_ids")
        if not self.evidence_ids:
            raise ValueError("evidence_ids must not be empty")
        require_instance(self.first_frame, FrameIdentity, "first_frame")
        require_instance(self.last_frame, FrameIdentity, "last_frame")
        require_probability(self.confidence, "confidence")
        require_non_negative_int(self.captured_ms, "captured_ms")
        if self.correction_id is not None:
            require_text(self.correction_id, "correction_id")
        if self.semantics is not None and not isinstance(
            self.semantics, ActionSemantics
        ):
            raise TypeError("semantics must be ActionSemantics or None")
        if self.kind is ActionKind.PASS and self.semantics is not None:
            raise ValueError("pass action cannot contain action semantics")

        before = self.version_before
        after = self.version_after
        if before.session_id != after.session_id:
            raise ValueError("game action cannot cross session")
        if after.state_revision != before.state_revision + 1:
            raise ValueError("game action must advance state_revision exactly once")
        if after.turn_index != before.turn_index + 1:
            raise ValueError("game action must advance turn_index exactly once")
        first_stream = (
            self.first_frame.session_id,
            self.first_frame.capture_generation,
            self.first_frame.roi_version,
            self.first_frame.source_id,
        )
        last_stream = (
            self.last_frame.session_id,
            self.last_frame.capture_generation,
            self.last_frame.roi_version,
            self.last_frame.source_id,
        )
        if first_stream != last_stream:
            raise ValueError("game action evidence must use one capture stream")
        for frame in (self.first_frame, self.last_frame):
            if frame.session_id != before.session_id:
                raise ValueError("game action evidence belongs to another session")
        if self.last_frame.frame_sequence < self.first_frame.frame_sequence:
            raise ValueError("game action evidence frames are reversed")
        if self.last_frame.captured_ms < self.first_frame.captured_ms:
            raise ValueError("game action evidence timestamps are reversed")
        if self.captured_ms != self.last_frame.captured_ms:
            raise ValueError("captured_ms must identify the final evidence frame")
        require_cards(self.cards, allow_empty=self.kind is ActionKind.PASS)
        if self.kind is ActionKind.PLAY:
            require_suit_options(
                self.cards, self.suit_options, require_selected_card=False
            )
        else:
            require_tuple(self.suit_options, "suit_options")
            if self.cards or self.suit_options:
                raise ValueError("pass action cannot contain cards or suit_options")

    @property
    def action_metadata(self) -> dict[str, object] | None:
        return None if self.semantics is None else self.semantics.to_metadata()
@dataclass(frozen=True, slots=True)
class SeatCardCount:
    seat: Seat
    count: int

    def __post_init__(self) -> None:
        require_enum(self.seat, Seat, "seat")
        require_non_negative_int(self.count, "count")
        if self.count > INITIAL_CARDS_PER_SEAT:
            raise ValueError("remaining card count cannot exceed initial hand size")
@dataclass(frozen=True, slots=True)
class TrustedGameSnapshot:
    """Self-contained rule state suitable for a version-bound model request."""
    version: VersionIdentity
    round_level: str
    wild_rank: str
    trick_index: int
    current_seat: Seat | None
    lead_seat: Seat | None
    my_hand: tuple[str, ...]
    play_history: tuple[GameAction, ...]
    current_trick: tuple[GameAction, ...]
    remaining: tuple[SeatCardCount, ...]
    finished: tuple[Seat, ...]
    trusted: bool
    terminal: bool
    captured_ms: int
    correction_history: tuple[ConfirmedCorrection, ...] = ()

    def __post_init__(self) -> None:
        require_instance(self.version, VersionIdentity, "version")
        require_text(self.round_level, "round_level")
        require_text(self.wild_rank, "wild_rank")
        if (
            isinstance(self.trick_index, bool)
            or not isinstance(self.trick_index, int)
            or self.trick_index < 1
        ):
            raise ValueError("trick_index must be a positive integer")
        if self.current_seat is not None:
            require_enum(self.current_seat, Seat, "current_seat")
        if self.lead_seat is not None:
            require_enum(self.lead_seat, Seat, "lead_seat")
        require_cards(self.my_hand, allow_empty=True)
        for field_name in (
            "play_history",
            "current_trick",
            "remaining",
            "finished",
            "correction_history",
        ):
            require_tuple(getattr(self, field_name), field_name)
        if not isinstance(self.trusted, bool) or not isinstance(self.terminal, bool):
            raise TypeError("trusted and terminal must be bool")
        require_non_negative_int(self.captured_ms, "captured_ms")

        self._validate_counts()
        self._validate_history()
        self._validate_lifecycle()
    def _validate_counts(self) -> None:
        if len(self.remaining) != len(_SEATS):
            raise ValueError("remaining must contain exactly four seat counts")
        for item in self.remaining:
            require_instance(item, SeatCardCount, "remaining item")
        seats = tuple(item.seat for item in self.remaining)
        if len(set(seats)) != len(_SEATS) or set(seats) != set(_SEATS):
            raise ValueError("remaining must contain every seat exactly once")
        for seat in self.finished:
            require_enum(seat, Seat, "finished seat")
        if len(set(self.finished)) != len(self.finished):
            raise ValueError("finished must not contain duplicate seats")

        played = {seat: 0 for seat in _SEATS}
        for action in self.play_history:
            if action.kind is ActionKind.PLAY:
                played[action.seat] += len(action.cards)
        counts = {item.seat: item.count for item in self.remaining}
        for seat in _SEATS:
            if played[seat] + counts[seat] != INITIAL_CARDS_PER_SEAT:
                raise ValueError("play history and remaining counts are inconsistent")
        if len(self.my_hand) != counts[Seat.SELF]:
            raise ValueError("my_hand must contain exactly self's remaining cards")
        zero_seats = {seat for seat, count in counts.items() if count == 0}
        if set(self.finished) != zero_seats:
            raise ValueError("finished seats must exactly match zero remaining counts")
    def _validate_history(self) -> None:
        state_version = self.version.state_version
        identifiers: set[str] = set()
        for action in self.play_history:
            require_instance(action, GameAction, "play_history action")
            if action.action_id in identifiers:
                raise ValueError("play_history action_id values must be unique")
            identifiers.add(action.action_id)
            if action.version_before.session_id != state_version.session_id:
                raise ValueError("play_history action belongs to another snapshot session")
            if action.version_after.state_revision > self.version.state_revision:
                raise ValueError("play_history is newer than its snapshot")
            if action.version_after.turn_index > self.version.turn_index:
                raise ValueError("play_history turn is newer than its snapshot")
            if action.captured_ms > self.captured_ms:
                raise ValueError("play_history evidence is newer than its snapshot")
        _validate_correction_timeline(
            self.play_history,
            self.correction_history,
            state_version,
        )
        if len(self.current_trick) > len(self.play_history) or (
            self.current_trick
            and self.play_history[-len(self.current_trick):] != self.current_trick
        ):
            raise ValueError("current_trick must be an exact suffix of play_history")
        if self.current_trick and self.current_trick[0].kind is ActionKind.PASS:
            raise ValueError("current_trick cannot begin with a pass")
        if self.current_trick and self.lead_seat is not self.current_trick[0].seat:
            raise ValueError("lead_seat must own the first play of current_trick")
        if self.correction_history and self.captured_ms < max(
            item.corrected_ms for item in self.correction_history
        ):
            raise ValueError("snapshot watermark precedes a confirmed correction")
    def _validate_lifecycle(self) -> None:
        if self.current_seat is not None and self.current_seat in self.finished:
            raise ValueError("finished seat cannot be the current actor")
        if self.terminal:
            if self.current_seat is not None:
                raise ValueError("terminal snapshot cannot have a current actor")
            if len(self.finished) < 2:
                raise ValueError("terminal snapshot requires a decided finishing order")
            if self.current_trick:
                raise ValueError("terminal snapshot cannot retain a current trick")
        elif len(self.finished) >= 3:
            raise ValueError("three finished seats require a terminal snapshot")
        elif self.current_trick:
            if self.lead_seat is None:
                raise ValueError("active current trick requires a lead_seat")
        elif self.current_seat is None:
            if self.lead_seat is not None:
                raise ValueError("waiting for opening lead requires both seats to be empty")
        elif self.lead_seat is not self.current_seat:
            raise ValueError("empty current trick must bind lead_seat to current_seat")
    def remaining_for(self, seat: Seat) -> int:
        require_enum(seat, Seat, "seat")
        return next(item.count for item in self.remaining if item.seat is seat)
    @property
    def opening_seat(self) -> Seat | None:
        """Derive the match's opening seat without overloading current lead state."""

        return self.play_history[0].seat if self.play_history else None
# Transitional public name for integration code that already imports GameSnapshot.
GameSnapshot = TrustedGameSnapshot
def _validate_correction_timeline(
    history: tuple[GameAction, ...],
    corrections: tuple[ConfirmedCorrection, ...],
    snapshot_version: StateVersion,
) -> None:
    if not history:
        if corrections:
            raise ValueError("correction history requires a confirmed action")
        return
    records = sorted(
        (*history, *corrections),
        key=lambda item: item.version_after.state_revision,
    )
    expected = records[0].version_before
    effective: dict[str, tuple[ActionKind, tuple[str, ...], tuple[tuple[str, ...], ...], str | None]] = {}
    for record in records:
        if record.version_before != expected:
            raise ValueError(
                "action and correction history must form one contiguous revision chain"
            )
        expected = record.version_after
        if isinstance(record, GameAction):
            first_correction = next(
                (
                    item
                    for item in corrections
                    if item.target_action_id == record.action_id
                ),
                None,
            )
            effective[record.action_id] = (
                record.kind if first_correction is None else first_correction.previous_kind,
                record.cards if first_correction is None else first_correction.previous_cards,
                (
                    record.suit_options
                    if first_correction is None
                    else first_correction.previous_suit_options
                ),
                None,
            )
            continue
        # A correction is appended at the current revision, but it may target
        # any earlier committed action. The reducer is rebuilt atomically and
        # the effective value below is then checked against the full history.
        # Requiring the target to be the latest action would reject the normal
        # delayed visual-reread case where a follower has already acted.
        previous = effective.get(record.target_action_id)
        if previous is None or previous[:3] != (
            record.previous_kind,
            record.previous_cards,
            record.previous_suit_options,
        ):
            raise ValueError("correction previous value does not match effective history")
        effective[record.target_action_id] = (
            record.corrected_kind,
            record.corrected_cards,
            record.corrected_suit_options,
            record.correction_id,
        )
    if expected != snapshot_version:
        raise ValueError("snapshot state_version must end at the latest action or correction")
    for action in history:
        if effective[action.action_id] != (
            action.kind,
            action.cards,
            action.suit_options,
            action.correction_id,
        ):
            raise ValueError("play_history does not expose the effective corrected action")
