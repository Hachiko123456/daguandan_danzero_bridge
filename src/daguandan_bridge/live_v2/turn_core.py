"""Small, deterministic turn core for the live game pipeline.

This module deliberately knows nothing about screenshots, OCR, GUI code, or
model workers.  It accepts one already-confirmed action at a time and lets a
rule port decide who acts next.  The important invariant is that visual code
can never manufacture a multi-action history: there is one current seat and
at most one pending action.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from .candidates import ActionKind
from .identity import (
    FrameIdentity,
    Seat,
    require_enum,
    require_instance,
    require_non_negative_int,
)


class TurnPhase(str, Enum):
    OPENING = "opening"
    WAIT_EXPECTED = "wait_expected"
    CONFIRMING_ACTION = "confirming_action"
    REPAIRING = "repairing"
    DESYNC = "desync"
    FINISHED = "finished"


class HistoryIntegrity(str, Enum):
    TRUSTED = "trusted"
    PARTIAL_SUITS = "partial_suits"
    DESYNC = "desync"
    UNTRUSTED = "untrusted"


def _same_stream(first: FrameIdentity, last: FrameIdentity) -> bool:
    return (
        first.session_id,
        first.capture_generation,
        first.roi_version,
        first.source_id,
    ) == (
        last.session_id,
        last.capture_generation,
        last.roi_version,
        last.source_id,
    )


def _strictly_after(later: FrameIdentity, earlier: FrameIdentity) -> bool:
    return (
        _same_stream(earlier, later)
        and later.frame_sequence > earlier.frame_sequence
        and later.captured_ms > earlier.captured_ms
    )


def _validate_cards(
    kind: ActionKind,
    cards: tuple[str, ...],
    suit_options: tuple[tuple[str, ...], ...],
) -> None:
    if kind is ActionKind.PASS:
        if cards or suit_options:
            raise ValueError("pass cannot contain cards or suit options")
        return
    if not cards:
        raise ValueError("play must contain cards")
    if len(cards) != len(suit_options):
        raise ValueError("suit_options must align with cards")
    if any(not isinstance(card, str) or not card.strip() for card in cards):
        raise ValueError("cards must contain non-empty text")
    for card, options in zip(cards, suit_options, strict=True):
        if not isinstance(options, tuple):
            raise TypeError("suit_options must contain tuples")
        if card in {"small_joker", "big_joker"}:
            if options:
                raise ValueError("joker suit options must be empty")
            continue
        if not options:
            raise ValueError("play suit options must not be empty")
        if len(set(options)) != len(options):
            raise ValueError("play suit options must be unique")
        if any(option not in {"S", "H", "C", "D"} for option in options):
            raise ValueError("play suit options must contain physical suits")


@dataclass(frozen=True, slots=True)
class TurnCursor:
    """The only authoritative pointer to the next actor."""

    trick_index: int
    lead_seat: Seat
    current_seat: Seat
    turn_token: int
    last_action_frame: FrameIdentity | None = None
    passes_in_trick: int = 0
    last_non_pass_seat: Seat | None = None

    def __post_init__(self) -> None:
        require_non_negative_int(self.trick_index, "trick_index")
        require_enum(self.lead_seat, Seat, "lead_seat")
        require_enum(self.current_seat, Seat, "current_seat")
        require_non_negative_int(self.turn_token, "turn_token")
        require_non_negative_int(self.passes_in_trick, "passes_in_trick")
        if self.passes_in_trick > 3:
            raise ValueError("passes_in_trick cannot exceed three")
        if self.last_non_pass_seat is not None:
            require_enum(self.last_non_pass_seat, Seat, "last_non_pass_seat")
        if self.last_action_frame is not None:
            require_instance(self.last_action_frame, FrameIdentity, "last_action_frame")


@dataclass(frozen=True, slots=True)
class PendingAction:
    """One expected-seat action being confirmed from independent evidence."""

    seat: Seat
    turn_token: int
    kind: ActionKind
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    first_frame: FrameIdentity
    last_frame: FrameIdentity
    confidence: float = 1.0

    def __post_init__(self) -> None:
        require_enum(self.seat, Seat, "seat")
        require_non_negative_int(self.turn_token, "turn_token")
        require_enum(self.kind, ActionKind, "kind")
        _validate_cards(self.kind, self.cards, self.suit_options)
        if not _strictly_after(self.last_frame, self.first_frame):
            raise ValueError("pending evidence frames must strictly increase")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")

    @property
    def partial_suits(self) -> bool:
        return self.kind is ActionKind.PLAY and any(
            len(options) != 1 for options in self.suit_options
        )

    def with_later_evidence(
        self,
        *,
        last_frame: FrameIdentity,
        suit_options: tuple[tuple[str, ...], ...] | None = None,
        confidence: float | None = None,
    ) -> PendingAction:
        """Extend confirmation without creating a second action."""

        return replace(
            self,
            last_frame=last_frame,
            suit_options=self.suit_options if suit_options is None else suit_options,
            confidence=self.confidence if confidence is None else confidence,
        )


@dataclass(frozen=True, slots=True)
class CommittedAction:
    """A single formal action in the immutable turn history."""

    action_id: str
    turn_index: int
    seat: Seat
    turn_token: int
    kind: ActionKind
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    first_frame: FrameIdentity
    last_frame: FrameIdentity
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("action_id must be non-empty text")
        require_non_negative_int(self.turn_index, "turn_index")
        require_enum(self.seat, Seat, "seat")
        require_non_negative_int(self.turn_token, "turn_token")
        require_enum(self.kind, ActionKind, "kind")
        _validate_cards(self.kind, self.cards, self.suit_options)
        if not _strictly_after(self.last_frame, self.first_frame):
            raise ValueError("committed evidence frames must strictly increase")
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")

    @classmethod
    def from_pending(
        cls, pending: PendingAction, *, action_id: str, turn_index: int
    ) -> CommittedAction:
        return cls(
            action_id=action_id,
            turn_index=turn_index,
            seat=pending.seat,
            turn_token=pending.turn_token,
            kind=pending.kind,
            cards=pending.cards,
            suit_options=pending.suit_options,
            first_frame=pending.first_frame,
            last_frame=pending.last_frame,
            confidence=pending.confidence,
        )

    @property
    def partial_suits(self) -> bool:
        return self.kind is ActionKind.PLAY and any(
            len(options) != 1 for options in self.suit_options
        )


@dataclass(frozen=True, slots=True)
class ActionRepairEvent:
    """A later suit observation that refines one existing action ID."""

    repair_id: str
    target_action_id: str
    previous_suit_options: tuple[tuple[str, ...], ...]
    repaired_suit_options: tuple[tuple[str, ...], ...]
    evidence_frame: FrameIdentity
    repaired_ms: int

    def __post_init__(self) -> None:
        for name, value in (
            ("repair_id", self.repair_id),
            ("target_action_id", self.target_action_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if len(self.previous_suit_options) != len(self.repaired_suit_options):
            raise ValueError("repair suit options must preserve card count")
        require_instance(self.evidence_frame, FrameIdentity, "evidence_frame")
        require_non_negative_int(self.repaired_ms, "repaired_ms")
        if self.repaired_ms < self.evidence_frame.captured_ms:
            raise ValueError("repair time cannot precede repair evidence")


@dataclass(frozen=True, slots=True)
class RuleAdvance:
    """The rule port's complete answer after one committed action."""

    next_seat: Seat | None
    lead_seat: Seat | None
    trick_index: int
    passes_in_trick: int = 0
    last_non_pass_seat: Seat | None = None
    finished: tuple[Seat, ...] = ()
    terminal: bool = False
    wind_catch: Seat | None = None

    def __post_init__(self) -> None:
        if self.next_seat is not None:
            require_enum(self.next_seat, Seat, "next_seat")
        if self.lead_seat is not None:
            require_enum(self.lead_seat, Seat, "lead_seat")
        require_non_negative_int(self.trick_index, "trick_index")
        require_non_negative_int(self.passes_in_trick, "passes_in_trick")
        if self.passes_in_trick > 3:
            raise ValueError("passes_in_trick cannot exceed three")
        if len(set(self.finished)) != len(self.finished):
            raise ValueError("finished seats must be unique")
        for seat in self.finished:
            require_enum(seat, Seat, "finished seat")
        if self.wind_catch is not None:
            require_enum(self.wind_catch, Seat, "wind_catch")
        if self.terminal and self.next_seat is not None:
            raise ValueError("terminal advance cannot have a next seat")
        if self.next_seat is not None and self.next_seat in self.finished:
            raise ValueError("next seat cannot already be finished")


class TurnRules(Protocol):
    def after_opening(self, action: CommittedAction) -> RuleAdvance: ...

    def after_action(
        self,
        action: CommittedAction,
        history: tuple[CommittedAction, ...],
        cursor: TurnCursor,
    ) -> RuleAdvance: ...


@dataclass(frozen=True, slots=True)
class TurnState:
    """Immutable state and transition facade for the minimal turn core."""

    phase: TurnPhase
    cursor: TurnCursor | None
    committed: tuple[CommittedAction, ...] = ()
    pending: PendingAction | None = None
    repairs: tuple[ActionRepairEvent, ...] = ()
    integrity: HistoryIntegrity = HistoryIntegrity.TRUSTED
    opening_seat: Seat | None = None
    finished: tuple[Seat, ...] = ()

    @classmethod
    def opening(cls) -> TurnState:
        return cls(TurnPhase.OPENING, None)

    def __post_init__(self) -> None:
        require_enum(self.phase, TurnPhase, "phase")
        require_enum(self.integrity, HistoryIntegrity, "integrity")
        if self.opening_seat is not None:
            require_enum(self.opening_seat, Seat, "opening_seat")
        if len(set(self.finished)) != len(self.finished):
            raise ValueError("finished seats must be unique")
        if self.cursor is not None and self.cursor.current_seat in self.finished:
            raise ValueError("finished seat cannot be current actor")
        if self.pending is not None:
            if self.cursor is None or self.phase is not TurnPhase.CONFIRMING_ACTION:
                raise ValueError("pending action requires confirming phase and cursor")
            if self.pending.seat is not self.cursor.current_seat:
                raise ValueError("pending action must belong to current seat")
            if self.pending.turn_token != self.cursor.turn_token:
                raise ValueError("pending action must belong to current turn token")
        if self.phase is TurnPhase.FINISHED and self.cursor is not None:
            raise ValueError("finished state cannot have cursor")
        if self.phase is TurnPhase.OPENING and (self.cursor is not None or self.committed):
            raise ValueError("opening barrier must precede formal history")
        if self.phase in {
            TurnPhase.WAIT_EXPECTED,
            TurnPhase.CONFIRMING_ACTION,
            TurnPhase.REPAIRING,
        } and self.cursor is None:
            raise ValueError("active turn phases require a cursor")
        if self.phase is TurnPhase.DESYNC and self.integrity is not HistoryIntegrity.DESYNC:
            raise ValueError("desync phase requires desync integrity")
        if any(item.turn_index != index for index, item in enumerate(self.committed)):
            raise ValueError("committed actions must have contiguous turn indexes")
        for previous, current in zip(self.committed, self.committed[1:], strict=False):
            if not _strictly_after(current.first_frame, previous.last_frame):
                raise ValueError("committed action evidence must be strictly ordered")

    @property
    def advice_allowed(self) -> bool:
        return bool(
            self.phase is TurnPhase.WAIT_EXPECTED
            and self.cursor is not None
            and self.cursor.current_seat is Seat.SELF
            and self.integrity is HistoryIntegrity.TRUSTED
            and self.pending is None
        )

    def confirm_opening(
        self, action: CommittedAction, *, rules: TurnRules
    ) -> TurnState:
        if self.phase is not TurnPhase.OPENING:
            raise ValueError("opening can only be confirmed once")
        if action.kind is not ActionKind.PLAY:
            raise ValueError("opening action must be a play")
        if action.turn_index != 0 or action.turn_token != 0:
            raise ValueError("opening action must be the first formal turn")
        advance = rules.after_opening(action)
        if advance.next_seat is None or advance.lead_seat is None:
            raise ValueError("opening rule advance must identify next and lead seats")
        cursor = TurnCursor(
            trick_index=advance.trick_index,
            lead_seat=advance.lead_seat,
            current_seat=advance.next_seat,
            turn_token=1,
            last_action_frame=action.last_frame,
            passes_in_trick=advance.passes_in_trick,
            last_non_pass_seat=advance.last_non_pass_seat or action.seat,
        )
        return TurnState(
            phase=TurnPhase.WAIT_EXPECTED,
            cursor=cursor,
            committed=(action,),
            integrity=HistoryIntegrity.PARTIAL_SUITS
            if action.partial_suits
            else HistoryIntegrity.TRUSTED,
            opening_seat=action.seat,
            finished=advance.finished,
        )

    def begin_action(self, pending: PendingAction) -> TurnState:
        if self.phase is not TurnPhase.WAIT_EXPECTED or self.cursor is None:
            raise ValueError("actions can only begin while waiting for expected seat")
        if pending.seat is not self.cursor.current_seat:
            raise ValueError("foreign seat cannot become pending")
        if self.cursor.last_action_frame is not None and not _strictly_after(
            pending.first_frame, self.cursor.last_action_frame
        ):
            raise ValueError("pending evidence must follow last formal action")
        return replace(self, phase=TurnPhase.CONFIRMING_ACTION, pending=pending)

    def commit_pending(
        self, *, action_id: str, rules: TurnRules
    ) -> tuple[TurnState, CommittedAction]:
        if self.phase is not TurnPhase.CONFIRMING_ACTION or self.cursor is None:
            raise ValueError("no action is pending")
        if self.pending is None:
            raise ValueError("no action is pending")
        action = CommittedAction.from_pending(
            self.pending, action_id=action_id, turn_index=len(self.committed)
        )
        if self.cursor.last_action_frame is not None and not _strictly_after(
            action.first_frame, self.cursor.last_action_frame
        ):
            raise ValueError("formal evidence must strictly follow previous action")
        advance = rules.after_action(action, self.committed, self.cursor)
        if advance.terminal:
            next_state = replace(
                self,
                phase=TurnPhase.FINISHED,
                cursor=None,
                committed=(*self.committed, action),
                pending=None,
                integrity=(
                    HistoryIntegrity.PARTIAL_SUITS
                    if action.partial_suits
                    else self.integrity
                ),
                finished=advance.finished,
            )
        else:
            if advance.next_seat is None or advance.lead_seat is None:
                raise ValueError("non-terminal advance must identify next and lead seats")
            finished = advance.finished
            if advance.next_seat in finished:
                raise ValueError("rule port returned a finished next seat")
            next_state = replace(
                self,
                phase=TurnPhase.WAIT_EXPECTED,
                cursor=TurnCursor(
                    trick_index=advance.trick_index,
                    lead_seat=advance.lead_seat,
                    current_seat=advance.next_seat,
                    turn_token=self.cursor.turn_token + 1,
                    last_action_frame=action.last_frame,
                    passes_in_trick=advance.passes_in_trick,
                    last_non_pass_seat=advance.last_non_pass_seat
                    or self.cursor.last_non_pass_seat,
                ),
                committed=(*self.committed, action),
                pending=None,
                integrity=(
                    HistoryIntegrity.PARTIAL_SUITS
                    if action.partial_suits
                    else self.integrity
                ),
                finished=finished,
            )
        return next_state, action

    def repair_action(
        self,
        *,
        repair_id: str,
        target_action_id: str,
        suit_options: tuple[tuple[str, ...], ...],
        evidence_frame: FrameIdentity,
        repaired_ms: int,
    ) -> tuple[TurnState, ActionRepairEvent]:
        if self.phase not in {
            TurnPhase.WAIT_EXPECTED,
            TurnPhase.REPAIRING,
            TurnPhase.FINISHED,
        }:
            raise ValueError("repairs are unavailable during opening or desync")
        index = next(
            (index for index, item in enumerate(self.committed)
             if item.action_id == target_action_id),
            None,
        )
        if index is None:
            raise ValueError("repair target action is absent")
        target = self.committed[index]
        if not target.partial_suits:
            raise ValueError("repair target does not contain partial suits")
        if not _strictly_after(evidence_frame, target.last_frame):
            raise ValueError("repair evidence must follow target action evidence")
        if self.cursor is not None and self.cursor.last_action_frame is not None:
            if not _strictly_after(evidence_frame, self.cursor.last_action_frame):
                raise ValueError("repair evidence must follow current formal watermark")
        _validate_cards(target.kind, target.cards, suit_options)
        for previous, repaired in zip(
            target.suit_options, suit_options, strict=True
        ):
            if not set(repaired).issubset(previous):
                raise ValueError("repair may only narrow previous suit options")
        event = ActionRepairEvent(
            repair_id=repair_id,
            target_action_id=target_action_id,
            previous_suit_options=target.suit_options,
            repaired_suit_options=suit_options,
            evidence_frame=evidence_frame,
            repaired_ms=repaired_ms,
        )
        repaired = replace(target, suit_options=suit_options)
        committed = (*self.committed[:index], repaired, *self.committed[index + 1 :])
        integrity = (
            HistoryIntegrity.PARTIAL_SUITS
            if any(item.partial_suits for item in committed)
            else HistoryIntegrity.TRUSTED
        )
        return replace(
            self,
            committed=committed,
            repairs=(*self.repairs, event),
            integrity=integrity,
        ), event

    def mark_desync(self) -> TurnState:
        return replace(
            self,
            phase=TurnPhase.DESYNC,
            pending=None,
            integrity=HistoryIntegrity.DESYNC,
        )

    def resync(self, *, current_seat: Seat, lead_seat: Seat | None = None) -> TurnState:
        if self.phase is not TurnPhase.DESYNC:
            raise ValueError("resync requires desync state")
        if current_seat in self.finished:
            raise ValueError("cannot resync to a finished seat")
        lead = current_seat if lead_seat is None else lead_seat
        require_enum(lead, Seat, "lead_seat")
        cursor = TurnCursor(
            trick_index=(self.cursor.trick_index if self.cursor else 0),
            lead_seat=lead,
            current_seat=current_seat,
            turn_token=(self.cursor.turn_token + 1 if self.cursor else 1),
            last_action_frame=(self.cursor.last_action_frame if self.cursor else None),
            last_non_pass_seat=(
                self.cursor.last_non_pass_seat if self.cursor else None
            ),
        )
        return replace(
            self,
            phase=TurnPhase.WAIT_EXPECTED,
            cursor=cursor,
            integrity=HistoryIntegrity.UNTRUSTED,
        )

    def confirm_resync_integrity(self) -> TurnState:
        """Make a manually resynchronised cursor advisory-safe after audit."""

        if self.phase is not TurnPhase.WAIT_EXPECTED or self.cursor is None:
            raise ValueError("resync integrity confirmation requires an active cursor")
        if self.integrity is not HistoryIntegrity.UNTRUSTED:
            raise ValueError("only an untrusted resynchronised state can be confirmed")
        integrity = (
            HistoryIntegrity.PARTIAL_SUITS
            if any(item.partial_suits for item in self.committed)
            else HistoryIntegrity.TRUSTED
        )
        return replace(self, integrity=integrity)


class SimpleSeatRules:
    """Reference four-seat rule port used by unit tests and smoke checks.

    Production rule adapters can implement :class:`TurnRules` when they need
    team completion or wind-catch details.  This port covers the deterministic
    seat rotation and the three-pass return to the last non-pass leader.
    """

    _ORDER = (Seat.SELF, Seat.RIGHT, Seat.OPPOSITE, Seat.LEFT)

    def _next(self, seat: Seat, finished: tuple[Seat, ...]) -> Seat:
        for offset in range(1, len(self._ORDER) + 1):
            candidate = self._ORDER[(self._ORDER.index(seat) + offset) % 4]
            if candidate not in finished:
                return candidate
        raise ValueError("all seats are finished")

    def after_opening(self, action: CommittedAction) -> RuleAdvance:
        return RuleAdvance(
            next_seat=self._next(action.seat, ()),
            lead_seat=action.seat,
            trick_index=0,
            last_non_pass_seat=action.seat,
        )

    def after_action(
        self,
        action: CommittedAction,
        history: tuple[CommittedAction, ...],
        cursor: TurnCursor,
    ) -> RuleAdvance:
        finished: tuple[Seat, ...] = ()
        if action.kind is ActionKind.PASS:
            passes = cursor.passes_in_trick + 1
            leader = cursor.last_non_pass_seat or cursor.lead_seat
            if passes >= 3:
                return RuleAdvance(
                    next_seat=leader,
                    lead_seat=leader,
                    trick_index=cursor.trick_index + 1,
                    last_non_pass_seat=leader,
                    finished=finished,
                )
            return RuleAdvance(
                next_seat=self._next(action.seat, finished),
                lead_seat=cursor.lead_seat,
                trick_index=cursor.trick_index,
                passes_in_trick=passes,
                last_non_pass_seat=leader,
                finished=finished,
            )
        return RuleAdvance(
            next_seat=self._next(action.seat, finished),
            lead_seat=cursor.lead_seat,
            trick_index=cursor.trick_index,
            last_non_pass_seat=action.seat,
            finished=finished,
        )


__all__ = [
    "ActionRepairEvent",
    "CommittedAction",
    "HistoryIntegrity",
    "PendingAction",
    "RuleAdvance",
    "SimpleSeatRules",
    "TurnCoreRules",
    "TurnCursor",
    "TurnPhase",
    "TurnRules",
    "TurnState",
]


TurnCoreRules = TurnRules
