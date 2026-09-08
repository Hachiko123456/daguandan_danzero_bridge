"""Pure opening handshake for the deterministic live turn pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from .identity import FrameIdentity, Seat
from .turn_core import CommittedAction, PendingAction


class OpeningPhase(str, Enum):
    WAITING_TABLE = "waiting_table"
    CONFIRMING_HAND_LEVEL = "confirming_hand_level"
    CONFIRMING_LEAD = "confirming_lead"
    CONFIRMING_OPENING_PLAY = "confirming_opening_play"
    COMPLETED = "completed"


def _strictly_after(later: FrameIdentity, earlier: FrameIdentity) -> bool:
    return bool(
        later.session_id == earlier.session_id
        and later.capture_generation == earlier.capture_generation
        and later.roi_version == earlier.roi_version
        and later.source_id == earlier.source_id
        and later.frame_sequence > earlier.frame_sequence
        and later.captured_ms > earlier.captured_ms
    )


@dataclass(frozen=True, slots=True)
class HandLevelEvidence:
    frame: FrameIdentity
    hand: tuple[str, ...]
    round_level: str
    visual: bool = True

    def __post_init__(self) -> None:
        if len(self.hand) != 27:
            raise ValueError("opening hand must contain exactly 27 cards")
        if not self.round_level.strip():
            raise ValueError("round level must be non-empty")
        if not self.visual:
            raise ValueError("opening hand/level confirmation requires visual evidence")


@dataclass(frozen=True, slots=True)
class LeadEvidence:
    frame: FrameIdentity
    scores: tuple[tuple[Seat, float], ...]
    visual: bool = True

    def __post_init__(self) -> None:
        if not self.visual:
            raise ValueError("lead confirmation requires visual evidence")
        if {seat for seat, _ in self.scores} != set(Seat):
            raise ValueError("lead evidence must contain all four seats")
        if any(not 0.0 <= float(score) <= 1.0 for _, score in self.scores):
            raise ValueError("lead scores must be probabilities")

    def score_map(self) -> Mapping[Seat, float]:
        return dict(self.scores)

    def unique_winner(self, *, threshold: float, margin: float) -> Seat | None:
        ordered = sorted(self.scores, key=lambda item: item[1], reverse=True)
        winner, best = ordered[0]
        second = ordered[1][1]
        return winner if best >= threshold and best - second >= margin else None


@dataclass(frozen=True, slots=True)
class OpeningResult:
    hand: tuple[str, ...]
    round_level: str
    lead_seat: Seat
    opening_action: CommittedAction
    reset_observer_baselines: bool = True


@dataclass(frozen=True, slots=True)
class OpeningState:
    phase: OpeningPhase = OpeningPhase.WAITING_TABLE
    table_frame: FrameIdentity | None = None
    hand_level_votes: tuple[HandLevelEvidence, ...] = ()
    hand: tuple[str, ...] = ()
    round_level: str = ""
    lead_votes: tuple[LeadEvidence, ...] = ()
    lead_seat: Seat | None = None
    result: OpeningResult | None = None

    def table_stable(self, frame: FrameIdentity) -> OpeningState:
        if self.phase is not OpeningPhase.WAITING_TABLE:
            raise ValueError("table stability can only be confirmed once")
        return replace(
            self,
            phase=OpeningPhase.CONFIRMING_HAND_LEVEL,
            table_frame=frame,
        )

    def observe_hand_level(
        self, evidence: HandLevelEvidence, *, confirmations: int = 2
    ) -> OpeningState:
        if self.phase is not OpeningPhase.CONFIRMING_HAND_LEVEL:
            raise ValueError("hand/level evidence is not expected")
        if self.table_frame is None or not _strictly_after(evidence.frame, self.table_frame):
            raise ValueError("hand/level evidence must follow stable table evidence")
        previous = self.hand_level_votes[-1] if self.hand_level_votes else None
        if previous is not None and not _strictly_after(evidence.frame, previous.frame):
            raise ValueError("hand/level evidence frames must strictly increase")
        signature = (evidence.hand, evidence.round_level)
        votes = self.hand_level_votes
        if previous is None or (previous.hand, previous.round_level) != signature:
            votes = (evidence,)
        else:
            votes = (*votes, evidence)
        if len(votes) < max(2, confirmations):
            return replace(self, hand_level_votes=votes)
        return replace(
            self,
            phase=OpeningPhase.CONFIRMING_LEAD,
            hand_level_votes=votes,
            hand=evidence.hand,
            round_level=evidence.round_level,
        )

    def observe_lead(
        self,
        evidence: LeadEvidence,
        *,
        confirmations: int = 2,
        threshold: float = 0.80,
        margin: float = 0.10,
    ) -> OpeningState:
        if self.phase is not OpeningPhase.CONFIRMING_LEAD:
            raise ValueError("lead evidence is not expected")
        previous_frame = (
            self.lead_votes[-1].frame
            if self.lead_votes
            else self.hand_level_votes[-1].frame
        )
        if not _strictly_after(evidence.frame, previous_frame):
            raise ValueError("lead evidence frames must strictly increase")
        winner = evidence.unique_winner(threshold=threshold, margin=margin)
        if winner is None:
            return replace(self, lead_votes=())
        previous_winner = (
            self.lead_votes[-1].unique_winner(threshold=threshold, margin=margin)
            if self.lead_votes
            else None
        )
        votes = (*self.lead_votes, evidence) if previous_winner is winner else (evidence,)
        if len(votes) < max(2, confirmations):
            return replace(self, lead_votes=votes)
        return replace(
            self,
            phase=OpeningPhase.CONFIRMING_OPENING_PLAY,
            lead_votes=votes,
            lead_seat=winner,
        )

    def complete_opening(
        self, pending: PendingAction, *, action_id: str
    ) -> OpeningState:
        if self.phase is not OpeningPhase.CONFIRMING_OPENING_PLAY:
            raise ValueError("opening play is not expected")
        if self.lead_seat is None or pending.seat is not self.lead_seat:
            raise ValueError("opening play must belong to the visually confirmed lead")
        if pending.turn_token != 0:
            raise ValueError("opening play must use turn token zero")
        last_lead_frame = self.lead_votes[-1].frame
        if not _strictly_after(pending.first_frame, last_lead_frame):
            raise ValueError("opening play evidence must follow lead confirmation")
        action = CommittedAction.from_pending(
            pending,
            action_id=action_id,
            turn_index=0,
        )
        return replace(
            self,
            phase=OpeningPhase.COMPLETED,
            result=OpeningResult(
                self.hand,
                self.round_level,
                self.lead_seat,
                action,
            ),
        )


__all__ = [
    "HandLevelEvidence",
    "LeadEvidence",
    "OpeningPhase",
    "OpeningResult",
    "OpeningState",
]
