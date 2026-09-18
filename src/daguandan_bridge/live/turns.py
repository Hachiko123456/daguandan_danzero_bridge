from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ..danzero.state import GameStateError, Seat


TURN_ORDER: tuple[Seat, ...] = ("self", "right", "opposite", "left")

# 接风时，出完牌一方把下一墩的领出权交给正对面的队友。
PARTNER_SEAT: dict[Seat, Seat] = {
    "self": "opposite",
    "opposite": "self",
    "right": "left",
    "left": "right",
}

class WindCatchPolicy(str, Enum):
    """How a finished leader hands the next trick to their partner."""

    AUTO_HANDOFF_TO_PARTNER = "auto_handoff_to_partner"
    PARTNER_MUST_EXPLICITLY_PASS = "partner_must_explicitly_pass"
    LEGACY_SKIP_RECEIVER = "legacy_skip_receiver"


TEAMS: tuple[frozenset[Seat], ...] = (
    frozenset({"self", "opposite"}),
    frozenset({"right", "left"}),
)


def round_is_decided(finished: Iterable[Seat]) -> bool:
    """Return whether the finish order already ends the current round."""

    finished_seats = frozenset(finished)
    return len(finished_seats) >= 3 or any(
        team.issubset(finished_seats) for team in TEAMS
    )


def next_active_seat(current: Seat, finished: frozenset[Seat]) -> Seat:
    """Return the next counter-clockwise seat that is still playing."""

    if current not in TURN_ORDER:
        raise GameStateError("当前座位不在逆时针座位表中")
    start = TURN_ORDER.index(current)
    for offset in range(1, len(TURN_ORDER) + 1):
        seat = TURN_ORDER[(start + offset) % len(TURN_ORDER)]
        if seat not in finished:
            return seat
    raise GameStateError("没有仍在对局中的下一位玩家")


@dataclass(frozen=True)
class TrickTurnProjection:
    """The single source of truth for a trick's response and wind-catch turn.

    ``passed`` must contain only PASS actions after the current trick leader's
    most recent play.  A finished leader's active partner is a *future* trick
    leader, but remains an ordinary respondent to that finishing play.  Every
    active seat must actually PASS before the next trick begins; no synthetic
    PASS is inferred for the future leader.
    """

    leader: Seat
    finished: frozenset[Seat]
    passed: frozenset[Seat]
    wind_receiver: Seat | None
    next_leader: Seat | None
    required_passers: frozenset[Seat]
    wind_receiver_must_pass: bool = False
    skip_wind_receiver_while_incomplete: bool = False

    @property
    def is_complete(self) -> bool:
        return bool(self.required_passers) and self.required_passers.issubset(
            self.passed
        )

    def expected_after(self, actor: Seat) -> Seat | None:
        """Return the actor after ``actor`` has completed this trick action."""

        if actor not in TURN_ORDER:
            raise GameStateError("当前座位不在逆时针座位表中")
        if self.is_complete:
            return self.next_leader
        try:
            next_player = next_active_seat(actor, self.finished)
        except GameStateError:
            return None
        # The future wind receiver still takes its natural turn when another
        # opponent later in seat order has not responded yet.  Historical
        # pre-policy timelines may explicitly request the former skip behavior.
        if self.skip_wind_receiver_while_incomplete and next_player == self.wind_receiver:
            try:
                return next_active_seat(next_player, self.finished)
            except GameStateError:
                return self.next_leader
        return next_player

    def closes_if(self, player: Seat) -> bool:
        """Return whether a PASS from ``player`` would close this trick."""

        return bool(self.required_passers) and self.required_passers.issubset(
            self.passed | {player}
        )


def project_trick_turn(
    leader: Seat,
    finished: Iterable[Seat],
    passed: Iterable[Seat] = (),
    *,
    wind_catch_policy: WindCatchPolicy = WindCatchPolicy.PARTNER_MUST_EXPLICITLY_PASS,
    wind_receiver_must_pass: bool | None = None,
) -> TrickTurnProjection:
    """Project one trick's turn ownership, including the full 接风 rule.

    Callers own their card/history state and pass in only the leader, finished
    seats and responders that have already PASSed.  They must not separately
    implement partner handoff or response counts.  In a wind catch, the
    receiver takes the next trick only after every active seat has responded.
    Tencent automatically hands the next lead to the finished player's partner;
    that partner does not need to emit an artificial PASS.  The explicit-pass
    policy remains available only for replaying older timelines that recorded it.
    """

    if leader not in TURN_ORDER:
        raise GameStateError("当前墩领出座位不在逆时针座位表中")
    policy = WindCatchPolicy(wind_catch_policy)
    if wind_receiver_must_pass is not None:
        # Preserve the legacy bool API: False meant unconditionally skipping
        # the future receiver, while True required its explicit PASS.
        policy = (
            WindCatchPolicy.PARTNER_MUST_EXPLICITLY_PASS
            if wind_receiver_must_pass
            else WindCatchPolicy.LEGACY_SKIP_RECEIVER
        )
    require_receiver_pass = (
        policy is WindCatchPolicy.PARTNER_MUST_EXPLICITLY_PASS
    )
    legacy_skip_receiver = policy is WindCatchPolicy.LEGACY_SKIP_RECEIVER
    finished_seats = frozenset(finished)
    passed_seats = frozenset(passed)
    active = frozenset(seat for seat in TURN_ORDER if seat not in finished_seats)

    if round_is_decided(finished_seats):
        return TrickTurnProjection(
            leader=leader,
            finished=finished_seats,
            passed=passed_seats,
            wind_receiver=None,
            next_leader=None,
            required_passers=frozenset(),
        )

    wind_receiver: Seat | None = None
    if leader in finished_seats:
        partner = PARTNER_SEAT[leader]
        if partner in active:
            wind_receiver = partner

    if wind_receiver is not None:
        next_leader: Seat | None = wind_receiver
    elif leader in active:
        next_leader = leader
    elif not active:
        next_leader = None
    else:
        try:
            next_leader = next_active_seat(leader, finished_seats)
        except GameStateError:
            next_leader = None

    required_passers = (
        active
        if wind_receiver is not None and require_receiver_pass
        else active - {wind_receiver}
        if wind_receiver is not None
        else active - {next_leader}
        if next_leader is not None
        else frozenset()
    )
    return TrickTurnProjection(
        leader=leader,
        finished=finished_seats,
        passed=passed_seats,
        wind_receiver=wind_receiver,
        next_leader=next_leader,
        required_passers=required_passers,
        wind_receiver_must_pass=require_receiver_pass,
        skip_wind_receiver_while_incomplete=legacy_skip_receiver,
    )
