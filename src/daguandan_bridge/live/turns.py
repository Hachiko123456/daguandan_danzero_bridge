from __future__ import annotations

from ..danzero.state import GameStateError, Seat


TURN_ORDER: tuple[Seat, ...] = ("self", "right", "opposite", "left")


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
