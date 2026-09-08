from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from daguandan_bridge.live_v2.game_state import (
    GameAction,
    SeatCardCount,
    TrustedGameSnapshot,
)
from daguandan_bridge.live_v2.types import (
    ActionKind,
    FrameIdentity,
    Seat,
    StateVersion,
    VersionIdentity,
)


HAND = tuple(f"CARD-{index}" for index in range(27))


def version(**changes: object) -> VersionIdentity:
    values = dict(
        session_id="session",
        capture_generation=3,
        state_revision=0,
        update_sequence=0,
        turn_index=0,
    )
    values.update(changes)
    return VersionIdentity(**values)  # type: ignore[arg-type]


def action(
    action_id: str = "a-1",
    *,
    seat: Seat = Seat.RIGHT,
    cards: tuple[str, ...] = ("3H",),
    kind: ActionKind = ActionKind.PLAY,
    before: StateVersion | None = None,
    capture_generation: int = 3,
    captured_ms: int = 110,
) -> GameAction:
    before = before or version().state_version
    after = replace(
        before,
        state_revision=before.state_revision + 1,
        turn_index=before.turn_index + 1,
    )
    first = FrameIdentity(
        before.session_id, capture_generation, 10, captured_ms - 10, "roi"
    )
    last = FrameIdentity(
        before.session_id, capture_generation, 11, captured_ms, "roi"
    )
    return GameAction(
        action_id=action_id,
        version_before=before,
        version_after=after,
        seat=seat,
        kind=kind,
        cards=cards,
        suit_options=(
            tuple((card,) for card in cards) if kind is ActionKind.PLAY else ()
        ),
        action_epoch=7,
        evidence_ids=(f"{action_id}-e1", f"{action_id}-e2"),
        first_frame=first,
        last_frame=last,
        confidence=0.95,
        captured_ms=captured_ms,
    )


def counts(**changes: int) -> tuple[SeatCardCount, ...]:
    return tuple(
        SeatCardCount(seat, changes.get(seat.value, 27))
        for seat in Seat
    )


def stream_version(
    state: StateVersion,
    *,
    capture_generation: int = 3,
    update_sequence: int = 9,
) -> VersionIdentity:
    return VersionIdentity.from_state(
        state,
        capture_generation=capture_generation,
        update_sequence=update_sequence,
    )


def snapshot(**changes: object) -> TrustedGameSnapshot:
    first = action()
    values = dict(
        version=stream_version(first.version_after),
        round_level="6",
        wild_rank="6",
        trick_index=1,
        current_seat=Seat.OPPOSITE,
        lead_seat=Seat.RIGHT,
        my_hand=HAND,
        play_history=(first,),
        current_trick=(first,),
        remaining=counts(right=26),
        finished=(),
        trusted=True,
        terminal=False,
        captured_ms=120,
    )
    values.update(changes)
    return TrustedGameSnapshot(**values)  # type: ignore[arg-type]


def test_snapshot_is_complete_immutable_and_keeps_physical_duplicates() -> None:
    duplicate = action(cards=("7H", "7H"))
    state = snapshot(
        version=stream_version(duplicate.version_after, update_sequence=4),
        play_history=(duplicate,),
        current_trick=(duplicate,),
        remaining=counts(right=25),
        captured_ms=duplicate.captured_ms,
    )
    assert state.play_history[0].cards == ("7H", "7H")
    assert state.play_history[0].suit_options == (("7H",), ("7H",))
    assert state.play_history[0].action_epoch == 7
    assert state.play_history[0].evidence_ids == ("a-1-e1", "a-1-e2")
    assert state.remaining_for(Seat.RIGHT) == 25
    with pytest.raises(FrozenInstanceError):
        state.trusted = False  # type: ignore[misc]


def test_lead_seat_tracks_current_trick_not_match_opening_seat() -> None:
    specs = (
        (Seat.RIGHT, ActionKind.PLAY, ("3H",)),
        (Seat.OPPOSITE, ActionKind.PLAY, ("4H",)),
        (Seat.LEFT, ActionKind.PASS, ()),
        (Seat.SELF, ActionKind.PASS, ()),
        (Seat.RIGHT, ActionKind.PASS, ()),
        (Seat.OPPOSITE, ActionKind.PLAY, ("5H",)),
    )
    history: list[GameAction] = []
    before = version().state_version
    for index, (seat, kind, cards) in enumerate(specs, start=1):
        item = action(
            f"two-tricks-{index}",
            seat=seat,
            kind=kind,
            cards=cards,
            before=before,
            captured_ms=100 + index * 10,
        )
        history.append(item)
        before = item.version_after
    state = TrustedGameSnapshot(
        version=stream_version(before, update_sequence=20),
        round_level="6",
        wild_rank="6",
        trick_index=2,
        current_seat=Seat.LEFT,
        lead_seat=Seat.OPPOSITE,
        my_hand=HAND,
        play_history=tuple(history),
        current_trick=(history[-1],),
        remaining=counts(right=26, opposite=25),
        finished=(),
        trusted=True,
        terminal=False,
        captured_ms=170,
    )
    assert state.opening_seat is Seat.RIGHT
    assert state.lead_seat is Seat.OPPOSITE
    assert state.trick_index == 2
    with pytest.raises(ValueError, match="first play of current_trick"):
        replace(state, lead_seat=Seat.RIGHT)


def test_empty_trick_binds_known_current_seat_as_next_leader() -> None:
    first = action()
    state = snapshot(
        version=stream_version(first.version_after),
        current_seat=Seat.RIGHT,
        lead_seat=Seat.RIGHT,
        current_trick=(),
        captured_ms=first.captured_ms,
    )
    assert state.lead_seat is state.current_seat
    with pytest.raises(ValueError, match="empty current trick"):
        replace(state, lead_seat=Seat.OPPOSITE)
    waiting = TrustedGameSnapshot(
        version=version(), round_level="6", wild_rank="6", trick_index=1,
        current_seat=None, lead_seat=None, my_hand=HAND,
        play_history=(), current_trick=(), remaining=counts(), finished=(),
        trusted=True, terminal=False, captured_ms=0,
    )
    assert waiting.current_seat is waiting.lead_seat is None


def test_trick_index_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        snapshot(trick_index=0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"remaining": counts(left=26)}, "play history and remaining"),
        ({"my_hand": HAND[:-1]}, "my_hand"),
        ({"remaining": (
            SeatCardCount(Seat.SELF, 27),
            SeatCardCount(Seat.RIGHT, 26),
            SeatCardCount(Seat.OPPOSITE, 27),
            SeatCardCount(Seat.OPPOSITE, 27),
        )}, "every seat exactly once"),
        ({"finished": (Seat.RIGHT,)}, "zero remaining"),
    ],
)
def test_snapshot_rejects_inconsistent_hand_history_and_counts(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        snapshot(**changes)


def test_current_trick_must_be_an_exact_history_suffix() -> None:
    first = action()
    second = action(
        "a-2", seat=Seat.OPPOSITE, cards=(), kind=ActionKind.PASS,
        before=first.version_after, captured_ms=130,
    )
    with pytest.raises(ValueError, match="exact suffix"):
        snapshot(
            version=stream_version(second.version_after, update_sequence=8),
            play_history=(first, second),
            current_trick=(first,),
            remaining=counts(right=26),
            captured_ms=140,
        )


def test_snapshot_rejects_foreign_or_non_contiguous_history_versions() -> None:
    first = action()
    foreign_before = StateVersion("other", 1, 1)
    foreign = action("foreign", before=foreign_before, captured_ms=130)
    with pytest.raises(ValueError, match="another snapshot session"):
        snapshot(
            version=version(state_revision=2, update_sequence=5, turn_index=2),
            play_history=(foreign,),
            current_trick=(foreign,),
            remaining=counts(right=26),
            captured_ms=140,
        )
    skipped = action("skipped", before=replace(first.version_after, state_revision=2),
                     captured_ms=130)
    with pytest.raises(ValueError, match="contiguous"):
        snapshot(
            version=stream_version(skipped.version_after, update_sequence=8),
            play_history=(first, skipped),
            current_trick=(first, skipped),
            remaining=counts(right=25),
            captured_ms=140,
        )


def test_reconnected_snapshot_accepts_contiguous_history_from_prior_generation() -> None:
    first = action(capture_generation=1)
    second = action(
        "a-2",
        seat=Seat.OPPOSITE,
        cards=("4H",),
        before=first.version_after,
        capture_generation=1,
        captured_ms=130,
    )
    rebound = TrustedGameSnapshot(
        version=stream_version(second.version_after, capture_generation=2),
        round_level="6",
        wild_rank="6",
        trick_index=1,
        current_seat=Seat.LEFT,
        lead_seat=Seat.RIGHT,
        my_hand=HAND,
        play_history=(first, second),
        current_trick=(first, second),
        remaining=counts(right=26, opposite=26),
        finished=(),
        trusted=True,
        terminal=False,
        captured_ms=140,
    )
    assert rebound.version.capture_generation == 2
    assert {item.first_frame.capture_generation for item in rebound.play_history} == {1}
    assert rebound.play_history[-1].version_after == rebound.version.state_version


def test_terminal_and_current_actor_must_be_consistent() -> None:
    with pytest.raises(ValueError, match="terminal snapshot cannot"):
        snapshot(terminal=True)
    all_cards = tuple(f"R-{index}" for index in range(27))
    finish = action(cards=all_cards)
    with pytest.raises(ValueError, match="finished seat cannot"):
        snapshot(
            version=stream_version(finish.version_after),
            current_seat=Seat.RIGHT,
            play_history=(finish,),
            current_trick=(finish,),
            remaining=counts(right=0),
            finished=(Seat.RIGHT,),
            captured_ms=finish.captured_ms,
        )
