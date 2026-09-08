from __future__ import annotations

from daguandan_bridge.fabledan.advisor import FableDanAdvisor
from daguandan_bridge.infrastructure.live_v2_advice_worker import replay_trusted_snapshot
from daguandan_bridge.live_v2.events import ActionKind
from daguandan_bridge.live_v2.game_state import (
    GameAction,
    SeatCardCount,
    TrustedGameSnapshot,
)
from daguandan_bridge.live_v2.identity import (
    FrameIdentity,
    Seat,
    StateVersion,
    VersionIdentity,
)


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("3", "4", "5", "6", "7", "8", "9")
    for suit in "HDSC"
)[:27]


def _two_trick_snapshot() -> TrustedGameSnapshot:
    actions: list[GameAction] = []
    before = StateVersion("advice-session", 0, 0)
    specs = (
        (Seat.RIGHT, ActionKind.PLAY, ("3D",)),
        (Seat.OPPOSITE, ActionKind.PLAY, ("4D",)),
        (Seat.LEFT, ActionKind.PASS, ()),
        (Seat.SELF, ActionKind.PASS, ()),
        (Seat.RIGHT, ActionKind.PASS, ()),
        (Seat.OPPOSITE, ActionKind.PLAY, ("5D",)),
        (Seat.LEFT, ActionKind.PASS, ()),
    )
    for index, (seat, kind, cards) in enumerate(specs, start=1):
        after = StateVersion(
            before.session_id,
            before.state_revision + 1,
            before.turn_index + 1,
        )
        frame = FrameIdentity("advice-session", 1, index, 100 + index, "roi")
        actions.append(GameAction(
            f"two-tricks-{index}", before, after, seat, kind, cards,
            tuple((card,) for card in cards), index, (f"e-two-{index}",),
            frame, frame, 0.99, frame.captured_ms,
        ))
        before = after
    return TrustedGameSnapshot(
        version=VersionIdentity.from_state(
            before,
            capture_generation=2,
            update_sequence=len(actions),
        ),
        round_level="6",
        wild_rank="6",
        trick_index=2,
        current_seat=Seat.SELF,
        lead_seat=Seat.OPPOSITE,
        my_hand=HAND,
        play_history=tuple(actions),
        current_trick=tuple(actions[-2:]),
        remaining=tuple(
            SeatCardCount(
                seat,
                26 if seat is Seat.RIGHT else 25 if seat is Seat.OPPOSITE else 27,
            )
            for seat in Seat
        ),
        finished=(),
        trusted=True,
        terminal=False,
        captured_ms=110,
    )


def test_two_trick_replay_preserves_current_lead_for_fabledan(tmp_path) -> None:
    snapshot = _two_trick_snapshot()
    state = replay_trusted_snapshot(snapshot)
    assert snapshot.opening_seat is Seat.RIGHT
    assert snapshot.lead_seat is Seat.OPPOSITE
    assert snapshot.version.capture_generation == 2
    assert {item.first_frame.capture_generation for item in snapshot.play_history} == {1}
    assert snapshot.play_history[-1].version_after == snapshot.version.state_version
    assert state.lead_player == "opposite"
    assert state.current_player == "self"
    assert state.trick_plays[0].player == "opposite"

    advice = FableDanAdvisor(
        tmp_path,
        "profile",
        runtime_policy="rule_only",
        write_decision_log=False,
    ).recommend(state, request_id="second-trick-current-lead")
    assert advice.engine_input is not None
    projected = advice.engine_input["project_snapshot"]
    assert projected["lead_player"] == "opposite"
    assert projected["current_player"] == "self"
