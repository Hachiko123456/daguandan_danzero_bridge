from __future__ import annotations

from daguandan_bridge.application.replay_turn_draft import ReplayTurnDraftAssembler
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _log(turns=()):
    return TruthLog("game", TruthInitialState("2", "left", HAND), tuple(turns))


def test_streaming_assembler_appends_each_confirmed_turn_in_memory():
    assembler = ReplayTurnDraftAssembler(_log())

    first = assembler.append(
        {
            "turn_id": 8,
            "trick_id": 1,
            "frame_index": 140,
            "actor": "left",
            "recognized_pass": False,
            "recognized_cards": ["3D"],
        }
    )
    second = assembler.append(
        {
            "turn_id": 9,
            "trick_id": 1,
            "frame_index": 155,
            "actor": "self",
            "recognized_pass": True,
            "recognized_cards": [],
        }
    )

    assert first.accepted and second.accepted
    assert first.status == "扫描确认"
    assert [(turn.index, turn.actor, turn.frame_index) for turn in second.truth_log.turns] == [
        (1, "left", 140),
        (2, "self", 155),
    ]
    assert not assembler.append({"turn_id": 9, "actor": "self", "is_pass": True}).accepted
    malformed = assembler.append(
        {"turn_id": "bad", "frame_index": "bad", "actor": "self", "is_pass": True}
    )
    assert not malformed.accepted
    assert malformed.truth_log == second.truth_log
