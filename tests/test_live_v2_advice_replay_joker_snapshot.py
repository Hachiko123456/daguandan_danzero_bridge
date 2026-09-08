from __future__ import annotations

import json

from daguandan_bridge.infrastructure.live_v2_advice_worker import (
    replay_trusted_snapshot,
)
from daguandan_bridge.live_v2.identity import Seat
from daguandan_bridge.live_v2.reducer_snapshot import physical_action_entities

from test_live_v2_visual_joker_rule_boundary import (
    SESSION,
    MemoryStore,
    _close_runtime_workers,
    _runtime,
)


def _action_signatures(values) -> tuple[tuple[object, ...], ...]:
    signatures = []
    for item in values:
        kind = getattr(item, "kind", None)
        is_pass = getattr(item, "is_pass", None)
        signatures.append((
            getattr(getattr(item, "seat", None), "value", None)
            or getattr(item, "player", None),
            bool(is_pass) if is_pass is not None else kind.value == "pass",
            physical_action_entities(
                tuple(item.cards), tuple(getattr(item, "suit_options", ()))
            ),
        ))
    return tuple(signatures)


def test_advice_replay_preserves_truth_history_through_turn53_jokers() -> None:
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    runtime, rules = _runtime(
        MemoryStore("advice-replay-big-joker-turn53"), truth["initial_state"]
    )
    try:
        for turn in truth["turns"][:53]:
            runtime.commit_trusted_action(
                actor=str(turn["actor"]),
                cards=tuple(str(card) for card in turn["cards"]),
                is_pass=bool(turn["is_pass"]),
                monotonic_ms=int(turn["turn_id"]),
                confidence=1.0,
            )
        trusted = rules.snapshot(captured_ms=53)
        assert len(trusted.play_history) == 53
        assert trusted.play_history[46].cards == ("big_joker",)
        assert trusted.play_history[47].cards == ("4C", "4C", "4D", "4S")

        replayed = replay_trusted_snapshot(trusted)

        assert _action_signatures(replayed.play_history) == _action_signatures(
            trusted.play_history
        )
        assert _action_signatures(replayed.trick_plays) == _action_signatures(
            trusted.current_trick
        )
        assert replayed.current_player == trusted.current_seat.value
    finally:
        _close_runtime_workers(runtime)


def test_advice_replay_preserves_small_joker_history_and_current_trick() -> None:
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    initial = dict(truth["initial_state"])
    initial["lead_player"] = Seat.RIGHT.value
    runtime, rules = _runtime(MemoryStore("advice-replay-small-joker"), initial)
    try:
        runtime.commit_trusted_action(
            actor=Seat.RIGHT.value,
            cards=("small_joker",),
            is_pass=False,
            monotonic_ms=1,
            confidence=1.0,
        )
        trusted = rules.snapshot(captured_ms=1)

        replayed = replay_trusted_snapshot(trusted)

        assert _action_signatures(replayed.play_history) == _action_signatures(
            trusted.play_history
        )
        assert _action_signatures(replayed.trick_plays) == _action_signatures(
            trusted.current_trick
        )
        assert replayed.play_history[-1].cards == ("small_joker",)
        assert replayed.trick_plays[-1].cards == ("small_joker",)
    finally:
        _close_runtime_workers(runtime)
