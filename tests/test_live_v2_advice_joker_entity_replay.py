from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.application.live_v2_runtime_updates import trusted_candidate
from daguandan_bridge.infrastructure.live_v2_advice_worker import (
    replay_trusted_snapshot,
)
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.types import (
    CandidateReason,
    CommitReason,
    EvidenceOrigin,
    ProjectionReason,
    Seat,
)


ROOT = Path(__file__).parents[1]
SESSION = (
    ROOT
    / "data/profiles/tencent_daguandan/sessions/game_20260814_004447_aab3dc"
)


class _MemoryStore:
    persistence_enabled = True
    automatic_log_delivery_enabled = False
    directory = Path("memory-advice-joker-entity-replay")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def append_event_batch(self, events) -> None:
        del events


def _commit(
    session: ProductionRuleSession,
    *,
    seat: Seat,
    cards: tuple[str, ...],
    is_pass: bool,
    captured_ms: int,
) -> None:
    binding = session.bind_generation(session.version.capture_generation)
    candidate = trusted_candidate(
        version=binding.version,
        seat=seat,
        cards=cards,
        is_pass=is_pass,
        captured_ms=captured_ms,
        processing_ms=captured_ms,
        confidence=1.0,
        origin=EvidenceOrigin.TRUSTED,
        reason=CandidateReason.LOCAL_ACTION_CONFIRMED,
        evidence_refs=(f"trusted-{captured_ms}",),
        sequence=captured_ms,
    )
    projection = binding.adapter.project(
        base_version=binding.version,
        candidates=(candidate,),
        processing_ms=captured_ms,
    )
    assert projection.reason is ProjectionReason.ACCEPTED
    commit = binding.adapter.commit(
        expected_version=binding.version,
        actions=projection.actions,
    )
    assert commit.reason is CommitReason.COMMITTED


def test_turn54_snapshot_with_big_joker_history_replays_for_advice() -> None:
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    initial = truth["initial_state"]
    session = ProductionRuleSession(_MemoryStore("turn54-big-joker-replay"))
    session.initialize(
        round_level=str(initial["round_level"]),
        hand=tuple(str(card) for card in initial["my_hand"]),
        lead_player=Seat(str(initial["lead_player"])),
        monotonic_ms=0,
        capture_generation=1,
    )
    for turn in truth["turns"][:53]:
        _commit(
            session,
            seat=Seat(str(turn["actor"])),
            cards=tuple(str(card) for card in turn["cards"]),
            is_pass=bool(turn["is_pass"]),
            captured_ms=int(turn["turn_id"]),
        )

    snapshot = session.snapshot(captured_ms=53)

    assert snapshot.play_history[46].cards == ("big_joker",)
    assert replay_trusted_snapshot(snapshot).current_player == "self"


def test_small_joker_history_replays_for_advice() -> None:
    hand = tuple(f"{rank}{suit}" for suit in "SH" for rank in (
        "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
    )) + ("3C",)
    session = ProductionRuleSession(_MemoryStore("small-joker-replay"))
    session.initialize(
        round_level="6",
        hand=hand,
        lead_player=Seat.RIGHT,
        monotonic_ms=0,
        capture_generation=1,
    )
    _commit(
        session,
        seat=Seat.RIGHT,
        cards=("small_joker",),
        is_pass=False,
        captured_ms=1,
    )

    snapshot = session.snapshot(captured_ms=1)

    assert replay_trusted_snapshot(snapshot).play_history[0].cards == (
        "small_joker",
    )
