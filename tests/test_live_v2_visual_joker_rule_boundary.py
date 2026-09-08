from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    CommitReason,
    FrameIdentity,
    ProjectionReason,
    Seat,
)


ROOT = Path(__file__).parents[1]
SESSION = (
    ROOT
    / "data/profiles/tencent_daguandan/sessions/game_20260814_004447_aab3dc"
)


class MemoryStore:
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.directory = Path(f"memory-{session_id}")

    def start(self, manifest): self.manifest = manifest
    def append_event(self, event): pass
    def append_event_batch(self, events): pass
    def append_advice(self, record): pass
    def append_observation(self, record): pass
    def append_recognition_trace(self, record): pass
    def update_runtime_identity(self, identity): pass
    def upsert_decision(self, record): pass
    def create_incident(self, **kwargs): return self.directory
    def append_incident_occurrence(self, *args, **kwargs): pass
    def seal(self, **kwargs): pass
    def append_post_seal_health_audit(self, report, **kwargs): pass
    def record_automatic_log_delivery(self, result): pass


class MemoryRecorder:
    frame_count = 0

    def write_frame(self, frame, monotonic_ms, wall_time): return None

    def close(self):
        return RecordingResult(Path("game.avi"), Path("frame_index.jsonl"), 0, 0)


class IdleWorker:
    def start(self, **kwargs): pass
    def close(self, **kwargs): pass
    def submit(self, *args, **kwargs): return ()
    def drain_results(self): return ()


def _runtime(store: MemoryStore, initial: dict[str, object]):
    store.start({"schema": "test.visual-joker-rule-boundary/1"})
    rules = ProductionRuleSession(store)
    runtime = LiveV2SessionRuntime(
        rule_session=rules,
        store=store,
        recorder=MemoryRecorder(),
        recognition_service=object(),
        vision_factory=lambda version: IdleWorker(),
        advice_runtime_factory=lambda version: IdleWorker(),
        local_hint_window_ms=0,
    )
    runtime.start(
        round_level=str(initial["round_level"]),
        hand=tuple(str(card) for card in initial["my_hand"]),
        lead_player=str(initial["lead_player"]),
        monotonic_ms=0,
    )
    runtime.bind_capture_generation(1)
    return runtime, rules


def _visual_candidate(rules, turn: dict[str, object], sequence: int) -> ActionCandidate:
    version = rules.version
    first = FrameIdentity(
        version.session_id, version.capture_generation, sequence,
        sequence * 10, "historical-visual", "truth-replay",
    )
    last = FrameIdentity(
        version.session_id, version.capture_generation, sequence + 1,
        sequence * 10 + 5, "historical-visual", "truth-replay",
    )
    cards = tuple(str(card) for card in turn["cards"])
    return ActionCandidate(
        candidate_id=f"visual-turn-{turn['turn_id']}",
        version=version,
        seat=Seat(str(turn["actor"])),
        kind=ActionKind.PLAY,
        cards=cards,
        suit_options=tuple((card,) for card in cards),
        evidence_ids=(f"visual-{sequence}", f"visual-{sequence + 1}"),
        action_epoch=int(turn["turn_id"]),
        first_frame=first,
        last_frame=last,
        processing_ms=last.captured_ms + 1,
        confidence=0.99,
        reason=CandidateReason.STABLE_PLAY,
    )


def _commit_visual(rules, candidate: ActionCandidate):
    binding = rules.bind_generation(candidate.version.capture_generation)
    projection = binding.adapter.project(
        base_version=binding.version,
        candidates=(candidate,),
        processing_ms=candidate.processing_ms,
    )
    assert projection.reason is ProjectionReason.ACCEPTED, (
        f"{candidate.cards} visual PLAY was rejected as {projection.reason.value}"
    )
    commit = binding.adapter.commit(
        expected_version=binding.version,
        actions=projection.actions,
    )
    assert commit.reason is CommitReason.COMMITTED, (
        f"{candidate.cards} VISUAL PLAY projection was accepted but cross-engine "
        f"snapshot matching rejected its commit as {commit.reason.value}"
    )
    return rules.snapshot(captured_ms=candidate.last_captured_ms)


def _close_runtime_workers(runtime: LiveV2SessionRuntime) -> None:
    runtime._close_detached(runtime._detach_workers())


def test_truth_turn47_big_joker_visual_play_and_turn48_bomb_both_commit() -> None:
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    runtime, rules = _runtime(
        MemoryStore("visual-big-joker-turn47"), truth["initial_state"]
    )
    try:
        for turn in truth["turns"][:46]:
            runtime.commit_trusted_action(
                actor=str(turn["actor"]),
                cards=tuple(str(card) for card in turn["cards"]),
                is_pass=bool(turn["is_pass"]),
                monotonic_ms=int(turn["turn_id"]),
                confidence=1.0,
            )
        assert len(rules.confirmed_actions) == 46
        assert rules.snapshot(captured_ms=46).current_seat is Seat.RIGHT

        turn47 = truth["turns"][46]
        after_joker = _commit_visual(
            rules, _visual_candidate(rules, turn47, 1_047)
        )
        assert len(after_joker.play_history) == 47
        assert after_joker.play_history[-1].kind is ActionKind.PLAY
        assert after_joker.play_history[-1].cards == ("big_joker",)
        assert after_joker.current_seat is Seat.OPPOSITE

        turn48 = truth["turns"][47]
        after_bomb = _commit_visual(
            rules, _visual_candidate(rules, turn48, 1_050)
        )
        assert len(after_bomb.play_history) == 48
        assert after_bomb.play_history[-1].kind is ActionKind.PLAY
        assert after_bomb.play_history[-1].cards == ("4C", "4C", "4D", "4S")
        assert after_bomb.current_seat is Seat.LEFT
    finally:
        _close_runtime_workers(runtime)


def test_small_joker_visual_play_matches_legacy_reducer_snapshot() -> None:
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    initial = dict(truth["initial_state"])
    initial["lead_player"] = Seat.RIGHT.value
    runtime, rules = _runtime(MemoryStore("visual-small-joker"), initial)
    try:
        turn = {
            "turn_id": 1,
            "actor": Seat.RIGHT.value,
            "cards": ["small_joker"],
        }
        snapshot = _commit_visual(
            rules, _visual_candidate(rules, turn, 2_001)
        )
        assert snapshot.play_history[-1].kind is ActionKind.PLAY
        assert snapshot.play_history[-1].cards == ("small_joker",)
        assert snapshot.current_seat is Seat.OPPOSITE
    finally:
        _close_runtime_workers(runtime)
