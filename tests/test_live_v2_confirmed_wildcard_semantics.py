from __future__ import annotations

import json
from pathlib import Path

import pytest

from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.fabledan import FableDanAdvisor, FableDanStateError
from daguandan_bridge.infrastructure.live_v2_advice_worker import (
    replay_trusted_snapshot,
)
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.identity import Seat


ROOT = Path(__file__).parents[1]
SESSION = (
    ROOT
    / "data/profiles/tencent_daguandan/sessions/game_20260814_004447_aab3dc"
)
AMBIGUOUS_PHYSICAL_CARDS = ("10C", "6C", "6H", "7C", "9C")


class MemoryStore:
    session_id = "confirmed-wildcard-semantics"
    directory = Path("memory-confirmed-wildcard-semantics")
    persistence_enabled = True
    automatic_log_delivery_enabled = False

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


class IdleVision:
    def start(self, **kwargs): pass
    def close(self, **kwargs): pass
    def submit(self, *args, **kwargs): return ()
    def drain_results(self): return ()


class IdleAdvice:
    def start(self, **kwargs): pass
    def close(self, **kwargs): pass
    def submit(self, *args, **kwargs): return ()
    def drain_results(self): return ()


def test_confirmed_turn23_semantics_reach_fabledan_at_turn34(tmp_path: Path) -> None:
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    initial = truth["initial_state"]
    store = MemoryStore()
    store.start({"schema": "test.confirmed-wildcard-semantics/1"})
    rules = ProductionRuleSession(store)
    runtime = LiveV2SessionRuntime(
        rule_session=rules,
        store=store,
        recorder=MemoryRecorder(),
        recognition_service=object(),
        vision_factory=lambda version: IdleVision(),
        advice_runtime_factory=lambda version: IdleAdvice(),
        local_hint_window_ms=0,
    )
    try:
        runtime.start(
            round_level=str(initial["round_level"]),
            hand=tuple(str(card) for card in initial["my_hand"]),
            lead_player=str(initial["lead_player"]),
            monotonic_ms=0,
        )
        runtime.bind_capture_generation(1)
        for turn in truth["turns"][:33]:
            turn_id = int(turn["turn_id"])
            runtime.commit_trusted_action(
                actor=str(turn["actor"]),
                cards=tuple(str(card) for card in turn["cards"]),
                is_pass=bool(turn["is_pass"]),
                monotonic_ms=turn_id,
                confidence=1.0,
            )

        trusted = rules.snapshot(captured_ms=33)
        assert trusted.current_seat is Seat.SELF
        assert len(trusted.play_history) == 33
        assert trusted.play_history[22].cards == AMBIGUOUS_PHYSICAL_CARDS
        state = replay_trusted_snapshot(trusted)
        try:
            decision = FableDanAdvisor(
                tmp_path,
                "profile",
                runtime_policy="rule_only",
                diagnostics="full",
                write_decision_log=False,
            ).recommend_detailed(state, request_id="turn34")
        except FableDanStateError as exc:
            pytest.fail(
                "turn34 FableDan rejected confirmed turn23 semantics: "
                f"{exc}"
            )

        metadata = trusted.play_history[22].action_metadata
        selected = metadata["selected_interpretation"]
        assert metadata["selection_source"] == "exact_engine_state"
        assert {
            int(item["type_id"])
            for item in metadata["candidate_interpretations"]
        } == {5, 9}
        assert selected == {
            "type_id": 9,
            "key": 6,
            "claim_ranks": ["6", "7", "8", "9", "10"],
        }
        resolution = decision.advice.engine_input["fabledan_trace"][
            "adapter_observation"
        ]["events"][22]["semantic_resolution"]
        assert decision.advice.engine_input["validation_status"] == "accepted"
        assert resolution["selection_source"] == "exact_engine_state"
        assert resolution["selected_interpretation"]["type_id"] == 9
        assert "metadata_warning" not in resolution
    finally:
        if runtime.status != "sealed":
            runtime.finish()
