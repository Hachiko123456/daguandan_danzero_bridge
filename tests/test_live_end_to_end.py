from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.recognition_service import FastSignalResult, PlayRegionResult


FIXTURE = Path(__file__).parent / "fixtures" / "live_sessions" / "golden_observations.json"
HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("JC", "JD", "9S")


class ScriptedRecognition:
    def __init__(self, actions):
        self.actions = list(actions)
        self.sample_counts: dict[str, int] = {}

    def recognize_fast_signals(self, _image, expected_player):
        return FastSignalResult(
            expected_player=expected_player,
            active_player=expected_player,
            pass_visible=False,
            self_action_buttons_visible=expected_player == "self",
            effect_visible=False,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank):
        del wild_rank
        action = self.actions[0]
        assert action["player"] == seat
        self.sample_counts[seat] = self.sample_counts.get(seat, 0) + 1
        if self.sample_counts[seat] == 3:
            self.actions.pop(0)
        return PlayRegionResult(
            player=seat,
            cards=tuple(action["cards"]),
            is_pass=bool(action["is_pass"]),
            confidence=0.95,
            diagnostics=(),
            annotations=(),
            source="golden_fixture",
            post_hand=(
                tuple(card for card in HAND if card not in {"JC", "JD"})
                if seat == "self"
                else ()
            ),
            post_hand_confidence=0.96 if seat == "self" else 0.0,
        )


class FakeAdvisor:
    def __init__(self):
        self.calls = 0

    def recommend(self, state, *, request_id=""):
        self.calls += 1
        return LocalAdvice(
            strategy="fake",
            cards=("JC", "JD"),
            play_type="Pair",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=20.0,
            request_id=request_id,
            engine_input={"request_id": request_id, "state_revision": state.revision},
            timings={"agent_step": 15.0},
        )


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_golden_session_produces_exact_events_advice_and_replay_data(tmp_path):
    fixture = json.loads(FIXTURE.read_text("utf-8"))
    store = LiveSessionStore(
        tmp_path / "profiles",
        "tencent_daguandan",
        session_id="golden",
    )
    store.start(
        {
            "application_version": "test",
            "configuration_hash": "config",
            "template_manifest_hash": "templates",
            "target_fps": 10,
            "codec": "MJPG",
        }
    )
    recognition = ScriptedRecognition(fixture["actions"])
    advisor = FakeAdvisor()
    runner = LiveOrchestrator(
        reducer=LiveReducer("golden"),
        store=store,
        recorder=SessionRecorder(store.directory, size=(64, 32), fps=10),
        recognition_service=recognition,
        advisor=advisor,
        settle_ms=100,
        burst_sample_interval_ms=50,
        minimum_free_bytes=0,
    )
    runner.start(
        round_level=fixture["round_level"],
        hand=HAND,
        lead_player=fixture["lead_player"],
        monotonic_ms=0,
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    now = 0

    def feed(*, occupied, motion=0.001, effect=False):
        nonlocal now
        now += 100
        runner.ingest_frame(
            frame,
            monotonic_ms=now,
            wall_time=f"t{now}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=now,
                occupied=occupied,
                motion_score=motion,
                pass_visible=False,
                effect_visible=effect,
            ),
        )

    def normal_action(*, needs_clear):
        if needs_clear:
            feed(occupied=False)
        feed(occupied=True, motion=0.2)
        feed(occupied=True)
        feed(occupied=True)
        feed(occupied=True)
        feed(occupied=True)

    normal_action(needs_clear=False)  # right
    normal_action(needs_clear=True)   # opposite pass

    feed(occupied=False)              # clear old left-zone residue
    feed(occupied=True, motion=0.2)   # animation starts
    feed(occupied=True)               # false settle begins
    feed(occupied=True, effect=True)  # effect is ignored by the minimal gate
    assert recognition.sample_counts.get("left", 0) >= 1
    feed(occupied=True)               # continue the same settle/read window
    feed(occupied=True)               # burst sample 1
    feed(occupied=True)               # burst sample 2
    feed(occupied=True)               # burst sample 3 / commit

    _wait_until(lambda: runner.latest_advice is not None and runner.latest_advice.status == "ready")
    feed(occupied=False)              # self-turn corroborator makes advice visible
    _wait_until(lambda: runner.latest_advice is not None and runner.latest_advice.visible)
    normal_action(needs_clear=False)  # self plays the advised pair

    actions = [
        (event.actor, event.event_type, tuple(event.payload.get("cards", ())))
        for event in runner.events
        if event.event_type in {"player_played", "player_passed"}
    ]
    assert actions == [
        ("right", "player_played", ("7H", "7S")),
        ("opposite", "player_passed", ()),
        ("left", "player_played", ("9C", "9D")),
        ("self", "player_played", ("JC", "JD")),
    ]
    assert [event.seq for event in runner.events] == list(
        range(1, len(runner.events) + 1)
    )
    assert runner.snapshot.current_player == fixture["expected_current_player"]
    assert advisor.calls == 1
    assert runner.metrics.advice_visible_latency_ms is not None
    assert runner.metrics.advice_visible_latency_ms <= 3_000
    assert len(read_json_lines(store.advice_path)) >= 2
    timeline = store.timeline_markdown_path.read_text("utf-8")
    assert "右家出牌：7H 7S" in timeline
    assert "对家不出" in timeline
    assert "DanZero 建议已就绪" in timeline

    result = runner.finish()
    assert result.status == "sealed"
    manifest = json.loads(store.manifest_path.read_text("utf-8"))
    assert manifest["performance_metrics"]["confirmed_action_count"] == 4
    assert manifest["performance_metrics"]["advice_visible_latency_ms"] <= 3_000
    assert store.observations_gzip_path.is_file()
    assert (store.directory / "video" / "game.avi").is_file()
