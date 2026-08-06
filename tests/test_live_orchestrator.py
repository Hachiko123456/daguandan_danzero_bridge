from __future__ import annotations

import threading

import numpy as np

from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.recognition_service import FastSignalResult, PlayRegionResult
from daguandan_bridge.gui.workers import LatestOnlyWorker


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


class FakeRecognitionService:
    def __init__(self, samples: list[PlayRegionResult]):
        self.samples = list(samples)
        self.targeted_calls = 0
        self.fast_calls = 0

    def recognize_fast_signals(self, _image, expected_player):
        self.fast_calls += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=expected_player,
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank):
        del wild_rank
        self.targeted_calls += 1
        sample = self.samples.pop(0)
        assert sample.player == seat
        return sample


def _play(*cards: str) -> PlayRegionResult:
    return PlayRegionResult(
        player="right",
        cards=cards,
        is_pass=False,
        confidence=0.94,
        diagnostics=(),
        annotations=(),
        source="fake",
    )


def _orchestrator(tmp_path, samples):
    store = LiveSessionStore(tmp_path / "profiles", "tencent_daguandan", session_id="game")
    store.start(
        {
            "application_version": "test",
            "configuration_hash": "config",
            "template_manifest_hash": "templates",
            "target_fps": 10,
            "codec": "MJPG",
        }
    )
    recorder = SessionRecorder(store.directory, size=(64, 32), fps=10)
    orchestrator = LiveOrchestrator(
        reducer=LiveReducer("game"),
        store=store,
        recorder=recorder,
        recognition_service=FakeRecognitionService(samples),
        settle_ms=100,
        burst_sample_limit=5,
        burst_sample_interval_ms=50,
        minimum_free_bytes=0,
    )
    orchestrator.start(
        round_level="2",
        hand=HAND,
        lead_player="right",
        monotonic_ms=0,
    )
    return orchestrator


def _feed_action(orchestrator, *, count: int = 7):
    frame = np.zeros((32, 64, 3), np.uint8)
    for index in range(count):
        timestamp = 100 + index * 100
        motion = 0.2 if index == 0 else 0.001
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"t{index}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=motion,
                pass_visible=False,
                effect_visible=False,
            ),
        )


def test_orchestrator_commits_each_turn_once(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S", "7H") for _ in range(5)])

    _feed_action(orchestrator)

    plays = [event for event in orchestrator.events if event.event_type == "player_played"]
    assert len(plays) == 1
    assert orchestrator.snapshot.current_player == "opposite"
    assert len(read_json_lines(orchestrator.store.timeline_path)) == 2
    orchestrator.finish()


def test_uncertain_action_pauses_reducer_but_recording_continues(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [_play("3S"), _play("4S"), _play("5S"), _play("6S"), _play("7S")],
    )

    _feed_action(orchestrator)
    before = orchestrator.recorder.frame_count
    orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=900,
        wall_time="after-review",
    )

    assert orchestrator.status == "review_required"
    assert orchestrator.snapshot.current_player == "right"
    assert orchestrator.recorder.frame_count == before + 1
    assert orchestrator.latest_review is not None
    assert list(orchestrator.store.incidents_directory.iterdir())
    orchestrator.finish()


def test_latest_only_worker_replaces_pending_recognition_batch():
    first_started = threading.Event()
    release_first = threading.Event()
    completed = threading.Event()
    processed: list[int] = []
    results: list[int] = []

    def operation(value: int) -> int:
        processed.append(value)
        if value == 1:
            first_started.set()
            assert release_first.wait(2)
        return value * 10

    def on_result(value: int) -> None:
        results.append(value)
        if len(results) == 2:
            completed.set()

    worker = LatestOnlyWorker(operation, on_result=on_result)
    worker.start()
    worker.submit(1)
    assert first_started.wait(2)
    worker.submit(2)
    worker.submit(3)
    release_first.set()
    assert completed.wait(2)
    worker.stop(timeout=2)

    assert processed == [1, 3]
    assert results == [10, 30]
    assert not worker.is_running
