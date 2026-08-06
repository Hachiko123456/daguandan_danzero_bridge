from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from daguandan_bridge.live.orchestrator import (
    LiveOrchestrator,
    ReviewCandidate,
    ReviewRequest,
)
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
    timeline = read_json_lines(orchestrator.store.timeline_path)
    assert [item["event_type"] for item in timeline] == [
        "initial_state_confirmed",
        "turn_started",
        "player_played",
        "turn_started",
    ]
    assert [item["seq"] for item in timeline] == [1, 2, 3, 4]
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


def test_latest_only_worker_suppresses_result_after_stop_request():
    started = threading.Event()
    release = threading.Event()
    results: list[int] = []

    def operation(value: int) -> int:
        started.set()
        assert release.wait(2)
        return value

    worker = LatestOnlyWorker(operation, on_result=results.append)
    worker.start()
    worker.submit(1)
    assert started.wait(2)

    assert not worker.stop(timeout=0.01)
    release.set()
    assert worker.stop(timeout=2)

    assert results == []


def test_recording_path_does_not_run_slow_recognition(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S") for _ in range(3)])
    recognition = orchestrator.recognition_service

    warning = orchestrator.record_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="record-only",
    )

    assert warning is None
    assert orchestrator.recorder.frame_count == 1
    assert recognition.fast_calls == 0
    assert recognition.targeted_calls == 0
    orchestrator.finish()


def test_finish_seals_session_when_pending_incident_media_fails(tmp_path, monkeypatch):
    orchestrator = _orchestrator(tmp_path, [_play("7S") for _ in range(3)])
    orchestrator.record_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="trigger",
    )
    incident_dir = orchestrator.store.incidents_directory / "INC-test"
    orchestrator.recorder.schedule_incident_media(
        incident_dir,
        trigger_ms=100,
        after_ms=5_000,
    )
    monkeypatch.setattr(
        orchestrator.recorder,
        "_write_incident_media",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("codec full")),
    )

    update = orchestrator.finish()

    manifest = json.loads(orchestrator.store.manifest_path.read_text("utf-8"))
    assert update.status == "sealed"
    assert manifest["status"] == "sealed"
    assert manifest["incident_media_failures"][0]["reason"] == (
        "incident_media_finalize_failed"
    )


def test_pause_does_not_wait_for_slow_vision_and_stale_analysis_is_discarded(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S") for _ in range(3)])
    recognition = orchestrator.recognition_service
    started = threading.Event()
    release = threading.Event()
    original = recognition.recognize_fast_signals

    def slow_fast(frame, expected_player):
        started.set()
        assert release.wait(2)
        return original(frame, expected_player)

    recognition.recognize_fast_signals = slow_fast
    updates = []
    analysis = threading.Thread(
        target=lambda: updates.append(
            orchestrator.analyze_frame(
                np.zeros((32, 64, 3), np.uint8),
                monotonic_ms=100,
            )
        )
    )
    analysis.start()
    assert started.wait(2)

    started_at = time.perf_counter()
    paused = orchestrator.pause()
    elapsed = time.perf_counter() - started_at
    assert paused.status == "paused"
    assert elapsed < 0.2
    release.set()
    analysis.join(2)

    assert not analysis.is_alive()
    assert updates[0].status == "paused"
    assert recognition.targeted_calls == 0
    orchestrator.finish()


def test_persistent_empty_zone_and_next_turn_signal_offer_one_click_inferred_pass(
    tmp_path,
):
    orchestrator = _orchestrator(tmp_path, [_play("7S") for _ in range(5)])
    _feed_action(orchestrator)
    recognition = orchestrator.recognition_service

    def next_turn_fast(_frame, expected_player):
        return FastSignalResult(
            expected_player=expected_player,
            active_player="left",
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
        )

    recognition.recognize_fast_signals = next_turn_fast
    update = None
    for timestamp in (1_000, 1_100, 1_200, 1_300, 1_400):
        update = orchestrator.analyze_frame(
            np.zeros((32, 64, 3), np.uint8),
            monotonic_ms=timestamp,
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=False,
                motion_score=0.0,
                pass_visible=False,
                effect_visible=False,
            ),
        )

    assert update is not None
    assert update.status == "review_required"
    assert update.review is not None
    assert update.review.reason == "pass_template_missing"
    assert update.review.candidates[0].is_pass
    orchestrator.finish()


def test_repeated_recorder_warnings_share_one_incident_package(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S") for _ in range(3)])
    bad_frame = np.zeros((10, 10, 3), np.uint8)

    orchestrator.record_frame(bad_frame, monotonic_ms=100, wall_time="bad-1")
    orchestrator.record_frame(bad_frame, monotonic_ms=200, wall_time="bad-2")

    incidents = list(orchestrator.store.incidents_directory.iterdir())
    assert len(incidents) == 1
    occurrences = read_json_lines(incidents[0] / "occurrences.jsonl")
    assert len(occurrences) == 2
    assert occurrences[-1]["coalesced"] is True
    orchestrator.finish()


def test_pause_and_resume_preserve_pending_review(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [_play("3S"), _play("4S"), _play("5S"), _play("6S"), _play("7S")],
    )
    _feed_action(orchestrator)
    assert orchestrator.status == "review_required"

    assert orchestrator.pause().status == "paused"
    resumed = orchestrator.resume(monotonic_ms=1_000)

    assert resumed.status == "review_required"
    assert resumed.review is not None
    orchestrator.finish()


def test_invalid_review_candidate_cannot_be_published_by_one_click(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S") for _ in range(3)])
    orchestrator.status = "review_required"
    orchestrator.latest_review = ReviewRequest(
        reason="does_not_beat_table",
        player="right",
        candidates=(
            ReviewCandidate(
                candidate_id="CAND-invalid",
                cards=("6S",),
                is_pass=False,
                votes=3,
                confidence=0.95,
                valid=False,
                rejected_reason="does_not_beat_table",
            ),
        ),
    )

    with pytest.raises(ValueError, match="候选未通过"):
        orchestrator.confirm_candidate("CAND-invalid")

    assert orchestrator.snapshot.current_player == "right"
    orchestrator.finish()
