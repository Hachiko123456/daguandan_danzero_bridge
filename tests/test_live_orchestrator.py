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
from daguandan_bridge.recognition_service import (
    FastSignalResult,
    OpeningSignal,
    PlayRegionResult,
)
from daguandan_bridge.gui.workers import LatestOnlyWorker


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


class FakeRecognitionService:
    def __init__(
        self,
        samples: list[PlayRegionResult],
        *,
        super_double_visible: bool = False,
    ):
        self.samples = list(samples)
        self.super_double_visible = super_double_visible
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
            super_double_visible=self.super_double_visible,
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


def _orchestrator(
    tmp_path,
    samples,
    *,
    recognition=None,
    lead_player="right",
    settle_ms=100,
    **options,
):
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
        recognition_service=recognition or FakeRecognitionService(samples),
        settle_ms=settle_ms,
        burst_sample_limit=5,
        burst_sample_interval_ms=50,
        minimum_free_bytes=0,
        **options,
    )
    orchestrator.start(
        round_level="2",
        hand=HAND,
        lead_player=lead_player,
        monotonic_ms=0,
    )
    return orchestrator


def _orchestrator_with_default_settle(
    tmp_path,
    samples,
    *,
    recognition_strategy="two_valid_streak",
):
    """Create the production default path without a test-only settle override."""

    store = LiveSessionStore(
        tmp_path / "profiles",
        "tencent_daguandan",
        session_id="default-settle",
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
    recorder = SessionRecorder(store.directory, size=(64, 32), fps=10)
    recognition = FakeRecognitionService(samples)
    orchestrator = LiveOrchestrator(
        reducer=LiveReducer("default-settle"),
        store=store,
        recorder=recorder,
        recognition_service=recognition,
        burst_sample_limit=5,
        burst_sample_interval_ms=50,
        minimum_free_bytes=0,
        recognition_strategy=recognition_strategy,
    )
    orchestrator.start(
        round_level="2",
        hand=HAND,
        lead_player="right",
        monotonic_ms=0,
    )
    return orchestrator, recognition


def _feed_changed_action(orchestrator, timestamp):
    return orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=timestamp,
        wall_time=f"default-settle-{timestamp}",
        metrics=ZoneFrameMetrics(
            monotonic_ms=timestamp,
            occupied=True,
            motion_score=0.20 if timestamp == 100 else 0.001,
            pass_visible=False,
            effect_visible=False,
        ),
    )


def test_default_two_valid_streak_starts_sampling_when_action_region_changes(tmp_path):
    """Prevent a global delay from hiding a short-lived valid play."""

    orchestrator, recognition = _orchestrator_with_default_settle(
        tmp_path,
        [_play("7S")] * 3,
    )

    _feed_changed_action(orchestrator, 100)

    assert recognition.targeted_calls == 1
    orchestrator.finish()


def test_reference_single_shot_keeps_its_explicit_1000ms_delay(tmp_path):
    """Only the reference strategy should wait a full second before sampling."""

    orchestrator, recognition = _orchestrator_with_default_settle(
        tmp_path,
        [_play("7S")],
        recognition_strategy="reference_single_shot",
    )

    _feed_changed_action(orchestrator, 100)
    _feed_changed_action(orchestrator, 1_099)
    assert recognition.targeted_calls == 0

    _feed_changed_action(orchestrator, 1_100)

    assert recognition.targeted_calls == 1
    orchestrator.finish()


class StrictFirstActionRecognition(FakeRecognitionService):
    """Records whether the first action path attempted pass recognition."""

    def __init__(self, samples):
        super().__init__(samples)
        self.fast_allow_pass: list[bool] = []
        self.play_allow_pass: list[bool] = []

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        self.fast_calls += 1
        self.fast_allow_pass.append(bool(allow_pass))
        return FastSignalResult(
            expected_player=expected_player,
            active_player=expected_player,
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank, allow_pass=True):
        del wild_rank
        self.targeted_calls += 1
        self.play_allow_pass.append(bool(allow_pass))
        sample = self.samples.pop(0)
        assert sample.player == seat
        return sample


def _feed_visible_action(orchestrator, *, first_motion_at=100):
    frame = np.zeros((32, 64, 3), np.uint8)
    update = None
    for timestamp in range(100, 2_100, 100):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"strategy-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.20 if timestamp == first_motion_at else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )
        if any(event.event_type == "player_played" for event in orchestrator.events):
            break
    return update


def test_first_action_never_scans_pass_template(tmp_path):
    recognition = StrictFirstActionRecognition([_play("7S")] * 8)
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        recognition_strategy="two_valid_streak",
    )

    _feed_visible_action(orchestrator)

    assert recognition.fast_allow_pass
    assert not any(recognition.fast_allow_pass)
    assert recognition.play_allow_pass
    assert not any(recognition.play_allow_pass)
    orchestrator.finish()


@pytest.mark.parametrize(
    ("strategy", "samples"),
    (
        ("reference_single_shot", [_play("7S")] * 8),
        ("two_valid_streak", [_play("3S", "4S", "5S", "6S"), _play("7S")] * 4),
        ("stable_single_shot", [_play("7S")] * 8),
        ("valid_candidate_vote", [_play("3S", "4S", "5S", "6S"), _play("7S")] * 4),
    ),
)
def test_each_recognition_strategy_can_commit_a_valid_first_play(
    tmp_path,
    strategy,
    samples,
):
    orchestrator = _orchestrator(
        tmp_path,
        samples,
        recognition_strategy=strategy,
    )

    _feed_visible_action(orchestrator)

    events = [event for event in orchestrator.events if event.event_type == "player_played"]
    assert len(events) == 1
    assert events[0].actor == "right"
    assert events[0].payload["cards"] == ["7S"]
    orchestrator.finish()


def test_empty_samples_wait_for_a_later_valid_action_without_emitting_retry(tmp_path):
    empty = PlayRegionResult(
        player="right",
        cards=(),
        is_pass=False,
        confidence=0.0,
        diagnostics=(),
        annotations=(),
        source="",
    )
    orchestrator = _orchestrator(
        tmp_path,
        [empty] * 5 + [_play("7S")] * 3,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    for index, timestamp in enumerate(range(100, 2_000, 100)):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"empty-then-play-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.2 if index == 0 else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )
        if orchestrator.snapshot.current_player != "right":
            break

    assert orchestrator.snapshot.current_player == "opposite"
    assert not any(
        event.event_type == "recognition_retry" for event in orchestrator.events
    )
    orchestrator.finish()


def test_new_action_window_ignores_stale_pixels_from_the_same_player_region(tmp_path):
    recognition = FakeRecognitionService([_play("7S")] * 4)
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition)
    stale = np.full((32, 64), 255, np.uint8)
    orchestrator._baseline_by_seat["right"] = stale.copy()
    orchestrator._previous_by_seat["right"] = stale.copy()
    orchestrator._content_prev_by_seat["right"] = np.full((16, 32), 255, np.uint8)
    orchestrator._activate_zone(0)

    for timestamp in (100, 200, 300, 400):
        orchestrator.ingest_frame(
            np.zeros((32, 64, 3), np.uint8),
            monotonic_ms=timestamp,
            wall_time=f"stale-{timestamp}",
        )

    assert recognition.targeted_calls == 0
    orchestrator.finish()


def test_running_super_double_pauses_action_pipeline_without_review(tmp_path):
    recognition = FakeRecognitionService(
        [_play("7S") for _ in range(7)],
        super_double_visible=True,
    )


def _pass(player: str = "right") -> PlayRegionResult:
    return PlayRegionResult(
        player=player,
        cards=(),
        is_pass=True,
        confidence=0.98,
        diagnostics=(),
        annotations=(),
        source="pass-template",
    )
    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S") for _ in range(7)],
        recognition=recognition,
    )
    before = orchestrator.snapshot
    updates = []
    frame = np.zeros((32, 64, 3), np.uint8)

    update = None
    for index in range(7):
        timestamp = 100 + index * 100
        updates.append(
            orchestrator.ingest_frame(
                frame,
                monotonic_ms=timestamp,
                wall_time=f"super-double-{timestamp}",
                metrics=ZoneFrameMetrics(
                    monotonic_ms=timestamp,
                    occupied=True,
                    motion_score=0.2 if index == 0 else 0.001,
                    pass_visible=False,
                    effect_visible=False,
                ),
            )
        )

    assert updates[-1].status == "running"
    assert updates[-1].fast_signals is not None
    assert updates[-1].fast_signals.super_double_visible is True
    assert updates[-1].review is None
    assert orchestrator.snapshot == before
    assert recognition.targeted_calls == 0
    assert [event.event_type for event in orchestrator.events] == [
        "initial_state_confirmed",
        "turn_started",
    ]
    orchestrator.finish()


def test_running_super_double_resumes_targeted_recognition_after_clear(tmp_path):
    recognition = FakeRecognitionService(
        [_play("7S") for _ in range(7)],
        super_double_visible=True,
    )
    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S") for _ in range(7)],
        recognition=recognition,
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    orchestrator.ingest_frame(
        frame,
        monotonic_ms=100,
        wall_time="super-double",
        metrics=ZoneFrameMetrics(
            monotonic_ms=100,
            occupied=True,
            motion_score=0.2,
            pass_visible=False,
            effect_visible=False,
        ),
    )
    recognition.super_double_visible = False

    for index in range(1, 5):
        timestamp = 100 + index * 100
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"after-super-double-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.2 if index == 1 else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )

    assert recognition.targeted_calls >= 1
    orchestrator.finish()


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


def test_uncertain_action_retries_without_pausing_reducer(tmp_path):
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

    assert orchestrator.status == "running"
    assert orchestrator.snapshot.current_player == "right"
    assert orchestrator.recorder.frame_count == before + 1
    assert orchestrator.latest_review is None
    assert orchestrator.events[-1].event_type == "recognition_retry"
    assert list(orchestrator.store.incidents_directory.iterdir())
    orchestrator.finish()


def test_pass_marker_commits_for_current_player_without_clear_gate(tmp_path):
    recognition = FakeRecognitionService(
        [_play("7S") for _ in range(2)] + [_pass("opposite") for _ in range(5)]
    )
    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S") for _ in range(2)] + [_pass("opposite") for _ in range(5)],
        recognition=recognition,
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    _feed_action(orchestrator)

    for index in range(7):
        timestamp = 1_000 + index * 100
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.001,
                pass_visible=True,
                effect_visible=False,
            ),
        )
        if any(event.event_type == "player_passed" for event in orchestrator.events):
            break

    assert update.status == "running"
    assert any(event.event_type == "player_passed" for event in orchestrator.events)
    assert orchestrator.snapshot.current_player == "left"
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


def test_latest_only_worker_preserves_first_action_window_in_order():
    first_started = threading.Event()
    release_first = threading.Event()
    completed = threading.Event()
    processed: list[int] = []

    def operation(value: int) -> int:
        processed.append(value)
        if value == 1:
            first_started.set()
            assert release_first.wait(2)
        if len(processed) == 4:
            completed.set()
        return value

    worker = LatestOnlyWorker(operation)
    worker.start()
    worker.submit(1)
    assert first_started.wait(2)
    worker.submit(2, preserve=True, max_preserved=3)
    worker.submit(3, preserve=True, max_preserved=3)
    worker.submit(4, preserve=True, max_preserved=3)
    worker.submit(5)
    release_first.set()
    assert completed.wait(2)
    assert worker.stop(timeout=2)

    assert processed == [1, 2, 3, 4, 5]


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


def test_persistent_empty_zone_and_next_turn_signal_do_not_infer_a_pass(
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
    assert update.status == "running"
    assert update.review is None
    assert orchestrator.snapshot.current_player == "opposite"
    assert not [
        event
        for event in orchestrator.events
        if event.event_type in {"player_passed", "recognition_retry"}
    ]
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


def test_pause_and_resume_preserve_automatic_retry_state(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [_play("3S"), _play("4S"), _play("5S"), _play("6S"), _play("7S")],
    )
    _feed_action(orchestrator)
    assert orchestrator.status == "running"

    assert orchestrator.pause().status == "paused"
    resumed = orchestrator.resume(monotonic_ms=1_000)

    assert resumed.status == "running"
    assert resumed.review is None
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


class FakeLeadRecognitionService:
    def __init__(
        self,
        *,
        lead_seat=None,
        active_seat=None,
        super_double_visible=False,
    ):
        self.lead_seat = lead_seat
        self.active_seat = active_seat if active_seat is not None else lead_seat
        self.super_double_visible = super_double_visible
        self.fast_calls = 0
        self.opening_calls = 0

    def recognize_fast_signals(self, _image, expected_player):
        self.fast_calls += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=None,
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
            super_double_visible=self.super_double_visible,
        )

    def recognize_opening_signal(self, _image):
        self.opening_calls += 1
        return OpeningSignal(
            super_double_visible=self.super_double_visible,
            marker_player=self.lead_seat,
            active_player=self.active_seat,
            self_action_buttons_visible=False,
        )


def _lead_orchestrator(
    tmp_path,
    recognition,
    *,
    lead_player=None,
    timeout_ms=30_000,
):
    store = LiveSessionStore(tmp_path / "profiles", "tencent_daguandan", session_id="game-lead")
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
        reducer=LiveReducer("game-lead"),
        store=store,
        recorder=recorder,
        recognition_service=recognition,
        settle_ms=100,
        minimum_free_bytes=0,
        lead_wait_timeout_ms=timeout_ms,
    )
    update = orchestrator.start(
        round_level="2",
        hand=HAND,
        lead_player=lead_player,
        monotonic_ms=0,
    )
    return orchestrator, update


def test_waiting_lead_ignores_marker_until_doubling_controls_clear(tmp_path):
    recognition = FakeLeadRecognitionService(
        lead_seat="right",
        super_double_visible=True,
    )
    orchestrator, update = _lead_orchestrator(tmp_path, recognition, timeout_ms=500)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        update = orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"double-{timestamp}")

    assert update.status == "waiting_lead"
    assert update.snapshot.lead_player is None
    assert recognition.opening_calls == 3
    assert "deal_complete" not in [event.event_type for event in orchestrator.events]

    recognition.super_double_visible = False
    for timestamp in (400, 500, 600):
        update = orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")

    assert update.status == "running"
    assert update.snapshot.lead_player == "right"
    assert recognition.opening_calls == 6
    assert "deal_complete" in [event.event_type for event in orchestrator.events]
    orchestrator.finish()


def test_waiting_lead_records_deal_complete_once_when_controls_briefly_appear(tmp_path):
    recognition = FakeLeadRecognitionService(lead_seat=None)
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition, timeout_ms=5_000)
    frame = np.zeros((32, 64, 3), np.uint8)

    orchestrator.ingest_frame(frame, monotonic_ms=100, wall_time="before-controls")
    recognition.super_double_visible = True
    orchestrator.ingest_frame(frame, monotonic_ms=200, wall_time="controls-visible")
    recognition.super_double_visible = False
    orchestrator.ingest_frame(frame, monotonic_ms=300, wall_time="controls-cleared")

    assert [event.event_type for event in orchestrator.events].count("deal_complete") == 1
    orchestrator.finish()


def test_waiting_lead_auto_confirms_from_first_play_marker(tmp_path):
    recognition = FakeLeadRecognitionService(lead_seat="right")
    orchestrator, update = _lead_orchestrator(tmp_path, recognition)

    assert update.status == "waiting_lead"
    assert update.snapshot.current_player is None

    frame = np.zeros((32, 64, 3), np.uint8)
    update = orchestrator.ingest_frame(frame, monotonic_ms=100, wall_time="t0")
    assert update.status == "waiting_lead"

    update = orchestrator.ingest_frame(frame, monotonic_ms=200, wall_time="t1")
    assert update.status == "waiting_lead"
    update = orchestrator.ingest_frame(frame, monotonic_ms=300, wall_time="t2")

    assert update.status == "running"
    assert update.snapshot.current_player == "right"
    assert update.snapshot.lead_player == "right"
    event_types = [event.event_type for event in orchestrator.events]
    assert "lead_player_confirmed" in event_types
    assert "turn_started" in event_types
    orchestrator.finish()


def test_lead_confirmation_preserves_opening_baseline_for_the_first_action(tmp_path):
    recognition = FakeLeadRecognitionService(lead_seat="right")
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"baseline-{timestamp}")

    assert orchestrator.status == "running"
    assert "right" in orchestrator._baseline_by_seat
    orchestrator.finish()


def test_waiting_lead_keeps_waiting_when_first_marker_and_active_timer_conflict(tmp_path):
    recognition = FakeLeadRecognitionService(
        lead_seat="right",
        active_seat="opposite",
    )
    orchestrator, update = _lead_orchestrator(tmp_path, recognition, timeout_ms=1_000)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        update = orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"conflict-{timestamp}")

    assert update.status == "waiting_lead"
    assert update.snapshot.lead_player is None
    assert "lead_player_confirmed" not in [event.event_type for event in orchestrator.events]
    orchestrator.finish()


class FakeDeferredLeadPlayRecognitionService(FakeLeadRecognitionService):
    def __init__(self, *, lead_seat, cards):
        super().__init__(lead_seat=lead_seat)
        self.cards = cards
        self.targeted_calls = 0

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
        return PlayRegionResult(
            player=seat,
            cards=self.cards,
            is_pass=False,
            confidence=0.95,
            diagnostics=(),
            annotations=(),
            source="fake_deferred_lead_play",
        )


class FakeSelfLeadRecognitionService(FakeLeadRecognitionService):
    def __init__(self, *, cards: tuple[str, ...], post_hand: tuple[str, ...] | None = None):
        super().__init__(lead_seat="self")
        self.cards = cards
        self.post_hand = post_hand
        self.controls_visible = True
        self.targeted_calls = 0

    def recognize_fast_signals(self, _image, expected_player):
        self.fast_calls += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=expected_player,
            pass_visible=False,
            self_action_buttons_visible=(
                expected_player == "self" and self.controls_visible
            ),
            effect_visible=False,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank):
        del wild_rank
        self.targeted_calls += 1
        post_hand = self.post_hand
        if post_hand is None:
            post_hand = tuple(card for card in HAND if card not in set(self.cards))
        return PlayRegionResult(
            player=seat,
            cards=self.cards,
            is_pass=False,
            confidence=0.95,
            diagnostics=(),
            annotations=(),
            source="fake_self_lead_play",
            post_hand=post_hand,
            post_hand_confidence=0.95,
        )


def test_self_lead_waits_for_action_controls_to_clear(tmp_path):
    recognition = FakeSelfLeadRecognitionService(cards=("7S",))
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")
    for timestamp in (400, 500, 600, 700, 800):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"buttons-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.2,
                pass_visible=False,
                effect_visible=False,
            ),
        )

    assert update.status == "running"
    assert update.snapshot.current_player == "self"
    assert recognition.targeted_calls == 0
    assert orchestrator.latest_review is None
    orchestrator.finish()


def test_self_lead_empty_reads_wait_for_the_real_action_without_using_post_hand(tmp_path):
    changed_hand = tuple(card for card in HAND if card != "7S")
    recognition = FakeSelfLeadRecognitionService(cards=(), post_hand=changed_hand)
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=400,
        wall_time="buttons-visible",
        metrics=ZoneFrameMetrics(400, True, 0.2, False, False),
    )
    recognition.controls_visible = False
    for timestamp in (500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"changed-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.001, False, False),
        )

    assert update.status == "running"
    assert update.review is None
    assert not any(
        event.event_type == "recognition_retry" for event in orchestrator.events
    )
    orchestrator.finish()


def test_self_lead_commits_after_action_controls_clear(tmp_path):
    recognition = FakeSelfLeadRecognitionService(cards=("7S",))
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=400,
        wall_time="buttons-visible",
        metrics=ZoneFrameMetrics(400, True, 0.2, False, False),
    )
    recognition.controls_visible = False
    for index, timestamp in enumerate((500, 600, 700, 800, 900, 1000, 1100, 1200)):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"played-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )

    assert update.status == "running"
    assert orchestrator.snapshot.current_player == "right"
    assert recognition.targeted_calls >= 2
    assert any(
        event.event_type == "player_played" and event.actor == "self"
        for event in orchestrator.events
    )
    orchestrator.finish()


def test_self_play_commits_when_post_hand_template_is_wrong(tmp_path):
    # The play-zone result is correct, while the old second recognition pass
    # over the hand is deliberately wrong. It must not block the action.
    recognition = FakeSelfLeadRecognitionService(cards=("7S",), post_hand=HAND)
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(
            frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}"
        )
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=400,
        wall_time="controls-visible",
        metrics=ZoneFrameMetrics(400, True, 0.2, False, False),
    )
    recognition.controls_visible = False
    for timestamp in (500, 600, 700, 800, 900, 1000, 1100, 1200, 1300):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"action-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.001, False, False),
        )

    assert orchestrator.snapshot.current_player == "right"
    assert any(event.event_type == "player_played" for event in orchestrator.events)
    assert not any(
        "self_hand_delta_unverified" in str(event.payload)
        for event in orchestrator.events
    )
    orchestrator.finish()


def test_deferred_lead_captures_already_visible_first_action(tmp_path):
    recognition = FakeDeferredLeadPlayRecognitionService(
        lead_seat="opposite",
        cards=("7S", "7H"),
    )
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")

    assert orchestrator.status == "running"
    assert orchestrator.snapshot.current_player == "opposite"

    for index, timestamp in enumerate((400, 500, 600, 700, 800, 900, 1000, 1100)):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"play-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.2 if index == 0 else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )

    plays = [event for event in orchestrator.events if event.event_type == "player_played"]
    assert len(plays) == 1
    assert plays[0].actor == "opposite"
    assert orchestrator.snapshot.current_player == "left"
    assert recognition.targeted_calls >= 2
    orchestrator.finish()


def test_deferred_lead_empty_first_action_waits_for_the_real_action(tmp_path):
    recognition = FakeDeferredLeadPlayRecognitionService(
        lead_seat="right",
        cards=(),
    )
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")
    for index, timestamp in enumerate(
        (400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300)
    ):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"play-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.2 if index == 0 else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )

    assert update.status == "running"
    assert update.review is None
    assert not any(
        event.event_type == "recognition_retry" for event in orchestrator.events
    )
    orchestrator.finish()


def test_waiting_lead_timeout_retries_without_manual_confirmation(tmp_path):
    recognition = FakeLeadRecognitionService()
    orchestrator, _update = _lead_orchestrator(
        tmp_path,
        recognition,
        timeout_ms=500,
    )

    frame = np.zeros((32, 64, 3), np.uint8)
    update = orchestrator.ingest_frame(frame, monotonic_ms=600, wall_time="t0")

    assert update.status == "waiting_lead"
    assert update.event is not None
    assert update.event.event_type == "recognition_retry"
    assert update.review is None
    assert update.event.payload["reason"].startswith("lead_player")
    orchestrator.finish()
