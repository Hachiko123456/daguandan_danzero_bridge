from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from daguandan_bridge.live.orchestrator import (
    LiveOrchestrator,
    ReviewCandidate,
    ReviewRequest,
)
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.recognition_service import (
    FastSignalResult,
    OpeningSignal,
    PlacementSignal,
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
    round_level="2",
    settle_ms=100,
    hand=HAND,
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
        round_level=round_level,
        hand=hand,
        lead_player=lead_player,
        monotonic_ms=0,
    )
    return orchestrator


def test_latest_game_wildcard_play_is_committed_and_removed_from_hand(tmp_path):
    latest_hand = (
        "10D", "10H", "2D", "2H", "4C", "4D", "4S", "5D", "5S",
        "6C", "7C", "7D", "7H", "8C", "8H", "8S", "9C", "9D", "9H",
        "AC", "AS", "JC", "JD", "KS", "QC", "big_joker", "small_joker",
    )
    played = ("JC", "JD", "9H", "2D", "2H")
    samples = [
        PlayRegionResult(
            player="self",
            cards=played,
            is_pass=False,
            confidence=0.94,
            diagnostics=(),
            annotations=(),
            source="latest-game-regression",
        )
        for _ in range(2)
    ]
    orchestrator = _orchestrator(
        tmp_path,
        samples,
        lead_player="self",
        round_level="9",
        settle_ms=0,
        hand=latest_hand,
    )
    orchestrator.commit_trusted_action(
        actor="self",
        cards=("4C", "4D", "4S", "5D", "5S"),
        is_pass=False,
        monotonic_ms=10,
    )
    orchestrator.commit_trusted_action(
        actor="right",
        cards=("10C", "10H", "10S", "KH", "KS"),
        is_pass=False,
        monotonic_ms=20,
    )
    orchestrator.commit_trusted_action(
        actor="opposite", is_pass=True, monotonic_ms=30
    )
    orchestrator.commit_trusted_action(
        actor="left", is_pass=True, monotonic_ms=40
    )

    updates = []
    for timestamp in (100, 200):
        updates.append(
            orchestrator.ingest_frame(
                np.zeros((32, 64, 3), np.uint8),
                monotonic_ms=timestamp,
                wall_time=f"wildcard-{timestamp}",
                metrics=ZoneFrameMetrics(
                    monotonic_ms=timestamp,
                    occupied=True,
                    motion_score=0.20 if timestamp == 100 else 0.0,
                    pass_visible=False,
                    effect_visible=False,
                    content_changed=timestamp == 100,
                ),
            )
        )

    event = updates[-1].event
    assert event is not None
    assert event.event_type == "player_played"
    assert event.actor == "self"
    assert tuple(event.payload["cards"]) == tuple(sorted(played))
    assert event.payload["logical_label"] == "JJJ22"
    assert event.payload["beats_table"] is True
    assert event.payload["wildcard_substitutions"] == [
        {"card": "9H", "as_rank": "J"}
    ]
    assert orchestrator.snapshot.current_player == "right"
    assert not (set(played) & set(orchestrator.snapshot.my_hand))
    orchestrator.finish()


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


def test_empty_action_window_timeout_rearms_silently(tmp_path):
    """A reset window without a single new sample is not a failed action."""

    orchestrator = _orchestrator(
        tmp_path,
        [],
        action_timeout_ms=500,
    )
    update = orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=600,
        wall_time="empty-window-timeout",
        metrics=ZoneFrameMetrics(
            monotonic_ms=600,
            occupied=False,
            motion_score=0.0,
            pass_visible=False,
            effect_visible=False,
        ),
    )

    assert update.status == "running"
    assert update.event is None
    assert update.snapshot.current_player == "right"
    assert not any(
        event.event_type == "recognition_retry" for event in orchestrator.events
    )
    orchestrator.finish()


def test_first_action_confirmation_survives_a_transient_zone_reset(tmp_path):
    """A live queue may put two reads of the same opening play in two bursts."""

    orchestrator = _orchestrator(
        tmp_path,
        [_play("3D"), _play("3D")],
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    orchestrator.ingest_frame(
        frame,
        monotonic_ms=100,
        wall_time="opening-read-1",
        metrics=ZoneFrameMetrics(100, True, 0.20, False, False),
    )
    # The effect/ROI change clears the ordinary burst between the two samples.
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=200,
        wall_time="opening-transition",
        metrics=ZoneFrameMetrics(200, False, 0.0, False, False),
    )
    update = orchestrator.ingest_frame(
        frame,
        monotonic_ms=300,
        wall_time="opening-read-2",
        metrics=ZoneFrameMetrics(300, True, 0.20, False, False),
    )

    assert update.event is not None
    assert update.event.event_type == "player_played"
    assert update.event.actor == "right"
    assert update.event.payload["cards"] == ["3D"]
    assert orchestrator.snapshot.current_player == "opposite"
    orchestrator.finish()


def test_first_action_timeout_keeps_observed_cards_and_opening_guard(tmp_path):
    """A failed opening read must be diagnosable and continue as an opening turn."""

    invalid = PlayRegionResult(
        player="right",
        cards=("3S", "4S"),
        is_pass=False,
        confidence=0.94,
        diagnostics=(),
        annotations=(),
        source="fake-invalid-opening",
    )
    orchestrator = _orchestrator(
        tmp_path,
        [invalid, _play("7S"), _play("7S")],
        settle_ms=0,
        action_timeout_ms=300,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    orchestrator.ingest_frame(
        frame,
        monotonic_ms=100,
        wall_time="invalid-opening-read",
        metrics=ZoneFrameMetrics(100, True, 0.20, False, False),
    )
    retry = orchestrator.ingest_frame(
        frame,
        monotonic_ms=301,
        wall_time="opening-timeout",
        metrics=ZoneFrameMetrics(301, True, 0.0, False, False),
    )

    assert retry.event is not None
    assert retry.event.event_type == "recognition_retry"
    assert retry.event.payload["reason"] == "action_timeout"
    assert "3S 4S" in retry.event.payload["message"]
    assert "没有可用候选" not in retry.event.payload["message"]
    assert orchestrator._first_action_pending

    for timestamp in (400, 500):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"opening-recovery-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.20, False, False),
        )

    assert update.event is not None
    assert update.event.event_type == "player_played"
    assert not orchestrator._first_action_pending
    assert orchestrator.snapshot.current_player == "opposite"
    orchestrator.finish()


def test_timeout_commits_two_matching_legal_first_action_reads(tmp_path, monkeypatch):
    """Do not discard stable first-play evidence merely because the timer won."""

    orchestrator = _orchestrator(
        tmp_path,
        [_play("7D", "7S"), _play("7D", "7S")],
        settle_ms=0,
        action_timeout_ms=300,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    # Simulate a live handoff whose ordinary per-frame decision was
    # invalidated after samples were logged but before it could commit.
    monkeypatch.setattr(orchestrator, "_decide_if_ready", lambda *_args: None)
    for timestamp in (100, 200):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"deferred-first-read-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.20, False, False),
        )

    update = orchestrator.ingest_frame(
        frame,
        monotonic_ms=301,
        wall_time="deferred-first-timeout",
        metrics=ZoneFrameMetrics(301, True, 0.0, False, False),
    )

    assert update.event is not None
    assert update.event.event_type == "player_played"
    assert update.event.actor == "right"
    assert update.event.payload["cards"] == ["7D", "7S"]
    assert not any(
        event.event_type == "recognition_retry" for event in orchestrator.events
    )
    assert orchestrator.snapshot.current_player == "opposite"
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


def test_continue_game_control_emits_one_game_end_event(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S")])

    first = orchestrator.ingest_fast_signal(
        active_player="right",
        game_end_control="continue_game",
    )
    second = orchestrator.ingest_fast_signal(
        active_player="right",
        game_end_control="continue_game",
    )

    assert first.event is not None
    assert first.event.event_type == "game_end_detected"
    assert first.event.payload["control"] == "continue_game"
    assert set(first.event.payload["remaining_cards"]) == {
        "self", "right", "opposite", "left"
    }
    assert all(
        type(count) is int
        for count in first.event.payload["remaining_cards"].values()
    )
    assert isinstance(first.event.payload["finished_seats"], list)
    assert second.event is None
    orchestrator.finish()


def test_finished_player_and_wind_catch_are_emitted_to_the_timeline(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S")])
    before = replace(
        orchestrator.snapshot,
        remaining_cards={"self": 27, "left": 27, "opposite": 27, "right": 1},
        lead_player="right",
        trick_id=1,
        finished_seats=frozenset(),
    )
    after_finish = replace(
        before,
        remaining_cards={"self": 27, "left": 27, "opposite": 27, "right": 0},
        finished_seats=frozenset({"right"}),
    )
    finished = orchestrator._append_action_outcomes(
        before,
        after_finish,
        trigger_action_event_id="EVT-RIGHT-FINAL",
        trigger_actor="right",
    )
    assert any(
        event.event_type == "player_finished"
        and event.payload["placement"] == "head"
        and event.payload["trigger_action_event_id"] == "EVT-RIGHT-FINAL"
        and event.actor == "right"
        for event in finished
    )
    after_wind = replace(
        after_finish,
        lead_player="left",
        trick_id=2,
    )
    latest = orchestrator._append_action_outcomes(after_finish, after_wind)
    assert any(
        event.event_type == "wind_caught"
        and event.payload == {"from_player": "right", "to_player": "left"}
        for event in latest
    )
    orchestrator.finish()


def test_visual_head_badge_recovers_finish_and_wind_after_two_frames(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S")],
        lead_player="left",
    )
    orchestrator.commit_trusted_action(
        actor="left",
        cards=("9S",),
        is_pass=False,
        monotonic_ms=1,
    )
    fast = FastSignalResult(
        expected_player="self",
        active_player="self",
        pass_visible=False,
        self_action_buttons_visible=True,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="left",
                placement="head",
                confidence=0.99,
                source="template:head",
            ),
        ),
    )

    assert orchestrator._apply_visual_placements(fast) == ()
    events = orchestrator._apply_visual_placements(fast)

    assert len(events) == 1
    assert events[0].event_type == "player_finished"
    assert events[0].actor == "left"
    assert events[0].payload["placement"] == "head"
    assert orchestrator.snapshot.remaining_cards["left"] == 0

    orchestrator.commit_trusted_action(
        actor="self",
        cards=(),
        is_pass=True,
        monotonic_ms=2,
    )
    orchestrator.commit_trusted_action(
        actor="right",
        cards=(),
        is_pass=True,
        monotonic_ms=3,
    )
    update = orchestrator.commit_trusted_action(
        actor="opposite",
        cards=(),
        is_pass=True,
        monotonic_ms=4,
    )

    assert orchestrator.snapshot.current_player == "right"
    assert any(
        event.event_type == "wind_caught"
        and event.payload == {"from_player": "left", "to_player": "right"}
        for event in update.events
    )
    orchestrator.finish()


def test_visual_second_teammate_ends_round_without_turning_to_finished_head(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S")],
        lead_player="right",
    )
    orchestrator.reducer.confirm_player_finished(
        "right",
        placement="head",
        source="test",
    )
    orchestrator._finish_order.append("right")
    fast = FastSignalResult(
        expected_player="opposite",
        active_player="opposite",
        pass_visible=False,
        self_action_buttons_visible=False,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="left",
                placement="second",
                confidence=0.99,
                source="template:second",
            ),
        ),
    )

    assert orchestrator._apply_visual_placements(fast) == ()
    events = orchestrator._apply_visual_placements(fast)

    assert len(events) == 1
    assert events[0].event_type == "player_finished"
    assert events[0].actor == "left"
    assert orchestrator.snapshot.current_player is None
    assert "right" in orchestrator.snapshot.finished_seats
    orchestrator.finish()


def test_current_player_badge_waits_for_final_card_path_before_fallback(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [_play("AS")],
        lead_player="self",
    )
    fast = FastSignalResult(
        expected_player="self",
        active_player="self",
        pass_visible=False,
        self_action_buttons_visible=True,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="self",
                placement="third",
                confidence=0.99,
                source="template:third",
            ),
        ),
    )
    orchestrator.reducer.confirm_player_finished(
        "left",
        placement="head",
        confidence=1.0,
        source="test",
    )
    orchestrator.reducer.confirm_player_finished(
        "opposite",
        placement="second",
        confidence=1.0,
        source="test",
    )
    orchestrator._finish_order.extend(("left", "opposite"))

    for _ in range(5):
        assert orchestrator._apply_visual_placements(fast, defer_player="self") == ()
        assert "self" not in orchestrator.snapshot.finished_seats

    events = orchestrator._apply_visual_placements(fast, defer_player="self")

    assert len(events) == 2
    assert events[0].event_type == "player_finished"
    assert events[0].actor == "self"
    assert events[0].payload["placement"] == "third"
    assert events[1].event_type == "player_finished"
    assert events[1].actor == "right"
    assert events[1].payload["placement"] == "last"
    orchestrator.finish()


def test_out_of_order_visual_placement_never_changes_game_state(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S")],
        lead_player="self",
    )
    fast = FastSignalResult(
        expected_player="self",
        active_player="self",
        pass_visible=False,
        self_action_buttons_visible=True,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="left",
                placement="third",
                confidence=0.99,
                source="template:false-third",
            ),
        ),
    )

    for _ in range(8):
        assert orchestrator._apply_visual_placements(fast) == ()

    assert orchestrator.snapshot.finished_seats == frozenset()
    assert orchestrator._finish_order == []
    assert orchestrator._placement_streaks == {}
    orchestrator.finish()


def test_post_self_finish_left_suit_probe_only_publishes_a_visual_correction(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S")])
    target = LiveEvent(
        event_id="EVT-LEFT-UNKNOWN",
        event_type="player_played",
        session_id="game",
        seq=1,
        monotonic_ms=1,
        wall_time="2026-08-10T00:00:00+08:00",
        trick_id=1,
        turn_id=1,
        actor="left",
        payload={"cards": ["3?", "3?", "4?", "4?", "5?", "5?"]},
        confidence=0.7,
        source="test",
        state_revision_before=1,
        state_revision_after=1,
    )
    probe = PlayRegionResult(
        player="left",
        cards=("3H", "3C", "4H", "4C", "5H", "5C"),
        is_pass=False,
        confidence=0.93,
        diagnostics=(),
        annotations=(),
        source="left-probe",
    )
    before = orchestrator.snapshot.semantic_dict()

    assert orchestrator._apply_left_suit_correction(target, probe) is None
    correction = orchestrator._apply_left_suit_correction(target, probe)

    assert correction is not None
    assert correction.event_type == "suit_corrected"
    assert correction.payload["target_event_id"] == target.event_id
    assert correction.payload["cards"] == ["3C", "3H", "4C", "4H", "5C", "5H"]
    assert orchestrator.snapshot.semantic_dict() == before
    orchestrator.finish()


def test_pre_self_turn_left_suit_probe_corrects_only_after_two_matching_reads(tmp_path):
    left_probe = PlayRegionResult(
        player="left",
        cards=("3S",),
        is_pass=False,
        confidence=0.95,
        diagnostics=(),
        annotations=(),
        source="left-sidecar",
    )
    self_play = PlayRegionResult(
        player="self",
        cards=("4H",),
        is_pass=False,
        confidence=0.95,
        diagnostics=(),
        annotations=(),
        source="self-play",
    )
    recognition = FakeRecognitionService([item for _ in range(8) for item in (left_probe, self_play)])
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    left_update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("3?",),
        suit_options=(("S", "C"),),
        is_pass=False,
        monotonic_ms=1,
    )
    left_event = left_update.event
    assert left_event is not None
    assert orchestrator.snapshot.current_player == "self"
    assert orchestrator._left_suit_correction_target(orchestrator.snapshot) == left_event

    frame = np.zeros((32, 64, 3), np.uint8)
    update = None
    for timestamp in range(100, 1_200, 100):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"left-sidecar-{timestamp}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.20 if timestamp == 100 else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )
        if any(event.event_type == "suit_corrected" for event in orchestrator.events):
            break

    corrections = [
        event for event in orchestrator.events if event.event_type == "suit_corrected"
    ]
    assert len(corrections) == 1
    assert corrections[0].payload == {
        "target_event_id": left_event.event_id,
        "cards": ["3S"],
        "reason": "two_frame_left_sidecar_probe",
    }
    assert update is not None
    assert any(event.event_type == "suit_corrected" for event in update.events)
    assert next(
        play for play in orchestrator.snapshot.play_history if play.player == "left"
    ).cards == ("3?",)
    assert orchestrator.snapshot.current_player in {"self", "right"}
    orchestrator.finish()


def test_left_suit_probe_resets_when_the_full_card_read_changes(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S")])
    target = LiveEvent(
        event_id="EVT-LEFT-UNKNOWN",
        event_type="player_played",
        session_id="game",
        seq=1,
        monotonic_ms=1,
        wall_time="2026-08-10T00:00:00+08:00",
        trick_id=1,
        turn_id=1,
        actor="left",
        payload={"cards": ["5?"]},
        confidence=0.7,
        source="test",
        state_revision_before=1,
        state_revision_after=1,
    )
    spade = replace(_play("5S"), player="left", source="left-probe")
    club = replace(_play("5C"), player="left", source="left-probe")

    assert orchestrator._apply_left_suit_correction(target, spade) is None
    assert orchestrator._apply_left_suit_correction(target, club) is None
    assert orchestrator._apply_left_suit_correction(target, spade) is None
    correction = orchestrator._apply_left_suit_correction(target, spade)

    assert correction is not None
    assert correction.payload["cards"] == ["5S"]
    orchestrator.finish()


def test_known_three_places_wait_for_game_end_control_without_review_or_advice(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S")])
    orchestrator.reducer._finished_seats.update({"self", "opposite", "right"})
    orchestrator.reducer._current_player = None
    orchestrator.advisor = object()

    update = orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="round-complete",
        metrics=ZoneFrameMetrics(
            monotonic_ms=100,
            occupied=False,
            motion_score=0.0,
            pass_visible=False,
            effect_visible=False,
        ),
    )

    assert update.status == "running"
    assert update.review is None
    assert not any(event.event_type == "advice_requested" for event in orchestrator.events)
    assert orchestrator.start_self_advice() is None
    orchestrator.finish()


def test_live_consensus_keeps_a_rank_when_its_suit_is_occluded(tmp_path):
    obscured = PlayRegionResult(
        player="right",
        cards=("3H", "3S", "4D", "4S", "5?", "7H"),
        suit_options=(("H",), ("S",), ("D",), ("S",), ("H", "D"), ("H",)),
        is_pass=False,
        confidence=0.94,
        diagnostics=("5 花色被遮挡，按未知花色保留",),
        annotations=(),
        source="fake:occluded-suit",
    )
    orchestrator = _orchestrator(
        tmp_path,
        [obscured] * 8,
        round_level="7",
        recognition_strategy="two_valid_streak",
    )

    _feed_visible_action(orchestrator)

    event = next(
        event for event in orchestrator.events if event.event_type == "player_played"
    )
    assert event.payload["cards"] == ["3H", "3S", "4D", "4S", "5?", "7H"]
    assert event.payload["suit_options"][4] == ["H", "D"]
    assert orchestrator.snapshot.remaining_cards["right"] == 21
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
    # The recognizer is blocked for two seconds.  Keep a generous scheduler
    # margin while still proving pause does not wait for vision/that timeout.
    assert elapsed < 0.5
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


def test_opening_self_action_buttons_do_not_identify_self_as_lead():
    signal = OpeningSignal(
        super_double_visible=False,
        marker_player=None,
        active_player=None,
        self_action_buttons_visible=True,
    )

    assert LiveOrchestrator._lead_candidate_from_opening(signal) is None


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
    def __init__(
        self,
        *,
        cards: tuple[str, ...],
        post_hand: tuple[str, ...] | None = None,
        suit_options: tuple[tuple[str, ...], ...] = (),
    ):
        super().__init__(lead_seat="self")
        self.cards = cards
        self.post_hand = post_hand
        self.suit_options = suit_options
        self.controls_visible = True
        self.active_player = None
        self.targeted_calls = 0

    def recognize_fast_signals(self, _image, expected_player):
        self.fast_calls += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=self.active_player or expected_player,
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
            suit_options=self.suit_options,
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


def test_self_lead_buttons_do_not_erase_the_opening_roi_baseline(tmp_path):
    recognition = FakeSelfLeadRecognitionService(cards=("7C",))
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    opening = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(
            opening,
            monotonic_ms=timestamp,
            wall_time=f"lead-{timestamp}",
        )
    baseline = orchestrator._baseline_by_seat["self"].copy()

    orchestrator.ingest_frame(
        np.full((32, 64, 3), 255, np.uint8),
        monotonic_ms=400,
        wall_time="buttons-visible",
    )

    assert np.array_equal(orchestrator._baseline_by_seat["self"], baseline)
    orchestrator.finish()


def test_explicit_self_lead_initializes_the_roi_baseline_only_once(tmp_path):
    recognition = FakeSelfLeadRecognitionService(cards=("7C",))
    orchestrator, _update = _lead_orchestrator(
        tmp_path,
        recognition,
        lead_player="self",
    )

    orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="first-buttons-frame",
    )
    baseline = orchestrator._baseline_by_seat["self"].copy()
    orchestrator.ingest_frame(
        np.full((32, 64, 3), 255, np.uint8),
        monotonic_ms=200,
        wall_time="later-buttons-frame",
    )

    assert np.array_equal(orchestrator._baseline_by_seat["self"], baseline)
    orchestrator.finish()


def test_self_lead_recovers_when_worker_skips_all_buttons_visible_frames(tmp_path):
    recognition = FakeSelfLeadRecognitionService(cards=("7C",))
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"lead-{timestamp}",
        )
    recognition.controls_visible = False
    recognition.active_player = "right"
    for timestamp in (400, 500, 600):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"played-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.001, False, False),
        )

    actions = [
        event
        for event in orchestrator.events
        if event.event_type in {"player_played", "player_passed"}
    ]
    assert [(event.actor, event.payload["cards"]) for event in actions] == [
        ("self", ["7C"]),
    ]
    assert orchestrator.snapshot.current_player == "right"
    orchestrator.finish()


class FakeSelfThenRightRecognitionService(FakeSelfLeadRecognitionService):
    def recognize_play_region(self, _image, seat, *, wild_rank):
        del wild_rank
        self.targeted_calls += 1
        cards = self.cards if seat == "self" else ("KS",)
        post_hand = (
            tuple(card for card in HAND if card not in set(self.cards))
            if seat == "self"
            else ()
        )
        return PlayRegionResult(
            player=seat,
            cards=cards,
            is_pass=False,
            confidence=0.95,
            diagnostics=(),
            annotations=(),
            source="fake_self_then_right",
            post_hand=post_hand,
            post_hand_confidence=0.95 if seat == "self" else 0.0,
        )


class CountingAdviceService:
    strategy_id = "counting_test"
    display_name = "计数测试建议"

    def __init__(self) -> None:
        self.calls = 0

    def recommend(self, _state, *, request_id: str):
        self.calls += 1
        raise AssertionError(f"suppressed advice must not run: {request_id}")


def test_self_first_play_handoff_captures_already_visible_right_play(tmp_path):
    recognition = FakeSelfThenRightRecognitionService(cards=("7C",))
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"lead-{timestamp}",
        )
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=400,
        wall_time="buttons-visible",
    )
    recognition.controls_visible = False
    recognition.active_player = "right"
    for timestamp in (500, 600, 700, 800, 900, 1000):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"action-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.001, False, False),
        )

    actions = [
        event
        for event in orchestrator.events
        if event.event_type in {"player_played", "player_passed"}
    ]
    assert [
        (event.actor, tuple(event.payload["cards"])) for event in actions[:2]
    ] == [("self", ("7C",)), ("right", ("KS",))]
    assert orchestrator.snapshot.current_player == "opposite"
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


def test_self_lead_active_next_seat_overrides_stale_action_buttons(tmp_path):
    cards = ("4S", "4H", "4D", "8S", "8H")
    recognition = FakeSelfLeadRecognitionService(cards=cards)
    recognition.active_player = "right"
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)
    reset_calls: list[int] = []
    original_reset = orchestrator._reset_waiting_self_lead

    def capture_reset(*args, **kwargs):
        reset_calls.append(1)
        return original_reset(*args, **kwargs)

    orchestrator._reset_waiting_self_lead = capture_reset
    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")
    for index, timestamp in enumerate((400, 500, 600, 700, 800, 900, 1000, 1100)):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"stale-buttons-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )

    actions = [event for event in orchestrator.events if event.event_type == "player_played"]
    assert reset_calls == []
    assert [(event.actor, tuple(event.payload["cards"])) for event in actions] == [
        ("self", tuple(sorted(cards))),
    ]
    assert update.snapshot.current_player == "right"
    assert len(update.snapshot.my_hand) == 22
    assert orchestrator.latest_review is None
    orchestrator.finish()


def test_visual_finish_withholds_advice_without_starting_an_advice_job(tmp_path):
    advisor = CountingAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="left",
        advisor=advisor,
    )
    fast = FastSignalResult(
        expected_player="left",
        active_player="self",
        pass_visible=False,
        self_action_buttons_visible=False,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="left",
                placement="head",
                confidence=0.99,
                source="template:head",
            ),
        ),
    )

    assert orchestrator._apply_visual_placements(fast) == ()
    assert len(orchestrator._apply_visual_placements(fast)) == 1
    assert orchestrator.snapshot.current_player == "self"
    assert orchestrator._request_advice_if_needed() is None
    assert orchestrator._request_advice_if_needed() is None

    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "withheld"
    assert advisor.calls == 0
    events = [event.event_type for event in orchestrator.events]
    assert events.count("advice_withheld") == 1
    assert "advice_requested" not in events
    records = read_json_lines(orchestrator.store.directory / "advice.jsonl")
    assert [record["status"] for record in records] == ["withheld"]
    assert records[0]["reason"] == "visual_finish_without_complete_history"
    orchestrator.finish()


def test_manual_correction_can_clear_withhold_after_reconstructing_counts(tmp_path):
    advisor = CountingAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="left",
        advisor=advisor,
    )
    # First create the exact scenario the visual fallback protects: the
    # latest known left action did not account for the finish badge.
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(
        actor="left",
        cards=("3S",),
        is_pass=False,
        monotonic_ms=1,
    )
    orchestrator.advisor = advisor
    fast = FastSignalResult(
        expected_player="self",
        active_player="self",
        pass_visible=False,
        self_action_buttons_visible=True,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="left",
                placement="head",
                confidence=0.99,
                source="template:head",
            ),
        ),
    )
    assert orchestrator._apply_visual_placements(fast) == ()
    assert len(orchestrator._apply_visual_placements(fast)) == 1
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "withheld"

    submitted: list[object] = []
    worker = orchestrator._advice_worker
    assert worker is not None
    worker.submit = submitted.append
    update = orchestrator.correct_latest(
        cards=HAND,
        is_pass=False,
        reason="manual_complete_left_history",
    )

    assert orchestrator._advice_history_is_complete(update.snapshot)
    assert update.advice is not None
    assert update.advice.status == "requested"
    assert len(submitted) == 1
    assert [
        event.event_type for event in orchestrator.events
    ].count("advice_withhold_cleared") == 1
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


def test_unknown_suit_self_play_advances_with_a_hand_constrained_card(tmp_path):
    recognition = FakeSelfLeadRecognitionService(
        cards=("7?",),
        suit_options=(("S", "C"),),
    )
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(frame, monotonic_ms=timestamp, wall_time=f"lead-{timestamp}")
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=400,
        wall_time="controls-visible",
        metrics=ZoneFrameMetrics(400, True, 0.2, False, False),
    )
    recognition.controls_visible = False
    for timestamp in (500, 600, 700, 800, 900, 1000, 1100, 1200):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"unknown-suit-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.001, False, False),
        )

    event = next(event for event in orchestrator.events if event.event_type == "player_played")
    assert event.actor == "self"
    assert event.payload["cards"] == ["7S"]
    assert orchestrator.snapshot.current_player == "right"
    assert "7S" not in orchestrator.snapshot.my_hand
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
