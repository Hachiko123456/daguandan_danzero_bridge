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
    _TurnOwnershipWindow,
)
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.consensus import ConsensusResult, RecognitionSample
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


def test_confirmed_expected_roi_handoff_precedes_fast_turn_recovery():
    cards = ("10D", "10S", "QD", "QH", "QS")
    samples = [
        RecognitionSample(cards, False, 0.91, "template:cards", f"OBS-{index}")
        for index in (1, 2)
    ]
    window = _TurnOwnershipWindow(
        key=("session", 31, 32, "right"),
        expected_player="right",
        handoff_detected_ms=1_000,
        turn_recovery_pending=True,
        turn_recovery_detected_ms=1_188,
        handoff_samples=samples,
    )
    result = ConsensusResult(
        status="confirmed",
        cards=cards,
        is_pass=False,
        confidence=0.91,
        source="two_valid_streak",
        vote_count=2,
        candidates=(),
    )

    assert LiveOrchestrator._confirmed_handoff_precedes_turn_recovery(
        window,
        result,
    )
    assert not LiveOrchestrator._confirmed_handoff_precedes_turn_recovery(
        replace(window, handoff_detected_ms=None),
        result,
    )
    assert not LiveOrchestrator._confirmed_handoff_precedes_turn_recovery(
        window,
        replace(result, is_pass=True, cards=()),
    )


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


class ScheduledActiveRecognitionService(FakeRecognitionService):
    """Drive fast active-seat/effect evidence independently from ROI reads."""

    def __init__(
        self,
        samples,
        active_players,
        *,
        effect_visible=(),
        pass_marker_players=(),
    ):
        super().__init__(samples)
        self.active_players = list(active_players)
        self.effect_visible = list(effect_visible)
        self.pass_marker_players = list(pass_marker_players)
        self._active_index = 0

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        del allow_pass
        self.fast_calls += 1
        index = min(self._active_index, len(self.active_players) - 1)
        active = self.active_players[index]
        effect = (
            bool(self.effect_visible[min(index, len(self.effect_visible) - 1)])
            if self.effect_visible
            else False
        )
        pass_marker_player = (
            self.pass_marker_players[
                min(index, len(self.pass_marker_players) - 1)
            ]
            if self.pass_marker_players
            else None
        )
        self._active_index += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=active,
            pass_visible=pass_marker_player is not None,
            self_action_buttons_visible=False,
            effect_visible=effect,
            pass_marker_player=pass_marker_player,
        )


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


def _seat_play(seat: str, *cards: str) -> PlayRegionResult:
    return PlayRegionResult(
        player=seat,
        cards=cards,
        is_pass=False,
        confidence=0.94,
        diagnostics=(),
        annotations=(),
        source="scheduled-active",
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


def test_visual_opening_anchor_commits_the_already_visible_first_action(tmp_path):
    orchestrator = _orchestrator(tmp_path, [], lead_player="right")

    update = orchestrator.bootstrap_opening_action(
        actor="right",
        cards=("9S",),
        expected_next_player="opposite",
        monotonic_ms=10,
        confidence=0.91,
        source="template:right_play",
    )

    action = next(event for event in update.events if event.event_type == "player_played")
    assert action.actor == "right"
    assert action.source == "visual_opening_anchor"
    assert action.payload["cards"] == ["9S"]
    assert orchestrator.snapshot.current_player == "opposite"
    assert orchestrator._first_action_pending is False


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


def test_first_action_scans_pass_markers_but_never_treats_lead_as_pass(tmp_path):
    recognition = StrictFirstActionRecognition([_play("7S")] * 8)
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        recognition_strategy="two_valid_streak",
    )

    _feed_visible_action(orchestrator)

    assert recognition.fast_allow_pass
    assert all(recognition.fast_allow_pass)
    assert recognition.play_allow_pass
    assert not any(recognition.play_allow_pass)
    orchestrator.finish()


def test_every_new_trick_leader_scans_pass_markers_but_is_not_passable(tmp_path):
    """A wind-catch lead retains PASS evidence without becoming passable."""

    recognition = StrictFirstActionRecognition([_play("7S")] * 8)
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(
        actor="opposite", is_pass=True, monotonic_ms=20
    )
    orchestrator.commit_trusted_action(
        actor="left", is_pass=True, monotonic_ms=30
    )
    orchestrator.commit_trusted_action(
        actor="self", is_pass=True, monotonic_ms=40
    )
    assert orchestrator.snapshot.current_player == "right"
    assert not orchestrator.snapshot.trick_plays

    _feed_visible_action(orchestrator)

    assert recognition.fast_allow_pass
    assert all(recognition.fast_allow_pass)
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


def test_owner_window_quarantines_non_next_active_samples_and_enters_recovery(
    tmp_path,
):
    advisor = CountingAdviceService()
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("left", "AH", "AD", "AS", "6C", "6S")] * 4,
        ["right", "opposite", "opposite"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=10_000,
        advisor=advisor,
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    first = orchestrator.ingest_frame(
        frame,
        monotonic_ms=100,
        wall_time="foreign-active-first",
        metrics=ZoneFrameMetrics(100, True, 0.2, False, False),
    )
    second = orchestrator.ingest_frame(
        frame,
        monotonic_ms=200,
        wall_time="foreign-active-second",
        metrics=ZoneFrameMetrics(200, True, 0.001, False, False),
    )
    later = orchestrator.ingest_frame(
        frame,
        monotonic_ms=10_000,
        wall_time="foreign-active-later",
        metrics=ZoneFrameMetrics(10_000, True, 0.001, False, False),
    )

    assert first.event is None
    assert second.event is None
    assert later.event is None
    assert orchestrator.snapshot.current_player == "left"
    assert not any(
        event.event_type
        in {"player_played", "player_passed", "recognition_retry", "turn_desynchronized"}
        for event in orchestrator.events
    )
    assert advisor.calls == 0
    assert not any(
        event.event_type == "advice_requested" for event in orchestrator.events
    )
    ownership_records = read_json_lines(orchestrator.store.observations_part_path)
    ownership = ownership_records[-1]["ownership"]
    assert ownership["owner"] == "left"
    assert ownership["active"] == "opposite"
    assert ownership["window_key"][-1] == "left"
    assert ownership["owner_active_streak"] == 0
    assert ownership["turn_recovery"]["pending"] is True
    assert ownership["authenticated"] is False
    assert ownership["disposition"] == "turn_recovery"
    assert ownership["accepted_for_consensus"] is True

    def terminal_fast(_image, expected_player, *, allow_pass=True):
        del allow_pass
        return FastSignalResult(
            expected_player=expected_player,
            active_player="opposite",
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
            game_end_control="continue_game",
        )

    recognition.recognize_fast_signals = terminal_fast
    terminal = orchestrator.ingest_frame(
        frame,
        monotonic_ms=10_100,
        wall_time="foreign-active-terminal",
        metrics=ZoneFrameMetrics(10_100, False, 0.0, False, False),
    )
    assert terminal.event is not None
    assert terminal.event.event_type == "game_end_detected"
    orchestrator.finish()


def test_authenticated_owner_allows_a_brief_direct_next_handoff_to_commit(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [
            _seat_play("left"),
            _seat_play("left"),
            _seat_play("left", "2C"),
            _seat_play("left", "2C"),
            _seat_play("left", "2C"),
        ],
        ["left", "left", "self", "self"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    for index, timestamp in enumerate((100, 200, 300, 400, 500)):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"direct-handoff-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )
        if update.snapshot.current_player != "left":
            break

    assert update.snapshot.current_player == "self"
    assert any(
        event.event_type == "player_played"
        and event.actor == "left"
        and event.payload["cards"] == ["2C"]
        for event in orchestrator.events
    )
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    ownership_records = read_json_lines(orchestrator.store.observations_part_path)
    assert any(
        record.get("ownership", {}).get("disposition") == "direct_next_handoff"
        and record["ownership"].get("accepted_for_consensus") is True
        for record in ownership_records
    )
    orchestrator.finish()


def test_direct_handoff_waits_for_effect_to_settle_before_reading_and_committing(
    tmp_path,
):
    recognition = ScheduledActiveRecognitionService(
        [
            _seat_play("left"),
            _seat_play("left"),
            _seat_play("left", "2C"),
            _seat_play("left", "2C"),
        ],
        ["left", "left", "self", "self", "self", "self"],
        effect_visible=[False, False, True, True, False, False],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=10_000,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = []
    for index, timestamp in enumerate((100, 200, 300, 1_300, 2_300, 2_400)):
        updates.append(
            orchestrator.ingest_frame(
                frame,
                monotonic_ms=timestamp,
                wall_time=f"effect-handoff-{timestamp}",
                metrics=ZoneFrameMetrics(
                    timestamp,
                    True,
                    0.2 if index == 0 else (0.062 if timestamp == 2_300 else 0.001),
                    False,
                    timestamp in {300, 1_300},
                ),
            )
        )

    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert updates[-1].snapshot.current_player == "self"
    assert any(
        event.event_type == "player_played"
        and event.actor == "left"
        and event.payload["cards"] == ["2C"]
        for event in orchestrator.events
    )
    ownership_records = read_json_lines(orchestrator.store.observations_part_path)
    handoff_record = next(
        record
        for record in ownership_records
        if record["ownership"]["disposition"] == "direct_next_handoff"
    )
    assert handoff_record["ownership"]["handoff"] == {
        "detected_ms": 300,
        "global_deadline_ms": 3_300,
        "readable_since_ms": 2_300,
        "local_deadline_ms": 3_300,
        "sample_count": 1,
        "block_reason": "",
        "last_block_reason": "effect_settling",
        "deadline_kind": None,
    }
    assert handoff_record["zone"]["effect_visible"] is False
    assert handoff_record["zone"]["motion_score"] == 0.062
    orchestrator.finish()


def test_direct_handoff_unreadable_until_global_deadline_enters_recovery(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("left"), _seat_play("left")],
        ["left", "left", "self", "self"],
        effect_visible=[False, False, True, True],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=10_000,
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"global-deadline-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                index >= 2,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 3_300))
    ]

    assert updates[-1].event is None
    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.turn_recovery_pending is True
    handoff = orchestrator._handoff_telemetry(window)
    assert handoff["detected_ms"] == 300
    assert handoff["readable_since_ms"] is None
    assert handoff["global_deadline_ms"] == 3_300
    assert window.disposition == "turn_recovery_waiting_readable"
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "withheld"
    assert not any(
        event.event_type
        in {"player_played", "player_passed", "recognition_retry", "turn_desynchronized"}
        for event in orchestrator.events
    )
    orchestrator.finish()


def test_direct_handoff_global_deadline_starts_recovery_even_with_short_zone_timeout(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("left"), _seat_play("left")],
        ["left", "left", "self", "self"],
        effect_visible=[False, False, True, True],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=1_000,
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"bounded-global-deadline-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                index >= 2,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 1_000))
    ]

    assert updates[-1].event is None
    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.turn_recovery_pending is True
    handoff = orchestrator._handoff_telemetry(window)
    assert handoff["detected_ms"] == 300
    assert handoff["global_deadline_ms"] == 1_000
    orchestrator.finish()


def test_direct_handoff_crossing_keeps_two_frame_expected_play_before_recovery(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("left"), _seat_play("left")] + [_seat_play("left", "2C")] * 8,
        ["left", "left", "self", "right"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=10_000,
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"seat-cross-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 400))
    ]

    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_played"
    assert updates[-1].event.actor == "left"
    assert updates[-1].event.payload["cards"] == ["2C"]
    assert orchestrator.snapshot.current_player == "self"
    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.expected_player == "self"
    assert not any(event.event_type == "player_passed" for event in orchestrator.events)
    traces = read_json_lines(orchestrator.store.recognition_trace_path)
    assert any(
        row.get("outcome") == "confirmed_handoff_before_turn_recovery"
        for row in traces
    )
    orchestrator.finish()


def test_crossed_handoff_recovers_play_that_was_hidden_by_effect_until_next_pass(
    tmp_path, monkeypatch,
):
    """A quick right-play + opposite-PASS must not become a permanent gap."""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("right", "4S")] * 2 + [_seat_play("left")] * 10,
        ["opposite", "opposite", "left", "left", "left", "left", "left"],
        effect_visible=[True, True, False, False, False, False, False],
        pass_marker_players=[None, None, "opposite", "opposite", "opposite", "opposite", "opposite"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    assert orchestrator.snapshot.current_player == "right"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"crossed-effect-recovery-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                index >= 2,
                index < 2,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 800, 1_000, 1_200, 1_400))
    ]

    recovered = [
        event
        for update in updates
        for event in update.events
        if event.event_type == "player_played" and event.actor == "right"
    ]
    assert recovered
    assert recovered[-1].payload["cards"] == ["4S"]
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    orchestrator.finish()


def test_direct_handoff_keeps_recovering_when_active_player_is_unknown(
    tmp_path,
):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("left"), _seat_play("left")],
        ["left", "left", "self", None, None, None],
        effect_visible=[False, False, True, True, True, True],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=10_000,
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"unknown-active-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                index >= 2,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 1_000, 2_000, 3_300))
    ]

    assert all(update.event is None for update in updates)
    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.turn_recovery_pending is True
    handoff = orchestrator._handoff_telemetry(window)
    assert handoff["detected_ms"] == 300
    assert handoff["global_deadline_ms"] == 3_300
    assert handoff["readable_since_ms"] is None
    orchestrator.finish()


def test_direct_handoff_without_two_expected_roi_reads_keeps_recovering(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("left")] * 4,
        ["left", "left", "self", "self"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=10_000,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"direct-handoff-empty-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 1_400))
    ]

    assert updates[-1].event is None
    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.turn_recovery_pending is True
    assert window.disposition == "turn_recovery"
    assert orchestrator.snapshot.current_player == "left"
    assert not any(
        event.event_type
        in {"player_played", "player_passed", "recognition_retry", "turn_desynchronized"}
        for event in orchestrator.events
    )
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "withheld"
    orchestrator.finish()


def test_local_handoff_deadline_replays_only_retained_direct_samples(tmp_path, monkeypatch):
    recognition = ScheduledActiveRecognitionService(
        [
            _seat_play("left"),
            _seat_play("left"),
            _seat_play("left", "2C"),
            _seat_play("left", "2C"),
        ],
        ["left", "left", "self", "self", "self"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        action_timeout_ms=10_000,
        recognition_strategy="two_valid_streak",
    )
    monkeypatch.setattr(orchestrator, "_decide_if_ready", lambda *_args: None)
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"local-fallback-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 400, 1_400))
    ]

    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_played"
    assert updates[-1].event.actor == "left"
    assert updates[-1].event.payload["cards"] == ["2C"]
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    traces = read_json_lines(orchestrator.store.recognition_trace_path)
    fallback = next(item for item in traces if item["outcome"] == "local_deadline_fallback")
    assert fallback["fallback"] is True
    assert fallback["deadline_kind"] == "local"
    assert fallback["strategy"]["status"] == "confirmed"
    assert fallback["commit_attempted"] is True
    assert len(fallback["observation_refs"]) == 2
    assert any(
        item["outcome"] == "committed"
        and item["fallback"] is True
        and item["commit_event_id"] == updates[-1].event.event_id
        for item in traces
    )
    manifest = json.loads(orchestrator.store.manifest_path.read_text("utf-8"))
    assert manifest["runtime_identity"]["implementation_fingerprint"]
    assert manifest["runtime_identity"]["executable_path"]
    orchestrator.finish()


def test_unseen_direct_next_accepts_only_a_fresh_expected_pass_marker(tmp_path):
    """A new left turn may recover one pass before its own timer is observed."""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 2 + [_seat_play("left")] * 10,
        ["opposite", "opposite", "left", "self", "self"],
        # This reproduces the target replay: a zero-sample provisional left
        # frame observes no marker, then the timer is already at self.  The
        # following left marker is the fresh edge, not a stale baseline.
        pass_marker_players=[None, None, None, "left", "left"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="opposite",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = []
    for index, timestamp in enumerate((100, 200, 313, 400, 500)):
        pass_marker = timestamp in {400, 500}
        updates.append(
            orchestrator.ingest_frame(
                frame,
                monotonic_ms=timestamp,
                wall_time=f"unseen-pass-{timestamp}",
                metrics=ZoneFrameMetrics(
                    timestamp,
                    True,
                    0.2 if index == 0 else 0.001,
                    pass_marker,
                    False,
                ),
            )
        )

    assert updates[2].event is None
    assert updates[3].event is None
    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_passed"
    assert updates[-1].event.actor == "left"
    assert updates[-1].event.source == "unseen_direct_next_pass_marker"
    assert orchestrator.snapshot.current_player == "self"
    # Only opposite's normal action was read.  The left ROI was never offered
    # to generic card consensus during the unauthenticated recovery.
    assert recognition.targeted_calls >= 2
    traces = read_json_lines(orchestrator.store.recognition_trace_path)
    assert any(
        item["outcome"] == "unseen_direct_next_pass_pending"
        and item["unseen_direct_next_pass"]["fresh_edge_seen"] is True
        and item["unseen_direct_next_pass"]["marker_player"] == "left"
        for item in traces
    )
    assert any(
        item["outcome"] == "committed"
        and item["fallback"] is True
        and item["deadline_kind"] == "unseen_direct_next_pass"
        for item in traces
    )
    orchestrator.finish()


def test_unseen_direct_next_does_not_pass_an_already_visible_marker(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 2 + [_seat_play("left")] * 10,
        ["opposite", "opposite", "self", "self", "self", "self"],
        # This was already visible while opposite still owned the preceding
        # turn, so it is a stale baseline rather than a fresh left pass edge.
        pass_marker_players=[None, "left", "left", "left", "left", "left"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="opposite",
        settle_ms=0,
        action_timeout_ms=10_000,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = []
    for index, timestamp in enumerate((100, 200, 313, 400, 500, 3_313)):
        updates.append(
            orchestrator.ingest_frame(
                frame,
                monotonic_ms=timestamp,
                wall_time=f"stale-unseen-pass-{timestamp}",
                metrics=ZoneFrameMetrics(
                    timestamp,
                    True,
                    0.2 if index == 0 else 0.001,
                    index >= 2,
                    False,
                ),
            )
        )

    assert updates[-1].event is None
    assert orchestrator._turn_ownership_window is not None
    assert orchestrator._turn_ownership_window.turn_recovery_pending is True
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert not any(event.event_type == "player_passed" for event in orchestrator.events)
    assert recognition.targeted_calls >= 2
    orchestrator.finish()


def test_unseen_direct_next_requires_the_expected_seat_pass_marker(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 2 + [_seat_play("left")] * 10,
        ["opposite", "opposite", "self", "self", "self", "self"],
        # A visible marker attributed to another seat must not become left's
        # pass merely because the active timer has already reached self.
        pass_marker_players=[None, None, None, "right", "right", "right"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="opposite",
        settle_ms=0,
        action_timeout_ms=10_000,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"wrong-marker-unseen-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                index >= 3,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 313, 400, 500, 3_313))
    ]

    assert updates[-1].event is None
    assert orchestrator._turn_ownership_window is not None
    assert orchestrator._turn_ownership_window.turn_recovery_pending is True
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert not any(event.event_type == "player_passed" for event in orchestrator.events)
    assert recognition.targeted_calls >= 2
    orchestrator.finish()


def test_unseen_direct_next_marker_seen_during_effect_cannot_pass_later(tmp_path):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 2 + [_seat_play("left")] * 10,
        ["opposite", "opposite", "self", "self", "self", "self"],
        effect_visible=[False, False, False, True, False, False],
        pass_marker_players=[None, None, None, "left", "left", "left"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="opposite",
        settle_ms=0,
        action_timeout_ms=10_000,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"effect-unseen-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                index >= 3,
                timestamp == 400,
            ),
        )
        for index, timestamp in enumerate((100, 200, 313, 400, 900, 3_313))
    ]

    assert not any(event.event_type == "player_passed" for event in orchestrator.events)
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert recognition.targeted_calls >= 2
    orchestrator.finish()


def test_unseen_direct_next_crossing_a_non_next_active_seat_recovers_without_escalating(
    tmp_path,
):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 2 + [_seat_play("left")] * 10,
        ["opposite", "opposite", "self", "right", "left"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="opposite",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"cross-unseen-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 313, 400, 500))
    ]

    assert updates[-1].event is None
    assert orchestrator._turn_ownership_window is not None
    assert orchestrator._turn_ownership_window.turn_recovery_pending is True
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert not any(event.event_type == "player_passed" for event in orchestrator.events)
    assert recognition.targeted_calls == 2
    orchestrator.finish()


def test_unseen_direct_next_pass_recovers_the_opposite_to_left_rotation(
    tmp_path, monkeypatch
):
    """The no-card PASS recovery must work for every adjacent seat pair."""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 2,
        ["left", "left"],
        pass_marker_players=["opposite", "opposite"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    assert orchestrator.snapshot.current_player == "opposite"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )
    # The prior right-action frame had no opposite PASS marker.  The first
    # left-active frame is therefore a fresh opposite marker, not a stale one.
    orchestrator._last_pass_marker_players = frozenset()
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"opposite-left-unseen-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                True,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200))
    ]

    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_passed"
    assert updates[-1].event.actor == "opposite"
    assert updates[-1].event.source == "unseen_direct_next_pass_marker"
    assert orchestrator.snapshot.current_player == "left"
    assert recognition.targeted_calls == 0
    orchestrator.finish()


def test_authenticated_expected_pass_recovers_across_timer_gap_before_handoff(
    tmp_path,
    monkeypatch,
):
    """A fresh PASS must beat generic handoff after the owner was visible."""

    recognition = ScheduledActiveRecognitionService(
        [],
        ["opposite", "opposite", "left", None],
        pass_marker_players=[None, None, "opposite", "opposite"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    assert orchestrator.snapshot.current_player == "opposite"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )
    orchestrator._last_pass_marker_players = frozenset()
    frame = np.zeros((32, 64, 3), np.uint8)

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"authenticated-opposite-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                False,
                0.2 if index == 0 else 0.001,
                timestamp >= 300,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 400))
    ]

    assert updates[2].event is None
    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_passed"
    assert updates[-1].event.actor == "opposite"
    assert updates[-1].event.source == "unseen_direct_next_pass_marker"
    assert orchestrator.snapshot.current_player == "left"
    assert recognition.targeted_calls == 0
    orchestrator.finish()


def test_delayed_pass_marker_after_empty_handoff_recovers_without_timer_gap(
    tmp_path, monkeypatch
):
    """A PASS label may arrive one frame after its successor's timer."""

    recognition = ScheduledActiveRecognitionService(
        [],
        ["left", "left", "left"],
        effect_visible=[True, False, False],
        pass_marker_players=[None, "opposite", "opposite"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    assert orchestrator.snapshot.current_player == "opposite"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )
    orchestrator._last_pass_marker_players = frozenset()
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"delayed-opposite-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                index > 0,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300))
    ]

    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_passed"
    assert updates[-1].event.actor == "opposite"
    assert updates[-1].event.source == "unseen_direct_next_pass_marker"
    assert orchestrator.snapshot.current_player == "left"
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert recognition.targeted_calls == 0
    orchestrator.finish()


def test_turn_recovery_uses_eight_second_local_advice_target_without_global_latch(
    tmp_path,
):
    """Local controls start an advice target, not a session-wide shutdown."""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("right", "AH", "AD", "AS", "6C", "6S")] * 4,
        ["left", "left", "self", "self"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        action_timeout_ms=10_000,
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    for timestamp in (100, 200, 300, 8_300):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"local-recovery-target-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if timestamp == 100 else 0.001,
                False,
                False,
            ),
        )

    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.turn_recovery_pending is True
    assert window.turn_recovery_local_started_ms == 300
    assert window.turn_recovery_local_deadline_ms == 8_300
    assert window.turn_recovery_target_exceeded is True
    assert orchestrator._advice_suspended_reason is None
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    advice_records = read_json_lines(orchestrator.store.directory / "advice.jsonl")
    assert advice_records[-1]["outcome"] == "recovery_target_exceeded"
    assert advice_records[-1]["target_ms"] == 8_000
    orchestrator.finish()


def test_unseen_direct_next_pass_recovers_marker_that_appears_while_timer_unknown(
    tmp_path, monkeypatch,
):
    """A transition-frame timer gap must not turn a fresh PASS label stale."""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("self", "6S")] * 4,
        ["self", None, "right", "right"],
        pass_marker_players=[None, "self", "self", "self"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    for actor, cards, is_pass, timestamp in (
        ("right", ("3S",), False, 10),
        ("opposite", (), True, 20),
        ("left", (), True, 30),
    ):
        orchestrator.commit_trusted_action(
            actor=actor,
            cards=cards,
            is_pass=is_pass,
            monotonic_ms=timestamp,
        )
    assert orchestrator.snapshot.current_player == "self"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"unknown-timer-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                True,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300, 400))
    ]

    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_passed"
    assert updates[-1].event.actor == "self"
    assert updates[-1].event.source == "unseen_direct_next_pass_marker"
    assert orchestrator.snapshot.current_player == "right"
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    orchestrator.finish()


def test_expected_play_remains_eligible_when_active_badge_is_temporarily_unknown(
    tmp_path,
    monkeypatch,
):
    """A missing timer must not discard two valid cards in its own ROI."""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("right", "6S")] * 4,
        [None, None, None, None],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    assert orchestrator.snapshot.current_player == "right"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"unknown-timer-play-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                False,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200, 300))
    ]

    committed = next(
        event
        for update in updates
        for event in update.events
        if event.event_type == "player_played"
    )
    assert committed.actor == "right"
    assert committed.payload["cards"] == ["6S"]
    assert orchestrator.snapshot.current_player == "opposite"
    orchestrator.finish()


def test_wind_catch_recovers_final_opponent_pass_after_timer_already_reaches_partner(
    tmp_path, monkeypatch,
):
    """接风方可先亮计时器，但只能补录有两帧座位归属的最后 PASS。"""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 3,
        ["right", "right", "right"],
        pass_marker_players=["opposite", "opposite", "opposite"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)

    # Right leads; left wins and empties their hand.  All remaining seats,
    # including left's future-wind partner right, must still PASS before the
    # final opposite PASS catches wind.
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(
        actor="opposite", is_pass=True, monotonic_ms=20
    )
    orchestrator.commit_trusted_action(
        actor="left", cards=HAND, is_pass=False, monotonic_ms=30
    )
    orchestrator.commit_trusted_action(
        actor="self", is_pass=True, monotonic_ms=40
    )
    orchestrator.commit_trusted_action(
        actor="right", is_pass=True, monotonic_ms=50
    )
    assert orchestrator.snapshot.current_player == "opposite"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"wind-catch-pass-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.2 if index == 0 else 0.001,
                True,
                False,
            ),
        )
        for index, timestamp in enumerate((100, 200))
    ]

    assert updates[-1].event is not None
    assert updates[-1].event.event_type == "player_passed"
    assert updates[-1].event.actor == "opposite"
    assert updates[-1].event.source == "wind_catch_pass_marker"
    assert orchestrator.snapshot.current_player == "right"
    assert orchestrator.snapshot.lead_player == "right"
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert any(
        event.event_type == "wind_caught"
        and event.payload == {"from_player": "left", "to_player": "right"}
        for event in updates[-1].events
    )
    assert recognition.targeted_calls == 0
    traces = read_json_lines(orchestrator.store.recognition_trace_path)
    assert any(
        item["outcome"] == "committed"
        and item["deadline_kind"] == "wind_catch_pass"
        and item["strategy"]["is_pass"] is True
        for item in traces
    )
    orchestrator.finish()


def test_wind_catch_does_not_recover_without_the_expected_pass_marker(
    tmp_path, monkeypatch
):
    recognition = ScheduledActiveRecognitionService(
        [_seat_play("opposite", "6S")] * 3,
        ["right", "right", "right"],
        pass_marker_players=["self", "self", "self"],
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        action_timeout_ms=10_000,
        recognition_strategy="two_valid_streak",
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    for actor, cards, is_pass, timestamp in (
        ("right", ("3S",), False, 10),
        ("opposite", (), True, 20),
        ("left", HAND, False, 30),
        ("self", (), True, 40),
        ("right", (), True, 50),
    ):
        orchestrator.commit_trusted_action(
            actor=actor,
            cards=cards,
            is_pass=is_pass,
            monotonic_ms=timestamp,
        )
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )

    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"wind-catch-wrong-marker-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.001, True, False),
        )
        for timestamp in (100, 200, 3_100)
    ]

    assert updates[-1].event is None
    assert orchestrator._turn_ownership_window is not None
    assert orchestrator._turn_ownership_window.turn_recovery_pending is True
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    assert not any(
        event.event_type == "player_passed"
        and event.actor == "opposite"
        and event.source == "wind_catch_pass_marker"
        for event in orchestrator.events
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
        active_player=None,
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

    # With no unresolved action evidence, the normal two-frame visual
    # fallback remains available for a genuinely stalled terminal screen.
    assert orchestrator._apply_visual_placements(
        fast,
        defer_player="opposite",
    ) == ()
    events = orchestrator._apply_visual_placements(
        fast,
        defer_player="opposite",
    )

    assert events[0].event_type == "player_finished"
    assert events[0].actor == "left"
    assert orchestrator.snapshot.current_player is None
    assert "right" in orchestrator.snapshot.finished_seats
    assert not any(
        event.event_type == "advice_withheld" for event in orchestrator.events
    )
    terminal_gap = next(
        event
        for event in orchestrator.events
        if event.event_type == "terminal_history_gap"
    )
    assert terminal_gap.payload["mismatches"]["right"] == {
        "state_remaining_cards": 0,
        "reconstructed_remaining_cards": 27,
    }
    orchestrator.finish()


def test_visual_second_badge_waits_for_expected_pass_then_following_play(tmp_path):
    """A teammate rank badge must not terminate an actionable foreign turn."""

    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S")],
        lead_player="right",
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=1
    )
    orchestrator.commit_trusted_action(
        actor="opposite", cards=(), is_pass=True, monotonic_ms=2
    )
    orchestrator.commit_trusted_action(
        actor="left", cards=("5S",), is_pass=False, monotonic_ms=3
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("6S",), is_pass=False, monotonic_ms=4
    )
    assert orchestrator.snapshot.current_player == "right"
    orchestrator.reducer.confirm_player_finished(
        "right",
        placement="head",
        source="test",
    )
    orchestrator._finish_order.append("right")
    assert orchestrator.snapshot.current_player == "opposite"

    second_badge = FastSignalResult(
        expected_player="opposite",
        active_player="opposite",
        pass_visible=True,
        self_action_buttons_visible=False,
        effect_visible=False,
        pass_marker_player="opposite",
        pass_marker_players=("opposite",),
        placements=(
            PlacementSignal(
                player="left",
                placement="second",
                confidence=0.99,
                source="template:second",
            ),
        ),
    )
    assert orchestrator._apply_visual_placements(
        second_badge,
        defer_player="opposite",
    ) == ()
    assert orchestrator._apply_visual_placements(
        second_badge,
        defer_player="opposite",
    ) == ()
    assert orchestrator.snapshot.current_player == "opposite"
    assert "left" not in orchestrator.snapshot.finished_seats

    # An unresolved expected-seat PASS recovery, not a timer grace, owns the
    # visual fallback.  A blank fast frame must still keep its two-marker
    # decision window alive and cannot turn the rank badge terminal.
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None
    window.unseen_direct_next_pass_pending = True
    no_surface_badge = replace(
        second_badge,
        active_player=None,
        pass_visible=False,
        pass_marker_player=None,
        pass_marker_players=(),
    )
    assert orchestrator._apply_visual_placements(
        no_surface_badge,
        defer_player="opposite",
    ) == ()
    assert orchestrator.snapshot.current_player == "opposite"
    assert "left" not in orchestrator.snapshot.finished_seats

    pass_update = orchestrator.commit_trusted_action(
        actor="opposite", cards=(), is_pass=True, monotonic_ms=5
    )
    assert pass_update.event is not None
    assert pass_update.event.actor == "opposite"
    assert orchestrator.snapshot.current_player == "left"

    left_action_badge = replace(
        second_badge,
        expected_player="left",
        active_player="left",
        pass_visible=False,
        pass_marker_player=None,
        pass_marker_players=(),
    )
    assert orchestrator._apply_visual_placements(
        left_action_badge,
        defer_player="left",
    ) == ()
    play_update = orchestrator.commit_trusted_action(
        actor="left", cards=("7S",), is_pass=False, monotonic_ms=6
    )
    assert play_update.event is not None
    assert play_update.event.actor == "left"
    assert not any(
        event.event_type == "terminal_history_gap" for event in orchestrator.events
    )
    orchestrator.finish()


def test_current_player_badge_never_preempts_the_final_card_path(tmp_path):
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

    assert orchestrator._apply_visual_placements(fast, defer_player="self") == ()
    assert "self" not in orchestrator.snapshot.finished_seats
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


def test_post_self_finish_right_suit_probe_only_publishes_a_visual_correction(tmp_path):
    orchestrator = _orchestrator(tmp_path, [_play("7S")])
    target = LiveEvent(
        event_id="EVT-RIGHT-UNKNOWN",
        event_type="player_played",
        session_id="game",
        seq=1,
        monotonic_ms=1,
        wall_time="2026-08-10T00:00:00+08:00",
        trick_id=1,
        turn_id=1,
        actor="right",
        payload={"cards": ["3?", "3?", "4?", "4?", "5?", "5?"]},
        confidence=0.7,
        source="test",
        state_revision_before=1,
        state_revision_after=1,
    )
    probe = PlayRegionResult(
        player="right",
        cards=("3H", "3C", "4H", "4C", "5H", "5C"),
        is_pass=False,
        confidence=0.93,
        diagnostics=(),
        annotations=(),
        source="right-probe",
    )
    before = orchestrator.snapshot.semantic_dict()

    assert orchestrator._apply_suit_correction(target, probe) is None
    correction = orchestrator._apply_suit_correction(target, probe)

    assert correction is not None
    assert correction.event_type == "suit_corrected"
    assert correction.payload["target_event_id"] == target.event_id
    assert correction.payload["cards"] == ["3C", "3H", "4C", "4H", "5C", "5H"]
    assert orchestrator.snapshot.semantic_dict() == before
    orchestrator.finish()


def test_pre_self_turn_suit_probe_corrects_only_after_two_matching_reads(tmp_path):
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
    assert orchestrator._suit_correction_target(orchestrator.snapshot) == left_event

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
        "reason": "two_frame_sidecar_probe",
    }
    assert update is not None
    assert any(event.event_type == "suit_corrected" for event in update.events)
    assert next(
        play for play in orchestrator.snapshot.play_history if play.player == "left"
    ).cards == ("3?",)
    assert orchestrator.snapshot.current_player in {"self", "right"}
    orchestrator.finish()


def test_suit_probe_resets_when_the_full_card_read_changes(tmp_path):
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

    assert orchestrator._apply_suit_correction(target, spade) is None
    assert orchestrator._apply_suit_correction(target, club) is None
    assert orchestrator._apply_suit_correction(target, spade) is None
    correction = orchestrator._apply_suit_correction(target, spade)

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


def test_latest_only_worker_reports_discard_reasons_stats_and_wait_idle():
    first_started = threading.Event()
    release = threading.Event()
    discarded: list[tuple[int, str]] = []

    def operation(value: int) -> int:
        if value == 1:
            first_started.set()
            assert release.wait(2)
        return value

    worker = LatestOnlyWorker(
        operation,
        on_discard=lambda value, reason: discarded.append((value, reason)),
    )
    worker.start()
    worker.submit(1)
    assert first_started.wait(2)
    worker.submit(2)
    worker.submit(3)
    worker.submit(4, preserve=True, max_preserved=1)
    worker.submit(5, preserve=True, max_preserved=1)

    queued = worker.stats
    assert queued["submitted"] == 5
    assert queued["inflight"] == 1
    assert queued["pending_depth"] == 1
    assert queued["priority_depth"] == 1
    assert queued["latest_replaced"] == 1
    assert queued["preserved_evicted"] == 1
    assert worker.wait_idle(0.01) is False

    release.set()
    assert worker.wait_idle(2)
    assert worker.stop(timeout=2)
    assert discarded == [(2, "latest_replaced"), (4, "preserved_evicted")]
    final = worker.stats
    assert final["started"] == 3
    assert final["completed"] == 3
    assert final["failed"] == 0
    assert final["max_depth"] == 2
    assert final["inflight"] == 0


def test_latest_only_worker_attributes_pending_items_discarded_by_stop():
    started = threading.Event()
    release = threading.Event()
    discarded: list[tuple[int, str]] = []

    def operation(value: int) -> int:
        started.set()
        assert release.wait(2)
        return value

    worker = LatestOnlyWorker(
        operation,
        on_discard=lambda value, reason: discarded.append((value, reason)),
    )
    worker.start()
    worker.submit(1)
    assert started.wait(2)
    worker.submit(2)
    worker.submit(3, preserve=True, max_preserved=2)

    assert worker.stop(timeout=0.01) is False
    assert discarded == [
        (2, "stop_discarded"),
        (3, "stop_discarded"),
        (1, "stop_discarded"),
    ]
    assert worker.stats["stop_discarded"] == 3
    release.set()
    assert worker.stop(timeout=2)


def test_stopping_inflight_advice_persists_cancelled_terminal_and_signals(tmp_path):
    class BlockingAdvisor:
        strategy_id = "blocking"
        display_name = "阻塞建议"

        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def recommend(self, _state, *, request_id):
            del request_id
            self.started.set()
            assert self.release.wait(2)
            raise RuntimeError("released after cancellation")

    advisor = BlockingAdvisor()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="self",
        advisor=advisor,
    )
    assert advisor.started.wait(2)

    finished = threading.Event()

    def finish():
        orchestrator.finish()
        finished.set()

    thread = threading.Thread(target=finish)
    thread.start()
    deadline = time.monotonic() + 2
    statuses: list[str] = []
    while time.monotonic() < deadline:
        statuses = [
            str(row.get("status"))
            for row in read_json_lines(orchestrator.store.advice_path)
        ]
        if "cancelled" in statuses:
            break
        time.sleep(0.01)

    assert statuses[0] == "requested"
    assert "worker_started" in statuses
    assert statuses[-1] == "cancelled"
    assert any(event.event_type == "advice_cancelled" for event in orchestrator.events)
    decisions = read_json_lines(orchestrator.store.decisions_path)
    assert decisions[-1]["status"] == "cancelled"
    advisor.release.set()
    thread.join(2)
    assert finished.is_set()


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


class FakeOpeningDirectHandoffRecognitionService(FakeLeadRecognitionService):
    """Keep the opening marker on the lead while the live timer is next."""

    def __init__(self, cards: tuple[str, ...]):
        super().__init__(lead_seat="right")
        self.cards = cards
        self.targeted_calls = 0

    def recognize_fast_signals(self, _image, expected_player):
        self.fast_calls += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player="opposite",
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
            source="fake_opening_direct_handoff",
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


def _open_adjacent_action_reread(tmp_path, cards=("2H",)):
    advisor = CountingAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="opposite",
        advisor=advisor,
    )
    submitted: list[object] = []
    worker = orchestrator._advice_worker
    assert worker is not None
    worker.submit = submitted.append

    played = orchestrator.commit_trusted_action(
        actor="opposite",
        cards=tuple(cards),
        is_pass=False,
        monotonic_ms=10,
    )
    assert played.event is not None
    assert orchestrator._previous_action_verification_target(orchestrator.snapshot) is None
    followed = orchestrator.commit_trusted_action(
        actor="left",
        is_pass=True,
        monotonic_ms=20,
    )
    target = orchestrator._previous_action_verification_target(orchestrator.snapshot)
    assert target is not None
    assert target.target.event_id == played.event.event_id
    assert target.followup_event_id == followed.event.event_id
    return orchestrator, target, advisor, submitted


def test_adjacent_reread_opens_only_after_next_actor_and_withholds_self_advice(tmp_path):
    orchestrator, target, advisor, submitted = _open_adjacent_action_reread(tmp_path)

    assert target.expected_followup_actor == "left"
    assert orchestrator.snapshot.current_player == "self"
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "withheld"
    assert orchestrator.latest_advice.error == "上一手牌面待复核，暂停推荐"
    assert advisor.calls == 0
    assert submitted == []
    assert [event.event_type for event in orchestrator.events].count("advice_withheld") == 1
    orchestrator.finish()


def test_adjacent_reread_expands_single_to_222333_after_two_distinct_frames(tmp_path):
    orchestrator, target, _advisor, submitted = _open_adjacent_action_reread(tmp_path)
    reread = _seat_play("opposite", "2H", "2D", "2C", "3H", "3C", "3S")

    assert orchestrator._apply_previous_action_correction(
        target, reread, monotonic_ms=100
    ) is None
    # A duplicate analysis of the same capture must not become a second vote.
    assert orchestrator._apply_previous_action_correction(
        target, reread, monotonic_ms=100
    ) is None
    correction = orchestrator._apply_previous_action_correction(
        target, reread, monotonic_ms=200
    )

    assert correction is not None
    assert correction.event_type == "event_correction"
    assert correction.payload["target_event_id"] == target.target.event_id
    assert correction.payload["cards"] == ["2C", "2D", "2H", "3C", "3H", "3S"]
    assert orchestrator.snapshot.current_player == "self"
    assert orchestrator.snapshot.play_history[0].cards == (
        "2C", "2D", "2H", "3C", "3H", "3S"
    )
    assert len(submitted) == 1
    orchestrator.finish()


def test_adjacent_reread_never_rewrites_history_on_invalid_or_unconfirmed_read(tmp_path):
    orchestrator, target, advisor, submitted = _open_adjacent_action_reread(tmp_path)
    invalid = _seat_play("opposite", "3H")

    assert orchestrator._apply_previous_action_correction(
        target, invalid, monotonic_ms=100
    ) is None
    assert orchestrator._apply_previous_action_correction(
        target, invalid, monotonic_ms=200
    ) is None

    assert target.state == "open"
    assert orchestrator.snapshot.play_history[0].cards == ("2H",)
    assert not any(event.event_type == "event_correction" for event in orchestrator.events)
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "withheld"
    assert advisor.calls == 0
    assert submitted == []
    orchestrator.finish()


def test_adjacent_reread_accepts_two_candidate_compatible_7777_reads(tmp_path):
    original = ("7H", "7D", "7D", "7C")
    orchestrator, target, _advisor, submitted = _open_adjacent_action_reread(
        tmp_path,
        original,
    )
    stored_before = orchestrator.snapshot.play_history[0].cards
    reread = PlayRegionResult(
        player="opposite",
        cards=("7?", "7D", "7D", "7C"),
        is_pass=False,
        confidence=0.87,
        diagnostics=("first_suit_occluded_by_button",),
        annotations=(),
        source="test_occluded_button",
        suit_options=(("H", "D"), ("D",), ("D",), ("C",)),
    )

    assert orchestrator._apply_previous_action_correction(
        target,
        reread,
        monotonic_ms=100,
    ) is None
    # Reusing one captured frame is not a second read.
    assert orchestrator._apply_previous_action_correction(
        target,
        reread,
        monotonic_ms=100,
    ) is None
    verified = orchestrator._apply_previous_action_correction(
        target,
        reread,
        monotonic_ms=200,
    )

    assert verified is not None
    assert verified.event_type == "previous_action_verified"
    assert verified.payload["reason"] == "two_distinct_candidate_compatible_rereads"
    assert verified.payload["original_cards_preserved"] is True
    assert orchestrator.snapshot.play_history[0].cards == stored_before
    assert not any(event.event_type == "event_correction" for event in orchestrator.events)
    assert len(submitted) == 1
    orchestrator.finish()


def test_adjacent_reread_timeout_preserves_history_emits_once_and_resumes_advice(
    tmp_path,
):
    orchestrator, target, _advisor, submitted = _open_adjacent_action_reread(tmp_path)
    stored_before = orchestrator.snapshot.play_history[0].cards

    assert target.opened_monotonic_ms == 20
    assert orchestrator._apply_previous_action_correction(
        target,
        None,
        monotonic_ms=1_219,
    ) is None
    expired = orchestrator._apply_previous_action_correction(
        target,
        None,
        monotonic_ms=1_220,
    )
    repeated = orchestrator._apply_previous_action_correction(
        target,
        None,
        monotonic_ms=1_300,
    )

    assert expired is not None
    assert expired.event_type == "previous_action_verification_expired"
    assert expired.payload["reason"] == "reread_timeout_original_preserved"
    assert expired.payload["timeout_ms"] == 1_200
    assert repeated is None
    assert target.state == "expired"
    assert orchestrator.snapshot.play_history[0].cards == stored_before
    assert [event.event_type for event in orchestrator.events].count(
        "previous_action_verification_expired"
    ) == 1
    assert len(submitted) == 1
    orchestrator.finish()


def test_adjacent_reread_confirmation_wins_at_timeout_boundary(tmp_path):
    orchestrator, target, _advisor, submitted = _open_adjacent_action_reread(tmp_path)
    reread = _seat_play("opposite", "2H")

    assert orchestrator._apply_previous_action_correction(
        target,
        reread,
        monotonic_ms=100,
    ) is None
    verified = orchestrator._apply_previous_action_correction(
        target,
        reread,
        monotonic_ms=1_220,
    )

    assert verified is not None
    assert verified.event_type == "previous_action_verified"
    assert target.state == "confirmed"
    assert not any(
        event.event_type == "previous_action_verification_expired"
        for event in orchestrator.events
    )
    assert len(submitted) == 1
    orchestrator.finish()


def test_next_formal_action_explicitly_retires_open_reread_without_rewriting_history(
    tmp_path,
):
    orchestrator, target, _advisor, submitted = _open_adjacent_action_reread(tmp_path)
    original_cards = orchestrator.snapshot.play_history[0].cards

    update = orchestrator.commit_trusted_action(
        actor="self",
        cards=("3S",),
        is_pass=False,
        monotonic_ms=1_082,
    )
    retirement = next(
        event
        for event in update.events
        if event.event_type == "previous_action_verification_expired"
        and event.payload.get("target_event_id") == target.target.event_id
    )

    assert update.event is not None
    assert update.event.event_type == "player_played"
    assert update.event.actor == "self"
    assert update.events.index(update.event) < update.events.index(retirement)
    assert retirement.payload["reason"] == "next_formal_action_original_preserved"
    assert retirement.payload["retirement_action_event_id"] == update.event.event_id
    assert retirement.payload["elapsed_ms"] == 1_062
    assert retirement.payload["original_cards_preserved"] is True
    assert target.state == "expired"
    assert orchestrator.snapshot.play_history[0].cards == original_cards
    assert orchestrator.snapshot.play_history[-1].cards == ("3S",)
    assert orchestrator.snapshot.current_player == "right"
    assert submitted == []

    # Complete the normal trick.  The old target must never emit twice, and
    # reaching self again must schedule advice through the ordinary path.
    orchestrator.commit_trusted_action(
        actor="right",
        is_pass=True,
        monotonic_ms=1_100,
    )
    orchestrator.commit_trusted_action(
        actor="opposite",
        is_pass=True,
        monotonic_ms=1_120,
    )
    orchestrator.commit_trusted_action(
        actor="left",
        is_pass=True,
        monotonic_ms=1_140,
    )

    matching_retirements = [
        event
        for event in orchestrator.events
        if event.event_type == "previous_action_verification_expired"
        and event.payload.get("target_event_id") == target.target.event_id
    ]
    assert len(matching_retirements) == 1
    assert orchestrator.snapshot.current_player == "self"
    assert len(submitted) == 1
    orchestrator.finish()


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
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
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


def test_terminal_visual_finish_clears_an_earlier_midgame_withhold(tmp_path):
    """A prior visual gap cannot remain an active stop after double-down."""

    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="right",
    )
    head = FastSignalResult(
        expected_player="right",
        active_player="opposite",
        pass_visible=False,
        self_action_buttons_visible=False,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="right",
                placement="head",
                confidence=0.99,
                source="template:head",
            ),
        ),
    )
    assert orchestrator._apply_visual_placements(head) == ()
    assert len(orchestrator._apply_visual_placements(head)) == 1
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "withheld"

    second = FastSignalResult(
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
    assert orchestrator._apply_visual_placements(second) == ()
    events = orchestrator._apply_visual_placements(second)

    assert orchestrator.snapshot.current_player is None
    assert orchestrator.latest_advice is None
    assert any(
        event.event_type == "advice_withhold_cleared"
        and event.payload["clear_reason"] == "round_decided"
        for event in events
    )
    assert any(event.event_type == "terminal_history_gap" for event in events)
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


def test_opening_direct_handoff_commits_two_valid_anchor_after_auto_lead(
    tmp_path,
    monkeypatch,
):
    """Rescue the latest-session shape without relaxing ordinary consensus."""

    recognition = FakeOpeningDirectHandoffRecognitionService(("7D", "7S"))
    orchestrator, _update = _lead_orchestrator(tmp_path, recognition)
    frame = np.zeros((32, 64, 3), np.uint8)

    for timestamp in (100, 200, 300):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"opening-anchor-lead-{timestamp}",
        )
    assert orchestrator.snapshot.lead_player == "right"
    assert orchestrator.snapshot.current_player == "right"

    # The normal path has priority.  Simulate its async no-result outcome so
    # this test exercises only the intentionally narrow direct-handoff anchor.
    monkeypatch.setattr(orchestrator, "_decide_if_ready", lambda *_args: None)
    for index, timestamp in enumerate((400, 500, 600)):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"opening-anchor-read-{timestamp}",
            metrics=ZoneFrameMetrics(
                timestamp,
                True,
                0.20 if index == 0 else 0.001,
                False,
                False,
            ),
        )

    assert update.event is not None
    assert update.event.event_type == "player_played"
    assert update.event.actor == "right"
    assert update.event.source == "opening_handoff_two_valid_anchor"
    assert len(update.event.evidence_refs) == 2
    assert orchestrator.snapshot.play_history[0].cards == ("7D", "7S")
    assert orchestrator.snapshot.current_player == "opposite"
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    orchestrator.finish()


def test_opening_handoff_anchor_refuses_one_direct_handoff_read(tmp_path):
    orchestrator = _orchestrator(tmp_path, [], lead_player="right")
    # Isolate the one-read guard from the separate explicit-lead exclusion.
    orchestrator._lead_auto_confirmed_from_marker = True
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None
    window.disposition = "direct_next_handoff"
    window.handoff_samples.extend(
        [
            RecognitionSample(
                cards=("7S",),
                is_pass=False,
                confidence=0.95,
                source="test",
                evidence_ref="OBS-000001",
            ),
        ]
    )

    result = orchestrator._decide_opening_handoff_anchor(
        window,
        ZoneFrameMetrics(100, True, 0.001, False, False),
        FastSignalResult(
            expected_player="right",
            active_player="opposite",
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
        ),
    )

    assert result is None
    assert orchestrator.snapshot.play_history == ()
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
