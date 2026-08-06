from __future__ import annotations

from dataclasses import replace
import time

import cv2
import numpy as np
import pytest

from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.replay import (
    EventReplayer,
    VideoReplaySource,
    compare_timelines,
)
from daguandan_bridge.live.models import LiveEvent


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + (
    "8S",
    "8H",
    "8C",
)


def _events():
    reducer = LiveReducer("replay-game")
    initial = reducer.confirm_initial_state(
        round_level="2",
        hand=HAND,
        lead_player="right",
    )
    right = reducer.record_play("right", ("9S",))
    opposite = reducer.record_pass("opposite")
    left = reducer.record_pass("left")
    return (
        replace(left, monotonic_ms=300, seq=4),
        replace(initial, monotonic_ms=0, seq=1),
        replace(right, monotonic_ms=100, seq=2),
        replace(opposite, monotonic_ms=200, seq=3),
    )


def test_event_replay_sorts_by_virtual_time_and_never_sleeps(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("real sleep used"))

    result = EventReplayer(lambda: LiveReducer("replay-game")).replay(_events())

    assert result.final_snapshot.current_player == "self"
    assert len(result.snapshot_hashes) == 4
    assert result.ordered_event_ids == (
        "EVT-000001",
        "EVT-000002",
        "EVT-000003",
        "EVT-000004",
    )


def test_replaying_same_events_has_identical_snapshot_hashes():
    replayer = EventReplayer(lambda: LiveReducer("replay-game"))

    first = replayer.replay(_events())
    second = replayer.replay(_events())

    assert first.snapshot_hashes == second.snapshot_hashes


def test_video_replay_uses_index_timestamps_and_reports_missing_frames(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10)
    for index in range(3):
        recorder.write_frame(
            np.full((32, 64, 3), index * 50, np.uint8),
            captured_monotonic_ms=1000 + index * 137,
            wall_time=f"t{index}",
        )
    recording = recorder.close()

    source = VideoReplaySource(recording.video_path, recording.index_path)
    decoded = list(source.frames())

    assert [record.monotonic_ms for record, _ in decoded] == [1000, 1137, 1274]
    assert source.warnings == ()

    capture = cv2.VideoCapture(str(recording.video_path))
    first_frame_ok, first_frame = capture.read()
    capture.release()
    assert first_frame_ok
    short_path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(
        str(short_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 32)
    )
    writer.write(first_frame)
    writer.release()

    short_source = VideoReplaySource(short_path, recording.index_path)
    assert len(list(short_source.frames())) == 1
    assert short_source.warnings[0].reason == "missing_video_frames"


def _timeline_event(turn_id, cards, *, confidence=0.9, monotonic_ms=100):
    return LiveEvent(
        event_id=f"EVT-{turn_id}",
        event_type="player_played",
        session_id="comparison",
        seq=turn_id,
        monotonic_ms=monotonic_ms,
        wall_time="2026-08-06T12:00:00+08:00",
        trick_id=1,
        turn_id=turn_id,
        actor="right",
        payload={"cards": list(cards), "is_pass": False},
        confidence=confidence,
        source="test",
        state_revision_before=turn_id,
        state_revision_after=turn_id + 1,
    )


def test_compare_timelines_reports_changed_turn():
    result = compare_timelines(
        [_timeline_event(3, ("7S",))],
        [_timeline_event(3, ("8S",))],
    )

    assert result.changed[0].turn_id == 3
    assert result.missing == ()
    assert result.added == ()


def test_confidence_and_latency_delta_do_not_become_semantic_change():
    result = compare_timelines(
        [_timeline_event(3, ("7S",), confidence=0.9, monotonic_ms=100)],
        [_timeline_event(3, ("7S",), confidence=0.8, monotonic_ms=160)],
    )

    assert result.changed == ()
    assert result.identical_turn_ids == (3,)
    assert result.metric_deltas[0].confidence_delta == pytest.approx(-0.1)
    assert result.metric_deltas[0].latency_delta_ms == 60
