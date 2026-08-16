from __future__ import annotations

from dataclasses import replace
import json
import time

import cv2
import numpy as np
import pytest

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.replay import (
    EventReplayer,
    VideoReplaySource,
    _stable_visual_initial_state,
    compare_timelines,
    replay_truth_through_live_advisor,
    replay_video_through_live_pipeline,
)
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    save_truth_log,
)
from daguandan_bridge.recognition_service import FastSignalResult, OpeningSignal, PlayRegionResult


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + (
    "8S",
    "8H",
    "8C",
)


def test_visual_replay_initial_state_requires_two_matching_valid_frames():
    class Recognition:
        def recognize(self, _frame):
            return type(
                "Result",
                (),
                {"round_level": "8", "my_hand": HAND},
            )()

    frames = (
        np.zeros((32, 64, 3), np.uint8),
        np.ones((32, 64, 3), np.uint8),
    )

    assert _stable_visual_initial_state(Recognition(), frames) == (
        "8",
        tuple(sorted(HAND)),
    )


class TrustedReplayAdvisor:
    def __init__(self):
        self.calls = 0
        self.states = []

    def recommend(self, state, *, request_id=""):
        self.calls += 1
        self.states.append(state)
        return LocalAdvice(
            strategy="trusted-replay-test",
            cards=("2S",),
            play_type="Single",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
            engine_input={"request_id": request_id},
            timings={"test": 1.0},
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


def test_trusted_truth_replay_uses_live_advisor_path_without_video(tmp_path):
    source = tmp_path / "source-session"
    source.mkdir()
    truth = TruthLog(
        source_session_id="trusted-source",
        initial_state=TruthInitialState("2", "right", HAND),
        turns=(
            TruthTurn(
                1,
                "right",
                False,
                ("7S",),
                monotonic_ms=100,
                move_semantics={
                    "selected_interpretation": {
                        "move_type": "Single",
                        "key": "7",
                    },
                    "selection_source": "exact_engine_state",
                },
            ),
            TruthTurn(2, "opposite", True, (), monotonic_ms=200),
            TruthTurn(3, "left", True, (), monotonic_ms=300),
            TruthTurn(4, "self", False, ("8S",), monotonic_ms=400),
        ),
    )
    truth_path = source / "truth_log.json"
    save_truth_log(truth_path, truth)
    before = truth_path.read_bytes()
    advisor = TrustedReplayAdvisor()

    result = replay_truth_through_live_advisor(
        source,
        advisor,
        truth_log=truth,
        advice_timeout_sec=2.0,
    )

    summary = json.loads(result.summary_path.read_text("utf-8"))
    assert result.processed_turn_count == 4
    assert result.advice_requested == 1
    assert result.advice_ready == 1
    assert result.advice_failed == 0
    assert result.advice_stale == 0
    assert advisor.calls == 1
    assert advisor.states[0].play_history[0].action_metadata == {
        "selected_interpretation": {"move_type": "Single", "key": "7"},
        "selection_source": "exact_engine_state",
    }
    assert result.output_path.is_file()
    assert result.summary_path.is_file()
    assert summary["completed"] is True
    assert truth_path.read_bytes() == before


def test_trusted_truth_replay_rejects_wrong_actor(tmp_path):
    source = tmp_path / "source-session"
    source.mkdir()
    truth = TruthLog(
        source_session_id="trusted-source",
        initial_state=TruthInitialState("2", "right", HAND),
        turns=(TruthTurn(1, "self", False, ("8S",), monotonic_ms=100),),
    )

    with pytest.raises(ValueError, match="不能提交 self"):
        replay_truth_through_live_advisor(
            source,
            TrustedReplayAdvisor(),
            truth_log=truth,
            advice_timeout_sec=2.0,
        )


def test_trusted_truth_replay_uses_temporary_variants_for_unknown_suits(tmp_path):
    source = tmp_path / "source-session"
    source.mkdir()
    truth = TruthLog(
        source_session_id="trusted-source",
        initial_state=TruthInitialState("2", "right", HAND),
        turns=(
            TruthTurn(1, "right", False, ("7S",), monotonic_ms=100),
            TruthTurn(2, "opposite", True, (), monotonic_ms=200),
            TruthTurn(3, "left", False, ("8?",), monotonic_ms=300),
            TruthTurn(4, "self", False, ("8C",), monotonic_ms=400),
        ),
    )
    advisor = TrustedReplayAdvisor()

    result = replay_truth_through_live_advisor(
        source,
        advisor,
        truth_log=truth,
        advice_timeout_sec=2.0,
    )

    assert result.unknown_card_resolutions == 0
    assert advisor.calls == 4
    assert all(
        all("?" not in card for card in state.my_hand)
        and all("?" not in card for event in state.play_history for card in event.cards)
        for state in advisor.states
    )
    summary = json.loads(result.summary_path.read_text("utf-8"))
    assert summary["unknown_card_resolutions"] == []
    assert summary["unknown_card_policy"] == "temporary_suit_variants"


def test_persisted_timeline_round_trips_into_deterministic_replay(tmp_path):
    store = LiveSessionStore(
        tmp_path,
        "tencent_daguandan",
        session_id="replay-game",
    )
    store.start({"target_fps": 10})
    events = tuple(reversed(_events()))
    for event in events:
        store.append_event(event)
    store.seal(frame_count=0, dropped_frames=0)

    loaded = tuple(
        LiveEvent.from_dict(raw) for raw in read_json_lines(store.timeline_path)
    )
    result = EventReplayer(lambda: LiveReducer("replay-game")).replay(loaded)

    assert result.final_snapshot.current_player == "self"
    assert result.ordered_event_ids == (
        "EVT-000001",
        "EVT-000002",
        "EVT-000003",
        "EVT-000004",
    )


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


def test_video_visual_replay_runs_live_pipeline_and_compares_turns(tmp_path):
    session_store = LiveSessionStore(
        tmp_path,
        "tencent_daguandan",
        session_id="replay-game",
    )
    session_store.start({"target_fps": 10})
    initial, right, *_ = sorted(_events(), key=lambda event: event.seq)
    # Persisted opening metadata is intentionally wrong.  Pipeline replay must
    # re-enter the same visual opening state machine as a live session.
    initial = replace(
        initial,
        actor="opposite",
        payload={**initial.payload, "lead_player": "opposite"},
    )
    session_store.append_event(initial)
    session_store.append_event(right)
    recorder = SessionRecorder(session_store.directory, size=(64, 32), fps=10)
    recorder.write_frame(np.zeros((32, 64, 3), np.uint8), 0, "t0")
    # The live action gate waits one second after the visual change, then the
    # default strategy needs a second matching sample to confirm the play.
    for index in range(1, 15):
        recorder.write_frame(
            np.full((32, 64, 3), 255 if index else 0, np.uint8),
            index * 100,
            f"t{index}",
        )
    recording = recorder.close()
    session_store.seal(
        frame_count=recording.frame_count,
        dropped_frames=recording.dropped_frames,
    )

    class ScriptedRecognition:
        def recognize_opening_signal(self, _frame):
            return OpeningSignal(
                super_double_visible=False,
                marker_player="right",
                active_player="right",
                self_action_buttons_visible=False,
            )

        def recognize_fast_signals(self, _frame, expected_player):
            return FastSignalResult(
                expected_player=expected_player,
                active_player=expected_player,
                pass_visible=False,
                self_action_buttons_visible=False,
                effect_visible=False,
            )

        def recognize_play_region(self, _frame, seat, *, wild_rank):
            del wild_rank
            return PlayRegionResult(
                player=seat,
                cards=("9S",),
                is_pass=False,
                confidence=0.95,
                diagnostics=(),
                annotations=(),
                source="scripted-replay",
            )

    result = replay_video_through_live_pipeline(
        session_store.directory,
        ScriptedRecognition(),
        use_live_pipeline=True,
        sample_every_frame=True,
    )

    assert result.frame_count == 15
    assert result.comparison.identical_turn_ids == (1,)
    assert result.comparison.missing == ()
    assert result.comparison.changed == ()
    assert abs(result.comparison.metric_deltas[0].latency_delta_ms) < 2_000
    assert result.output_path.is_file()
    assert result.comparison_path.is_file()
    rows = list(read_json_lines(result.output_path))
    assert any(row.get("events") for row in rows)
    assert any(
        event.get("event_type") == "session_finalizing"
        for row in rows
        for event in row.get("events", ())
    )


def test_truth_log_video_replay_uses_manual_baseline(tmp_path):
    session_store = LiveSessionStore(
        tmp_path,
        "tencent_daguandan",
        session_id="truth-replay-game",
    )
    session_store.start({"target_fps": 10})
    recorder = SessionRecorder(session_store.directory, size=(64, 32), fps=10)
    for index in range(24):
        recorder.write_frame(
            np.full((32, 64, 3), 255 if index else 0, np.uint8),
            index * 100,
            f"t{index}",
        )
    recording = recorder.close()
    session_store.seal(
        frame_count=recording.frame_count,
        dropped_frames=recording.dropped_frames,
    )
    truth = TruthLog(
        source_session_id="truth-replay-game",
        initial_state=TruthInitialState("2", "right", HAND),
        turns=(TruthTurn(1, 1, "right", False, ("9S",), 500, 5),),
    )

    class ScriptedRecognition:
        def recognize_opening_signal(self, _frame):
            return OpeningSignal(
                super_double_visible=False,
                marker_player="right",
                active_player="right",
                self_action_buttons_visible=False,
            )

        def recognize_fast_signals(self, _frame, expected_player):
            return FastSignalResult(
                expected_player=expected_player,
                active_player=expected_player,
                pass_visible=False,
                self_action_buttons_visible=False,
                effect_visible=False,
            )

        def recognize_play_region(self, _frame, seat, *, wild_rank):
            del wild_rank
            return PlayRegionResult(
                player=seat,
                cards=("9S",),
                is_pass=False,
                confidence=0.95,
                diagnostics=(),
                annotations=(),
                source="truth-scripted-replay",
            )

    result = replay_video_through_live_pipeline(
        session_store.directory,
        ScriptedRecognition(),
        truth_log=truth,
    )

    assert result.comparison.identical_turn_ids == (1,)
    assert result.output_path.name == "truth_replay.jsonl"
    assert result.comparison_path.name == "truth_replay_comparison.json"


    original = _timeline_event(3, ("7S",))
    correction = replace(
        original,
        event_id="EVT-correction",
        event_type="event_correction",
        seq=4,
        payload={
            "target_event_id": original.event_id,
            "cards": ["8S"],
            "is_pass": False,
            "reason": "test",
        },
    )

    result = compare_timelines(
        [original, correction],
        [_timeline_event(3, ("8S",))],
    )

    assert result.identical_turn_ids == (3,)
    assert result.changed == ()
