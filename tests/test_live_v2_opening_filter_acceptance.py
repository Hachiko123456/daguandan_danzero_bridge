from __future__ import annotations

from pathlib import Path

from daguandan_bridge.application.live_v2_frame_types import FramePipelineResult
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.candidates import ActionCandidate, ActionKind, CandidateReason
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat


HAND = tuple(
    f"{rank}{suit}"
    for suit in "SHC"
    for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
)[:27]


class Store:
    session_id = "opening-filter"
    directory = Path("opening-filter-store")
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def __init__(self) -> None:
        self.batches = []
        self.advice = []

    def start(self, manifest): pass
    def append_event(self, event): pass
    def append_event_batch(self, events): self.batches.append(tuple(events))
    def append_advice(self, record): self.advice.append(record)
    def append_observation(self, record): pass
    def append_recognition_trace(self, record): pass
    def update_runtime_identity(self, identity): pass
    def upsert_decision(self, record): pass
    def create_incident(self, **kwargs): return self.directory
    def append_incident_occurrence(self, *args, **kwargs): pass
    def seal(self, **kwargs): pass
    def append_post_seal_health_audit(self, *args, **kwargs): pass
    def record_automatic_log_delivery(self, result): pass


class Recorder:
    frame_count = 0

    def write_frame(self, frame, monotonic_ms, wall_time): self.frame_count += 1
    def close(self): return RecordingResult(Path("game.avi"), Path("frames.jsonl"), self.frame_count, 0)


class Vision:
    def __init__(self): self.batches = []
    def start(self, **kwargs): pass
    def close(self, **kwargs): pass
    def queue(self, *candidates): self.batches.append(candidates)
    def process_frame(self, image, *, frame, version, wild_rank, expected_seat=None, now_ms=None, formal_action_boundary=None):
        del image, wild_rank, now_ms, formal_action_boundary
        values = []
        for index, (seat, cards) in enumerate(self.batches.pop(0) if self.batches else (), 1):
            first = FrameIdentity(frame.session_id, frame.capture_generation, frame.frame_sequence * 10 + index, frame.captured_ms - 10, frame.roi_version, frame.source_id)
            last = FrameIdentity(frame.session_id, frame.capture_generation, frame.frame_sequence * 10 + index + 1, frame.captured_ms, frame.roi_version, frame.source_id)
            values.append(ActionCandidate(
                f"candidate-{frame.frame_sequence}-{index}", version, seat,
                ActionKind.PLAY, cards, tuple((card,) for card in cards),
                (f"e-{frame.frame_sequence}-{index}-a", f"e-{frame.frame_sequence}-{index}-b"),
                version.turn_index, first, last, frame.captured_ms,
                0.9, CandidateReason.STABLE_PLAY,
            ))
        return FramePipelineResult(
            frame,
            FastSignalResult(expected_seat.value if expected_seat else None, None, False, False, False),
            (), (), tuple(values), (), (), 0,
        )


class Advice:
    def start(self, **kwargs): pass
    def close(self, **kwargs): pass


def _runtime(store, vision):
    value = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=Recorder(),
        recognition_service=object(), vision_factory=lambda version: vision,
        advice_runtime_factory=lambda version: Advice(), processing_clock_ms=lambda: 0,
        local_hint_window_ms=0,
    )
    value.start(round_level="2", hand=HAND, lead_player=None, monotonic_ms=0)
    value.bind_capture_generation(1)
    return value


def test_opening_conflicting_seat_reads_do_not_enter_normal_history() -> None:
    store, vision = Store(), Vision()
    runtime = _runtime(store, vision)
    try:
        vision.queue((Seat.SELF, ("4D", "4S")), (Seat.LEFT, ("2C",)))
        blocked = runtime.analyze_frame(object(), monotonic_ms=100)
        assert blocked.snapshot.play_history == ()
        assert blocked.snapshot.lead_player is None
        assert "opening" in blocked.block_reason

        vision.queue((Seat.LEFT, ("2C",)),)
        opened = runtime.analyze_frame(object(), monotonic_ms=200)
        assert opened.snapshot.lead_player == "left"
        assert opened.snapshot.play_history[-1].cards == ("2C",)
        assert [event.event_type for event in store.batches[-1]] == [
            "lead_player_confirmed", "player_played"
        ]
    finally:
        runtime.finish()


def test_normal_turn_commits_only_current_seat_from_same_frame_batch() -> None:
    store, vision = Store(), Vision()
    runtime = _runtime(store, vision)
    try:
        vision.queue((Seat.LEFT, ("3C",)),)
        runtime.analyze_frame(object(), monotonic_ms=100)
        vision.queue((Seat.SELF, ("4S",)), (Seat.RIGHT, ("5S",)))
        update = runtime.analyze_frame(object(), monotonic_ms=200)
        assert [event.cards for event in update.snapshot.play_history] == [("3C",), ("4S",)]
        assert update.snapshot.current_player == "right"
    finally:
        runtime.finish()
