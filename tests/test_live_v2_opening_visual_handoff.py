from __future__ import annotations

from pathlib import Path

import pytest

from daguandan_bridge.application.live_v2_frame_types import FramePipelineResult
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.candidates import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
)
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat
from daguandan_bridge.live_v2.observations import (
    ObservationKind,
    ObservationReason,
    SeatObservation,
)


HAND = tuple(
    f"{rank}{suit}"
    for suit in "SHC"
    for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
)[:27]


class _Store:
    session_id = "opening-visual-production"
    directory = Path("opening-visual-production")
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def __init__(self) -> None:
        self.batches: list[tuple[object, ...]] = []

    def start(self, manifest): pass
    def append_event(self, event): pass
    def append_event_batch(self, events): self.batches.append(tuple(events))
    def append_advice(self, record): pass
    def append_observation(self, record): pass
    def append_recognition_trace(self, record): pass
    def update_runtime_identity(self, identity): pass
    def upsert_decision(self, record): pass
    def create_incident(self, **kwargs): return self.directory
    def append_incident_occurrence(self, *args, **kwargs): pass
    def seal(self, **kwargs): pass
    def append_post_seal_health_audit(self, *args, **kwargs): pass
    def record_automatic_log_delivery(self, result): pass


class _Recorder:
    frame_count = 0

    def write_frame(self, frame, monotonic_ms, wall_time): return None
    def close(self):
        return RecordingResult(Path("video.avi"), Path("frames.jsonl"), 0, 0)


class _Advice:
    def start(self, *, timeout=10.0): pass
    def submit(self, *args, **kwargs): return ()
    def drain_results(self): return ()
    def close(self, *, timeout=5.0): pass


def _fast() -> FastSignalResult:
    return FastSignalResult("left", "left", False, False, False)


class _TwoFrameOpeningVision:
    def __init__(self, kind: ObservationKind) -> None:
        self.kind = kind
        self.first: FrameIdentity | None = None
        self.calls = 0

    def start(self): pass
    def close(self): pass

    def process_frame(
        self, image, *, frame, version, wild_rank, expected_seat=None, now_ms=None,
        formal_action_boundary=None,
    ) -> FramePipelineResult:
        del image, wild_rank, expected_seat, formal_action_boundary
        self.calls += 1
        observation = self._observation(frame, int(now_ms))
        candidates: tuple[ActionCandidate, ...] = ()
        if self.first is None:
            self.first = frame
        elif self.kind in {ObservationKind.PLAY, ObservationKind.PASS}:
            kind = ActionKind.PLAY if self.kind is ObservationKind.PLAY else ActionKind.PASS
            cards = ("5H", "5C") if kind is ActionKind.PLAY else ()
            candidates = (ActionCandidate(
                candidate_id="left-opening-two-frame",
                version=version,
                seat=Seat.LEFT,
                kind=kind,
                cards=cards,
                suit_options=tuple((card,) for card in cards),
                evidence_ids=("source-frame-83-left", "source-frame-86-left"),
                action_epoch=1,
                first_frame=self.first,
                last_frame=frame,
                processing_ms=int(now_ms),
                confidence=0.99,
                reason=(
                    CandidateReason.STABLE_PLAY
                    if kind is ActionKind.PLAY
                    else CandidateReason.FRESH_PASS_EDGE
                ),
            ),)
        return FramePipelineResult(
            frame=frame,
            fast_signals=_fast(),
            surface_metrics=(),
            observations=(observation,),
            candidates=candidates,
            drops=(),
            pending_seats=(),
            candidate_backlog=0,
        )

    def _observation(self, frame: FrameIdentity, now_ms: int) -> SeatObservation:
        cards = ("5H", "5C") if self.kind is ObservationKind.PLAY else ()
        reason = {
            ObservationKind.PLAY: ObservationReason.CARDS_RECOGNIZED,
            ObservationKind.PASS: ObservationReason.PASS_MARKER,
            ObservationKind.UNKNOWN: ObservationReason.UNREADABLE,
        }[self.kind]
        return SeatObservation(
            observation_id=f"source-frame-{83 if self.calls == 1 else 86}-left",
            frame=frame,
            seat=Seat.LEFT,
            kind=self.kind,
            cards=cards,
            confidence=0.99,
            reason=reason,
            processing_ms=now_ms,
            suit_options=tuple((card,) for card in cards),
        )


def _runtime(kind: ObservationKind):
    store = _Store()
    vision = _TwoFrameOpeningVision(kind)
    clock = [0]
    runtime = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store),
        store=store,
        recorder=_Recorder(),
        recognition_service=object(),
        vision_factory=lambda version: vision,
        advice_runtime_factory=lambda version: _Advice(),
        processing_clock_ms=lambda: clock[0],
        local_hint_window_ms=0,
    )
    runtime.start(
        round_level="6", hand=HAND, lead_player=None,
        monotonic_ms=0, wall_time="2026-09-06T00:00:00+08:00",
    )
    runtime.bind_capture_generation(1)
    return runtime, store, clock


def _feed_two(runtime, clock) -> None:
    for sequence, captured_ms in ((83, 1_000), (86, 1_100)):
        clock[0] = captured_ms
        runtime.analyze_frame(
            object(),
            monotonic_ms=captured_ms,
            trace_context={
                "capture_generation": 1,
                "capture_seq": sequence,
                "captured_ms": captured_ms,
            },
        )


def test_two_stable_left_opening_frames_confirm_lead_then_commit_action() -> None:
    runtime, store, clock = _runtime(ObservationKind.PLAY)
    try:
        _feed_two(runtime, clock)

        assert runtime.snapshot.lead_player == "left"
        assert runtime.snapshot.current_player == "self"
        assert [event.event_type for batch in store.batches for event in batch] == [
            "initial_state_confirmed",
            "lead_player_confirmed",
            "player_played",
        ]
        assert [event.event_type for event in store.batches[-1]] == [
            "lead_player_confirmed", "player_played",
        ]
        play = store.batches[-1][1]
        assert set(play.payload["cards"]) == {"5H", "5C"}
        assert play.evidence_refs == (
            "source-frame-83-left",
            "source-frame-86-left",
        )
    finally:
        runtime.finish()


@pytest.mark.parametrize("kind", [ObservationKind.PASS, ObservationKind.UNKNOWN])
def test_pass_or_unknown_cannot_start_an_unresolved_opening(kind) -> None:
    runtime, store, clock = _runtime(kind)
    try:
        _feed_two(runtime, clock)

        assert runtime.snapshot.lead_player is None
        assert runtime.snapshot.play_history == ()
        assert [event.event_type for batch in store.batches for event in batch] == [
            "initial_state_confirmed"
        ]
    finally:
        runtime.finish()
