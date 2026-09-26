from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from time import sleep

import pytest

from daguandan_bridge.application.live_v2_advice_protocol import (
    AdviceRequestIdentity, AdviceRuntimeResult, AdviceRuntimeStatus,
)
from daguandan_bridge.application.live_v2_frame_types import FramePipelineResult
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime, _resolve_visual_correction
from daguandan_bridge.application.live_v2_vision_protocol import (
    VisionRequestIdentity, VisionRuntimeResult, VisionRuntimeStatus,
)
from daguandan_bridge.application.ports import LiveRuntimePort
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.candidates import (
    ActionCandidate, ActionKind, CandidateReason, EvidenceOrigin,
)
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat
from daguandan_bridge.live_v2.observations import ObservationKind, ObservationReason, SeatObservation


HAND = tuple(
    f"{rank}{suit}"
    for suit in "SHC"
    for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
)[:27]
_LIVE_RUNTIMES: list[LiveV2SessionRuntime] = []


@pytest.fixture(autouse=True)
def _close_runtime_workers():
    yield
    for runtime in _LIVE_RUNTIMES:
        if runtime.status != "sealed":
            runtime.finish()
    _LIVE_RUNTIMES.clear()


class MemoryStore:
    session_id = "runtime-session"
    directory = Path("memory-session")
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def __init__(self) -> None:
        self.started = []
        self.events = []
        self.batches = []
        self.observations = []
        self.advice = []
        self.traces = []
        self.identities = []
        self.seals = []
        self.fail_batches = False

    def start(self, manifest): self.started.append(manifest)
    def append_event(self, event): self.events.append(event)
    def append_event_batch(self, events):
        if self.fail_batches: raise OSError("disk full")
        self.batches.append(tuple(events))
    def append_advice(self, record): self.advice.append(record)
    def append_observation(self, record): self.observations.append(record)
    def append_recognition_trace(self, record): self.traces.append(record)
    def update_runtime_identity(self, identity): self.identities.append(identity)
    def upsert_decision(self, record): pass
    def create_incident(self, **kwargs): return self.directory
    def append_incident_occurrence(self, *args, **kwargs): pass
    def seal(self, **kwargs): self.seals.append(kwargs)
    def append_post_seal_health_audit(self, *args, **kwargs): pass
    def record_automatic_log_delivery(self, result): pass


class MemoryRecorder:
    frame_count = 0

    def __init__(self, *, fail_write: bool = False) -> None:
        self.fail_write = fail_write
        self.closed = False

    def write_frame(self, frame, monotonic_ms, wall_time):
        if self.fail_write: raise OSError("codec failed")
        self.frame_count += 1
        return None

    def close(self):
        self.closed = True
        return RecordingResult(Path("video.avi"), Path("frames.jsonl"), self.frame_count, 0)


class FakeRecognition:
    pass


class ManualClock:
    def __init__(self) -> None: self.value = 0
    def __call__(self) -> int: return self.value


def _fast(expected: str = "right") -> FastSignalResult:
    return FastSignalResult(expected, None, False, False, False)


class ScriptedVision:
    def __init__(self) -> None:
        self.specs: list[tuple[Seat, ActionKind, tuple[str, ...]] | None] = []
        self.closed = False
        self.calls = 0
        self.formal_action_boundaries: list[FrameIdentity | None] = []

    def queue(self, seat: Seat, kind: ActionKind, cards: tuple[str, ...] = ()) -> None:
        self.specs.append((seat, kind, cards))

    def empty(self) -> None: self.specs.append(None)

    def start(self): pass

    def process_frame(self, image, *, frame, version, wild_rank,
                      expected_seat=None, now_ms=None,
                      formal_action_boundary=None):
        del image, wild_rank
        self.calls += 1
        self.formal_action_boundaries.append(formal_action_boundary)
        spec = self.specs.pop(0) if self.specs else None
        candidates = ()
        if spec is not None:
            seat, kind, cards = spec
            first = FrameIdentity(
                frame.session_id, frame.capture_generation,
                frame.frame_sequence * 2 - 1, max(0, frame.captured_ms - 10),
                frame.roi_version, frame.source_id,
            )
            last = FrameIdentity(
                frame.session_id, frame.capture_generation,
                frame.frame_sequence * 2, frame.captured_ms,
                frame.roi_version, frame.source_id,
            )
            candidates = (ActionCandidate(
                f"visual-{frame.capture_generation}-{frame.frame_sequence}",
                version, seat, kind, cards,
                tuple((card,) for card in cards) if kind is ActionKind.PLAY else (),
                (f"e-{self.calls}-1", f"e-{self.calls}-2"), self.calls,
                first, last, int(now_ms), 0.96,
                CandidateReason.STABLE_PLAY if kind is ActionKind.PLAY
                else CandidateReason.FRESH_PASS_EDGE,
            ),)
        return FramePipelineResult(
            frame, _fast(expected_seat.value if expected_seat else "right"),
            (), (), candidates, (), (), 0,
        )

    def close(self): self.closed = True


class FakeAdviceRuntime:
    def __init__(self, *, immediate: bool = True) -> None:
        self.immediate = immediate
        self.started = False
        self.closed = False
        self.submissions = []
        self.pending = []
        self.release_on_drain = False

    def start(self, *, timeout=10.0): self.started = True

    def submit(self, snapshot, opportunity, *, request_sequence, timeout_ms=3000):
        self.submissions.append((snapshot, opportunity, request_sequence))
        identity = AdviceRequestIdentity(
            opportunity.version, request_sequence, opportunity.opportunity_id
        )
        result = AdviceRuntimeResult(
            identity, AdviceRuntimeStatus.ADVICE, 1, 101, 5.0,
            AdviceResult(
                "fake", (snapshot.my_hand[0],), "Single", False,
                snapshot.version.state_revision, 5.0,
            ),
        )
        if self.immediate: return (result,)
        self.pending.append(result)
        return ()

    def drain_results(self):
        if not self.release_on_drain: return ()
        values, self.pending = tuple(self.pending), []
        return values

    def close(self, *, timeout=5.0): self.closed = True


class DelayedVisionRuntime:
    def __init__(
        self, result_seat: Seat = Seat.RIGHT,
        result_cards: tuple[str, ...] = ("3D",),
        *, pipeline=None, include_fault: bool = False,
    ) -> None:
        self.started = False
        self.closed = False
        self.pending = None
        self.calls = 0
        self.result_seat = result_seat
        self.result_cards = result_cards
        self.pipeline = pipeline
        self.include_fault = include_fault

    def start(self, *, timeout=10.0): self.started = True

    def submit(self, image, *, frame, version, expected_seat,
               visual_self_opportunity, wild_rank, request_sequence,
               formal_action_boundary=None, timeout_ms=2000):
        del image, visual_self_opportunity, wild_rank, formal_action_boundary, timeout_ms
        self.calls += 1
        completed = ()
        if self.pending is not None:
            old_frame, old_version, old_expected, old_sequence = self.pending
            scripted = self.pipeline
            if scripted is None:
                scripted = ScriptedVision()
                scripted.queue(self.result_seat, ActionKind.PLAY, self.result_cards)
            pipeline = scripted.process_frame(
                object(), frame=old_frame, version=old_version,
                wild_rank="2", expected_seat=old_expected, now_ms=frame.captured_ms,
            )
            completed = (VisionRuntimeResult(
                VisionRequestIdentity(old_frame, old_version, old_sequence),
                VisionRuntimeStatus.FRAME, 1, 202,
                pipeline_result=pipeline,
            ),)
        self.pending = (frame, version, expected_seat, request_sequence)
        if self.include_fault:
            completed += (VisionRuntimeResult(
                VisionRequestIdentity(frame, version, request_sequence),
                VisionRuntimeStatus.WORKER_ERROR, 1, 202,
                failure_code="test_worker_fault", message="receipt fault regression",
            ),)
        return completed

    def drain_results(self): return ()
    def close(self, *, timeout=5.0): self.closed = True




class TerminalVision:
    def __init__(self, controls=("continue_game", "continue_game")) -> None:
        self.controls = list(controls)
        self.closed = False

    def start(self):
        pass

    def process_frame(
        self, image, *, frame, version, wild_rank, expected_seat=None,
        now_ms=None, formal_action_boundary=None, repair_seats=(),
    ):
        del image, version, wild_rank, now_ms, formal_action_boundary, repair_seats
        control = self.controls.pop(0) if self.controls else None
        expected = expected_seat.value if expected_seat is not None else "self"
        fast = FastSignalResult(
            expected, None, False, False, False, game_end_control=control
        )
        return FramePipelineResult(frame, fast, (), (), (), (), (), 0)

    def close(self):
        self.closed = True


def test_terminal_control_seals_history_even_when_runtime_is_in_review() -> None:
    store = MemoryStore(); recorder = MemoryRecorder()
    store.start({"schema": "test.live-v2/1"})
    clock = ManualClock(); vision = TerminalVision(); advisers = []

    def advice_factory(_version):
        value = FakeAdviceRuntime(); advisers.append(value); return value

    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=lambda _version: vision,
        advice_runtime_factory=advice_factory, processing_clock_ms=clock,
    )
    _LIVE_RUNTIMES.append(live)
    live.start(
        round_level="2", hand=HAND, lead_player="right", monotonic_ms=0,
        wall_time="2026-09-18T00:00:00+08:00",
    )
    live.bind_capture_generation(1)
    live.status = "review_required"

    clock.value = 100
    first = live.analyze_frame(object(), monotonic_ms=100)
    assert first.event is None
    # A clear engine tick may recover an artificial review status. Re-arm it
    # to prove terminal evidence is accepted independently of a review/gap.
    live.status = "review_required"

    clock.value = 200
    terminal = live.analyze_frame(object(), monotonic_ms=200)
    assert terminal.event is None
    assert terminal.block_reason == "terminal_suspected_inconsistent_rule_state"
    assert terminal.status == "running"
    assert live.status == "running"
    assert not store.events

def runtime(*, lead="right", recorder=None, store=None,
            advice_immediate=True, bind=True, local_hint_window_ms=0):
    store = store or MemoryStore()
    recorder = recorder or MemoryRecorder()
    store.start({"schema": "test.live-v2/1"})
    clock = ManualClock()
    visions = []
    advisers = []

    def vision_factory(version):
        value = ScriptedVision(); visions.append(value); return value

    def advice_factory(version):
        value = FakeAdviceRuntime(immediate=advice_immediate)
        advisers.append(value); return value

    value = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=vision_factory,
        advice_runtime_factory=advice_factory, processing_clock_ms=clock,
        local_hint_window_ms=local_hint_window_ms,
    )
    update = value.start(
        round_level="2", hand=HAND, lead_player=lead,
        monotonic_ms=0, wall_time="2026-09-06T00:00:00+08:00",
    )
    if bind:
        value.bind_capture_generation(1)
    _LIVE_RUNTIMES.append(value)
    return value, update, store, recorder, clock, visions, advisers



class VisualCorrectionVision:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True

    def process_frame(
        self, image, *, frame, version, wild_rank, expected_seat=None,
        now_ms=None, formal_action_boundary=None, repair_seats=(),
    ):
        del image, wild_rank, formal_action_boundary, repair_seats
        self.calls += 1
        fast = _fast(expected_seat.value if expected_seat else "right")
        if self.calls == 1:
            first = FrameIdentity(
                frame.session_id, frame.capture_generation, 1,
                max(0, frame.captured_ms - 1), frame.roi_version, frame.source_id,
            )
            last = FrameIdentity(
                frame.session_id, frame.capture_generation, 2,
                frame.captured_ms, frame.roi_version, frame.source_id,
            )
            candidate = ActionCandidate(
                "uncertain-5", version, Seat.RIGHT, ActionKind.PLAY,
                ("5?",), (("5?", "5S", "5C"),), ("u1", "u2"),
                0, first, last, int(now_ms), 0.9, CandidateReason.STABLE_PLAY,
            )
            return FramePipelineResult(frame, fast, (), (), (candidate,), (), (), 0)
        observation = SeatObservation(
            "reread-5S", frame, Seat.RIGHT, ObservationKind.PLAY, ("5S",),
            0.95, ObservationReason.CARDS_RECOGNIZED, int(now_ms), (("5S",),),
        )
        return FramePipelineResult(frame, fast, (), (observation,), (), (), (), 0)


def test_visual_uncertain_action_is_repaired_without_creating_second_turn():
    store = MemoryStore(); recorder = MemoryRecorder(); store.start({"schema": "test.live-v2/1"})
    clock = ManualClock(); vision = VisualCorrectionVision()
    advisers = []

    def advice_factory(_version):
        value = FakeAdviceRuntime(); advisers.append(value); return value

    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=lambda _version: vision,
        advice_runtime_factory=advice_factory, processing_clock_ms=clock,
    )
    _LIVE_RUNTIMES.append(live)
    live.start(
        round_level="2", hand=HAND, lead_player="right", monotonic_ms=0,
        wall_time="2026-09-13T00:00:00+08:00",
    )
    live.bind_capture_generation(1)
    clock.value = 100
    first = live.analyze_frame(object(), monotonic_ms=100)
    assert first.event and first.event.event_type == "player_played"
    assert live.snapshot.play_history[-1].cards == ("5?",)

    clock.value = 200
    confirming = live.analyze_frame(object(), monotonic_ms=200)
    assert confirming.event is None

    clock.value = 300
    repaired = live.analyze_frame(object(), monotonic_ms=300)

    assert repaired.event and repaired.event.event_type == "event_correction"
    assert repaired.event.payload["target_action_id"] == live.rule_session.confirmed_actions[0].action_id
    assert live.snapshot.play_history[-1].cards == ("5S",)
    assert len(live.snapshot.play_history) == 1
    assert live.snapshot.current_player == "opposite"
    assert len(advisers) == 1
    assert not advisers[0].closed
    assert not vision.closed

class VisualCountCorrectionVision:
    def __init__(self) -> None:
        self.calls = 0

    def start(self):
        pass

    def close(self):
        pass

    def process_frame(
        self, image, *, frame, version, wild_rank, expected_seat=None,
        now_ms=None, formal_action_boundary=None, repair_seats=(),
    ):
        del image, wild_rank, formal_action_boundary, repair_seats
        self.calls += 1
        fast = _fast(expected_seat.value if expected_seat else "right")
        if self.calls == 1:
            first = FrameIdentity(
                frame.session_id, frame.capture_generation, 1,
                max(0, frame.captured_ms - 1), frame.roi_version, frame.source_id,
            )
            last = FrameIdentity(
                frame.session_id, frame.capture_generation, 2,
                frame.captured_ms, frame.roi_version, frame.source_id,
            )
            candidate = ActionCandidate(
                "partial-four-aces", version, Seat.RIGHT, ActionKind.PLAY,
                ("AH", "A?"), (("AH",), ("A?", "AH", "AD")),
                ("a1", "a2"), 0, first, last, int(now_ms), 0.7,
                CandidateReason.STABLE_PLAY,
            )
            return FramePipelineResult(
                frame, fast, (), (), (candidate,), (), (), 0
            )
        observation = SeatObservation(
            f"reread-four-aces-{self.calls}", frame, Seat.RIGHT,
            ObservationKind.PLAY, ("AH", "AD", "AD", "AC"), 0.95,
            ObservationReason.CARDS_RECOGNIZED, int(now_ms),
            (("AH",), ("AD",), ("AD",), ("AC",)),
        )
        return FramePipelineResult(
            frame, fast, (), (observation,), (), (), (), 0
        )


def test_visual_reread_can_repair_rank_suit_and_card_count_in_place():
    store = MemoryStore(); recorder = MemoryRecorder()
    store.start({"schema": "test.live-v2/1"})
    clock = ManualClock(); vision = VisualCountCorrectionVision()

    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=lambda _version: vision,
        advice_runtime_factory=lambda _version: FakeAdviceRuntime(),
        processing_clock_ms=clock,
    )
    _LIVE_RUNTIMES.append(live)
    live.start(
        round_level="2", hand=HAND, lead_player="right", monotonic_ms=0,
        wall_time="2026-09-13T00:00:00+08:00",
    )
    live.bind_capture_generation(1)
    clock.value = 100
    first = live.analyze_frame(object(), monotonic_ms=100)
    assert first.event and first.event.event_type == "player_played"
    assert live.snapshot.play_history[-1].cards == ("AH", "A?")

    clock.value = 200
    assert live.analyze_frame(object(), monotonic_ms=200).event is None
    clock.value = 300
    repaired = live.analyze_frame(object(), monotonic_ms=300)

    assert repaired.event and repaired.event.event_type == "event_correction"
    assert repaired.event.payload["correction_reason"] == "visual_reread"
    assert tuple(repaired.event.payload["cards"]) == ("AH", "AD", "AD", "AC")
    assert live.snapshot.play_history[-1].cards == ("AH", "AD", "AD", "AC")
    assert len(live.snapshot.play_history) == 1
    assert live.snapshot.current_player == "opposite"



class QueuedVisualCorrectionVision(VisualCorrectionVision):
    def process_frame(
        self, image, *, frame, version, wild_rank, expected_seat=None,
        now_ms=None, formal_action_boundary=None, repair_seats=(),
    ):
        if self.calls == 0:
            result = super().process_frame(
                image, frame=frame, version=version, wild_rank=wild_rank,
                expected_seat=expected_seat, now_ms=now_ms,
                formal_action_boundary=formal_action_boundary,
                repair_seats=repair_seats,
            )
            self.calls = 1
            return result
        self.calls += 1
        reread_frame = FrameIdentity(
            frame.session_id, frame.capture_generation, self.calls + 1,
            7_900 if self.calls == 2 else 7_999,
            frame.roi_version, frame.source_id,
        )
        observation = SeatObservation(
            "queued-reread-5S", reread_frame, Seat.RIGHT, ObservationKind.PLAY,
            ("5S",), 0.95, ObservationReason.CARDS_RECOGNIZED,
            int(now_ms), (("5S",),),
        )
        return FramePipelineResult(
            frame, _fast(expected_seat.value if expected_seat else "right"),
            (), (observation,), (), (), (), 0,
        )


def test_visual_repair_window_uses_turn_boundary_not_eight_second_timeout():
    store = MemoryStore(); recorder = MemoryRecorder(); store.start({"schema": "test.live-v2/1"})
    clock = ManualClock(); vision = QueuedVisualCorrectionVision()

    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=lambda _version: vision,
        advice_runtime_factory=lambda _version: FakeAdviceRuntime(),
        processing_clock_ms=clock,
    )
    _LIVE_RUNTIMES.append(live)
    live.start(
        round_level="2", hand=HAND, lead_player="right", monotonic_ms=0,
        wall_time="2026-09-13T00:00:00+08:00",
    )
    live.bind_capture_generation(1)

    clock.value = 100
    first = live.analyze_frame(object(), monotonic_ms=100)
    assert first.snapshot.play_history[-1].cards == ("5?",)

    clock.value = 7_900
    assert live.analyze_frame(object(), monotonic_ms=7_900).event is None

    # The second matching reread was captured before the old eight-second
    # deadline but delivered later. Natural turn boundaries, not wall time,
    # decide whether the repair still belongs to the original action.
    clock.value = 8_101
    repaired = live.analyze_frame(object(), monotonic_ms=8_101)

    assert repaired.event and repaired.event.event_type == "event_correction"
    assert repaired.snapshot.play_history[-1].cards == ("5S",)


def test_visual_repair_priority_rotates_between_pending_seats():
    live, _first, _store, _recorder, _clock, _visions, _advisers = runtime()
    for seat in (Seat.RIGHT, Seat.OPPOSITE):
        live._register_visual_correction(SimpleNamespace(
            action_id=f"repair-{seat.value}",
            kind=ActionKind.PLAY,
            evidence_origin=EvidenceOrigin.VISUAL,
            seat=seat,
            cards=("3?",),
            suit_options=(("3D", "3C"),),
            confidence=0.5,
            diagnostics=("play_confidence",),
        ))

    assert live._ordered_visual_repair_seats() == (Seat.RIGHT, Seat.OPPOSITE)
    assert live._ordered_visual_repair_seats() == (Seat.OPPOSITE, Seat.RIGHT)
    assert live._ordered_visual_repair_seats() == (Seat.RIGHT, Seat.OPPOSITE)



def test_complete_high_confidence_action_opens_expansion_only_watch():
    live, _first, _store, _recorder, _clock, _visions, _advisers = runtime()
    live._register_visual_correction(SimpleNamespace(
        action_id="complete",
        kind=ActionKind.PLAY,
        evidence_origin=EvidenceOrigin.VISUAL,
        seat=Seat.RIGHT,
        cards=("3D",),
        suit_options=(("3D",),),
        confidence=0.95,
        diagnostics=(),
    ))
    assert live._visual_corrections["complete"].allow_expansion
    assert not live._visual_corrections["complete"].allow_complete_rewrite


def test_terminal_state_closes_all_auto_correction_windows():
    live, _first, _store, _recorder, _clock, _visions, _advisers = runtime()
    live._register_visual_correction(SimpleNamespace(
        action_id="uncertain",
        kind=ActionKind.PLAY,
        evidence_origin=EvidenceOrigin.VISUAL,
        seat=Seat.RIGHT,
        cards=("3?",),
        suit_options=(("3D", "3C"),),
        confidence=0.5,
        diagnostics=("play_confidence",),
    ))
    assert live._visual_corrections
    live._expire_visual_corrections(None)
    assert live._visual_corrections == {}


def test_complete_multi_seat_chain_reaches_self_and_publishes_advice_once() -> None:
    live, first, store, _recorder, _clock, _visions, advisers = runtime()
    assert isinstance(live, LiveRuntimePort)
    updates = [first]
    updates.append(live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100
    ))
    updates.append(live.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=200))
    updates.append(live.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=300))
    assert live.snapshot.current_player == "self"
    assert live.latest_advice and live.latest_advice.visible
    assert len(advisers[0].submissions) == 1
    assert [row["status"] for row in store.advice] == [
        "requested", "worker_started", "ready",
    ]
    assert len([row for row in store.advice if row["status"] == "ready"]) == 1
    assert all(row["opportunity_id"] for row in store.advice)
    assert len(store.batches) == 4
    assert all(len(batch) == 1 for batch in store.batches)
    assert [item.update_sequence for item in updates] == sorted(
        item.update_sequence for item in updates
    )


def test_analyze_frame_passes_last_formal_action_boundary_to_vision() -> None:
    live, _first, _store, _recorder, clock, visions, _advisers = runtime()
    live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100,
    )
    boundary = live._trusted_snapshot().play_history[-1].last_frame
    visions[0].empty()
    clock.value = 200
    live.analyze_frame(object(), monotonic_ms=200)
    assert visions[0].formal_action_boundaries[-1] == boundary


def test_foreign_candidate_does_not_open_recovery_or_stop_future_observation() -> None:
    live, _first, _store, _recorder, clock, visions, _advisers = runtime()
    vision = visions[0]
    vision.queue(Seat.OPPOSITE, ActionKind.PASS)
    clock.value = 200
    blocked = live.analyze_frame(object(), monotonic_ms=200)
    assert not blocked.block_reason
    vision.empty(); clock.value = 8_300
    expired = live.analyze_frame(object(), monotonic_ms=8_300)
    assert not expired.block_reason
    calls_before = vision.calls
    vision.queue(Seat.RIGHT, ActionKind.PLAY, ("3D",))
    clock.value = 8_400
    recovered = live.analyze_frame(object(), monotonic_ms=8_400)
    assert vision.calls == calls_before + 1
    assert recovered.snapshot.current_player == "opposite"
    assert not recovered.block_reason


def test_async_vision_consumes_completed_prior_frame_under_current_flow_version() -> None:
    store, recorder, clock = MemoryStore(), MemoryRecorder(), ManualClock()
    store.start({"schema": "test.live-v2/1"})
    visions = []
    def factory(version):
        value = DelayedVisionRuntime(); visions.append(value); return value
    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=factory,
        advice_runtime_factory=lambda version: FakeAdviceRuntime(),
        processing_clock_ms=clock, local_hint_window_ms=0,
    )
    live.start(round_level="2", hand=HAND, lead_player="right", monotonic_ms=0)
    live.bind_capture_generation(1)
    _LIVE_RUNTIMES.append(live)
    clock.value = 100
    first = live.analyze_frame(object(), monotonic_ms=100)
    assert not first.snapshot.play_history
    assert first.processed_capture_seq is None
    assert first.processed_captured_ms is None
    clock.value = 200
    completed = live.analyze_frame(object(), monotonic_ms=200)
    assert completed.processed_capture_seq == 1
    assert completed.processed_captured_ms == 100  # consumed previous frame, not the new submission
    assert completed.snapshot.current_player == "opposite"
    assert completed.snapshot.play_history[-1].cards == ("3D",)
    assert visions[0].started


@pytest.mark.parametrize("include_fault", [False, True], ids=["completed", "faulted"])
@pytest.mark.parametrize("branch", ["normal", "correction", "opening", "opening_wait", "terminal"])
def test_processed_receipt_follows_consumed_frame_on_real_branches(branch, include_fault):
    """Use actual branch logic and async receipts, never the newly submitted frame."""
    store, recorder, clock = MemoryStore(), MemoryRecorder(), ManualClock()
    store.start({"schema": "test.live-v2/1"})
    if branch == "correction":
        pipeline = VisualCorrectionVision()
        # Its initial action ends at sequence 2; correction needs two newer reads.
        completed_frames = 4
    elif branch == "terminal":
        pipeline = TerminalVision()
        completed_frames = 2
    else:
        pipeline = ScriptedVision()
        if branch != "opening_wait":
            pipeline.queue(Seat.RIGHT, ActionKind.PLAY, ("3D",))
        completed_frames = 1
    vision = DelayedVisionRuntime(pipeline=pipeline, include_fault=include_fault)
    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=lambda _version: vision,
        advice_runtime_factory=lambda _version: FakeAdviceRuntime(),
        processing_clock_ms=clock, local_hint_window_ms=0,
    )
    _LIVE_RUNTIMES.append(live)
    live.start(
        round_level="2", hand=HAND,
        lead_player=None if branch in {"opening", "opening_wait"} else "right",
        monotonic_ms=0,
    )
    live.bind_capture_generation(1)

    clock.value = 100
    pending = live.analyze_frame(object(), monotonic_ms=100)
    # Covers both an empty poll and faults without any completed frame.
    assert pending.processed_capture_seq is None
    assert pending.processed_captured_ms is None
    for sequence in range(2, completed_frames + 2):
        clock.value = sequence * 100
        update = live.analyze_frame(object(), monotonic_ms=clock.value)

    # Assert the real semantic branch was reached before checking its receipt.
    if branch == "correction":
        assert update.event and update.event.event_type == "event_correction"
        assert update.snapshot.play_history[-1].cards == ("5S",)
    elif branch == "opening":
        assert update.snapshot.lead_player == "right"
        assert len(update.snapshot.play_history) == 1
        assert update.event and update.event.event_type == "player_played"
        assert update.status == "running"
    elif branch == "opening_wait":
        assert update.status == "waiting_lead"
        assert update.block_reason == "opening_waiting_for_unique_visual_action"
        assert not update.snapshot.play_history
    elif branch == "terminal":
        assert update.status == "running"
        assert update.block_reason == "terminal_suspected_inconsistent_rule_state"
        assert update.event is None
    else:
        assert update.event and update.event.event_type == "player_played"
        assert update.snapshot.play_history[-1].cards == ("3D",)

    if include_fault:
        # A mixed completed-result/fault batch must not manufacture a clean receipt.
        assert update.processed_capture_seq is None
        assert update.processed_captured_ms is None
    else:
        assert update.processed_capture_seq == completed_frames
        assert update.processed_captured_ms == completed_frames * 100
        assert update.processed_captured_ms != clock.value


def test_async_candidate_overlapping_committed_boundary_is_dropped() -> None:
    store, recorder, clock = MemoryStore(), MemoryRecorder(), ManualClock()
    store.start({"schema": "test.live-v2/1"})
    visions = []
    def factory(version):
        value = DelayedVisionRuntime(); visions.append(value); return value
    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=factory,
        advice_runtime_factory=lambda version: FakeAdviceRuntime(),
        processing_clock_ms=clock, local_hint_window_ms=0,
    )
    live.start(round_level="2", hand=HAND, lead_player="right", monotonic_ms=0)
    live.bind_capture_generation(1)
    _LIVE_RUNTIMES.append(live)
    clock.value = 100
    live.analyze_frame(
        object(), monotonic_ms=100,
        trace_context={"capture_generation": 1, "capture_seq": 10, "captured_ms": 100},
    )
    live.commit_trusted_action(
        actor="right", cards=("4D",), is_pass=False, monotonic_ms=150,
    )
    clock.value = 200
    update = live.analyze_frame(
        object(), monotonic_ms=200,
        trace_context={"capture_generation": 1, "capture_seq": 11, "captured_ms": 200},
    )
    assert [event.cards for event in update.snapshot.play_history] == [("4D",)]
    assert not update.block_reason
    assert not live._pending


def test_async_candidate_for_next_hand_survives_state_revision_advance() -> None:
    store, recorder, clock = MemoryStore(), MemoryRecorder(), ManualClock()
    store.start({"schema": "test.live-v2/1"})
    visions = []
    def factory(version):
        value = DelayedVisionRuntime(Seat.OPPOSITE, ("5D",))
        visions.append(value)
        return value
    live = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store, recorder=recorder,
        recognition_service=FakeRecognition(), vision_factory=factory,
        advice_runtime_factory=lambda version: FakeAdviceRuntime(),
        processing_clock_ms=clock, local_hint_window_ms=0,
    )
    live.start(round_level="2", hand=HAND, lead_player="right", monotonic_ms=0)
    live.bind_capture_generation(1)
    _LIVE_RUNTIMES.append(live)
    clock.value = 200
    live.analyze_frame(
        object(), monotonic_ms=200,
        trace_context={"capture_generation": 1, "capture_seq": 10, "captured_ms": 200},
    )
    live.commit_trusted_action(
        actor="right", cards=("4D",), is_pass=False, monotonic_ms=150,
    )
    clock.value = 300
    update = live.analyze_frame(
        object(), monotonic_ms=300,
        trace_context={"capture_generation": 1, "capture_seq": 11, "captured_ms": 300},
    )
    assert [event.cards for event in update.snapshot.play_history] == [
        ("4D",), ("5D",),
    ]
    assert update.snapshot.current_player == "left"


def test_opening_and_manual_actions_share_transactional_commit_path() -> None:
    live, first, store, _recorder, _clock, _visions, _advisers = runtime(lead=None)
    assert first.status == "waiting_lead"
    opened = live.bootstrap_opening_action(
        actor="left", cards=("3D",), expected_next_player="self",
        monotonic_ms=100, confidence=0.99, source="opening",
    )
    assert opened.snapshot.current_player == "self"
    assert [batch[0].event_type for batch in store.batches[:2]] == [
        "initial_state_confirmed", "lead_player_confirmed",
    ]
    assert len(store.batches) == 3
    manual = live.confirm_manual_action(cards=("4S",), is_pass=False)
    assert manual.snapshot.current_player == "right"
    assert len(store.batches) == 4


def test_generation_rebuild_seeds_history_and_drops_old_advice() -> None:
    delivered = []
    live, _first, _store, _recorder, _clock, visions, advisers = runtime(
        lead="right", advice_immediate=False
    )
    live._on_update = delivered.append
    live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100
    )
    live.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=200)
    live.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=300)
    old = advisers[0]
    assert len(old.submissions) == 1
    old.release_on_drain = True
    rebound = live.bind_capture_generation(2)
    assert rebound.capture_generation == 2
    assert len(rebound.snapshot.play_history) == 3
    assert old.closed and visions[0].closed
    first_request = [
        row for row in _store.advice if row["capture_generation"] == 1
    ]
    assert [row["status"] for row in first_request] == [
        "requested", "worker_started", "cancelled",
    ]
    assert live._opportunity_metrics["opportunity_valid"] == 0
    assert not delivered
    live.poll_deadlines()
    assert advisers[1].submissions[0][0].version.capture_generation == 2


def test_late_model_result_after_state_change_is_not_published() -> None:
    delivered = []
    live, _first, _store, _recorder, _clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False
    )
    live._on_update = delivered.append
    delayed = advisers[0]
    live.commit_trusted_action(
        actor="self", cards=("3S",), is_pass=False, monotonic_ms=100
    )
    delayed.release_on_drain = True
    live.poll_deadlines()
    sleep(0.03)
    assert not delivered
    assert not (live.latest_advice and live.latest_advice.visible)
    assert [row["status"] for row in _store.advice] == [
        "requested", "worker_started", "cancelled",
    ]
    assert live._opportunity_metrics["opportunity_valid"] == 0
    assert live._opportunity_metrics["opportunity_no_result"] == 1


def test_advice_result_survives_empty_frame_update_sequences() -> None:
    delivered = []
    live, _first, _store, _recorder, clock, visions, advisers = runtime(
        lead="self", advice_immediate=False
    )
    live._on_update = delivered.append
    visions[0].empty(); clock.value = 100
    live.analyze_frame(object(), monotonic_ms=100)
    visions[0].empty(); clock.value = 200
    live.analyze_frame(object(), monotonic_ms=200)
    assert advisers[0].submissions
    advisers[0].release_on_drain = True
    live.poll_deadlines()
    assert live.latest_advice and live.latest_advice.visible
    assert delivered and delivered[-1].advice.visible
    assert [row["status"] for row in _store.advice] == [
        "requested", "worker_started", "ready",
    ]
    assert live._opportunity_metrics["opportunity_valid"] == 1


def test_persistence_failure_never_mutates_authoritative_history() -> None:
    store = MemoryStore()
    live, _first, _store, _recorder, _clock, _visions, advisers = runtime(store=store)
    store.fail_batches = True
    failed = live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100
    )
    assert failed.snapshot.revision == 1
    assert not failed.snapshot.play_history
    assert failed.block_reason == "rule_rejection"
    assert not advisers[0].submissions


def test_recording_failure_is_warning_only_and_analysis_still_commits() -> None:
    recorder = MemoryRecorder(fail_write=True)
    live, _first, store, _recorder, _clock, _visions, _advisers = runtime(recorder=recorder)
    warning = live.record_frame(object(), monotonic_ms=50, wall_time="now")
    assert warning and warning.reason == "recording_failed"
    committed = live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100
    )
    assert committed.snapshot.revision == 2
    assert store.traces[-1]["kind"] == "recording_failed"


def test_control_hint_never_commits_pass_and_close_has_no_live_workers() -> None:
    live, _first, store, recorder, clock, visions, advisers = runtime(lead="self")
    before = live.snapshot.revision
    control = FastSignalResult(
        "self", "self", False, True, False,
        cannot_beat_visible=True, cannot_beat_confidence=0.95,
        cannot_beat_box=(100, 100, 80, 30),
    )
    clock.value = 100
    assert live.preview_controls(
        control, captured_ms=100, capture_generation=1, frame_size=(1280, 720)
    ) is None
    clock.value = 150
    hinted = live.preview_controls(
        control, captured_ms=150, capture_generation=1, frame_size=(1280, 720)
    )
    assert hinted and hinted.local_rule_hint and live.snapshot.revision == before
    live.begin_finalizing()
    frame_count = recorder.frame_count
    assert live.record_frame(object(), monotonic_ms=200, wall_time="now") is None
    assert recorder.frame_count == frame_count
    sealed = live.finish()
    assert sealed.status == "sealed" and recorder.closed and store.seals
    metrics = store.seals[-1]["metrics"]
    assert set(metrics) >= {
        "opportunity_total", "opportunity_valid", "opportunity_late",
        "opportunity_no_result", "opportunity_unrecoverable",
    }
    assert visions[0].closed and advisers[0].closed
    assert live.finish().status == "sealed"


def test_finish_drains_pending_advice_to_one_terminal_before_store_seal() -> None:
    live, _first, store, _recorder, _clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=0
    )
    pump = live._advice_pump
    assert pump is not None and advisers[0].submissions
    observed = []
    real_seal = store.seal

    def seal(**kwargs):
        terminals = [
            row["status"] for row in store.advice
            if row["status"] in {
                "ready", "timeout", "stale", "cancelled", "failed", "withheld"
            }
        ]
        observed.append((pump.wait_idle(0), terminals))
        return real_seal(**kwargs)

    store.seal = seal
    assert live.finish().status == "sealed"
    assert observed == [(True, ["cancelled"])]


def test_confirmed_cannot_beat_wins_race_against_speculative_model_without_committing_pass() -> None:
    delivered = []
    live, _first, store, _recorder, clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=200
    )
    live._on_update = delivered.append
    before = live.snapshot
    control = FastSignalResult(
        "self", "self", False, True, False,
        cannot_beat_visible=True, cannot_beat_confidence=0.96,
        cannot_beat_box=(100, 100, 80, 30),
    )
    clock.value = 100
    assert live.preview_controls(
        control, captured_ms=100, capture_generation=1, frame_size=(1280, 720)
    ) is None
    clock.value = 200
    update = live.preview_controls(
        control, captured_ms=200, capture_generation=1, frame_size=(1280, 720)
    )
    assert update and update.advice.visible and update.advice.advice.is_pass
    assert delivered and delivered[-1].advice.advice.is_pass
    assert len(advisers[0].submissions) == 1
    assert live.snapshot.revision == before.revision
    assert live.snapshot.play_history == before.play_history
    assert [row["status"] for row in store.advice] == [
        "requested", "worker_started", "local_pass",
    ]
    sleep(0.25)
    assert len(advisers[0].submissions) == 1


def test_hint_just_after_window_is_ui_only_and_model_remains_terminal() -> None:
    live, _first, store, _recorder, clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=200
    )
    control = FastSignalResult(
        "self", "self", False, True, False,
        cannot_beat_visible=True, cannot_beat_confidence=0.96,
        cannot_beat_box=(100, 100, 80, 30),
    )
    clock.value = 101
    live.preview_controls(
        control, captured_ms=101, capture_generation=1, frame_size=(1280, 720)
    )
    clock.value = 201
    update = live.preview_controls(
        control, captured_ms=201, capture_generation=1, frame_size=(1280, 720)
    )
    assert update and update.local_rule_hint is not None
    assert all(row["status"] != "local_pass" for row in store.advice)
    sleep(.25)
    assert len(advisers[0].submissions) == 1
    advisers[0].release_on_drain = True
    live.poll_deadlines()
    assert [row["status"] for row in store.advice][-2:] == ["worker_started", "ready"]


def test_model_starts_immediately_but_result_waits_for_local_hint_window() -> None:
    live, _first, store, _recorder, _clock, _visions, advisers = runtime(
        lead="self", advice_immediate=True, local_hint_window_ms=150
    )
    assert len(advisers[0].submissions) == 1
    assert [row["status"] for row in store.advice] == [
        "requested", "worker_started",
    ]
    sleep(0.2)
    assert [row["status"] for row in store.advice] == [
        "requested", "worker_started", "ready",
    ]
    assert live.latest_advice and live.latest_advice.visible


def test_late_local_pass_cannot_supersede_a_submitted_model() -> None:
    live, _first, store, _recorder, clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=150
    )
    sleep(0.2)
    assert len(advisers[0].submissions) == 1
    control = FastSignalResult(
        "self", "self", False, True, False,
        cannot_beat_visible=True, cannot_beat_confidence=0.96,
        cannot_beat_box=(100, 100, 80, 30),
    )
    clock.value = 200
    live.preview_controls(
        control, captured_ms=200, capture_generation=1, frame_size=(1280, 720)
    )
    clock.value = 250
    update = live.preview_controls(
        control, captured_ms=250, capture_generation=1, frame_size=(1280, 720)
    )
    assert update and update.local_rule_hint is not None
    assert not (update.advice and update.advice.visible)
    advisers[0].release_on_drain = True
    live.poll_deadlines()
    assert live.latest_advice and live.latest_advice.visible
    assert not live.latest_advice.advice.is_pass
    assert [row["status"] for row in store.advice] == [
        "requested", "worker_started", "ready",
    ]
    assert live._opportunity_metrics["opportunity_valid"] == 1


def test_hint_13_seconds_late_cannot_resurrect_timed_out_opportunity() -> None:
    live, _first, store, _recorder, clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=200
    )
    control = FastSignalResult(
        "self", "self", False, True, False,
        cannot_beat_visible=True, cannot_beat_confidence=0.96,
        cannot_beat_box=(100, 100, 80, 30),
    )
    clock.value = 13_500
    live.preview_controls(
        control, captured_ms=13_500, capture_generation=1, frame_size=(1280, 720)
    )
    clock.value = 13_550
    update = live.preview_controls(
        control, captured_ms=13_550, capture_generation=1, frame_size=(1280, 720)
    )
    assert update and update.local_rule_hint is not None
    assert all(row["status"] != "local_pass" for row in store.advice)
    assert live.wait_for_advice_idle(timeout=1)
    assert len(advisers[0].submissions) == 1
    terminals = [
        row for row in store.advice
        if row["status"] in {"ready", "local_pass", "failed", "timeout", "withheld"}
    ]
    assert [row["status"] for row in terminals] == ["timeout"]


def test_stale_generation_effect_and_terminal_controls_cannot_short_circuit() -> None:
    live, _first, store, _recorder, clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=250
    )
    unsafe = (
        FastSignalResult(
            "self", "self", False, True, True,
            cannot_beat_visible=True, cannot_beat_confidence=0.96,
            cannot_beat_box=(100, 100, 80, 30),
        ),
        FastSignalResult(
            "self", "self", False, True, False, game_end_control="continue",
            cannot_beat_visible=True, cannot_beat_confidence=0.96,
            cannot_beat_box=(100, 100, 80, 30),
        ),
    )
    clock.value = 100
    for fast in unsafe:
        assert live.preview_controls(
            fast, captured_ms=clock.value, capture_generation=1,
            frame_size=(1280, 720),
        ) is None
        clock.value += 10
    assert live.preview_controls(
        unsafe[0], captured_ms=clock.value, capture_generation=0,
        frame_size=(1280, 720),
    ) is None
    assert len(advisers[0].submissions) == 1
    assert [row["status"] for row in store.advice] == [
        "requested", "worker_started",
    ]


def test_explicit_correction_rebuilds_the_same_generation() -> None:
    live, _first, _store, _recorder, _clock, visions, advisers = runtime()
    live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100,
    )
    before = live.snapshot
    result = live.correct_latest(cards=("4D",), is_pass=False)
    assert result.event and result.event.event_type == "event_correction"
    assert live.snapshot.revision == before.revision + 1
    assert live.snapshot.play_history[-1].cards == ("4D",)
    assert result.capture_generation == 1
    assert len(visions) == 2 and visions[0].closed == 1
    assert len(advisers) == 2 and advisers[0].closed == 1


def test_action_metadata_is_audit_only_and_does_not_override_rules() -> None:
    live, _first, store, _recorder, _clock, _visions, _advisers = runtime()
    result = live.commit_trusted_action(
        actor="right", cards=("3D",), is_pass=False, monotonic_ms=100,
        action_metadata={"semantic": "legacy-only"},
    )
    assert not result.block_reason
    assert result.snapshot.revision == 2
    assert result.snapshot.play_history[-1].cards == ("3D",)
    assert any(item.get("kind") == "trusted_action_metadata"
               for item in store.observations)


def test_visual_correction_matches_same_rank_and_narrows_unknown_suit():
    corrected = _resolve_visual_correction(
        ("5?", "3S"),
        (("5S", "5C"), ("3S",)),
        ("5S", "3S"),
        (("5S",), ("3S",)),
    )

    assert corrected == ("5S", "3S")


def test_visual_correction_only_accepts_information_improvement():
    target = (("5?", "5S", "5C"),)
    assert _resolve_visual_correction(
        ("5?",), target, ("6S",), (("6S",),)
    ) == ("6S",)
    assert _resolve_visual_correction(
        ("AH", "A?"),
        (("AH",), ("A?", "AH", "AD")),
        ("AH", "AH", "AD", "AC"),
        (("AH",), ("AH",), ("AD",), ("AC",)),
    ) == ("AH", "AH", "AD", "AC")
    assert _resolve_visual_correction(
        ("3H", "4H", "5H", "6H", "7H"),
        (("3H",), ("4H",), ("5H",), ("6H",), ("7H",)),
        ("3H", "4D"), (("3H",), ("4D",)),
        allow_complete_rewrite=True, allow_expansion=True,
    ) is None
    assert _resolve_visual_correction(
        ("small_joker",), ((),),
        ("QH", "QH", "8H", "8C"),
        (("QH",), ("QH",), ("8H",), ("8C",)),
        allow_complete_rewrite=True,
    ) is None
    assert _resolve_visual_correction(
        ("7D", "7C"), (("7D",), ("7C",)),
        ("7D", "7C", "7S", "8H", "8C", "8S"),
        (("7D",), ("7C",), ("7S",), ("8H",), ("8C",), ("8S",)),
        allow_expansion=True,
    ) == ("7D", "7C", "7S", "8H", "8C", "8S")


def test_visual_correction_rejects_another_incomplete_reread():
    target = (("5?", "5S", "5C"),)
    assert _resolve_visual_correction(
        ("5?",), target, ("5?",), (("5?", "5S", "5C"),)
    ) is None
