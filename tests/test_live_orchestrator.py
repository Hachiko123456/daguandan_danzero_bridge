from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from daguandan_bridge.application.ports import LiveRuntimePort
from daguandan_bridge.domain.live_runtime import (
    AdviceRequestKey as DomainAdviceRequestKey,
    LiveAdvice as DomainLiveAdvice,
    LiveUpdate as DomainLiveUpdate,
)
from daguandan_bridge.live.orchestrator import (
    AdviceRequestKey,
    LiveOrchestrator,
    ReviewCandidate,
    ReviewRequest,
    _AdviceJob,
    _TurnOwnershipWindow,
)
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.consensus import ConsensusResult, RecognitionSample
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.replay import EventReplayer
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live.zone_lifecycle import ZoneDecision, ZoneFrameMetrics, ZonePhase
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.recognition_service import (
    FastSignalResult,
    OpeningSignal,
    PlacementSignal,
    PlayRegionResult,
    RecognitionAnnotation,
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
        RecognitionSample(cards, False, 0.91, "template:cards", f"OBS-{index}",
                          captured_ms=1_000 + index * 50, capture_seq=index,
                          action_epoch=("session", 31, 32, "right", 0))
        for index in (1, 2)
    ]
    window = _TurnOwnershipWindow(
        key=("session", 31, 32, "right"),
        expected_player="right",
        handoff_detected_ms=1_000,
        turn_recovery_pending=True,
        turn_recovery_detected_ms=1_188,
        handoff_samples=samples,
        evidence_epoch=("session", 31, 32, "right", 0),
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


def test_recovery_real_deadline_emits_under_busy_analysis_lock_with_generation(tmp_path):
    notices = []
    notice_ready = threading.Event()

    def on_update(update):
        notices.append(update)
        if update.block_reason == "turn_recovery_budget_exceeded":
            notice_ready.set()

    orchestrator = _orchestrator(tmp_path, [], lead_player="right", on_update=on_update)
    orchestrator._capture_generation = 2
    window = orchestrator._ensure_turn_ownership_window()
    fast = FastSignalResult("right", "self", False, True, False)
    orchestrator._begin_turn_recovery(window, fast, 100, reason="test_missing_lead")
    began = time.monotonic()
    with orchestrator._state_lock:
        orchestrator._withhold_advice_for_turn_recovery(window, 100)
        assert notice_ready.wait(2.6), "deadline notice must not wait for analysis lock"
        notice = notices[-1]
        assert time.monotonic() - began < 2.6
        assert notice.capture_generation == 2 and notice.update_sequence > 0
        assert notice.missing_player == "right" and notice.missing_action_kind == "lead"
        assert window.turn_recovery_deadline_expired
    orchestrator.poll_deadlines()
    assert window.turn_recovery_failed
    orchestrator.finish()


def test_obsolete_deadline_cannot_overwrite_new_revision_or_generation(tmp_path):
    notices = []
    orchestrator = _orchestrator(tmp_path, [], lead_player="right", on_update=notices.append)
    window = orchestrator._ensure_turn_ownership_window()
    fast = FastSignalResult("right", "self", False, True, False)
    orchestrator._begin_turn_recovery(window, fast, 100, reason="test")
    orchestrator._withhold_advice_for_turn_recovery(window, 100)
    callback = orchestrator._deadline_timer.function
    orchestrator.commit_trusted_action(actor="right", cards=("9C",), is_pass=False, monotonic_ms=200)
    before = len(notices)
    callback()
    assert len(notices) == before
    assert not orchestrator._turn_ownership_window.turn_recovery_failed
    orchestrator.finish()


@pytest.mark.parametrize("through_fast_signal", [False, True])
def test_new_capture_generation_clears_old_deadline_before_poll(tmp_path, through_fast_signal):
    clock = [1_000]
    orchestrator = _orchestrator(tmp_path, [], lead_player="right", processing_clock_ms=lambda: clock[0])
    window = orchestrator._ensure_turn_ownership_window()
    fast = FastSignalResult("right", "self", False, True, False)
    orchestrator._begin_turn_recovery(window, fast, 100, reason="test")
    orchestrator._withhold_advice_for_turn_recovery(window, 100)
    window.turn_recovery_deadline_expired = True
    clock[0] = 3_000
    if through_fast_signal:
        orchestrator._recognition_trace_context = {"capture_generation": 2}
        orchestrator._observe_local_rule_hint(fast, 2_900, frame_size=(64, 32))
    else:
        orchestrator._capture_generation = 2
    orchestrator.poll_deadlines()
    assert not window.turn_recovery_failed
    assert not window.turn_recovery_deadline_expired
    assert window.turn_recovery_processing_started_ms is None
    assert window.turn_recovery_deadline_identity == ()
    assert window.turn_recovery_pending, "unresolved history still must not admit a model"
    orchestrator.finish()


def test_deadline_listener_reentering_pause_has_no_publication_state_lock_inversion(tmp_path):
    callback_entered = threading.Event()
    worker_holds_state = threading.Event()
    worker_done = threading.Event()
    callback_done = threading.Event()
    orchestrator = _orchestrator(tmp_path, [], lead_player="right")

    def listener(update):
        if update.block_reason == "turn_recovery_budget_exceeded":
            callback_entered.set()
            orchestrator.pause()

    orchestrator._update_listener = listener
    window = orchestrator._ensure_turn_ownership_window()
    fast = FastSignalResult("right", "self", False, True, False)
    orchestrator._begin_turn_recovery(window, fast, 100, reason="test")
    orchestrator._withhold_advice_for_turn_recovery(window, 100)
    callback = orchestrator._deadline_timer.function
    orchestrator._deadline_timer.cancel()

    def state_worker():
        with orchestrator._state_lock:
            worker_holds_state.set()
            assert callback_entered.wait(1)
            orchestrator._update()
        worker_done.set()

    def run_callback():
        callback()
        callback_done.set()

    worker = threading.Thread(target=state_worker, daemon=True)
    timer = threading.Thread(target=run_callback, daemon=True)
    worker.start()
    assert worker_holds_state.wait(1)
    timer.start()
    assert worker_done.wait(1), "publication listener must not retain the small publication lock"
    assert callback_done.wait(1)
    assert orchestrator.status == "paused"
    orchestrator.finish()


def test_high_risk_verification_expires_without_new_capture(tmp_path):
    orchestrator, target, _advisor, submitted = _open_adjacent_action_reread(tmp_path)
    target.processing_opened_ms = 1_000
    orchestrator._processing_clock_ms = lambda: 2_200
    orchestrator.poll_deadlines()
    assert target.state == "expired"
    assert submitted
    orchestrator.finish()


def test_turn_recovery_waits_for_self_opportunity_beyond_capture_wall_clock(tmp_path):
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="self",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3C",), is_pass=False, monotonic_ms=10,
    )
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None and window.expected_player == "right"
    orchestrator._begin_turn_recovery(
        window,
        FastSignalResult("right", "opposite", False, False, False),
        100,
        reason="active_player_crossed_handoff",
    )

    result = orchestrator._advance_turn_recovery(
        window,
        FastSignalResult("right", "left", False, False, False),
        4_500,
        metrics=ZoneFrameMetrics(4_500, False, 0.0, False, False),
        decision=ZoneDecision(ZonePhase.WAIT_ACTION),
    )

    assert result is None
    assert window.turn_recovery_pending
    assert not window.turn_recovery_failed
    assert window.turn_recovery_processing_started_ms is None
    assert not any(
        event.event_type == "advice_recovery_failed"
        and event.payload.get("reason") == "capture_evidence_window_expired"
        for event in orchestrator.events
    )
    orchestrator.finish()


def test_ordinary_exact_play_does_not_arm_blocking_previous_action_reread(tmp_path):
    orchestrator = _orchestrator(tmp_path, [], lead_player="opposite")
    update = orchestrator.commit_trusted_action(
        actor="opposite", cards=("9C",), is_pass=False, monotonic_ms=10, confidence=.90,
    )
    assert update.event.event_id not in orchestrator._previous_action_verifications
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=20)
    assert orchestrator._previous_action_verification_target(orchestrator.snapshot) is None
    orchestrator.finish()


@pytest.mark.parametrize("case", ["fresh", "stale", "not_in_hand", "duplicate", "generation", "controls_visible", "effect", "late", "wrong_successor"])
def test_self_static_handoff_is_a_bounded_known_hand_probe_not_a_motion_bypass(tmp_path, case):
    class SelfPixels(FakeRecognitionService):
        active = "self"
        controls = True
        effect = False

        @staticmethod
        def play_roi(frame, seat):
            index = ("self", "right", "opposite", "left").index(seat)
            return frame[:, index * 16:(index + 1) * 16]

        def recognize_fast_signals(self, image, expected_player, **kwargs):
            return FastSignalResult(expected_player, self.active, False, self.controls, self.effect)

        def recognize_play_region(self, image, seat, **kwargs):
            self.targeted_calls += 1
            cards = ("small_joker",) if self.play_roi(image, seat).any() else ()
            return PlayRegionResult(seat, cards, False, .96 if cards else 0., (), (), source="self-pixel-code")

    recognition = SelfPixels([])
    hand = HAND if case == "not_in_hand" else (*HAND[:-1], "small_joker")
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition, lead_player="opposite", hand=hand, round_level="6")
    orchestrator.commit_trusted_action(actor="opposite", cards=("6D",), is_pass=False, monotonic_ms=10)
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=20)
    baseline = np.zeros((32, 64, 3), np.uint8)
    played = baseline.copy()
    played[3:29, 2:14] = 220
    for stamp in (100, 200):
        orchestrator.analyze_frame(played if case == "stale" else baseline, monotonic_ms=stamp,
            metrics=ZoneFrameMetrics(stamp, False, 0., False, False),
            trace_context={"capture_generation": 1, "capture_seq": stamp})
    before = orchestrator.snapshot
    recognition.active = "left" if case == "wrong_successor" else "right"
    recognition.controls = case == "controls_visible"
    recognition.effect = case == "effect"
    stamps = (300, 300) if case == "duplicate" else (300, 1_401) if case == "late" else (300, 400)
    for index, stamp in enumerate(stamps):
        update = orchestrator.analyze_frame(played, monotonic_ms=stamp,
            metrics=ZoneFrameMetrics(stamp, False, 0., False, recognition.effect),
            trace_context={"capture_generation": 2 if case == "generation" and index else 1, "capture_seq": stamp})
    if case == "fresh":
        assert update.event is not None and update.event.source == "self_static_handoff_known_hand"
        assert update.event.payload["cards"] == ["small_joker"]
        assert update.snapshot.current_player == "right"
        assert "small_joker" not in update.snapshot.my_hand
    else:
        assert orchestrator.snapshot == before
    assert recognition.targeted_calls <= 4
    orchestrator.finish()


@pytest.mark.parametrize("case,expected", [("golden_joker", "visible_nonpass"), ("changed_empty", "unknown"), ("exception", "unknown"), ("stable_empty", "empty")])
def test_pass_inference_distinguishes_weak_cards_and_unknown_from_stable_empty(tmp_path, case, expected):
    class SurfaceRecognition(FakeRecognitionService):
        def recognize_play_region(self, frame, seat, **kwargs):
            if case == "exception":
                raise RuntimeError("temporary matcher failure")
            cards = ("big_joker",) if case == "golden_joker" else ()
            return PlayRegionResult(seat, cards, False, .67 if cards else 0., (), (), source="surface-state-fixture")

    orchestrator = _orchestrator(tmp_path, [], recognition=SurfaceRecognition([]), lead_player="opposite")
    orchestrator.commit_trusted_action(actor="opposite", cards=("9C",), is_pass=False, monotonic_ms=10)
    window = orchestrator._ensure_turn_ownership_window()
    frame = np.zeros((32, 64, 3), np.uint8)
    window.surface_baseline_frames = {"left": frame.copy()}
    if case != "stable_empty":
        frame[:] = 200
    fast = FastSignalResult("left", "self", False, True, False)
    state = orchestrator._pass_inference_surface_state(window, orchestrator.reducer, frame, fast,
        ZoneFrameMetrics(100, True, 0., False, False), "left")
    assert state == expected
    window.temporal_confirmed_actives = ["left", "self"]
    before = orchestrator.snapshot
    candidates = orchestrator._temporal_turn_recovery_cycle(window, frame, fast,
        metrics=ZoneFrameMetrics(100, True, 0., False, False))
    assert orchestrator.snapshot == before
    assert (candidates is not None) == (case == "stable_empty")
    orchestrator.finish()


@pytest.mark.parametrize("expected,next_seat,leader", [
    ("self", "right", "left"), ("right", "opposite", "self"),
    ("opposite", "left", "right"), ("left", "self", "opposite"),
])
def test_normal_pass_rotations_bypass_generic_recovery(tmp_path, expected, next_seat, leader):
    class RotationSignals(FakeRecognitionService):
        def recognize_fast_signals(self, image, expected_player, **kwargs):
            self.fast_calls += 1
            passed = self.fast_calls >= 2
            active = next_seat if passed else expected
            return FastSignalResult(
                expected_player, active, passed, active == "self", False,
                pass_marker_player=expected if passed else None,
                pass_marker_players=(expected,) if passed else (),
            )

    recognition = RotationSignals([])
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition, lead_player=leader)
    orchestrator.commit_trusted_action(actor=leader, cards=("3C",), is_pass=False, monotonic_ms=10)
    frame = np.zeros((32, 64, 3), np.uint8)
    for stamp in (100, 200, 300):
        update = orchestrator.ingest_frame(frame, monotonic_ms=stamp, wall_time=str(stamp),
            metrics=ZoneFrameMetrics(stamp, False, 0.0, False, False))
    assert update.event is not None and update.event.event_type == "player_passed"
    assert update.event.actor == expected
    assert update.event.source == "unseen_direct_next_pass_marker"
    assert update.snapshot.current_player == next_seat
    assert not any(event.event_type == "advice_withheld" and event.payload.get("reason") == "turn_recovery_pending"
                   for event in orchestrator.events)
    assert recognition.targeted_calls == 0
    orchestrator.finish()


def test_pre_switch_surface_cache_recovers_already_static_fast_play(tmp_path):
    class StaticFastPlay(FakeRecognitionService):
        def play_roi(self, frame, seat):
            index = ("self", "right", "opposite", "left").index(seat)
            return frame[:, index * 16:(index + 1) * 16]

        def recognize_fast_signals(self, image, expected_player, **kwargs):
            return FastSignalResult(expected_player, "opposite", False, False, False)

    recognition = StaticFastPlay([_seat_play("right", "9C")] * 4)
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition, lead_player="self")
    baseline = np.zeros((32, 64, 3), np.uint8)
    changed = baseline.copy()
    changed[:, 16:32] = 255  # Right already played before self is formally committed.
    orchestrator._update_surface_generations(baseline, 100)
    orchestrator._update_surface_generations(changed, 200)
    orchestrator.commit_trusted_action(actor="self", cards=("3C",), is_pass=False, monotonic_ms=220)
    update = None
    for stamp in (300, 400, 500, 600):
        update = orchestrator.ingest_frame(changed, monotonic_ms=stamp, wall_time=str(stamp))
        if update.event is not None:
            break
    assert update is not None and update.event is not None
    assert update.event.actor == "right" and update.event.payload["cards"] == ["9C"]
    assert update.snapshot.current_player == "opposite"
    assert orchestrator._turn_evidence.retained_bytes <= 16 * 1024 * 1024
    orchestrator.finish()


def test_warm_cache_does_not_reread_consumed_static_cards_after_fast_full_cycle(tmp_path):
    class StaticResidual(FakeRecognitionService):
        def play_roi(self, frame, seat):
            index = ("self", "right", "opposite", "left").index(seat)
            return frame[:, index * 16:(index + 1) * 16]

        def recognize_fast_signals(self, image, expected_player, **kwargs):
            return FastSignalResult(expected_player, "opposite", False, False, False)

    recognition = StaticResidual([_seat_play("right", "9C")] * 10)
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition, lead_player="right")
    baseline = np.zeros((32, 64, 3), np.uint8)
    residual = baseline.copy()
    residual[:, 16:32] = 255
    orchestrator._update_surface_generations(baseline, 100)
    orchestrator._update_surface_generations(residual, 200)
    orchestrator.commit_trusted_action(actor="right", cards=("9C",), is_pass=False, monotonic_ms=220)
    for seat, stamp in (("opposite", 240), ("left", 260), ("self", 280)):
        orchestrator.commit_trusted_action(actor=seat, is_pass=True, monotonic_ms=stamp)
    assert orchestrator.snapshot.current_player == "right" and not orchestrator.snapshot.trick_plays
    for stamp in (300, 400, 500):
        orchestrator.ingest_frame(residual, monotonic_ms=stamp, wall_time=str(stamp))
    right_actions = [event for event in orchestrator.events if event.event_type == "player_played" and event.actor == "right"]
    assert len(right_actions) == 1
    assert recognition.targeted_calls == 0
    assert orchestrator.snapshot.current_player == "right"
    orchestrator.finish()


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


class CannotBeatRecognitionService(FakeRecognitionService):
    def __init__(
        self,
        signals: list[tuple[bool, float, tuple[int, int, int, int] | None, bool]],
        *,
        active_players: list[str | None] | None = None,
    ):
        super().__init__([])
        self.signals = list(signals)
        self.active_players = list(active_players or ["self"])
        self._index = 0

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        del allow_pass
        self.fast_calls += 1
        index = min(self._index, len(self.signals) - 1)
        visible, confidence, box, effect = self.signals[index]
        active = self.active_players[min(index, len(self.active_players) - 1)]
        self._index += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=active,
            pass_visible=False,
            self_action_buttons_visible=visible,
            effect_visible=effect,
            cannot_beat_visible=visible,
            cannot_beat_confidence=confidence,
            cannot_beat_box=box,
        )


class FastCycleRecognitionService(FakeRecognitionService):
    def __init__(self, first_cards: tuple[str, ...]):
        super().__init__([])
        self.first_cards = first_cards
        self.calls: list[str] = []

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        del allow_pass
        return FastSignalResult(
            expected_player=expected_player,
            active_player="self",
            pass_visible=False,
            self_action_buttons_visible=True,
            effect_visible=False,
            pass_marker_players=("opposite", "left"),
        )

    def recognize_play_region(
        self,
        _image,
        seat,
        *,
        wild_rank,
        allow_pass=True,
        allow_unknown_suit=True,
    ):
        del wild_rank, allow_pass, allow_unknown_suit
        self.calls.append(seat)
        if seat == "right":
            return PlayRegionResult(
                player=seat,
                cards=self.first_cards,
                is_pass=False,
                confidence=0.96,
                diagnostics=(),
                annotations=(),
                source="synthetic-fast-cycle",
            )
        return PlayRegionResult(
            player=seat,
            cards=(),
            is_pass=True,
            confidence=0.99,
            diagnostics=(),
            annotations=(),
            source="synthetic-seat-pass",
        )


class PixelFastCycleRecognitionService(FastCycleRecognitionService):
    _SLICES = {
        "self": slice(0, 16),
        "right": slice(16, 32),
        "opposite": slice(32, 48),
        "left": slice(48, 64),
    }

    def __init__(self, first_cards: tuple[str, ...]):
        super().__init__(first_cards)
        self.fast_index = 0

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        del allow_pass
        self.fast_index += 1
        returned = self.fast_index >= 2
        return FastSignalResult(
            expected_player=expected_player,
            active_player="self" if returned else expected_player,
            pass_visible=False,
            self_action_buttons_visible=returned,
            effect_visible=False,
            pass_marker_players=("opposite", "left") if returned else (),
        )

    def play_roi(self, image, seat):
        return image[:, self._SLICES[seat], :]

    def recognize_play_region(
        self,
        image,
        seat,
        *,
        wild_rank,
        allow_pass=True,
        allow_unknown_suit=True,
    ):
        if self.fast_index <= 1 and seat == "right":
            return PlayRegionResult(
                player=seat,
                cards=(),
                is_pass=False,
                confidence=0.0,
                diagnostics=(),
                annotations=(),
                source="empty-turn-baseline",
            )
        return super().recognize_play_region(
            image,
            seat,
            wild_rank=wild_rank,
            allow_pass=allow_pass,
            allow_unknown_suit=allow_unknown_suit,
        )


class SelfControlEdgeConsumptionRecognitionService(FakeRecognitionService):
    def __init__(self):
        super().__init__([])
        self.frames = [
            ("self", False),
            ("self", True),
            ("self", True),
            ("self", True),
            ("self", True),
        ]
        self.index = 0

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        del allow_pass
        index = min(self.index, len(self.frames) - 1)
        active, buttons = self.frames[index]
        self.index += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=active,
            pass_visible=False,
            self_action_buttons_visible=buttons,
            effect_visible=False,
        )

    def recognize_play_region(
        self,
        _image,
        seat,
        *,
        wild_rank,
        allow_pass=True,
        allow_unknown_suit=True,
    ):
        del wild_rank, allow_pass, allow_unknown_suit
        return PlayRegionResult(
            player=seat,
            cards=(),
            is_pass=False,
            confidence=0.0,
            diagnostics=(),
            annotations=(),
            source="empty-surface",
        )


class TemporalPassCycleRecognitionService(FakeRecognitionService):
    def __init__(self):
        super().__init__([])
        self.fast_frames = [
            ("self", True, ()),
            ("self", True, ()),
            ("right", False, ()),
            ("right", False, ()),
            ("opposite", False, ("right",)),
            ("opposite", False, ("right",)),
            ("left", False, ("right", "opposite")),
            ("left", False, ("right", "opposite")),
            ("self", True, ("right", "opposite")),
            ("self", True, ("right", "opposite")),
        ]
        self.fast_index = 0

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        del allow_pass
        index = min(self.fast_index, len(self.fast_frames) - 1)
        active, buttons, markers = self.fast_frames[index]
        self.fast_index += 1
        return FastSignalResult(
            expected_player=expected_player,
            active_player=active,
            pass_visible=expected_player in markers,
            self_action_buttons_visible=buttons,
            effect_visible=False,
            pass_marker_player=expected_player if expected_player in markers else None,
            pass_marker_players=markers,
        )

    def recognize_play_region(
        self,
        _image,
        seat,
        *,
        wild_rank,
        allow_pass=True,
        allow_unknown_suit=True,
    ):
        del wild_rank, allow_pass, allow_unknown_suit
        if seat == "right":
            return PlayRegionResult(
                player=seat,
                cards=("10C",),
                is_pass=False,
                confidence=0.95,
                diagnostics=(),
                annotations=(),
                source="residual-right-surface",
            )
        return PlayRegionResult(
            player=seat,
            cards=(),
            is_pass=False,
            confidence=0.0,
            diagnostics=(),
            annotations=(),
            source="empty-surface",
        )


class SuccessfulAdviceService:
    strategy_id = "synthetic-success"
    display_name = "合成建议"

    def __init__(self):
        self.calls = 0
        self.called = threading.Event()

    def recommend(self, state, *, request_id):
        self.calls += 1
        self.called.set()
        return AdviceResult(
            strategy=self.strategy_id,
            cards=(),
            play_type="PASS",
            is_pass=True,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
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


def test_live_orchestrator_satisfies_application_runtime_port(tmp_path):
    orchestrator = _orchestrator(tmp_path, [])
    try:
        assert isinstance(orchestrator, LiveRuntimePort)
    finally:
        orchestrator.finish()


def test_legacy_orchestrator_dto_imports_are_domain_aliases():
    from daguandan_bridge.live.orchestrator import LiveAdvice, LiveUpdate

    assert AdviceRequestKey is DomainAdviceRequestKey
    assert LiveAdvice is DomainLiveAdvice
    assert LiveUpdate is DomainLiveUpdate


def test_self_action_consumes_response_control_edge_before_next_foreign_turn(tmp_path):
    recognition = SelfControlEdgeConsumptionRecognitionService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="left",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="left", cards=("3C",), is_pass=False, monotonic_ms=10,
    )
    assert orchestrator.snapshot.current_player == "self"
    frame = np.zeros((32, 64, 3), np.uint8)

    orchestrator.ingest_frame(
        frame,
        monotonic_ms=100,
        wall_time="self-controls-absent",
        metrics=ZoneFrameMetrics(100, False, 0.0, False, False),
    )
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=200,
        wall_time="self-controls-visible-1",
        metrics=ZoneFrameMetrics(200, False, 0.0, False, False),
    )
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=300,
        wall_time="self-controls-visible-2",
        metrics=ZoneFrameMetrics(300, False, 0.0, False, False),
    )
    assert orchestrator._response_controls_streak >= 2

    orchestrator.commit_trusted_action(
        actor="self", cards=(), is_pass=True, monotonic_ms=350,
    )
    assert orchestrator.snapshot.current_player == "right"
    assert orchestrator._response_controls_visible is True
    assert orchestrator._response_controls_edge_identity == ()
    assert orchestrator._response_controls_streak == 0

    update = None
    for timestamp in (400, 500):
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"lingering-self-controls-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, False, 0.0, False, False),
        )

    assert update is not None
    assert update.snapshot.current_player == "right"
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None
    assert not window.turn_recovery_pending
    assert window.pre_recovery_chain_signature is None
    assert not any(
        event.event_type == "advice_withheld"
        and event.payload.get("reason") == "turn_recovery_pending"
        for event in orchestrator.events
    )
    orchestrator.finish()


@pytest.mark.parametrize("right_cards", (("10C",), ("5D",)))
def test_fast_foreign_cycle_is_atomically_recovered_before_local_advice(
    tmp_path,
    right_cards,
):
    recognition = PixelFastCycleRecognitionService(right_cards)
    advisor = SuccessfulAdviceService()
    # The user's log contains an earlier self AH action.  The recoverable
    # turn itself must still be rule-valid, so this minimal fixture uses a 3C
    # table card before the observed right 10C/5D response.
    hand = ("3C", "9S", *(card for card in HAND if card not in {"3C", "5D"}))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        hand=hand,
        settle_ms=0,
        recognition_strategy="two_valid_streak",
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="self",
        cards=("3C",),
        is_pass=False,
        monotonic_ms=10,
    )
    assert orchestrator.snapshot.current_player == "right"
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None
    baseline = np.zeros((32, 64, 3), np.uint8)
    frame = baseline.copy()
    frame[:, 16:32, :] = 255

    orchestrator.ingest_frame(
        baseline,
        monotonic_ms=200,
        wall_time="fast-cycle-baseline",
        metrics=ZoneFrameMetrics(200, False, 0.0, False, False),
    )
    first = orchestrator.ingest_frame(
        frame,
        monotonic_ms=300,
        wall_time="fast-cycle-first",
        metrics=ZoneFrameMetrics(300, True, 0.0, False, False),
    )
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=400,
        wall_time="fast-cycle-controls-confirmed",
        metrics=ZoneFrameMetrics(400, True, 0.0, False, False),
    )
    second = orchestrator.ingest_frame(
        frame,
        monotonic_ms=500,
        wall_time="fast-cycle-second",
        metrics=ZoneFrameMetrics(500, True, 0.0, False, False),
    )

    assert first.event is None
    assert second.snapshot.current_player == "self"
    actions = [
        event
        for event in orchestrator.events
        if event.event_type in {"player_played", "player_passed"}
    ]
    assert [(event.actor, event.payload.get("cards", [])) for event in actions[-4:]] == [
        ("self", ["3C"]),
        ("right", list(right_cards)),
        ("opposite", []),
        ("left", []),
    ]
    assert actions[-3].event_type == "player_played"
    assert actions[-2].event_type == "player_passed"
    assert actions[-1].event_type == "player_passed"
    assert recognition.calls[-6:] == [
        "right", "opposite", "left", "right", "opposite", "left"
    ]
    assert any(
        event.event_type == "turn_recovery_cycle_recovered"
        for event in orchestrator.events
    )
    assert any(
        event.event_type == "advice_requested"
        for event in orchestrator.events
    )
    recovered_ids = {
        event.event_id
        for event in orchestrator.events
        if event.event_type in {"player_played", "player_passed"}
        and event.source.startswith("turn_recovery_cycle")
    }
    reducer_by_id = {
        event.event_id: event
        for event in orchestrator.reducer.events
        if event.event_id in recovered_ids
    }
    published_by_id = {
        event.event_id: event
        for event in orchestrator.events
        if event.event_id in recovered_ids
    }
    assert set(reducer_by_id) == recovered_ids
    assert {
        event_id: event.to_dict()
        for event_id, event in reducer_by_id.items()
    } == {
        event_id: event.to_dict()
        for event_id, event in published_by_id.items()
    }

    semantic_events = tuple(
        event
        for event in orchestrator.events
        if event.event_type
        in {
            "initial_state_confirmed",
            "lead_player_confirmed",
            "player_played",
            "player_passed",
            "player_finished",
        }
    )
    replayed = EventReplayer(lambda: LiveReducer("game")).replay(semantic_events)
    assert replayed.final_snapshot == orchestrator.snapshot
    assert not any(
        event.event_type in {"terminal_history_gap", "turn_desynchronized"}
        for event in orchestrator.events
    )
    assert 400 - 300 <= 1_500
    assert orchestrator.wait_for_advice_idle(timeout=2.0)
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=500,
        wall_time="fast-cycle-advice-visible",
        metrics=ZoneFrameMetrics(500, False, 0.0, False, False),
    )
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.status == "ready"
    assert orchestrator.latest_advice.visible is True
    orchestrator.finish()
    health = json.loads(orchestrator.store.directory.joinpath("health_audit.json").read_text(encoding="utf-8"))
    assert health["status"] == "PASS"
    assert not any(
        issue["code"] == "HEALTH-ACTION-CHAIN-INCONSISTENT"
        for issue in health["issues"]
    )


def test_fast_cycle_never_commits_a_stale_non_pass_surface(tmp_path):
    recognition = PixelFastCycleRecognitionService(("10C",))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        hand=("3C", "9S", *(card for card in HAND if card not in {"3C", "5D"})),
        settle_ms=0,
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3C",), is_pass=False, monotonic_ms=10
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    for timestamp in (300, 400, 500):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"stale-fast-cycle-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.0, False, False),
        )

    assert orchestrator.snapshot.current_player == "right"
    assert not any(
        event.actor == "right"
        and event.event_type in {"player_played", "player_passed"}
        for event in orchestrator.events
    )
    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.withhold_reason == "turn_recovery_pending"
    orchestrator.finish()


def test_stale_non_pass_is_not_fresh_even_after_two_matching_reads(tmp_path):
    """Repeated recognition of the old card cannot substitute for a new surface."""

    recognition = PixelFastCycleRecognitionService(("10C",))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        hand=("3C", "9S", *(card for card in HAND if card not in {"3C", "5D"})),
        settle_ms=0,
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3C",), is_pass=False, monotonic_ms=10
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=200,
        wall_time="stale-read-baseline",
        metrics=ZoneFrameMetrics(200, False, 0.0, False, False),
    )
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None
    window.handoff_samples = [
        RecognitionSample(
            cards=("10C",),
            is_pass=False,
            confidence=0.95,
            source="stale-surface",
            evidence_ref=f"OBS-{index}",
        )
        for index in (1, 2)
    ]
    observation = recognition.recognize_play_region(
        frame,
        "right",
        wild_rank=orchestrator.snapshot.wild_rank,
    )
    candidate = ConsensusResult(
        status="confirmed",
        cards=("10C",),
        is_pass=False,
        confidence=0.95,
        source="stale-surface",
        vote_count=2,
        candidates=(),
    )

    assert not orchestrator._turn_recovery_non_pass_is_fresh(
        window,
        "right",
        candidate,
        frame,
        observation,
    )
    orchestrator.finish()


def test_recovery_freshness_prefers_blank_seat_roi_over_stale_full_frame_anchor(tmp_path):
    recognition = PixelFastCycleRecognitionService(("10C",))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        settle_ms=0,
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3C",), is_pass=False, monotonic_ms=10
    )
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None

    blank = np.zeros((32, 64, 3), np.uint8)
    current = blank.copy()
    current[:, 16:32, :] = 240
    # The old full-frame ring already contains the same visible card in the
    # absolute annotation box, reproducing the full-size stale-anchor failure.
    window.surface_baseline_full_frame = current.copy()
    window.surface_baseline_frames = {"right": blank[:, 16:32, :].copy()}
    candidate = ConsensusResult(
        status="confirmed",
        cards=("10C",),
        is_pass=False,
        confidence=0.95,
        source="synthetic",
        vote_count=2,
        candidates=(),
    )
    observation = PlayRegionResult(
        player="right",
        cards=("10C",),
        is_pass=False,
        confidence=0.95,
        diagnostics=(),
        annotations=(
            RecognitionAnnotation("10", (18, 6, 8, 10), 0.91, "play"),
        ),
        source="synthetic",
    )

    assert orchestrator._turn_recovery_non_pass_is_fresh(
        window,
        "right",
        candidate,
        current,
        observation,
    )
    orchestrator.finish()


def test_recovery_freshness_rejects_same_old_seat_roi_with_absolute_annotation(tmp_path):
    recognition = PixelFastCycleRecognitionService(("10C",))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        settle_ms=0,
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3C",), is_pass=False, monotonic_ms=10
    )
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None

    current = np.zeros((32, 64, 3), np.uint8)
    current[:, 16:32, :] = 240
    window.surface_baseline_full_frame = np.zeros_like(current)
    window.surface_baseline_frames = {"right": current[:, 16:32, :].copy()}
    candidate = ConsensusResult(
        status="confirmed",
        cards=("10C",),
        is_pass=False,
        confidence=0.95,
        source="synthetic",
        vote_count=2,
        candidates=(),
    )
    observation = PlayRegionResult(
        player="right",
        cards=("10C",),
        is_pass=False,
        confidence=0.95,
        diagnostics=(),
        annotations=(
            RecognitionAnnotation("10", (18, 6, 8, 10), 0.91, "play"),
        ),
        source="synthetic",
    )

    assert not orchestrator._turn_recovery_non_pass_is_fresh(
        window,
        "right",
        candidate,
        current,
        observation,
    )
    orchestrator.finish()


def test_real_pixel_change_creates_fresh_generation_and_recovers_cycle(tmp_path):
    recognition = PixelFastCycleRecognitionService(("10C",))
    hand = ("3C", "9S", *(card for card in HAND if card not in {"3C", "5D"}))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        hand=hand,
        settle_ms=0,
        recognition_strategy="two_valid_streak",
        advisor=SuccessfulAdviceService(),
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3C",), is_pass=False, monotonic_ms=10
    )
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None
    baseline = np.zeros((32, 64, 3), np.uint8)
    changed = baseline.copy()
    changed[:, 16:32, :] = 255

    orchestrator.ingest_frame(
        baseline,
        monotonic_ms=200,
        wall_time="pixel-baseline",
        metrics=ZoneFrameMetrics(200, False, 0.0, False, False),
    )
    assert orchestrator._surface_generation_by_seat["right"] == 0
    orchestrator.ingest_frame(
        changed,
        monotonic_ms=300,
        wall_time="pixel-fresh-first",
        metrics=ZoneFrameMetrics(300, True, 0.0, False, False),
    )
    assert orchestrator._surface_generation_by_seat["right"] == 1
    assert orchestrator._surface_generation_by_seat["opposite"] == 0
    orchestrator.ingest_frame(
        changed,
        monotonic_ms=400,
        wall_time="pixel-fresh-controls-confirmed",
        metrics=ZoneFrameMetrics(400, True, 0.0, False, False),
    )
    update = orchestrator.ingest_frame(
        changed,
        monotonic_ms=500,
        wall_time="pixel-fresh-second",
        metrics=ZoneFrameMetrics(500, True, 0.0, False, False),
    )

    assert update.snapshot.current_player == "self"
    actions = [
        event for event in orchestrator.events
        if event.event_type in {"player_played", "player_passed"}
    ]
    assert [(event.actor, event.event_type) for event in actions[-3:]] == [
        ("right", "player_played"),
        ("opposite", "player_passed"),
        ("left", "player_passed"),
    ]
    assert any(event.event_type == "advice_requested" for event in orchestrator.events)
    assert 500 - 300 <= 1_500
    orchestrator.finish()


def test_temporal_active_and_pass_edges_recover_real_sequence_without_residual_10c(
    tmp_path,
):
    recognition = TemporalPassCycleRecognitionService()
    hand = ("AH", *(card for card in HAND if card != "5D"))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        hand=hand,
        settle_ms=0,
        recognition_strategy="two_valid_streak",
        advisor=SuccessfulAdviceService(),
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("AH",), is_pass=False, monotonic_ms=10
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    update = None
    for index in range(10):
        timestamp = 100 + index * 100
        update = orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"temporal-pass-{index}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.0, False, False),
        )

    assert update is not None
    assert update.snapshot.current_player == "self"
    actions = [
        event for event in orchestrator.events
        if event.event_type in {"player_played", "player_passed"}
    ]
    assert [(event.actor, event.event_type) for event in actions[-4:]] == [
        ("self", "player_played"),
        ("right", "player_passed"),
        ("opposite", "player_passed"),
        ("left", "player_passed"),
    ]
    assert not any(
        event.actor == "right"
        and event.event_type == "player_played"
        and event.payload.get("cards") == ["10C"]
        for event in actions
    )
    left_pass = actions[-1]
    assert left_pass.source == "turn_recovery_derived_pass_from_active_transition"
    assert left_pass.payload["integrity_warnings"] == [
        "derived_pass_from_confirmed_active_transition"
    ]
    assert any(event.event_type == "advice_requested" for event in orchestrator.events)
    # Confirmed right at 400ms -> recovered on the second self frame at 1000ms.
    assert 1_000 - 400 <= 1_500
    orchestrator.finish()


@pytest.mark.parametrize("failure_point", (1, 2, 3, "store"))
def test_fast_cycle_transaction_failure_leaves_no_partial_history(
    tmp_path,
    monkeypatch,
    failure_point,
):
    recognition = PixelFastCycleRecognitionService(("10C",))
    hand = ("3C", "9S", *(card for card in HAND if card not in {"3C", "5D"}))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        hand=hand,
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    orchestrator.commit_trusted_action(
        actor="self", cards=("3C",), is_pass=False, monotonic_ms=10
    )
    window = orchestrator._ensure_turn_ownership_window()
    assert window is not None
    baseline = np.zeros((32, 64, 3), np.uint8)
    frame = baseline.copy()
    frame[:, 16:32, :] = 255
    orchestrator.ingest_frame(
        baseline,
        monotonic_ms=200,
        wall_time="atomic-failure-baseline",
        metrics=ZoneFrameMetrics(200, False, 0.0, False, False),
    )
    orchestrator.ingest_frame(
        frame,
        monotonic_ms=300,
        wall_time="atomic-failure-prime",
        metrics=ZoneFrameMetrics(300, True, 0.0, False, False),
    )
    # The first fresh new-control frame stores only a full-chain pre-vote. The
    # second distinct controls frame both authenticates the response and tests
    # the independent atomic card-chain transaction.
    before_snapshot = orchestrator.snapshot
    before_reducer_events = orchestrator.reducer.events
    before_all_events = tuple(orchestrator.events)
    before_timeline = orchestrator.store.timeline_path.read_bytes()
    before_advice = orchestrator.store.advice_path.read_bytes()
    before_verifications = deepcopy(orchestrator._previous_action_verifications)
    before_retirements = deepcopy(orchestrator._pending_previous_action_retirements)
    before_window = deepcopy(window.__dict__)
    before_generations = dict(orchestrator._surface_generation_by_seat)

    if failure_point == "store":
        monkeypatch.setattr(
            orchestrator.store,
            "append_event_batch",
            lambda _events: (_ for _ in ()).throw(RuntimeError("batch disk failure")),
        )
    else:
        original = orchestrator._stage_recovery_action
        calls = {"count": 0}

        def fail_staging(staged, candidate, index):
            calls["count"] += 1
            if index == failure_point:
                raise RuntimeError(f"stage {index} failure")
            return original(staged, candidate, index)

        monkeypatch.setattr(orchestrator, "_stage_recovery_action", fail_staging)

    with pytest.raises(RuntimeError):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=400,
            wall_time="atomic-failure-controls-confirmed",
            metrics=ZoneFrameMetrics(400, True, 0.0, False, False),
        )

    assert orchestrator.snapshot == before_snapshot
    assert orchestrator.reducer.events == before_reducer_events
    assert tuple(orchestrator.events) == before_all_events
    assert orchestrator.store.timeline_path.read_bytes() == before_timeline
    assert orchestrator.store.advice_path.read_bytes() == before_advice
    assert orchestrator._previous_action_verifications == before_verifications
    assert orchestrator._pending_previous_action_retirements == before_retirements
    for key, expected in before_window.items():
        actual = window.__dict__[key]
        if isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), key
        elif isinstance(expected, dict) and any(
            isinstance(value, np.ndarray) for value in expected.values()
        ):
            assert set(actual) == set(expected), key
            for nested_key, nested_expected in expected.items():
                assert np.array_equal(actual[nested_key], nested_expected), (
                    key,
                    nested_key,
                )
        else:
            assert actual == expected, key
    assert orchestrator._surface_generation_by_seat == before_generations
    orchestrator.finish()


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


def test_direct_handoff_crossing_does_not_relabel_late_sample_as_pre_recovery(tmp_path):
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

    # The second read happens after recovery began, so these two reads cannot
    # prove a confirmed handoff from before the active-seat crossing.
    assert updates[-1].event is None
    assert orchestrator.snapshot.current_player == "left"
    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.expected_player == "left"
    assert not any(event.event_type == "player_passed" for event in orchestrator.events)
    traces = read_json_lines(orchestrator.store.recognition_trace_path)
    assert not any(
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


def test_turn_recovery_has_two_second_terminal_without_global_latch(
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
    assert window.turn_recovery_local_deadline_ms == 2_300
    assert window.turn_recovery_target_exceeded is True
    assert window.turn_recovery_failed is True
    assert orchestrator._advice_suspended_reason is None
    assert not any(
        event.event_type == "turn_desynchronized" for event in orchestrator.events
    )
    terminals = [event for event in orchestrator.events if event.event_type == "advice_recovery_failed"]
    assert len(terminals) == 1
    assert terminals[0].payload["response_budget_ms"] == 2_000
    assert orchestrator.latest_advice.withhold_reason in {"turn_recovery_expired", "turn_recovery_budget_exceeded"}
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
        [_seat_play("opposite", "6S")] * 5,
        ["right"] * 5,
        pass_marker_players=[
            "opposite",
            "opposite",
            None,
            "opposite",
            "opposite",
        ],
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
        for index, timestamp in enumerate((100, 200, 300, 400, 500))
    ]

    assert updates[1].event is None
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


@pytest.mark.parametrize("case", [
    "persistent_left_pass", "persistent_left_pass_with_residual_j", "missing_prefix_marker",
    "stale_prefix_marker", "one_prefix_capture", "missing_tail_handoff", "wrong_tail_handoff",
    "no_self_controls", "fresh_legal_left_bomb",
])
def test_partial_timer_chain_does_not_relax_the_complete_temporal_cycle_gate(tmp_path, case):
    """20260814 frames617..628: opposite timer hidden by straight-flush effect.

    Only left->self timers survive. Opposite has a newly confirmed PASS; a
    persistent left badge must not need a disappearance animation, but neither
    the old badge alone nor a fresh legal left play can be inferred as PASS.
    """
    left_cards = (
        ("AC", "AC", "AD", "AD", "AH", "AS") if case == "fresh_legal_left_bomb"
        else ("JC",) if case == "persistent_left_pass_with_residual_j" else ()
    )
    recognition = FakeRecognitionService([PlayRegionResult(
        "left", left_cards, False, .95 if left_cards else 0., (), (), source="fixture-left-surface",
    )])
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition, lead_player="right", round_level="6")
    orchestrator._last_pass_marker_players = frozenset({"self", "left"})
    orchestrator.commit_trusted_action(
        actor="right", cards=("6C", "7C", "6H", "9C", "10C"), is_pass=False, monotonic_ms=10,
    )
    window = orchestrator._ensure_turn_ownership_window()
    window.temporal_confirmed_actives = ["left", "self"]
    window.temporal_confirmed_passes = {"opposite"}
    window.temporal_pass_seen_absent = {"opposite"}
    window.temporal_pass_marker_streaks = {"opposite": 2}
    frame = np.zeros((32, 64, 3), np.uint8)
    window.surface_baseline_full_frame = frame.copy()
    window.surface_baseline_frames = {"left": frame.copy()}
    if case == "missing_prefix_marker":
        window.temporal_confirmed_passes.clear()
    elif case == "stale_prefix_marker":
        window.temporal_pass_seen_absent.clear()
    elif case == "one_prefix_capture":
        window.temporal_pass_marker_streaks["opposite"] = 1
    elif case == "missing_tail_handoff":
        window.temporal_confirmed_actives = ["left"]
    elif case == "wrong_tail_handoff":
        window.temporal_confirmed_actives = ["right", "self"]
    elif case == "fresh_legal_left_bomb":
        frame[:] = 255
    fast = FastSignalResult("opposite", "self", True, case != "no_self_controls", False,
        pass_marker_player="opposite", pass_marker_players=("opposite", "left"))
    before = orchestrator.snapshot
    candidates = orchestrator._temporal_turn_recovery_cycle(
        window, frame, fast, metrics=ZoneFrameMetrics(1_000, True, 0., False, False),
    )
    assert orchestrator.snapshot == before, "proof must remain a staged preview"
    # Partial timers do not authorize this all-PASS temporal path. The static
    # path must independently prove badge disappearance/reappearance, or the
    # separate last-response path needs a preceding formal PASS + new controls.
    assert candidates is None
    orchestrator.finish()


@pytest.mark.parametrize("freshness", ["none", "absence_only", "single_marker", "two_markers"])
def test_static_cycle_replaces_old_pass_baseline_only_after_real_new_marker_lifecycle(tmp_path, freshness):
    recognition = FakeRecognitionService([
        PlayRegionResult(seat, (), True, .99, (), (), source="new-pass-marker")
        for seat in ("opposite", "left")
    ])
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition, lead_player="right", round_level="6")
    orchestrator._last_pass_marker_players = frozenset({"self", "left"})
    orchestrator.commit_trusted_action(actor="right", cards=("6C", "7C", "6H", "9C", "10C"),
                                      is_pass=False, monotonic_ms=10)
    window = orchestrator._ensure_turn_ownership_window()
    assert "left" in window.pass_marker_baseline
    if freshness != "none":
        window.temporal_pass_seen_absent = {"left"}
    if freshness in {"single_marker", "two_markers"}:
        window.temporal_pass_marker_streaks = {"left": 1 if freshness == "single_marker" else 2}
        window.temporal_confirmed_passes = {"left"}
    fast = FastSignalResult("opposite", "self", True, True, False,
        pass_marker_player="opposite", pass_marker_players=("opposite", "left"))
    before = orchestrator.snapshot
    result = orchestrator._scan_turn_recovery_cycle(window, np.zeros((32, 64, 3), np.uint8), fast,
        metrics=ZoneFrameMetrics(1_000, True, 0., False, False))
    assert orchestrator.snapshot == before
    if freshness == "two_markers":
        assert result is not None and [(seat, candidate.is_pass) for seat, candidate in result] == [("opposite", True), ("left", True)]
    else:
        assert result is None
    orchestrator.finish()


def test_opposite_head_wind_catch_reaches_self_before_generic_recovery(
    tmp_path, monkeypatch,
):
    """对家头游后，右家最后一个 PASS 必须直接接风给自己。"""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("right", "6S")] * 3,
        ["self", "self", "self"],
        pass_marker_players=[None, "right", "right"],
    )
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="opposite",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="opposite", cards=HAND, is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(
        actor="left", is_pass=True, monotonic_ms=20
    )
    orchestrator.commit_trusted_action(
        actor="self", is_pass=True, monotonic_ms=30
    )
    assert orchestrator.snapshot.finished_seats == frozenset({"opposite"})
    assert orchestrator.snapshot.current_player == "right"
    assert orchestrator.reducer.wind_receiver_after_current_pass("right") == "self"
    monkeypatch.setattr(
        orchestrator, "_probe_previous_action", lambda *_args, **_kwargs: None
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    updates = [
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"opposite-head-wind-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, True, 0.001, True, False),
        )
        for timestamp in (100, 200, 300)
    ]

    assert updates[-1].event is not None
    assert updates[-1].event.actor == "right"
    assert updates[-1].event.source == "wind_catch_pass_marker"
    assert orchestrator.snapshot.current_player == "self"
    assert orchestrator.snapshot.lead_player == "self"
    assert any(
        event.event_type == "wind_caught"
        and event.payload == {"from_player": "opposite", "to_player": "self"}
        for event in updates[-1].events
    )
    assert not any(
        event.event_type == "advice_withheld"
        and event.payload.get("reason") == "turn_recovery_pending"
        for event in orchestrator.events
    )
    assert any(event.event_type == "advice_requested" for event in orchestrator.events)
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
    window = orchestrator._turn_ownership_window
    assert window.turn_recovery_pending is False
    assert window.wind_catch_pass_recovery_pending is False
    assert window.wind_catch_pass_recovery_failed is True
    assert window.wind_catch_pass_recovery_failure_reason == (
        "wind_catch_pass_marker_deadline"
    )
    assert window.wind_catch_pass_recovery_deadline_ms == 1_900
    assert not any(
        event.event_type == "advice_withheld"
        and event.payload.get("reason") == "turn_recovery_pending"
        for event in orchestrator.events
    )
    wind_withheld = [
        event
        for event in orchestrator.events
        if event.event_type == "advice_withheld"
        and event.payload.get("reason") == "wind_catch_pass_recovery_pending"
    ]
    assert len(wind_withheld) == 1
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


def test_late_opposite_head_placement_gets_one_short_wind_recovery_attempt(
    tmp_path,
):
    """现场末段：对家头游标志晚到时，不得再启动普通八秒恢复。"""

    recognition = ScheduledActiveRecognitionService(
        [_seat_play("right", "6S")] * 6,
        ["self", "self", "left", "right", None, "self"],
        pass_marker_players=[None] * 6,
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="opposite",
        settle_ms=0,
        action_timeout_ms=20_000,
    )
    orchestrator.commit_trusted_action(
        actor="opposite",
        cards=("8C", "8D", "8H", "8S"),
        is_pass=False,
        monotonic_ms=10,
    )
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=20)
    orchestrator.commit_trusted_action(actor="self", is_pass=True, monotonic_ms=30)
    assert orchestrator.snapshot.current_player == "right"
    placement = FastSignalResult(
        expected_player="right",
        active_player="self",
        pass_visible=False,
        self_action_buttons_visible=True,
        effect_visible=False,
        placements=(
            PlacementSignal(
                player="opposite",
                placement="head",
                confidence=0.98,
                source="late-head-fixture",
            ),
        ),
    )
    assert orchestrator._apply_visual_placements(placement) == ()
    assert len(orchestrator._apply_visual_placements(placement)) == 1
    assert orchestrator.snapshot.finished_seats == frozenset({"opposite"})
    assert orchestrator.reducer.wind_receiver_after_current_pass("right") == "self"

    frame = np.zeros((32, 64, 3), np.uint8)
    for timestamp in (100, 1_000, 2_000, 2_100, 2_200, 10_000):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"late-opposite-head-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, False, 0.0, False, False),
        )

    window = orchestrator._turn_ownership_window
    assert window is not None
    assert window.wind_catch_pass_recovery_failed is True
    assert window.turn_recovery_pending is False
    assert window.disposition == "wind_catch_pass_recovery_failed"
    assert window.wind_catch_pass_recovery_detected_ms == 100
    assert window.wind_catch_pass_recovery_deadline_ms == 1_900
    assert orchestrator.latest_advice is not None
    assert "未能确认接风前右家的不出" in orchestrator.latest_advice.error
    assert sum(
        event.event_type == "advice_withheld"
        and event.payload.get("reason") == "wind_catch_pass_recovery_pending"
        for event in orchestrator.events
    ) == 1
    failures = [
        event
        for event in orchestrator.events
        if event.event_type == "wind_catch_pass_recovery_failed"
    ]
    assert len(failures) == 1
    assert failures[0].payload["reason"] == "wind_catch_pass_marker_deadline"
    assert failures[0].payload["elapsed_ms"] <= 1_900
    assert not any(
        event.event_type == "advice_recovery_target_exceeded"
        for event in orchestrator.events
    )
    orchestrator.finish()


@pytest.mark.parametrize("active_players", (["self", "self"], [None, None]))
def test_cannot_beat_two_stable_frames_advise_pass_without_advancing_turn(
    tmp_path,
    active_players,
):
    recognition = CannotBeatRecognitionService(
        [
            (True, 0.94, (410, 620, 88, 36), False),
            (True, 0.93, (412, 619, 88, 36), False),
        ],
        active_players=active_players,
    )
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    orchestrator._request_advice_if_needed()
    assert orchestrator.snapshot.current_player == "self"
    assert advisor.calls == 0
    before = orchestrator.snapshot
    requested_before = sum(
        event.event_type == "advice_requested" for event in orchestrator.events
    )
    frame = np.zeros((32, 64, 3), np.uint8)
    first = orchestrator.ingest_frame(
        frame,
        monotonic_ms=100,
        wall_time="cannot-beat-first",
        metrics=ZoneFrameMetrics(100, False, 0.0, False, False),
    )
    assert first.event is None
    assert advisor.calls == 0
    second = orchestrator.ingest_frame(
        frame,
        monotonic_ms=200,
        wall_time="cannot-beat-second",
        metrics=ZoneFrameMetrics(200, False, 0.0, False, False),
    )

    assert second.event is not None
    assert second.event.event_type == "button_advice_ready"
    assert second.advice.status == "ready"
    assert second.advice.visible is True
    assert second.advice.advice.is_pass is True
    assert second.advice.advice.strategy == "button_cannot_beat"
    assert second.event.payload["model_required"] is False
    assert second.event.payload["model_call_skipped_for_button"] is True
    assert second.event.payload["model_requested_for_state"] is False
    assert "model_called" not in second.event.payload
    assert second.advice.advice.engine_input["model_requested_for_state"] is False
    assert orchestrator.snapshot == before
    assert sum(
        event.event_type == "advice_requested" for event in orchestrator.events
    ) == requested_before
    assert not advisor.called.wait(0.40)
    assert advisor.calls == 0
    for timestamp in (300, 400, 500):
        orchestrator.ingest_frame(
            frame, monotonic_ms=timestamp, wall_time=str(timestamp),
            metrics=ZoneFrameMetrics(timestamp, False, 0.0, False, False),
        )
    assert orchestrator.snapshot == before
    assert sum(e.event_type == "button_advice_ready" for e in orchestrator.events) == 1
    assert not any(e.event_type == "player_passed" and e.actor == "self"
                   for e in orchestrator.events)
    # Only a later seat-bound execution marker may become a formal action.
    orchestrator.recognition_service = ScheduledActiveRecognitionService(
        [_pass("self")] * 4, ["right"] * 4,
        pass_marker_players=["self"] * 4,
    )
    for index, timestamp in enumerate((600, 700, 800, 900)):
        orchestrator.ingest_frame(
            frame, monotonic_ms=timestamp, wall_time=str(timestamp),
            metrics=ZoneFrameMetrics(timestamp, False, 0.0, True, False),
        )
        if orchestrator.snapshot.current_player != "self":
            break
    passed = [e for e in orchestrator.events
              if e.event_type == "player_passed" and e.actor == "self"]
    assert len(passed) == 1
    assert passed[0].source != "button_cannot_beat"
    assert orchestrator.snapshot.current_player == "right"
    orchestrator.finish()


def _button_response_fixture(tmp_path, advisor=None):
    advisor = advisor or SuccessfulAdviceService()
    recognition = CannotBeatRecognitionService([(True, 0.95, (410, 620, 88, 36), False)])
    orchestrator = _orchestrator(tmp_path, [], recognition=recognition, advisor=advisor, settle_ms=0)
    orchestrator.commit_trusted_action(actor="right", cards=("3S",), is_pass=False, monotonic_ms=10)
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    fast = recognition.recognize_fast_signals(np.zeros((32, 64, 3), np.uint8), "self")
    return orchestrator, advisor, fast


def test_cannot_beat_duplicate_capture_timestamp_never_counts_as_two_frames(tmp_path):
    orchestrator, advisor, fast = _button_response_fixture(tmp_path)
    before = orchestrator.snapshot
    for timestamp in (100, 100, 99, 100):
        assert orchestrator._advance_cannot_beat_confirmation(fast, timestamp) is None
    assert orchestrator._turn_ownership_window.cannot_beat_streak == 1
    confirmed = orchestrator._advance_cannot_beat_confirmation(fast, 101)
    assert confirmed.event.event_type == "button_advice_ready"
    assert orchestrator.snapshot == before
    assert advisor.calls == 0
    orchestrator.finish()


@pytest.mark.parametrize("overrides", [
    {"cannot_beat_confidence": float("inf")},
    {"cannot_beat_confidence": float("-inf")},
    {"cannot_beat_confidence": float("nan")},
    {"cannot_beat_confidence": 1.00001},
    {"cannot_beat_confidence": -0.5},
    {"cannot_beat_confidence": "invalid"},
    {"cannot_beat_confidence": True},
    {"cannot_beat_box": (float("inf"), 620, 88, 36)},
    {"cannot_beat_box": (410, float("nan"), 88, 36)},
    {"cannot_beat_box": (410, 620, float("inf"), 36)},
    {"cannot_beat_box": (410, 620, -88, 36)},
    {"cannot_beat_box": (410, 620, 88, 0)},
    {"cannot_beat_box": (410, 620, 88, -1)},
    {"cannot_beat_box": (-1, 620, 88, 36)},
    {"cannot_beat_box": (410.5, 620, 88, 36)},
    {"cannot_beat_box": (410, 620, 88)},
    {"cannot_beat_box": (410, 620, 88, 36, 0)},
    {"cannot_beat_box": "410,620,88,36"},
    {"cannot_beat_box": (410, 620, 88, None)},
    {"cannot_beat_box": (410, 620, 88, True)},
    {"cannot_beat_box": (410, 620, 88, 2**10000)},
])
def test_invalid_button_numeric_evidence_revokes_half_vote_even_at_same_timestamp(tmp_path, overrides):
    orchestrator, advisor, fast = _button_response_fixture(tmp_path)
    before = orchestrator.snapshot
    assert orchestrator._advance_cannot_beat_confirmation(fast, 100) is None
    invalid = replace(fast, **overrides)
    assert orchestrator._advance_cannot_beat_confirmation(invalid, 100) is None
    window = orchestrator._turn_ownership_window
    assert window.cannot_beat_streak == 0
    assert window.cannot_beat_last_capture_ms == 100
    assert orchestrator._button_advice_key is None
    assert not any(event.event_type == "button_advice_ready" for event in orchestrator.events)
    # Replaying the valid version of the already-invalidated capture cannot
    # restore its half vote; a later single capture is still insufficient.
    assert orchestrator._advance_cannot_beat_confirmation(fast, 100) is None
    assert window.cannot_beat_streak == 0
    assert orchestrator._advance_cannot_beat_confirmation(fast, 200) is None
    assert window.cannot_beat_streak == 1
    assert orchestrator._advance_cannot_beat_confirmation(fast, 201).event.event_type == "button_advice_ready"
    assert advisor.calls == 0
    assert orchestrator.snapshot == before
    orchestrator.finish()


@pytest.mark.parametrize("confidence", [0.8, 1.0])
def test_cannot_beat_confidence_closed_valid_range_remains_supported(tmp_path, confidence):
    orchestrator, advisor, fast = _button_response_fixture(tmp_path)
    fast = replace(fast, cannot_beat_confidence=confidence)
    assert orchestrator._advance_cannot_beat_confirmation(fast, 100) is None
    assert orchestrator._advance_cannot_beat_confirmation(fast, 200).event.event_type == "button_advice_ready"
    assert advisor.calls == 0
    orchestrator.finish()


@pytest.mark.parametrize("overrides", [
    {"cannot_beat_confidence": float("inf")},
    {"cannot_beat_box": (410, 620, float("nan"), 36)},
    {"cannot_beat_box": (410, 620, 88, -36)},
])
def test_invalid_button_control_releases_preflight_to_normal_model(tmp_path, overrides):
    orchestrator, advisor, fast = _button_response_fixture(tmp_path)
    orchestrator._request_advice_if_needed()
    assert orchestrator._self_response_preflight_key is not None
    fast = replace(fast, **overrides)
    assert orchestrator._advance_cannot_beat_confirmation(fast, 100) is None
    orchestrator._resolve_self_response_preflight_from_fast(fast, 100)
    assert orchestrator.wait_for_advice_idle(timeout=3)
    assert advisor.calls == 1
    assert orchestrator.latest_advice.advice.strategy == "synthetic-success"
    assert not any(event.event_type == "button_advice_ready" for event in orchestrator.events)
    orchestrator.finish()


def test_confirmed_button_disappears_without_model_inflight_restarts_real_advice(tmp_path):
    orchestrator, advisor, fast = _button_response_fixture(tmp_path)
    before = orchestrator.snapshot
    orchestrator._advance_cannot_beat_confirmation(fast, 100)
    orchestrator._advance_cannot_beat_confirmation(fast, 200)
    assert orchestrator.latest_advice.advice.strategy == "button_cannot_beat"
    orchestrator._advance_cannot_beat_confirmation(replace(fast, cannot_beat_visible=False), 300)
    assert orchestrator.wait_for_advice_idle(timeout=3)
    assert advisor.calls == 1
    assert orchestrator.latest_advice.status == "ready"
    assert orchestrator.latest_advice.advice.strategy == "synthetic-success"
    assert orchestrator.snapshot == before
    orchestrator.finish()


def test_pause_clears_confirmed_button_and_resume_does_not_leave_stale_advice(tmp_path):
    orchestrator, advisor, fast = _button_response_fixture(tmp_path)
    orchestrator._advance_cannot_beat_confirmation(fast, 100)
    orchestrator._advance_cannot_beat_confirmation(fast, 200)
    update = orchestrator.pause()
    assert orchestrator._button_advice_key is None
    assert update.advice.status != "ready"
    assert update.advice.visible is False
    assert orchestrator._turn_ownership_window.cannot_beat_last_capture_ms is None
    orchestrator.resume(monotonic_ms=300)
    assert orchestrator.wait_for_advice_idle(timeout=3)
    assert advisor.calls == 1
    assert orchestrator.latest_advice.advice.strategy == "synthetic-success"
    orchestrator.finish()


def test_late_model_discarded_for_button_can_be_requested_again_when_button_disappears(tmp_path):
    release = threading.Event()

    class GatedAdvisor(SuccessfulAdviceService):
        def recommend(self, state, *, request_id):
            self.called.set()
            assert release.wait(3)
            return super().recommend(state, request_id=request_id)

    advisor = GatedAdvisor()
    orchestrator, _, fast = _button_response_fixture(tmp_path, advisor)
    orchestrator._request_advice_if_needed(bypass_response_preflight=True)
    assert advisor.called.wait(1)
    orchestrator._advance_cannot_beat_confirmation(fast, 100)
    confirmed = orchestrator._advance_cannot_beat_confirmation(fast, 200)
    assert confirmed.event.payload["model_required"] is False
    assert confirmed.event.payload["model_call_skipped_for_button"] is True
    assert confirmed.event.payload["model_requested_for_state"] is True
    assert "model_called" not in confirmed.event.payload
    assert orchestrator.latest_advice.advice.engine_input["model_requested_for_state"] is True
    release.set()
    assert orchestrator.wait_for_advice_idle(timeout=3)
    assert orchestrator.latest_advice.advice.strategy == "button_cannot_beat"
    assert not orchestrator._pending_model_advice
    orchestrator._advance_cannot_beat_confirmation(replace(fast, cannot_beat_visible=False), 300)
    assert orchestrator.wait_for_advice_idle(timeout=3)
    assert advisor.calls == 2
    assert orchestrator.latest_advice.status == "ready"
    assert orchestrator.latest_advice.advice.strategy == "synthetic-success"
    orchestrator.finish()


def test_button_disappearing_while_model_inflight_does_not_create_duplicate_job(tmp_path):
    release = threading.Event()

    class GatedAdvisor(SuccessfulAdviceService):
        def recommend(self, state, *, request_id):
            self.called.set()
            assert release.wait(3)
            return super().recommend(state, request_id=request_id)

    advisor = GatedAdvisor()
    orchestrator, _, fast = _button_response_fixture(tmp_path, advisor)
    orchestrator._request_advice_if_needed(bypass_response_preflight=True)
    assert advisor.called.wait(1)
    orchestrator._advance_cannot_beat_confirmation(fast, 100)
    orchestrator._advance_cannot_beat_confirmation(fast, 200)
    orchestrator._advance_cannot_beat_confirmation(replace(fast, cannot_beat_visible=False), 300)
    assert orchestrator.latest_advice.status == "requested"
    release.set()
    assert orchestrator.wait_for_advice_idle(timeout=3)
    assert advisor.calls == 1
    assert orchestrator.latest_advice.advice.strategy == "synthetic-success"
    orchestrator.finish()


def test_existing_button_advice_never_bypasses_new_recovery_hold(tmp_path):
    orchestrator, advisor, fast = _button_response_fixture(tmp_path)
    before = orchestrator.snapshot
    orchestrator._advance_cannot_beat_confirmation(fast, 100)
    orchestrator._advance_cannot_beat_confirmation(fast, 200)
    orchestrator._turn_ownership_window.turn_recovery_pending = True
    assert orchestrator._request_advice_if_needed(bypass_response_preflight=True) is None
    assert orchestrator._button_advice_key is None
    assert orchestrator.latest_advice.status != "ready"
    assert orchestrator.latest_advice.visible is False
    assert advisor.calls == 0
    assert orchestrator.snapshot == before
    assert orchestrator._request_advice_if_needed(bypass_response_preflight=True) is None
    assert advisor.calls == 0
    orchestrator.finish()


def test_terminal_signal_stops_frame_admission_before_sealing(tmp_path):
    orchestrator = _orchestrator(tmp_path, [])
    frame = np.zeros((32, 64, 3), np.uint8)
    orchestrator.record_frame(frame, monotonic_ms=1, wall_time="before")
    count = orchestrator.recorder.frame_count
    orchestrator._game_end_detected = True
    for timestamp in range(2, 20):
        orchestrator.record_frame(frame, monotonic_ms=timestamp, wall_time="settlement")
    assert orchestrator.recorder.frame_count == count
    assert len(read_json_lines(orchestrator.recorder.index_path)) == count
    orchestrator.finish()


def test_cannot_beat_foreign_active_never_commits_local_pass(tmp_path):
    recognition = CannotBeatRecognitionService(
        [
            (True, 0.94, (410, 620, 88, 36), False),
            (True, 0.93, (411, 620, 88, 36), False),
        ],
        active_players=["right", "right"],
    )
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
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)

    frame = np.zeros((32, 64, 3), np.uint8)
    for timestamp in (100, 200):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"cannot-beat-foreign-{timestamp}",
            metrics=ZoneFrameMetrics(timestamp, False, 0.0, False, False),
        )

    assert not any(
        event.event_type == "player_passed"
        and event.source == "button_cannot_beat"
        for event in orchestrator.events
    )
    orchestrator.finish()


@pytest.mark.parametrize("interruption", ["pause", "capture_interrupted"])
def test_cannot_beat_confirmation_does_not_cross_resumable_interruption(
    tmp_path,
    interruption,
):
    recognition = CannotBeatRecognitionService(
        [(True, 0.95, (10, 20, 80, 32), False)] * 3
    )
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
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    frame = np.zeros((32, 64, 3), np.uint8)
    first = orchestrator.ingest_frame(
        frame,
        monotonic_ms=100,
        wall_time="cannot-beat-before-interruption",
        metrics=ZoneFrameMetrics(100, False, 0.0, False, False),
    )
    assert first.event is None
    assert orchestrator._turn_ownership_window.cannot_beat_streak == 1

    if interruption == "pause":
        orchestrator.pause()
    else:
        orchestrator.capture_interrupted("test", monotonic_ms=150)
    assert orchestrator._turn_ownership_window.cannot_beat_streak == 0
    orchestrator.resume(monotonic_ms=200)
    second = orchestrator.ingest_frame(
        frame,
        monotonic_ms=300,
        wall_time="cannot-beat-after-interruption-first",
        metrics=ZoneFrameMetrics(300, False, 0.0, False, False),
    )
    assert second.event is None
    third = orchestrator.ingest_frame(
        frame,
        monotonic_ms=400,
        wall_time="cannot-beat-after-interruption-second",
        metrics=ZoneFrameMetrics(400, False, 0.0, False, False),
    )
    assert third.event is not None
    assert third.event.event_type == "button_advice_ready"
    assert third.advice.advice.strategy == "button_cannot_beat"
    assert orchestrator.snapshot.current_player == "self"
    orchestrator.finish()


@pytest.mark.parametrize("interruption", ["begin_finalizing", "finish"])
def test_cannot_beat_confirmation_is_cleared_by_terminal_interruption(
    tmp_path,
    interruption,
):
    recognition = CannotBeatRecognitionService(
        [(True, 0.95, (10, 20, 80, 32), False)]
    )
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
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="cannot-beat-before-terminal",
        metrics=ZoneFrameMetrics(100, False, 0.0, False, False),
    )
    window = orchestrator._turn_ownership_window
    assert window is not None and window.cannot_beat_streak == 1

    getattr(orchestrator, interruption)()
    assert window.cannot_beat_streak == 0
    if interruption == "begin_finalizing":
        orchestrator.finish()


def test_response_preflight_seen_turn_waits_until_transient_recovery_gate_clears(
    tmp_path,
):
    orchestrator, target, _advisor, submitted = _open_adjacent_action_reread(
        tmp_path
    )
    response_turn = (
        orchestrator.snapshot.session_id,
        orchestrator.snapshot.turn_id,
    )

    assert orchestrator._request_advice_if_needed() is None
    assert orchestrator._self_response_preflight_seen_turn is None
    assert orchestrator._self_response_preflight_key is None
    assert orchestrator._request_advice_if_needed(bypass_response_preflight=True) is None
    assert orchestrator._self_response_preflight_seen_turn is None

    target.state = "expired"
    assert (
        orchestrator._request_advice_if_needed(bypass_response_preflight=True)
        is not None
    )
    assert len(submitted) == 1
    assert orchestrator._self_response_preflight_key is None
    assert orchestrator._self_response_preflight_seen_turn == response_turn
    orchestrator.finish()


def test_self_response_preflight_releases_model_on_first_non_cannot_frame(
    tmp_path,
):
    recognition = CannotBeatRecognitionService(
        [(False, 0.0, None, False)]
    )
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    orchestrator._request_advice_if_needed()
    assert advisor.calls == 0

    orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="preflight-non-cannot",
        metrics=ZoneFrameMetrics(100, False, 0.0, False, False),
    )
    assert advisor.called.wait(1.0)
    assert orchestrator.wait_for_advice_idle(timeout=1.0)
    assert advisor.calls == 1
    assert sum(
        event.event_type == "advice_requested" for event in orchestrator.events
    ) == 1
    orchestrator.finish()


def test_self_response_preflight_hard_deadline_releases_model_by_300ms(
    tmp_path,
):
    recognition = CannotBeatRecognitionService(
        [(True, 0.95, (10, 20, 80, 32), False)]
    )
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    orchestrator._request_advice_if_needed()
    assert orchestrator._self_response_preflight_started_ms == 30
    assert orchestrator._self_response_preflight_deadline_ms == 330

    orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=330,
        wall_time="preflight-hard-deadline",
        metrics=ZoneFrameMetrics(330, False, 0.0, False, False),
    )
    assert advisor.called.wait(1.0)
    assert orchestrator.wait_for_advice_idle(timeout=1.0)
    assert advisor.calls == 1
    assert not any(
        event.event_type == "player_passed"
        and event.source == "button_cannot_beat"
        for event in orchestrator.events
    )
    orchestrator.finish()


def test_self_response_preflight_real_timer_survives_missing_capture_frame(
    tmp_path,
):
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="right",
        settle_ms=0,
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    orchestrator._request_advice_if_needed()

    assert advisor.calls == 0
    assert advisor.called.wait(0.80)
    assert orchestrator.wait_for_advice_idle(timeout=1.0)
    assert advisor.calls == 1
    assert orchestrator._self_response_preflight_key is None
    orchestrator.finish()


def test_self_response_preflight_is_cancelled_by_pause(tmp_path):
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="right",
        settle_ms=0,
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    orchestrator._request_advice_if_needed()
    assert orchestrator._self_response_preflight_key is not None

    orchestrator.pause()
    assert orchestrator._self_response_preflight_key is None
    assert not advisor.called.wait(0.40)
    assert advisor.calls == 0
    orchestrator.finish()


def test_self_lead_does_not_wait_for_response_preflight(tmp_path):
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="self",
        settle_ms=0,
        advisor=advisor,
    )

    assert advisor.called.wait(1.0)
    assert orchestrator.wait_for_advice_idle(timeout=1.0)
    assert advisor.calls == 1
    assert orchestrator._self_response_preflight_key is None
    orchestrator.finish()


@pytest.mark.parametrize(
    ("signals", "lead_player", "prepare_trick", "mark_recovery"),
    [
        ([(True, 0.94, (1, 2, 30, 10), False)], "right", True, False),
        (
            [
                (True, 0.60, (1, 2, 30, 10), False),
                (True, 0.60, (1, 2, 30, 10), False),
            ],
            "right",
            True,
            False,
        ),
        (
            [
                (True, 0.94, (1, 2, 30, 10), True),
                (True, 0.94, (1, 2, 30, 10), True),
            ],
            "right",
            True,
            False,
        ),
        (
            [
                (True, 0.94, (1, 2, 30, 10), False),
                (True, 0.94, (40, 2, 30, 10), False),
            ],
            "right",
            True,
            False,
        ),
        (
            [
                (True, 0.94, (1, 2, 30, 10), False),
                (True, 0.94, (1, 2, 30, 10), False),
            ],
            "right",
            False,
            False,
        ),
        (
            [
                (True, 0.94, (1, 2, 30, 10), False),
                (True, 0.94, (1, 2, 30, 10), False),
            ],
            "self",
            False,
            False,
        ),
        (
            [
                (True, 0.94, (1, 2, 30, 10), False),
                (True, 0.94, (1, 2, 30, 10), False),
            ],
            "right",
            True,
            True,
        ),
    ],
)
def test_cannot_beat_rejects_unstable_illegal_or_recovery_evidence(
    tmp_path,
    signals,
    lead_player,
    prepare_trick,
    mark_recovery,
):
    recognition = CannotBeatRecognitionService(signals)
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player=lead_player,
        settle_ms=0,
    )
    if prepare_trick:
        orchestrator.commit_trusted_action(
            actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
        )
        orchestrator.commit_trusted_action(
            actor="opposite", is_pass=True, monotonic_ms=20
        )
        orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    if mark_recovery:
        window = orchestrator._ensure_turn_ownership_window()
        assert window is not None
        window.turn_recovery_pending = True
    frame = np.zeros((32, 64, 3), np.uint8)
    for index in range(len(signals)):
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=100 + index * 100,
            wall_time=f"cannot-beat-reject-{index}",
            metrics=ZoneFrameMetrics(100 + index * 100, False, 0.0, False, False),
        )

    assert not any(
        event.event_type == "player_passed"
        and event.actor == "self"
        and event.source == "button_cannot_beat"
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


def test_preflight_expiry_and_advice_discard_have_bounded_lock_order(
    tmp_path,
    monkeypatch,
):
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="right",
        settle_ms=0,
        advisor=advisor,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.advisor = None
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    snapshot = orchestrator.snapshot
    expiry_key = AdviceRequestKey(
        snapshot.session_id,
        snapshot.turn_id,
        snapshot.revision,
    )
    orchestrator._self_response_preflight_key = expiry_key
    orchestrator._self_response_preflight_started_ms = 30
    orchestrator._self_response_preflight_deadline_ms = 330
    discarded_key = AdviceRequestKey("game", 999, 999)
    discarded_job = _AdviceJob(
        discarded_key,
        orchestrator.reducer.to_guandan_state(),
    )

    discard_has_advice_lock = threading.Event()
    release_discard = threading.Event()
    expiry_entered_request = threading.Event()
    original_append_advice = orchestrator.store.append_advice
    original_request = orchestrator._request_advice_if_needed

    def gated_append_advice(document):
        if (
            document.get("request_id") == discarded_key.request_id
            and document.get("status") == "cancelled"
        ):
            discard_has_advice_lock.set()
            assert release_discard.wait(1.0)
        original_append_advice(document)

    def marked_request(*args, **kwargs):
        expiry_entered_request.set()
        return original_request(*args, **kwargs)

    monkeypatch.setattr(orchestrator.store, "append_advice", gated_append_advice)
    monkeypatch.setattr(orchestrator, "_request_advice_if_needed", marked_request)
    discard_thread = threading.Thread(
        target=orchestrator._discard_advice_job,
        args=(discarded_job, "stop_discarded"),
    )
    discard_thread.start()
    assert discard_has_advice_lock.wait(1.0)
    expiry_thread = threading.Thread(
        target=orchestrator._expire_self_response_preflight,
        args=(expiry_key,),
    )
    expiry_thread.start()
    assert expiry_entered_request.wait(1.0)
    release_discard.set()

    discard_thread.join(1.0)
    expiry_thread.join(1.0)
    assert not discard_thread.is_alive()
    assert not expiry_thread.is_alive()
    assert advisor.called.wait(1.0)
    assert sum(
        event.event_type == "advice_cancelled"
        and event.payload.get("request_id") == discarded_key.request_id
        for event in orchestrator.events
    ) == 1
    assert sum(
        event.event_type == "advice_requested"
        and event.payload.get("request_id") == expiry_key.request_id
        for event in orchestrator.events
    ) == 1
    orchestrator.finish()


def test_preflight_expiry_waiting_on_finish_cannot_submit_advice(
    tmp_path,
    monkeypatch,
):
    advisor = SuccessfulAdviceService()
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="right",
        settle_ms=0,
    )
    orchestrator.commit_trusted_action(
        actor="right", cards=("3S",), is_pass=False, monotonic_ms=10
    )
    orchestrator.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=20)
    orchestrator.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=30)
    orchestrator.advisor = advisor
    snapshot = orchestrator.snapshot
    key = AdviceRequestKey(snapshot.session_id, snapshot.turn_id, snapshot.revision)
    orchestrator._self_response_preflight_key = key
    orchestrator._self_response_preflight_started_ms = 30
    orchestrator._self_response_preflight_deadline_ms = 330

    finish_inside_state = threading.Event()
    release_finish = threading.Event()
    original_clear = orchestrator._clear_self_response_preflight

    def gated_clear():
        if threading.current_thread().name == "bounded-finish":
            finish_inside_state.set()
            assert release_finish.wait(1.0)
        original_clear()

    monkeypatch.setattr(orchestrator, "_clear_self_response_preflight", gated_clear)
    finish_thread = threading.Thread(
        target=orchestrator.finish,
        name="bounded-finish",
    )
    finish_thread.start()
    assert finish_inside_state.wait(1.0)
    expiry_thread = threading.Thread(
        target=orchestrator._expire_self_response_preflight,
        args=(key,),
    )
    expiry_thread.start()
    release_finish.set()

    finish_thread.join(2.0)
    expiry_thread.join(1.0)
    assert not finish_thread.is_alive()
    assert not expiry_thread.is_alive()
    assert orchestrator.status == "sealed"
    assert advisor.calls == 0
    assert not any(
        event.event_type == "advice_requested"
        and event.payload.get("request_id") == key.request_id
        for event in orchestrator.events
    )


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
    manifest = json.loads(orchestrator.store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["lead_player"] == "right"
    assert manifest["lead_player_event_id"]
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
        confidence=0.75,  # Explicitly high-risk fixture: ordinary exact plays no longer reread.
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
