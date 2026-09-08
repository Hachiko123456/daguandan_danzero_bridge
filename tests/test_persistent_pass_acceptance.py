"""Bounded regression for the Aug14 frame617->630 PASS transition.

Evidence inspected outside the repository, never used as unverified truth:
  frame611 / 58640078ms: opposite AC AC, left old PASS, right timer13;
  frame617 / 58640984ms: right 6C 7C 6H 9C 10C (level6 straight flush),
                        left old PASS, opposite timer14/PASS animation;
  frame624 / 58642015ms: opposite PASS at top, LEFT TIMER14 REPLACES PASS;
  frame630 / 58642937ms: left PASS again, self cannot-beat control/timer5.

Thus physical PASS persistence is NOT established by endpoint images. The
challenging case is sparse analysis that misses frame624's short badge-clear
and left turn. Full and sparse evidence are separate tests. Old timeline's
opposite J? recognition is deliberately excluded. Only small synthetic ROI
shapes are used here; no screenshot/video is copied into the repository.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from daguandan_bridge.domain.recognition import FastSignalResult, PlayRegionResult
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")
RIGHT_FLUSH = ("6C", "7C", "6H", "9C", "10C")
LEFT_HIGH_BOMB = ("KC", "KC", "KD", "KD", "KH", "KH")


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class Signals:
    def __init__(self):
        self.active = "right"
        self.markers = {"left", "self"}
        self.buttons = False
        self.left_cards = ()
        self.probed = []

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        return FastSignalResult(
            expected_player, self.active, expected_player in self.markers,
            self.buttons, False,
            pass_marker_player=expected_player if expected_player in self.markers else None,
            pass_marker_players=tuple(seat for seat in ("self", "right", "opposite", "left") if seat in self.markers),
            cannot_beat_visible=self.buttons, cannot_beat_confidence=.99,
            cannot_beat_box=(32, 23, 10, 5) if self.buttons else None,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank, allow_pass=True):
        self.probed.append(seat)
        cards = self.left_cards if seat == "left" else ()
        is_pass = bool(not cards and seat in self.markers and allow_pass)
        return PlayRegionResult(seat, cards, is_pass, .99 if cards or is_pass else 0., (), (), source="pixel-supported-schedule")

    @staticmethod
    def play_roi(frame, player):
        index = {"left": 0, "opposite": 1, "right": 2, "self": 3}[player]
        return frame[:, index * 16:(index + 1) * 16]

    def image(self):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        for index, seat in enumerate(("left", "opposite", "right", "self")):
            if seat in self.markers:
                frame[9:17, index * 16 + 4:index * 16 + 12] = 190
        if self.left_cards:
            frame[3:28, 1:15] = 240  # Distinct, readable new legal left play.
        return frame


@pytest.fixture
def pass_rig(tmp_path):
    clock, signals = Clock(), Signals()
    store = LiveSessionStore(tmp_path, "acceptance", session_id="persistent-pass", automatic_log_delivery_enabled=False)
    store.start({"application_version": "persistent-pass-acceptance"})
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
        recognition_service=signals, advisor=None, minimum_free_bytes=0, settle_ms=0,
        burst_sample_interval_ms=50, processing_clock_ms=clock,
    )
    core.start(round_level="6", hand=HAND, lead_player="opposite", monotonic_ms=900)
    core.commit_trusted_action(actor="opposite", cards=("AC", "AC"), is_pass=False, monotonic_ms=910)
    core.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=920)
    core.commit_trusted_action(actor="self", is_pass=True, monotonic_ms=930)

    class Rig:
        def step(self, timestamp, *, active, markers, buttons=False, left_cards=()):
            clock.now = timestamp
            signals.active, signals.markers, signals.buttons, signals.left_cards = active, set(markers), buttons, tuple(left_cards)
            frame = signals.image()
            expected = core.snapshot.current_player
            return core.analyze_frame(
                frame, monotonic_ms=timestamp,
                metrics=ZoneFrameMetrics(timestamp, bool(left_cards) if expected == "left" else False, 0., expected in markers, False, content_changed=bool(left_cards)),
                trace_context={"capture_generation": 3, "capture_seq": timestamp},
            )

        def right_flush(self):
            core.commit_trusted_action(actor="right", cards=RIGHT_FLUSH, is_pass=False, monotonic_ms=1_100)

    rig = Rig()
    rig.core, rig.signals = core, signals
    yield rig
    core.finish()


def left_passes_after(core, revision):
    return [event for event in core.reducer.events
            if event.actor == "left" and event.event_type == "player_passed"
            and event.state_revision_after > revision]


def before_left_response(rig, *, stale_self_controls=False):
    # Seed the previous trick's visible left badge through actual analysis,
    # then commit only the clearly readable right straight flush.
    rig.step(1_000, active="right", markers={"left", "self"}, buttons=stale_self_controls)
    rig.right_flush()
    rig.step(1_150, active="opposite", markers={"left", "self"}, buttons=stale_self_controls)
    # This opponent PASS is supported by the pixels (frame624/top marker),
    # not the erroneous old J? timeline entry. Formal state advances normally.
    rig.core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=1_200)
    return rig.core.snapshot.revision


def test_complete_pixel_supported_badge_clear_then_left_pass_is_normal_handoff(pass_rig):
    rig = pass_rig
    revision = before_left_response(rig)
    rig.step(1_250, active="left", markers={"opposite", "self"})  # frame624: left PASS absent.
    rig.step(1_350, active="left", markers={"opposite", "self"})
    rig.step(1_450, active="self", markers={"opposite", "left"}, buttons=True)
    rig.step(1_550, active="self", markers={"opposite", "left"}, buttons=True)
    assert rig.core.snapshot.current_player == "self"
    assert len(left_passes_after(rig.core, revision)) == 1
    assert not any(event.event_type == "advice_recovery_failed" for event in rig.core.events)


def test_sparse_sampling_missed_clear_uses_fresh_self_controls_and_same_trick_without_freeze(pass_rig):
    rig = pass_rig
    revision = before_left_response(rig)
    # No sampled left timer/badge-clear: only the old baseline and a new
    # authenticated self control edge, with left still displaying PASS.
    for timestamp in (1_300, 1_400, 1_500):
        rig.step(timestamp, active="self", markers={"opposite", "left"}, buttons=True)
    assert rig.core.snapshot.current_player == "self", "same-trick persistent PASS left the canonical actor frozen at left"
    assert len(left_passes_after(rig.core, revision)) == 1
    assert not rig.core._turn_ownership_window.turn_recovery_failed


def test_old_pass_without_new_self_control_edge_does_not_create_action(pass_rig):
    rig = pass_rig
    revision = before_left_response(rig, stale_self_controls=True)
    before = rig.core.snapshot
    for timestamp in (1_300, 1_400, 1_500):
        rig.step(timestamp, active="self", markers={"opposite", "left"}, buttons=True)
    assert not left_passes_after(rig.core, revision)
    assert rig.core.snapshot == before


def test_fresh_legal_left_play_blocks_persistent_pass_inference(pass_rig):
    rig = pass_rig
    revision = before_left_response(rig)
    for timestamp in (1_300, 1_400, 1_500):
        rig.step(timestamp, active="self", markers={"opposite", "left"}, buttons=True, left_cards=LEFT_HIGH_BOMB)
    # A new readable card surface may be rejected conservatively without a
    # heavy probe; whichever mechanism is used, it must never become PASS.
    assert not left_passes_after(rig.core, revision), "fresh legal bomb was converted to a PASS"


def test_empty_new_trick_leader_cannot_pass_from_old_badge_or_self_control(pass_rig):
    rig = pass_rig
    # Finish the AA response cycle: right passes, opposite becomes the lead.
    rig.core.commit_trusted_action(actor="right", is_pass=True, monotonic_ms=1_000)
    before = rig.core.snapshot
    assert before.current_player == "opposite" and not before.trick_plays
    for timestamp in (1_100, 1_200, 1_300):
        rig.step(timestamp, active="self", markers={"opposite", "left", "right"}, buttons=True)
    assert rig.core.snapshot == before


def test_control_edge_from_previous_trick_cannot_be_reused_in_later_cycle(pass_rig):
    rig = pass_rig
    revision = before_left_response(rig)
    rig.step(1_250, active="left", markers={"opposite", "self"})
    rig.step(1_350, active="self", markers={"opposite", "left"}, buttons=True)
    rig.step(1_450, active="self", markers={"opposite", "left"}, buttons=True)
    assert len(left_passes_after(rig.core, revision)) == 1
    rig.core.commit_trusted_action(actor="self", is_pass=True, monotonic_ms=1_500)
    rig.core.commit_trusted_action(actor="right", cards=("9D",), is_pass=False, monotonic_ms=1_550)
    rig.core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=1_600)
    before = rig.core.snapshot
    for timestamp in (1_700, 1_800, 1_900):
        rig.step(timestamp, active="self", markers={"opposite", "left"}, buttons=True)
    assert rig.core.snapshot == before, "previous trick's self-control edge or PASS evidence leaked into the new cycle"
