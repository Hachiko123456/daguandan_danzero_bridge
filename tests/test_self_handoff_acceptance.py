"""Low-motion self-action handoff acceptance with actual tiny ROI pixel reads.

Primary's third1x replay: original frame1230 visibly has small_joker (0.9615),
right is already finished and opposite is active; frame1235 retains the joker,
opposite PASS, left active. Runtime had no joker observations and repeatedly
reported zone_wait_action/sample0. This fixture does NOT fake a finish/one-card
opponent: ordinary self->right and self->right PASS->opposite test the mechanism.

The passed lifecycle metrics deliberately say occupied=False, low motion and
content_changed=False. A real4x4 pixel glyph changes inside a16x32 self ROI,
and recognition decodes those pixels plus an actual absolute play bbox. No
freshness helper is stubbed and no user model runs.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from daguandan_bridge.domain.recognition import FastSignalResult, PlayRegionResult, RecognitionAnnotation
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("small_joker", "small_joker", "9C")
PIXELS = {None: 0, "small_joker": 220, "big_joker": 190, "7D": 160}
CARDS = {value: key for key, value in PIXELS.items()}


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class PixelRecognition:
    def __init__(self):
        self.active = "self"
        self.controls = True
        self.effect = False
        self.markers = set()
        self.self_reads = []

    @staticmethod
    def play_roi(frame, seat):
        offset = {"self": 0, "right": 16, "opposite": 32, "left": 48}[seat]
        return frame[:, offset:offset + 16]

    @staticmethod
    def image(card):
        frame = np.zeros((32, 64, 3), np.uint8)
        frame[10:14, 5:9] = PIXELS[card]
        return frame

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        return FastSignalResult(
            expected_player, self.active, expected_player in self.markers, self.controls, self.effect,
            pass_marker_player=expected_player if expected_player in self.markers else None,
            pass_marker_players=tuple(self.markers),
        )

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_pass=True):
        if seat == "self":
            card = CARDS.get(int(frame[11, 6, 0]))
            self.self_reads.append(card)
            cards = (card,) if card else ()
            annotations = (RecognitionAnnotation(card, (5, 10, 4, 4), .9615, "play"),) if card else ()
            return PlayRegionResult("self", cards, False, .9615 if cards else 0., (), annotations, source="tiny-real-pixel-bbox")
        is_pass = seat in self.markers and allow_pass
        return PlayRegionResult(seat, (), is_pass, .99 if is_pass else 0., (), (), source="seat-pass-evidence")


@pytest.fixture
def self_rig(tmp_path):
    (tmp_path / "profile").mkdir()
    store = InMemoryLiveSessionStore(tmp_path, "profile")
    clock, recognition = Clock(), PixelRecognition()
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
        recognition_service=recognition, advisor=None, minimum_free_bytes=0, settle_ms=0,
        burst_sample_interval_ms=50, processing_clock_ms=clock,
    )
    core.start(round_level="6", hand=HAND, lead_player="right", monotonic_ms=900)
    core.bind_capture_generation(3)
    core.commit_trusted_action(actor="right", cards=("JS",), is_pass=False, monotonic_ms=910)
    core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=920)
    core.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=930)
    assert core.snapshot.current_player == "self" and not core.snapshot.finished_seats

    class Rig:
        def step(self, timestamp, *, card=None, active="self", controls=True, effect=False, markers=()):
            clock.now = timestamp
            recognition.active, recognition.controls, recognition.effect = active, controls, effect
            recognition.markers = set(markers)
            return core.analyze_frame(
                recognition.image(card), monotonic_ms=timestamp,
                metrics=ZoneFrameMetrics(timestamp, False, .001, False, effect, content_changed=False),
                trace_context={"capture_generation": 3, "capture_seq": timestamp},
            )

        def authenticate(self, *, old_card=None):
            self.step(1_000, card=old_card)
            self.step(1_100, card=old_card)
            assert core._turn_ownership_window.authenticated
            return core.snapshot

    rig = Rig()
    rig.core, rig.recognition = core, recognition
    yield rig
    core.finish()


def self_actions_after(core, revision):
    return [event for event in core.reducer.events if event.event_type == "player_played"
            and event.actor == "self" and event.state_revision_after > revision]


def test_low_motion_known_hand_joker_is_semantically_read_and_committed_within_four_captures(self_rig):
    rig = self_rig
    before = rig.authenticate()
    reads_before = len(rig.recognition.self_reads)
    for timestamp in (1_200, 1_300, 1_400, 1_500):
        rig.step(timestamp, card="small_joker", active="right", controls=False)
        if rig.core.snapshot.current_player != "self":
            break
    actions = self_actions_after(rig.core, before.revision)
    assert len(actions) == 1, "self known-hand play stayed at zone_wait_action with no bounded semantic confirmation"
    assert tuple(actions[0].payload["cards"]) == ("small_joker",)
    assert rig.core.snapshot.current_player == "right"
    assert rig.core.snapshot.my_hand.count("small_joker") == before.my_hand.count("small_joker") - 1
    assert 2 <= len(rig.recognition.self_reads) - reads_before <= 4


def test_low_motion_self_joker_survives_successor_pass_to_opposite_without_fake_finish(self_rig):
    rig = self_rig
    before = rig.authenticate()
    # First direct successor frame supplies one joker read. Right then
    # genuinely PASSES before the second/third view; no player is finished.
    rig.step(1_200, card="small_joker", active="right", controls=False)
    for timestamp in (1_300, 1_400, 1_500, 1_600):
        rig.step(timestamp, card="small_joker", active="opposite", controls=False, markers=("right",))
        if rig.core.snapshot.current_player == "opposite":
            break
    actions = self_actions_after(rig.core, before.revision)
    assert len(actions) == 1 and tuple(actions[0].payload["cards"]) == ("small_joker",)
    assert rig.core.snapshot.current_player == "opposite", "self play plus actual right PASS was lost on crossed handoff"
    after = [event for event in rig.core.reducer.events if event.state_revision_after > before.revision
             and event.event_type in {"player_played", "player_passed"}]
    assert [(event.actor, event.event_type) for event in after] == [("self", "player_played"), ("right", "player_passed")]
    assert not rig.core.snapshot.finished_seats


def test_old_unchanged_joker_surface_is_not_replayed_even_with_same_value_copy_in_hand(self_rig):
    rig = self_rig
    before = rig.authenticate(old_card="small_joker")
    assert before.my_hand.count("small_joker") == 2
    for timestamp in (1_200, 1_300, 1_400, 1_500, 1_600, 1_700):
        rig.step(timestamp, card="small_joker", active="right", controls=False)
    assert rig.core.snapshot == before
    assert not self_actions_after(rig.core, before.revision)


@pytest.mark.parametrize("card", ["big_joker", "7D"])
def test_unknown_hand_card_or_illegal_response_cannot_commit_from_handoff_probe(self_rig, card):
    rig = self_rig
    before = rig.authenticate()
    assert (card not in before.my_hand) if card == "big_joker" else (card in before.my_hand)
    for timestamp in (1_200, 1_300, 1_400, 1_500, 1_600, 1_700):
        rig.step(timestamp, card=card, active="right", controls=False)
    assert rig.core.snapshot == before
    assert not self_actions_after(rig.core, before.revision)


def test_repeated_capture_cannot_create_second_semantic_vote(self_rig):
    rig = self_rig
    before = rig.authenticate()
    for _ in range(5):
        rig.step(1_200, card="small_joker", active="right", controls=False)
    assert rig.core.snapshot == before
    assert not self_actions_after(rig.core, before.revision)


@pytest.mark.parametrize("controls,effect", [(True, False), (False, True)])
def test_local_controls_not_cleared_or_animation_never_authorizes_low_motion_action(self_rig, controls, effect):
    rig = self_rig
    before = rig.authenticate()
    for timestamp in (1_200, 1_300, 1_400, 1_500):
        rig.step(timestamp, card="small_joker", active="right", controls=controls, effect=effect)
    assert rig.core.snapshot == before
    assert not self_actions_after(rig.core, before.revision)
