"""Never turn weak/unknown card recognition into derived PASS evidence.

Primary's warm real-video probe: source1244 visibly contains a golden left
big_joker; read_play returns cards=[big_joker], confidence=.6703. The prior
boolean 'fresh legal nonpass?' returned False below.80 and was misused as
'empty', producing a left player_passed at source1243. The low score is kept
exactly here, not inflated to make the recognition path pass.

Real core/table/ROI comparison is used; synthetic tiny pixels provide bounded
frame contents. The alternative actions may remain unknown/blocked, but must
never become a PASS or clear the current small_joker trick.
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


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("small_joker", "8S", "8H")
MODES = {"empty": 0, "weak_joker": 170, "exception": 190, "illegal_nonempty": 210, "unclassified_change": 230, "strong_pass": 90}


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class Recognition:
    def __init__(self):
        self.active, self.controls, self.marker = "left", False, False
        self.probes = []

    @staticmethod
    def play_roi(frame, seat):
        offset = {"self": 0, "right": 16, "opposite": 32, "left": 48}[seat]
        return frame[:, offset:offset + 16]

    @staticmethod
    def image(mode):
        frame = np.zeros((32, 64, 3), np.uint8)
        frame[8:24, 52:60] = MODES[mode]
        return frame

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        markers = ("left",) if self.marker else ()
        return FastSignalResult(expected_player, self.active, self.marker and expected_player == "left", self.controls, False,
                                pass_marker_player="left" if self.marker and expected_player == "left" else None,
                                pass_marker_players=markers)

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_pass=True):
        if seat != "left":
            return PlayRegionResult(seat, (), False, 0., (), (), source="acceptance")
        code = int(frame[12, 55, 0])
        self.probes.append(code)
        if code == MODES["exception"]:
            raise RuntimeError("synthetic left matcher failure")
        if code == MODES["weak_joker"]:
            return PlayRegionResult("left", ("big_joker",), False, .6703, ("weak golden template",),
                                    (RecognitionAnnotation("big_joker", (52, 8, 8, 16), .6703, "play"),), source="golden-score-0.6703")
        if code == MODES["illegal_nonempty"]:
            return PlayRegionResult("left", ("7D",), False, .99, (),
                                    (RecognitionAnnotation("7D", (52, 8, 8, 16), .99, "play"),), source="illegal-over-joker")
        if code == MODES["unclassified_change"]:
            return PlayRegionResult("left", (), False, 0., ("new visible surface could not classify",), (), source="unknown-not-empty")
        is_pass = code == MODES["strong_pass"] and allow_pass
        return PlayRegionResult("left", (), is_pass, .99 if is_pass else 0., (), (), source="strong-new-pass-marker" if is_pass else "empty")


@pytest.fixture
def unknown_rig(tmp_path):
    (tmp_path / "profile").mkdir()
    store = InMemoryLiveSessionStore(tmp_path, "profile")
    clock, recognition = Clock(), Recognition()
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
        recognition_service=recognition, advisor=None, minimum_free_bytes=0, settle_ms=0,
        burst_sample_interval_ms=50, processing_clock_ms=clock,
    )
    core.start(round_level="6", hand=HAND, lead_player="self", monotonic_ms=900)
    core.bind_capture_generation(3)
    core.commit_trusted_action(actor="self", cards=("small_joker",), is_pass=False, monotonic_ms=910)
    core.commit_trusted_action(actor="right", is_pass=True, monotonic_ms=920)
    core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=930)
    assert core.snapshot.current_player == "left" and core.snapshot.trick_plays

    class Rig:
        def step(self, timestamp, mode="empty", *, active="left", controls=False, marker=False):
            clock.now = timestamp
            recognition.active, recognition.controls, recognition.marker = active, controls, marker
            return core.analyze_frame(
                recognition.image(mode), monotonic_ms=timestamp,
                metrics=ZoneFrameMetrics(timestamp, False, 0., marker, False, content_changed=False),
                trace_context={"capture_generation": 3, "capture_seq": timestamp},
            )

    rig = Rig()
    rig.core, rig.recognition = core, recognition
    rig.step(1_000)
    rig.step(1_100)
    assert core._turn_ownership_window.authenticated
    yield rig
    core.finish()


@pytest.mark.parametrize("mode", ["weak_joker", "exception", "illegal_nonempty", "unclassified_change"])
def test_unknown_or_nonempty_left_surface_is_never_converted_to_derived_pass(unknown_rig, mode):
    rig = unknown_rig
    before = rig.core.snapshot
    for timestamp in (1_200, 1_300, 1_400, 1_500):
        rig.step(timestamp, mode, active="self", controls=True, marker=False)
    left_passes = [event for event in rig.core.reducer.events if event.actor == "left"
                   and event.event_type == "player_passed" and event.state_revision_after > before.revision]
    assert not left_passes, f"{mode} was incorrectly treated as empty and converted to left PASS"
    assert rig.core.snapshot.current_player == "left"
    assert rig.core.snapshot.trick_plays == before.trick_plays, "uncertain surface cleared the joker trick"
    assert rig.core.snapshot.play_history == before.play_history
    assert MODES[mode] in rig.recognition.probes, "fixture did not reach left semantic uncertainty check"
    assert not rig.core._advice_requested_at_ms and not rig.core._pending_model_advice


def test_new_strong_left_pass_marker_still_commits_and_closes_trick_normally(unknown_rig):
    rig = unknown_rig
    before = rig.core.snapshot
    for timestamp in (1_200, 1_300, 1_400):
        rig.step(timestamp, "strong_pass", active="self", controls=True, marker=True)
        if rig.core.snapshot.current_player == "self":
            break
    actions = [event for event in rig.core.reducer.events if event.state_revision_after > before.revision
               and event.event_type in {"player_played", "player_passed"}]
    assert [(event.actor, event.event_type) for event in actions] == [("left", "player_passed")]
    assert rig.core.snapshot.current_player == "self"
    assert not rig.core.snapshot.trick_plays
