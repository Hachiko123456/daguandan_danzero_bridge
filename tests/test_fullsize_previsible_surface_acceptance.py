"""Full-size pre-visible surface acceptance for per-seat turn evidence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from daguandan_bridge.domain.recognition import (
    FastSignalResult,
    PlayRegionResult,
    RecognitionAnnotation,
)
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live.turn_evidence import TurnEvidenceCache
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics


ROIS = {
    "self": (520, 610, 240, 56),
    "right": (1040, 300, 120, 80),
    "opposite": (520, 72, 240, 56),
    "left": (120, 300, 120, 80),
}
CARD_CODES = {(): 0, ("10C",): 30, ("JS",): 70, ("KH",): 110, ("6D",): 190}
CODE_CARDS = {code: cards for cards, code in CARD_CODES.items()}
HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "10C")


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self) -> int:
        return self.now


class FullFrameRecognition:
    def __init__(self) -> None:
        self.active_player = "right"
        self.controls_visible = False

    @staticmethod
    def play_roi(frame: np.ndarray, seat: str) -> np.ndarray:
        x, y, width, height = ROIS[seat]
        return frame[y : y + height, x : x + width]

    @classmethod
    def cards_from_frame(cls, frame: np.ndarray, seat: str) -> tuple[str, ...]:
        roi = cls.play_roi(frame, seat)
        return CODE_CARDS.get(int(roi[roi.shape[0] // 2, roi.shape[1] // 2, 0]), ())

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        return FastSignalResult(expected_player, self.active_player, False, self.controls_visible, False)

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_pass=True):
        cards = self.cards_from_frame(frame, seat)
        x, y, width, height = ROIS[seat]
        annotations = ()
        if cards:
            annotations = (
                RecognitionAnnotation(
                    label=" ".join(cards),
                    box=(x + 6, y + 6, width - 12, height - 12),
                    confidence=0.99,
                    category="play",
                ),
            )
        return PlayRegionResult(
            seat,
            cards,
            False,
            0.99 if cards else 0.0,
            (),
            annotations,
            source="full-frame-fixed-play-roi-code",
        )

    @staticmethod
    def image(cards_by_seat: dict[str, tuple[str, ...]], *, salt: int = 0) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[0, 0, 0] = salt % 251
        for seat, cards in cards_by_seat.items():
            x, y, width, height = ROIS[seat]
            frame[y + 6 : y + height - 6, x + 6 : x + width - 6] = CARD_CODES[tuple(cards)]
        return frame


@pytest.fixture
def fullsize_rig(tmp_path):
    clock = Clock()
    recognition = FullFrameRecognition()
    store = LiveSessionStore(
        tmp_path,
        "acceptance",
        session_id="fullsize-previsible-surface",
        automatic_log_delivery_enabled=False,
    )
    store.start({"application_version": "fullsize-previsible-acceptance"})
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id),
        store=store,
        recorder=InMemorySessionRecorder(store.directory),
        recognition_service=recognition,
        advisor=None,
        minimum_free_bytes=0,
        settle_ms=0,
        burst_sample_interval_ms=50,
        processing_clock_ms=clock,
    )
    core.start(round_level="6", hand=HAND, lead_player="self", monotonic_ms=900)
    core.commit_trusted_action(actor="self", cards=("10C",), is_pass=False, monotonic_ms=950)

    class Rig:
        def step(self, timestamp: int, cards: dict[str, tuple[str, ...]], *, active: str | None, controls: bool, content_changed: bool = False):
            clock.now = timestamp
            recognition.active_player = active
            recognition.controls_visible = controls
            frame = recognition.image(cards, salt=timestamp)
            expected = core.snapshot.current_player
            update = core.analyze_frame(
                frame,
                monotonic_ms=timestamp,
                metrics=ZoneFrameMetrics(timestamp, bool(expected and cards.get(expected, ())), 0.0, False, False, content_changed=content_changed),
                trace_context={"capture_generation": 3, "capture_seq": timestamp},
            )
            return update

    rig = Rig()
    rig.core = core
    yield rig
    core.finish()


def action_events_after(core: LiveOrchestrator, revision: int):
    return [event for event in core.reducer.events if event.state_revision_after > revision and event.event_type in {"player_played", "player_passed"}]


def test_previsible_opposite_surface_survives_full_frame_ring_pressure_and_recovers(fullsize_rig):
    rig = fullsize_rig
    core = rig.core
    for timestamp in range(1_000, 1_800, 100):
        rig.step(timestamp, {"self": ("10C",)}, active="right", controls=False)

    previsible = {"self": ("10C",), "opposite": ("KH",)}
    for timestamp in range(1_800, 2_400, 100):
        rig.step(timestamp, previsible, active="right", controls=False)

    assert core._turn_evidence.retained_bytes <= 16 * 1024 * 1024
    old_opposite = core._turn_evidence.before_roi(2_500, generation=3, seat="opposite")
    full_anchor = core._turn_evidence.before(2_500, generation=3, seat="opposite")
    assert old_opposite is not None and old_opposite.captured_ms == 1_000
    assert full_anchor is not None and full_anchor.captured_ms >= 2_000
    assert old_opposite.captured_ms < full_anchor.captured_ms
    assert not old_opposite.frame.any()
    assert not old_opposite.frame.flags.writeable
    assert FullFrameRecognition.cards_from_frame(full_anchor.frame, "opposite") == ("KH",)

    surfaces = {"self": ("10C",), "right": ("JS",), "opposite": ("KH",)}
    before_revision = core.snapshot.revision
    assert rig.step(2_400, surfaces, active="right", controls=False, content_changed=True).snapshot.current_player == "right"
    assert rig.step(2_500, surfaces, active="right", controls=False).snapshot.current_player == "opposite"
    continued = {**surfaces, "left": ("6D",)}
    assert rig.step(2_600, continued, active="self", controls=False).snapshot.current_player == "opposite"
    assert rig.step(2_700, continued, active="self", controls=True).snapshot.current_player == "opposite"
    assert rig.step(2_800, continued, active="self", controls=True).snapshot.current_player == "self"

    assert [(event.actor, tuple(event.payload["cards"])) for event in action_events_after(core, before_revision)] == [
        ("right", ("JS",)),
        ("opposite", ("KH",)),
        ("left", ("6D",)),
    ]
    assert core.snapshot.current_player == "self"


def test_previsible_roi_cache_boundaries_are_readonly_consumed_and_generation_scoped():
    cache = TurnEvidenceCache(max_frames=4, max_bytes=16 * 1024 * 1024)
    full = np.zeros((720, 1280, 3), dtype=np.uint8)
    blank = np.zeros((56, 240, 3), dtype=np.uint8)
    kh = np.full((56, 240, 3), CARD_CODES[("KH",)], dtype=np.uint8)

    for index in range(9):
        frame = full.copy()
        frame[0, 0, 0] = index
        cache.observe(frame, captured_ms=1_000 + index * 100, generation=3, seat_rois={"opposite": blank})

    anchor = cache.before_roi(1_900, generation=3, seat="opposite")
    assert cache.retained_bytes <= 16 * 1024 * 1024
    assert anchor is not None and anchor.captured_ms == 1_000
    assert not anchor.frame.flags.writeable
    assert cache.before_roi(3_401, generation=3, seat="opposite") is None

    cache.observe(full, captured_ms=1_900, generation=3, seat_rois={"opposite": kh})
    cache.mark_consumed("opposite", 1_901)
    assert cache.before_roi(2_000, generation=3, seat="opposite") is None

    cache.observe(full, captured_ms=2_100, generation=4, seat_rois={"opposite": kh})
    assert cache.before_roi(2_200, generation=3, seat="opposite") is None
    assert cache.before_roi(2_200, generation=4, seat="opposite").captured_ms == 2_100

    cache.mark_consumed("opposite", 2_101)
    assert cache.before_roi(2_200, generation=4, seat="opposite") is None
