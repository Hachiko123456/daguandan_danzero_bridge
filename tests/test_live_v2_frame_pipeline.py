from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from daguandan_bridge.application.live_v2_frame_pipeline import (
    FramePipelineConfig,
    LiveV2FramePipeline,
)
from daguandan_bridge.application.live_v2_surface_probe import FourSeatSurfaceProbe
from daguandan_bridge.domain.recognition import (
    FastSignalResult, PlacementSignal, PlayRegionResult, RecognitionAnnotation,
)
from daguandan_bridge.live_v2.types import (
    ActionKind,
    CandidateReason,
    FrameIdentity,
    ObservationKind,
    Seat,
    VersionIdentity,
)


SEATS = tuple(Seat)
APPLICATION = Path(__file__).parents[1] / "src" / "daguandan_bridge" / "application"
SLICE = {
    Seat.SELF: slice(0, 40),
    Seat.RIGHT: slice(40, 80),
    Seat.OPPOSITE: slice(80, 120),
    Seat.LEFT: slice(120, 160),
}


def frame(sequence: int, *, generation: int = 1) -> FrameIdentity:
    return FrameIdentity(
        "pipeline-session", generation, sequence, sequence * 100,
        "roi-v7", "capture-window-9",
    )


def version(
    *, generation: int = 1, revision: int = 0, turn: int = 0
) -> VersionIdentity:
    return VersionIdentity("pipeline-session", generation, revision, revision, turn)


def table(*, cards: tuple[Seat, ...] = ()) -> np.ndarray:
    image = np.zeros((40, 160, 3), dtype=np.uint8)
    image[:, :, :] = (40, 100, 40)
    for seat in cards:
        region = image[:, SLICE[seat]]
        region[8:32, 8:32] = 245
        region[14:20, 14:26] = 20
    return image


class FakeRecognition:
    def __init__(self) -> None:
        self.fast_calls = 0
        self.roi_calls: list[Seat] = []
        self.deep_calls: list[Seat] = []
        self.deep_kwargs: list[dict[str, object]] = []
        self.pass_seats: set[Seat] = set()
        self.effect_seats: set[Seat] = set()
        self.active_player: Seat | None = None
        self.self_buttons = False
        self.super_double = False
        self.game_end_control: str | None = None
        self.placements: tuple[PlacementSignal, ...] = ()
        self.card_by_seat = {seat: (f"{index + 5}S",) for index, seat in enumerate(SEATS)}
        self.advance = None

    def play_roi(self, image, seat):
        seat = Seat(seat)
        self.roi_calls.append(seat)
        return image[:, SLICE[seat]]

    def recognize_fast_signals(self, image, expected_player, *, allow_pass=True):
        del image
        assert allow_pass is True
        self.fast_calls += 1
        expected = Seat(expected_player)
        return FastSignalResult(
            expected_player=expected.value,
            active_player=self.active_player.value if self.active_player else None,
            pass_visible=expected in self.pass_seats,
            self_action_buttons_visible=self.self_buttons,
            effect_visible=expected in self.effect_seats,
            pass_marker_player=expected.value if expected in self.pass_seats else None,
            pass_marker_players=tuple(seat.value for seat in SEATS if seat in self.pass_seats),
            super_double_visible=self.super_double,
            game_end_control=self.game_end_control,
            placements=self.placements,
        )

    def recognize_play_region(self, image, seat, **kwargs):
        seat = Seat(seat)
        self.deep_calls.append(seat)
        self.deep_kwargs.append(dict(kwargs))
        if self.advance is not None:
            self.advance(800)
        roi = image[:, SLICE[seat]]
        if seat in self.pass_seats:
            return PlayRegionResult(seat.value, (), True, 0.96, (), ())
        cards = self.card_by_seat[seat] if np.max(roi) >= 200 else ()
        return PlayRegionResult(
            seat.value,
            cards,
            False,
            0.95 if cards else 0.0,
            (),
            (),
        )


class ManualClock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> int:
        return self.value

    def advance(self, amount: int) -> None:
        self.value += amount


def make_pipeline(fake: FakeRecognition, *, reads: int = 4, **kwargs) -> LiveV2FramePipeline:
    config = FramePipelineConfig(max_deep_reads_per_frame=reads, **kwargs)
    return LiveV2FramePipeline(fake, config=config)


def process(
    pipeline, image, sequence, *, expected=Seat.LEFT, generation=1,
    revision=0, turn=0, formal_action_boundary=None,
):
    return pipeline.process_frame(
        image,
        frame=frame(sequence, generation=generation),
        version=version(generation=generation, revision=revision, turn=turn),
        wild_rank="H",
        expected_seat=expected,
        now_ms=sequence * 100,
        formal_action_boundary=formal_action_boundary,
    )


def test_one_full_frame_probes_all_seats_and_preserves_capture_identity() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake, reads=2)
    result = process(pipeline, table(cards=(Seat.LEFT,)), 1)

    assert fake.fast_calls == 1
    assert fake.roi_calls == list(SEATS)
    assert len(result.surface_metrics) == 4
    assert len(result.observations) == 2
    assert all(metric.frame.roi_version == "roi-v7" for metric in result.surface_metrics)
    assert all(metric.frame.source_id == "capture-window-9" for metric in result.surface_metrics)
    assert all(call["allow_pass"] is True for call in fake.deep_kwargs)
    assert all(call["allow_unknown_suit"] is True for call in fake.deep_kwargs)
    with pytest.raises(FrozenInstanceError):
        result.candidate_backlog = 99


def test_first_static_card_surface_is_not_calibrated_as_empty() -> None:
    fake = FakeRecognition()
    probe = FourSeatSurfaceProbe(fake.play_roi)
    first = probe.probe(table(cards=(Seat.LEFT,)), frame=frame(1))
    left = next(item for item in first if item.seat is Seat.LEFT)

    assert left.visible_surface
    assert not left.stable_empty
    assert left.empty_streak == 0


def test_expected_seat_is_priority_only_and_budget_eventually_reads_all_seats() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake, reads=1, probe_starvation_ms=1_000)
    image = table(cards=SEATS)
    for sequence in range(1, 9):
        process(pipeline, image, sequence, expected=Seat.LEFT)

    assert fake.deep_calls[:2] == [Seat.LEFT, Seat.LEFT]
    assert set(fake.deep_calls) == set(SEATS)


def test_four_simultaneous_changes_are_retained_until_all_form_candidates() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    process(pipeline, table(), 1)
    process(pipeline, table(), 2)
    changed = process(pipeline, table(cards=SEATS), 3)
    assert all(item.animating and item.content_changed for item in changed.surface_metrics)
    assert not changed.candidates

    process(pipeline, table(cards=SEATS), 4)
    confirmed = process(pipeline, table(cards=SEATS), 5)
    assert {candidate.seat for candidate in confirmed.candidates} == set(SEATS)
    assert len(pipeline.pending_candidates(now_ms=500)) == 4
    assert len(pipeline.evidence_buffer.candidates()) == 4


def test_same_cards_can_emit_again_only_after_independent_blank_epoch() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    card_image = table(cards=(Seat.LEFT,))
    process(pipeline, card_image, 1)
    first = process(pipeline, card_image, 2)
    first_candidate = next(item for item in first.candidates if item.seat is Seat.LEFT)

    for sequence in range(3, 7):
        process(pipeline, table(), sequence)
    process(pipeline, card_image, 7)
    process(pipeline, card_image, 8)
    replay = process(pipeline, card_image, 9)
    second = next(item for item in replay.candidates if item.seat is Seat.LEFT)

    assert first_candidate.cards == second.cards
    assert first_candidate.action_epoch == 0
    assert second.action_epoch == 1


def test_pass_needs_two_distinct_frames_and_unknown_never_becomes_pass() -> None:
    fake = FakeRecognition()
    fake.pass_seats = {Seat.RIGHT}
    pipeline = make_pipeline(fake)
    assert not process(pipeline, table(), 1, expected=Seat.RIGHT).candidates
    second = process(pipeline, table(), 2, expected=Seat.RIGHT)
    candidate = next(item for item in second.candidates if item.seat is Seat.RIGHT)
    assert candidate.kind is ActionKind.PASS

    fake.pass_seats.clear()
    unknown_pipeline = make_pipeline(fake)
    unknown = process(unknown_pipeline, table(), 10)
    assert all(item.kind is not ObservationKind.PASS for item in unknown.observations)
    assert not unknown.candidates


def test_pass_evidence_must_be_strictly_after_formal_action_boundary() -> None:
    fake = FakeRecognition()
    fake.pass_seats = {Seat.RIGHT}
    pipeline = make_pipeline(fake)

    same_capture = process(
        pipeline, table(), 1, expected=Seat.RIGHT,
        formal_action_boundary=FrameIdentity(
            "pipeline-session", 1, 0, 100, "roi-v7", "capture-window-9",
        ),
    )
    same_sequence = process(
        pipeline, table(), 2, expected=Seat.RIGHT,
        formal_action_boundary=FrameIdentity(
            "pipeline-session", 1, 2, 150, "roi-v7", "capture-window-9",
        ),
    )
    assert not same_capture.candidates and not same_sequence.candidates
    assert all(item.kind is ObservationKind.UNKNOWN for item in same_capture.observations)
    assert all(item.kind is ObservationKind.UNKNOWN for item in same_sequence.observations)

    boundary = frame(2)
    assert not process(
        pipeline, table(), 3, expected=Seat.RIGHT,
        formal_action_boundary=boundary,
    ).candidates
    confirmed = process(
        pipeline, table(), 4, expected=Seat.RIGHT,
        formal_action_boundary=boundary,
    )
    candidate = next(item for item in confirmed.candidates if item.seat is Seat.RIGHT)
    assert candidate.first_frame.frame_sequence == 3
    assert candidate.first_frame.captured_ms > boundary.captured_ms


def test_pass_can_use_exact_boundary_frame_only_as_first_witness() -> None:
    fake = FakeRecognition()
    fake.pass_seats = {Seat.RIGHT}
    pipeline = make_pipeline(fake)
    boundary = frame(2)

    first = process(
        pipeline, table(), 2, expected=Seat.RIGHT,
        formal_action_boundary=boundary,
    )
    assert not first.candidates
    assert any(item.kind is ObservationKind.PASS for item in first.observations)

    confirmed = process(
        pipeline, table(), 3, expected=Seat.RIGHT,
        formal_action_boundary=boundary,
    )
    candidate = next(item for item in confirmed.candidates if item.seat is Seat.RIGHT)
    assert candidate.first_frame == boundary
    assert candidate.last_frame == frame(3)


def test_persistent_pass_rearms_after_marker_clear_not_formal_turnover() -> None:
    fake = FakeRecognition()
    fake.pass_seats = {Seat.RIGHT}
    pipeline = make_pipeline(fake)
    process(pipeline, table(), 1, expected=Seat.RIGHT)
    initial = process(pipeline, table(), 2, expected=Seat.RIGHT)
    assert any(item.seat is Seat.RIGHT for item in initial.candidates)

    # A formal turn change while the marker stays visible is not a new PASS.
    marker_only = process(
        pipeline, table(), 3, expected=Seat.RIGHT, revision=1, turn=1
    )
    assert not marker_only.candidates

    fake.pass_seats.clear()
    process(pipeline, table(), 4, expected=Seat.RIGHT, revision=1, turn=1)
    cleared = process(pipeline, table(), 5, expected=Seat.RIGHT, revision=1, turn=1)
    assert not cleared.candidates

    fake.pass_seats = {Seat.RIGHT}
    first = process(
        pipeline, table(), 6, expected=Seat.RIGHT, revision=1, turn=1
    )
    assert not first.candidates
    boundary = process(
        pipeline, table(), 7, expected=Seat.RIGHT, revision=1, turn=1
    )
    candidate = next(item for item in boundary.candidates if item.seat is Seat.RIGHT)
    assert candidate.action_epoch == 1
    assert candidate.first_frame.frame_sequence == 6
    assert candidate.last_frame.frame_sequence == 7
    assert "pass_marker_rearmed:right" in cleared.diagnostics


def test_game181845_single_pass_frame_cross_confirms_after_active_moves() -> None:
    """Regression for game_20260918_181845_8d432b around frame 2202."""

    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    fake.active_player = Seat.SELF
    process(pipeline, table(), 1, expected=Seat.SELF, revision=7, turn=57)

    fake.pass_seats = {Seat.SELF}
    fake.active_player = Seat.RIGHT
    result = process(
        pipeline, table(), 2, expected=Seat.SELF, revision=7, turn=57,
        formal_action_boundary=frame(1),
    )

    candidate = next(item for item in result.candidates if item.seat is Seat.SELF)
    assert candidate.kind is ActionKind.PASS
    assert candidate.reason is CandidateReason.CROSS_SOURCE_PASS
    assert candidate.first_frame == candidate.last_frame == frame(2)
    assert len(candidate.evidence_ids) == 2
    assert any(
        item == "pass_cross_source_confirmed:self->right"
        for item in candidate.diagnostics
    )


def test_cross_confirm_skips_only_same_frame_visibly_finished_seats() -> None:
    fake = FakeRecognition()
    fake.placements = (
        PlacementSignal(Seat.RIGHT.value, "head", 0.99, "right-finished"),
    )
    pipeline = make_pipeline(fake)
    fake.active_player = Seat.SELF
    process(
        pipeline, table(), 1, expected=Seat.SELF, revision=8, turn=58,
    )
    fake.pass_seats = {Seat.SELF}
    fake.active_player = Seat.OPPOSITE

    result = process(
        pipeline, table(), 2, expected=Seat.SELF, revision=8, turn=58,
        formal_action_boundary=frame(1),
    )

    candidate = next(item for item in result.candidates if item.seat is Seat.SELF)
    assert candidate.reason is CandidateReason.CROSS_SOURCE_PASS
    assert "pass_cross_source_confirmed:self->opposite" in candidate.diagnostics


def test_cross_confirm_rejects_jump_without_finished_seat_proof() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    fake.active_player = Seat.SELF
    process(
        pipeline, table(), 1, expected=Seat.SELF, revision=8, turn=58,
    )
    fake.pass_seats = {Seat.SELF}
    fake.active_player = Seat.OPPOSITE

    result = process(
        pipeline, table(), 2, expected=Seat.SELF, revision=8, turn=58,
        formal_action_boundary=frame(1),
    )

    assert not result.candidates
    assert pipeline.trackers[Seat.SELF].snapshot().pending_count == 1


def test_cross_confirm_never_reuses_latched_old_pass() -> None:
    fake = FakeRecognition()
    fake.pass_seats = {Seat.SELF}
    pipeline = make_pipeline(fake)
    process(pipeline, table(), 1, expected=Seat.SELF)
    initial = process(pipeline, table(), 2, expected=Seat.SELF)
    assert {item.seat for item in initial.candidates} == {Seat.SELF}

    # A still-visible old marker is not a new PASS in a later formal turn.
    fake.active_player = Seat.RIGHT
    stale = process(
        pipeline, table(), 3, expected=Seat.SELF, revision=1, turn=1,
        formal_action_boundary=frame(2),
    )
    assert not stale.candidates

    fake.pass_seats.clear()
    fake.active_player = Seat.SELF
    process(pipeline, table(), 4, expected=Seat.SELF, revision=1, turn=1)
    process(pipeline, table(), 5, expected=Seat.SELF, revision=1, turn=1)

    fake.pass_seats = {Seat.SELF}
    fake.active_player = Seat.RIGHT
    fresh = process(
        pipeline, table(), 6, expected=Seat.SELF, revision=1, turn=1,
        formal_action_boundary=frame(2),
    )
    candidate = next(item for item in fresh.candidates if item.seat is Seat.SELF)
    assert candidate.reason is CandidateReason.CROSS_SOURCE_PASS
    assert candidate.action_epoch == 1


def test_cross_confirm_requires_strictly_post_boundary_current_turn_frame() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    boundary = frame(2)
    fake.active_player = Seat.SELF
    at_boundary = process(
        pipeline, table(), 2, expected=Seat.SELF, revision=3, turn=9,
        formal_action_boundary=boundary,
    )
    assert not at_boundary.candidates

    owner_seen = process(
        pipeline, table(), 3, expected=Seat.SELF, revision=3, turn=9,
        formal_action_boundary=boundary,
    )
    assert not owner_seen.candidates
    fake.pass_seats = {Seat.SELF}
    fake.active_player = Seat.RIGHT
    after = process(
        pipeline, table(), 4, expected=Seat.SELF, revision=3, turn=9,
        formal_action_boundary=boundary,
    )
    candidate = next(item for item in after.candidates if item.seat is Seat.SELF)
    assert candidate.reason is CandidateReason.CROSS_SOURCE_PASS
    assert candidate.first_frame == candidate.last_frame == frame(4)


@pytest.mark.parametrize("interference", ["effect", "terminal", "super_double"])
def test_cross_confirm_is_disabled_by_effect_or_terminal_interference(
    interference: str,
) -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    fake.active_player = Seat.SELF
    process(
        pipeline, table(), 1, expected=Seat.SELF, revision=4, turn=10,
    )
    fake.pass_seats = {Seat.SELF}
    fake.active_player = Seat.RIGHT
    if interference == "effect":
        fake.effect_seats = {Seat.SELF}
    elif interference == "terminal":
        fake.game_end_control = "continue_game"
    else:
        fake.super_double = True

    result = process(
        pipeline, table(), 2, expected=Seat.SELF, revision=4, turn=10,
        formal_action_boundary=frame(1),
    )

    assert not result.candidates
    assert all(
        item.reason is not CandidateReason.CROSS_SOURCE_PASS
        for item in pipeline.pending_candidates(now_ms=200)
    )


def test_cross_confirm_expires_when_active_transition_is_too_late() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    fake.active_player = Seat.SELF
    process(
        pipeline, table(), 2, expected=Seat.SELF, revision=5, turn=11,
        formal_action_boundary=frame(1),
    )
    fake.pass_seats = {Seat.SELF}
    fake.active_player = Seat.RIGHT
    late = process(
        pipeline, table(), 20, expected=Seat.SELF, revision=5, turn=11,
        formal_action_boundary=frame(1),
    )
    assert not late.candidates
    assert pipeline.trackers[Seat.SELF].snapshot().pending_count == 1


def test_nonempty_low_confidence_result_stays_unknown_even_on_stable_blank_probe() -> None:
    class LowConfidenceRecognition(FakeRecognition):
        def recognize_play_region(self, image, seat, **kwargs):
            del image, kwargs
            seat = Seat(seat)
            self.deep_calls.append(seat)
            return PlayRegionResult(seat.value, ("5?",), False, 0.4, (), ())

    fake = LowConfidenceRecognition()
    pipeline = make_pipeline(fake, probe_starvation_ms=100)
    process(pipeline, table(), 1)
    result = process(pipeline, table(), 2)

    assert result.observations
    assert all(item.kind is ObservationKind.UNKNOWN for item in result.observations)
    assert not result.candidates


def test_frame235_audited_half_confidence_play_forms_stable_candidate() -> None:
    cards = ("3C", "6H", "4H", "4H", "5D", "5C")

    class AuditedRecognition(FakeRecognition):
        def recognize_play_region(self, image, seat, **kwargs):
            seat = Seat(seat)
            self.deep_calls.append(seat)
            self.deep_kwargs.append(dict(kwargs))
            roi = image[:, SLICE[seat]]
            if seat is not Seat.SELF or np.max(roi) < 200:
                return PlayRegionResult(seat.value, (), False, 0.0, (), ())
            annotations = tuple(
                RecognitionAnnotation(
                    card, (index * 10, 2, 8, 12),
                    0.5 if card == "6H" else 0.9, "play",
                )
                for index, card in enumerate(cards)
            )
            return PlayRegionResult(
                seat.value, cards, False, 0.5,
                ("source_frame=235", "wild_rank=6"), annotations,
                suit_options=tuple((card,) for card in cards),
            )

    fake = AuditedRecognition()
    fake.active_player = Seat.SELF
    fake.self_buttons = True
    pipeline = make_pipeline(fake)
    process(pipeline, table(), 1, expected=Seat.SELF)
    process(pipeline, table(), 2, expected=Seat.SELF)
    process(pipeline, table(cards=(Seat.SELF,)), 3, expected=Seat.SELF)
    single = process(
        pipeline, table(cards=(Seat.SELF,)), 4, expected=Seat.SELF
    )
    assert not single.candidates
    result = process(
        pipeline, table(cards=(Seat.SELF,)), 5, expected=Seat.SELF
    )
    candidate = next(item for item in result.candidates if item.seat is Seat.SELF)
    assert candidate.cards == cards
    assert candidate.suit_options == tuple((card,) for card in cards)
    assert candidate.confidence == 0.5
    assert "play_quality=audited_two_frame_eligible" in candidate.diagnostics
    assert "source_frame=235" in candidate.diagnostics
    assert any("confidence=0.500" in detail for detail in result.diagnostics)


def test_animation_overrides_recognized_cards_until_surface_is_stable() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    process(pipeline, table(), 1)
    moving = process(pipeline, table(cards=(Seat.OPPOSITE,)), 2, expected=Seat.OPPOSITE)
    opposite = next(item for item in moving.observations if item.seat is Seat.OPPOSITE)
    assert opposite.kind is ObservationKind.ANIMATING
    assert not moving.candidates

    process(pipeline, table(cards=(Seat.OPPOSITE,)), 3, expected=Seat.OPPOSITE)
    settled = process(pipeline, table(cards=(Seat.OPPOSITE,)), 4, expected=Seat.OPPOSITE)
    assert any(item.seat is Seat.OPPOSITE for item in settled.candidates)


def test_older_generation_is_rejected_without_fast_probe_or_deep_read() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    process(pipeline, table(), 1, generation=2)
    counts = (fake.fast_calls, len(fake.roi_calls), len(fake.deep_calls))
    stale = process(pipeline, table(cards=SEATS), 2, generation=1)

    assert (fake.fast_calls, len(fake.roi_calls), len(fake.deep_calls)) == counts
    assert not stale.observations and not stale.candidates
    assert {item.reason for item in stale.drops} == {"older_capture_generation"}


def test_slow_recognition_expires_other_raw_frames_instead_of_ignoring_budget() -> None:
    fake = FakeRecognition()
    clock = ManualClock()
    fake.advance = clock.advance
    pipeline = LiveV2FramePipeline(
        fake,
        config=FramePipelineConfig(max_deep_reads_per_frame=4, raw_max_age_ms=750),
        clock_ms=clock,
    )
    result = process(pipeline, table(cards=SEATS), 1)

    assert len(result.observations) == 1
    assert {drop.reason for drop in result.drops} == {"raw_expired"}
    assert not result.pending_seats


def test_self_visual_opportunity_preempts_expected_seat_without_suppressing_it() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake, reads=1, probe_starvation_ms=1_000)
    process(pipeline, table(), 1, expected=Seat.LEFT)
    fake.active_player = Seat.SELF
    fake.self_buttons = True
    result = process(pipeline, table(), 2, expected=Seat.LEFT)

    assert result.observations[0].seat is Seat.SELF
    assert Seat.RIGHT in result.pending_seats or Seat.OPPOSITE in result.pending_seats


def test_surface_metrics_and_candidate_fifo_are_not_formal_state_mutations() -> None:
    fake = FakeRecognition()
    pipeline = make_pipeline(fake)
    process(pipeline, table(cards=(Seat.RIGHT,)), 1, expected=Seat.RIGHT)
    result = process(pipeline, table(cards=(Seat.RIGHT,)), 2, expected=Seat.RIGHT)
    candidate = next(item for item in result.candidates if item.seat is Seat.RIGHT)

    assert pipeline.pop_candidate(now_ms=200) == candidate
    assert pipeline.pop_candidate(now_ms=200) is None
    assert pipeline.evidence_buffer.candidates(seat=Seat.RIGHT) == (candidate,)


def test_frame_adapter_is_split_into_small_single_responsibility_modules() -> None:
    modules = (
        "live_v2_frame_types.py",
        "live_v2_surface_probe.py",
        "live_v2_frame_dispatch.py",
        "live_v2_frame_pipeline.py",
    )
    for name in modules:
        source = (APPLICATION / name).read_text(encoding="utf-8")
        # Keep the frame adapter modules reviewable without making a few
        # dozen lines of cohesive plumbing a reason to restart the refactor.
        assert len(source.splitlines()) < 600, name

    pipeline_source = (APPLICATION / "live_v2_frame_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "recognize_play_region(" not in pipeline_source
    assert "submit_raw(" not in pipeline_source
    assert "class FramePipelineResult" not in pipeline_source
