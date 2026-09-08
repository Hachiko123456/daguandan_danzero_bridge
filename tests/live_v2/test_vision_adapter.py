from __future__ import annotations

from daguandan_bridge.domain.recognition import PlayRegionResult, RecognitionAnnotation
from daguandan_bridge.live_v2.types import (
    FrameIdentity,
    ObservationKind,
    ObservationReason,
    Seat,
)
from daguandan_bridge.live_v2.vision_adapter import VisionAdapter


def frame(seq: int = 1) -> FrameIdentity:
    return FrameIdentity("s", 2, seq, seq * 100, "roi-v3", "window-3")


def result(
    *,
    cards: tuple[str, ...] = (),
    is_pass: bool = False,
    confidence: float = 0.9,
    source: str = "test",
    suit_options: tuple[tuple[str, ...], ...] = (),
    diagnostics: tuple[str, ...] = ("legacy-diagnostic",),
    annotations: tuple[RecognitionAnnotation, ...] = (),
) -> PlayRegionResult:
    return PlayRegionResult(
        player="left",
        cards=cards,
        is_pass=is_pass,
        confidence=confidence,
        diagnostics=diagnostics,
        annotations=annotations,
        source=source,
        suit_options=suit_options,
    )


def test_normalizes_confident_play_without_formal_state_side_effects() -> None:
    observation = VisionAdapter().normalize_play_region(
        result(cards=("5D", "5C")), frame=frame(), processing_ms=110
    )
    assert observation.kind is ObservationKind.PLAY
    assert observation.reason is ObservationReason.CARDS_RECOGNIZED
    assert observation.seat is Seat.LEFT
    assert observation.cards == ("5D", "5C")
    assert observation.observation_id == "s:2:1:roi-v3:window-3:left"
    assert observation.suit_options == (("5D",), ("5C",))
    assert observation.diagnostics == ("legacy-diagnostic",)


def test_low_confidence_play_is_unknown_not_empty_or_pass() -> None:
    observation = VisionAdapter(play_confidence=0.8).normalize_play_region(
        result(cards=("8H",), confidence=0.79), frame=frame(), processing_ms=110
    )
    assert observation.kind is ObservationKind.UNKNOWN
    assert observation.reason is ObservationReason.UNREADABLE
    assert observation.cards == ()


def test_audited_half_confidence_play_is_two_frame_eligible() -> None:
    cards = ("3C", "6H", "4H", "4H", "5D", "5C")
    annotations = tuple(
        RecognitionAnnotation(card, (index * 10, 2, 8, 12), 0.5 if card == "6H" else 0.9, "play")
        for index, card in enumerate(cards)
    )
    observation = VisionAdapter().normalize_play_region(
        result(
            cards=cards, confidence=0.5,
            suit_options=tuple((card,) for card in cards),
            annotations=annotations,
        ),
        frame=frame(), processing_ms=110,
    )
    assert observation.kind is ObservationKind.PLAY
    assert observation.cards == cards
    assert "play_quality=audited_two_frame_eligible" in observation.diagnostics
    assert "play_annotation_min_confidence=0.500" in observation.diagnostics


def test_relaxed_play_rejects_incomplete_or_subminimum_annotations() -> None:
    card = "6H"
    adapter = VisionAdapter()
    missing = adapter.normalize_play_region(
        result(cards=(card,), confidence=0.5, suit_options=((card,),)),
        frame=frame(), processing_ms=110,
    )
    weak = adapter.normalize_play_region(
        result(
            cards=(card,), confidence=0.5, suit_options=((card,),),
            annotations=(RecognitionAnnotation(card, (0, 0, 8, 12), 0.49, "play"),),
        ),
        frame=frame(2), processing_ms=210,
    )
    assert missing.kind is weak.kind is ObservationKind.UNKNOWN
    assert "play_annotations_do_not_cover_cards" in missing.diagnostics
    assert "play_annotation_below_audited_minimum" in weak.diagnostics


def test_pass_requires_explicit_high_confidence_pass_result() -> None:
    adapter = VisionAdapter(pass_confidence=0.8)
    assert adapter.normalize_play_region(
        result(is_pass=True, confidence=0.81), frame=frame(), processing_ms=110
    ).kind is ObservationKind.PASS
    assert adapter.normalize_play_region(
        result(is_pass=True, confidence=0.79), frame=frame(2), processing_ms=210
    ).kind is ObservationKind.UNKNOWN


def test_empty_must_be_independently_confirmed() -> None:
    adapter = VisionAdapter()
    assert adapter.normalize_play_region(
        result(), frame=frame(), processing_ms=110
    ).kind is ObservationKind.UNKNOWN
    empty = adapter.normalize_play_region(
        result(), frame=frame(2), processing_ms=210, empty_confirmed=True
    )
    assert empty.kind is ObservationKind.EMPTY
    assert empty.reason is ObservationReason.STABLE_EMPTY


def test_animation_and_conflicting_signals_never_become_pass() -> None:
    adapter = VisionAdapter()
    animated = adapter.normalize_play_region(
        result(is_pass=True), frame=frame(), processing_ms=110, animating=True
    )
    conflict = adapter.normalize_play_region(
        result(cards=("9S",), is_pass=True), frame=frame(2), processing_ms=210
    )
    assert animated.kind is ObservationKind.ANIMATING
    assert animated.reason is ObservationReason.ANIMATION_DETECTED
    assert conflict.kind is ObservationKind.UNKNOWN
    assert conflict.reason is ObservationReason.CONFLICTING_SIGNALS


def test_custom_evidence_id_is_preserved() -> None:
    observation = VisionAdapter().from_play_region(
        result(cards=("AH",)), frame=frame(), processing_ms=110,
        evidence_id="capture-evidence-7",
    )
    assert observation.observation_id == "capture-evidence-7"


def test_preserves_explicit_uncertain_suit_options_one_to_one() -> None:
    observation = VisionAdapter().normalize_play_region(
        result(
            cards=("5?", "5C"),
            suit_options=(("5H", "5D"), ("5C",)),
            diagnostics=("first_suit_occluded",),
        ),
        frame=frame(),
        processing_ms=110,
    )
    assert observation.kind is ObservationKind.PLAY
    assert observation.cards == ("5?", "5C")
    assert observation.suit_options == (("5?", "5H", "5D"), ("5C",))
    assert observation.diagnostics == ("first_suit_occluded",)


def test_misaligned_suit_options_are_unknown_not_silently_repaired() -> None:
    observation = VisionAdapter().normalize_play_region(
        result(cards=("5D", "5C"), suit_options=(("5D",),)),
        frame=frame(),
        processing_ms=110,
    )
    assert observation.kind is ObservationKind.UNKNOWN
    assert observation.cards == ()
    assert observation.suit_options == ()
    assert "suit_options_misaligned" in observation.diagnostics
