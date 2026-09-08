"""Normalize existing recognition output into live-v2 observations.

This is an anti-corruption layer only. It does not decide turn ownership,
advance formal state, infer missing PASS actions, or import image/UI toolkits.
"""

from __future__ import annotations

from collections import Counter

from ..domain.recognition import PlayRegionResult
from .types import (
    FrameIdentity,
    ObservationKind,
    ObservationReason,
    Seat,
    SeatObservation,
)


class VisionAdapter:
    """Convert current recognizer records into the strict observation schema."""

    def __init__(
        self,
        *,
        play_confidence: float = 0.80,
        audited_play_confidence: float = 0.50,
        pass_confidence: float = 0.80,
    ) -> None:
        if not 0.0 <= play_confidence <= 1.0:
            raise ValueError("play_confidence must be between zero and one")
        if not 0.0 <= pass_confidence <= 1.0:
            raise ValueError("pass_confidence must be between zero and one")
        if not 0.0 <= audited_play_confidence <= play_confidence:
            raise ValueError("audited_play_confidence must not exceed play_confidence")
        self.play_confidence = play_confidence
        self.audited_play_confidence = audited_play_confidence
        self.pass_confidence = pass_confidence

    def normalize_play_region(
        self,
        result: PlayRegionResult,
        *,
        frame: FrameIdentity,
        evidence_id: str | None = None,
        processing_ms: int,
        animating: bool = False,
        empty_confirmed: bool = False,
    ) -> SeatObservation:
        """Normalize one seat-local recognition call without changing state."""

        diagnostics = list(result.diagnostics)
        suit_options: tuple[tuple[str, ...], ...] | None = ()
        if animating:
            kind = ObservationKind.ANIMATING
            reason = ObservationReason.ANIMATION_DETECTED
        elif result.cards and result.is_pass:
            kind = ObservationKind.UNKNOWN
            reason = ObservationReason.CONFLICTING_SIGNALS
        elif result.cards:
            suit_options = self._normalize_suit_options(
                tuple(result.cards), result.suit_options
            )
            accepted, quality = self._play_quality(result, suit_options)
            diagnostics.extend(quality)
            kind = (
                ObservationKind.PLAY
                if accepted
                else ObservationKind.UNKNOWN
            )
            if kind is ObservationKind.UNKNOWN:
                reason = ObservationReason.UNREADABLE
                if suit_options is None:
                    diagnostics.append("suit_options_misaligned")
                suit_options = ()
            else:
                reason = ObservationReason.CARDS_RECOGNIZED
        elif result.is_pass:
            kind = (
                ObservationKind.PASS
                if result.confidence >= self.pass_confidence
                else ObservationKind.UNKNOWN
            )
            if kind is ObservationKind.UNKNOWN:
                reason = ObservationReason.UNREADABLE
            else:
                reason = ObservationReason.PASS_MARKER
        elif empty_confirmed:
            kind = ObservationKind.EMPTY
            reason = ObservationReason.STABLE_EMPTY
        else:
            # An empty card tuple from the legacy matcher can mean true empty,
            # an occluded surface, or recognition failure. The caller must
            # provide independent empty evidence before it becomes EMPTY.
            kind = ObservationKind.UNKNOWN
            reason = ObservationReason.UNREADABLE

        return SeatObservation(
            observation_id=evidence_id or self._evidence_id(frame, result.player),
            frame=frame,
            seat=Seat(result.player),
            kind=kind,
            cards=tuple(result.cards) if kind is ObservationKind.PLAY else (),
            confidence=float(result.confidence),
            reason=reason,
            processing_ms=processing_ms,
            suit_options=suit_options or (),
            diagnostics=tuple(diagnostics),
        )

    # Friendly aliases for callers migrating from different naming schemes.
    from_play_region = normalize_play_region
    normalize = normalize_play_region

    @staticmethod
    def _evidence_id(frame: FrameIdentity, seat: str) -> str:
        return (
            f"{frame.session_id}:{frame.capture_generation}:"
            f"{frame.frame_sequence}:{frame.roi_version}:{frame.source_id}:{seat}"
        )

    @staticmethod
    def _normalize_suit_options(
        cards: tuple[str, ...],
        raw_options: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...] | None:
        if not raw_options:
            return tuple((card,) for card in cards)
        if len(raw_options) != len(cards):
            return None
        normalized: list[tuple[str, ...]] = []
        for card, raw_values in zip(cards, raw_options, strict=True):
            values: list[str] = [card]
            rank = card[:-1] if len(card) >= 2 else card
            for raw_value in raw_values:
                option = str(raw_value)
                if option in {"S", "H", "C", "D"} and card not in {
                    "small_joker",
                    "big_joker",
                }:
                    option = f"{rank}{option}"
                if option and option not in values:
                    values.append(option)
            normalized.append(tuple(values))
        return tuple(normalized)

    def _play_quality(
        self,
        result: PlayRegionResult,
        suit_options: tuple[tuple[str, ...], ...] | None,
    ) -> tuple[bool, tuple[str, ...]]:
        confidence = float(result.confidence)
        diagnostics: list[str] = []
        if suit_options is None:
            return False, tuple(diagnostics)
        if confidence >= self.play_confidence:
            return True, tuple(diagnostics)
        diagnostics.append(f"play_confidence={confidence:.3f}")
        if confidence < self.audited_play_confidence:
            diagnostics.append("play_confidence_below_audited_minimum")
            return False, tuple(diagnostics)

        annotations = tuple(
            item for item in result.annotations if item.category == "play"
        )
        if Counter(item.label for item in annotations) != Counter(result.cards):
            diagnostics.append("play_annotations_do_not_cover_cards")
            return False, tuple(diagnostics)
        if any(not self._auditable_annotation(item) for item in annotations):
            diagnostics.append("play_annotation_not_auditable")
            return False, tuple(diagnostics)
        minimum = min((float(item.confidence) for item in annotations), default=0.0)
        if minimum < self.audited_play_confidence:
            diagnostics.append("play_annotation_below_audited_minimum")
            return False, tuple(diagnostics)
        diagnostics.extend((
            "play_quality=audited_two_frame_eligible",
            f"play_annotation_count={len(annotations)}",
            f"play_annotation_min_confidence={minimum:.3f}",
        ))
        return True, tuple(diagnostics)

    @staticmethod
    def _auditable_annotation(annotation: object) -> bool:
        try:
            score = float(getattr(annotation, "confidence"))
            box = tuple(getattr(annotation, "box"))
        except (TypeError, ValueError, AttributeError):
            return False
        return bool(
            len(box) == 4
            and all(type(value) is int for value in box)
            and box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0
            and 0.0 <= score <= 1.0
        )
