from __future__ import annotations

from dataclasses import dataclass

from ..danzero.state import Seat


PLAY_REGION_TO_SEAT: dict[str, Seat] = {
    "my_play": "self",
    "left_play": "left",
    "opposite_play": "opposite",
    "right_play": "right",
}
SEATS_IN_ORDER: tuple[Seat, ...] = ("self", "right", "opposite", "left")


@dataclass(frozen=True)
class RecognitionAnnotation:
    label: str
    box: tuple[int, int, int, int]
    confidence: float
    category: str


@dataclass(frozen=True)
class RecognizedEvent:
    player: Seat
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    source: str


@dataclass(frozen=True)
class RecognitionResult:
    round_level: str | None
    wild_rank: str | None
    current_player: Seat | None
    lead_player: Seat | None
    my_hand: tuple[str, ...]
    events: tuple[RecognizedEvent, ...]
    field_confidences: dict[str, float]
    sources: dict[str, str]
    unresolved_fields: tuple[str, ...]
    diagnostics: tuple[str, ...]
    annotations: tuple[RecognitionAnnotation, ...] = ()
    buttons: tuple[str, ...] = ()
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class PlayRegionResult:
    player: Seat
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    diagnostics: tuple[str, ...]
    annotations: tuple[RecognitionAnnotation, ...]
    source: str = ""
    post_hand: tuple[str, ...] = ()
    post_hand_confidence: float = 0.0
    suit_options: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class PlacementSignal:
    player: Seat
    placement: str
    confidence: float
    source: str


@dataclass(frozen=True)
class FastSignalResult:
    expected_player: Seat
    active_player: Seat | None
    pass_visible: bool
    self_action_buttons_visible: bool
    effect_visible: bool
    # ``pass_visible`` is retained for the zone lifecycle.  The live
    # ownership guard additionally needs the seat-bound marker identity so it
    # never converts a stale marker from another action into a pass.
    pass_marker_player: Seat | None = None
    # The expected-seat fields above retain their original meaning.  Recovery
    # also needs the complete visible status set so it can prove intervening
    # passes after an expected action has been delayed.
    pass_marker_players: tuple[Seat, ...] = ()
    super_double_visible: bool = False
    game_end_control: str | None = None
    placements: tuple[PlacementSignal, ...] = ()
    # ``cannot_beat`` is a local-action hint only.  It is deliberately kept
    # separate from ``pass_visible``: the former is a button on the local
    # action bar, while the latter is a persistent seat-bound marker.  The
    # live state machine must still corroborate this signal across frames
    # before committing a PASS event.  These fields are appended after the
    # legacy defaults so existing positional constructors retain their
    # original meaning.
    cannot_beat_visible: bool = False
    cannot_beat_confidence: float = 0.0
    cannot_beat_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class OpeningSignal:
    """Raw opening evidence, before the live state machine commits a lead."""

    super_double_visible: bool
    marker_player: Seat | None
    active_player: Seat | None
    self_action_buttons_visible: bool
    game_end_control: str | None = None
