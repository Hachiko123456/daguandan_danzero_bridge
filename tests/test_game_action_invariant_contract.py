"""Focused construction invariants for immutable live-v2 GameAction values."""

from __future__ import annotations

import pytest

from daguandan_bridge.live_v2.game_state import GameAction
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, StateVersion
from daguandan_bridge.live_v2.reducer_snapshot import ReducedActionState, _same_action
from daguandan_bridge.live_v2.types import ActionKind


def _action(
    *,
    kind: ActionKind = ActionKind.PLAY,
    cards: tuple[str, ...] = ("3H",),
    suit_options: tuple[tuple[str, ...], ...] = (("3H",),),
    captured_ms: int = 110,
) -> GameAction:
    before = StateVersion("game-action-contract", 3, 7)
    after = StateVersion("game-action-contract", 4, 8)
    first = FrameIdentity(
        "game-action-contract", 2, 10, 100, "roi", "window"
    )
    last = FrameIdentity(
        "game-action-contract", 2, 11, 110, "roi", "window"
    )
    return GameAction(
        action_id="action-8",
        version_before=before,
        version_after=after,
        seat=Seat.RIGHT,
        kind=kind,
        cards=cards,
        suit_options=suit_options,
        action_epoch=7,
        evidence_ids=("frame-10", "frame-11"),
        first_frame=first,
        last_frame=last,
        confidence=0.99,
        captured_ms=captured_ms,
    )


def test_game_action_captured_ms_must_equal_final_evidence_timestamp():
    with pytest.raises(
        ValueError,
        match="captured_ms must identify the final evidence frame",
    ):
        _action(captured_ms=111)


@pytest.mark.parametrize(
    ("cards", "suit_options", "message"),
    (
        pytest.param((), (), "cards must not be empty", id="empty-play"),
        pytest.param(
            ("3H",),
            (),
            "suit_options must align one-to-one",
            id="missing-physical-entity",
        ),
    ),
)
def test_game_action_play_requires_valid_cards_and_aligned_physical_entities(
    cards: tuple[str, ...],
    suit_options: tuple[tuple[str, ...], ...],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        _action(cards=cards, suit_options=suit_options)


@pytest.mark.parametrize(
    ("cards", "suit_options"),
    (
        pytest.param(("3H",), (), id="pass-with-card"),
        pytest.param((), (("3H",),), id="pass-with-physical-option"),
    ),
)
def test_game_action_pass_cannot_carry_cards_or_physical_options(
    cards: tuple[str, ...],
    suit_options: tuple[tuple[str, ...], ...],
):
    with pytest.raises(ValueError, match="pass action cannot contain cards or suit_options"):
        _action(kind=ActionKind.PASS, cards=cards, suit_options=suit_options)


@pytest.mark.parametrize("joker", ("small_joker", "big_joker"))
def test_joker_game_action_matches_equivalent_reduced_physical_entity(joker: str):
    formal = _action(cards=(joker,), suit_options=((joker,),))
    reduced = ReducedActionState(
        seat=Seat.RIGHT,
        kind=ActionKind.PLAY,
        cards=(joker,),
        suit_options=((joker,),),
    )

    assert _same_action(formal, reduced) is True

