"""Cross-engine ActionInterpretation matching must remain discriminating.

DanZero records project names and rank labels while FableDan exposes numeric
type/key identifiers plus claimed ranks.  Joker aliases need a narrow bridge;
they must not turn every cross-schema comparison into a type-only match.
"""

from __future__ import annotations

import pytest

from daguandan_bridge.infrastructure.live_v2_legacy_gateway import (
    _matches_requested,
)
from daguandan_bridge.live_v2 import ActionInterpretation


def _project(
    move_type: str,
    key: str,
    *,
    label: str = "",
    wildcard: tuple[tuple[str, str], ...] = (),
) -> ActionInterpretation:
    return ActionInterpretation(
        move_type=move_type,
        key=key,
        logical_label=label,
        wildcard_assignments=wildcard,
    )


def _fabledan(
    type_id: int,
    key: int,
    claims: tuple[str, ...],
) -> ActionInterpretation:
    return ActionInterpretation(
        move_type="",
        type_id=type_id,
        key=key,
        claim_ranks=claims,
    )


@pytest.mark.parametrize(
    ("requested", "actual"),
    (
        pytest.param(
            _project("Single", "B"),
            _fabledan(1, 13, ("sj",)),
            id="small-joker-alias",
        ),
        pytest.param(
            _project("SINGLE", "R"),
            _fabledan(1, 14, ("BJ",)),
            id="big-joker-alias",
        ),
        pytest.param(
            _project("Single", "7", label="7"),
            _fabledan(1, 4, ("7",)),
            id="ordinary-single",
        ),
        pytest.param(
            _project("Single", "6", label="6"),
            _fabledan(1, 12, ("6",)),
            id="level-card-single",
        ),
        pytest.param(
            _project("Pair", "7", label="77"),
            _fabledan(2, 4, ("7", "7")),
            id="pair",
        ),
        pytest.param(
            _project("Bomb", "7", label="7777"),
            _fabledan(8, 4, ("7", "7", "7", "7")),
            id="bomb",
        ),
        pytest.param(
            _project(
                "ThreeWithTwo",
                "2",
                label="222JJ",
                wildcard=(("9H", "2"),),
            ),
            _fabledan(4, 0, ("2", "2", "2", "J", "J")),
            id="wildcard-full-house-key-2",
        ),
        pytest.param(
            _project(
                "ThreeWithTwo",
                "J",
                label="JJJ22",
                wildcard=(("9H", "J"),),
            ),
            _fabledan(4, 8, ("J", "J", "J", "2", "2")),
            id="wildcard-full-house-key-j",
        ),
        pytest.param(
            _project(
                "StraightFlush",
                "6",
                label="678910",
                wildcard=(("6H", "8"),),
            ),
            _fabledan(9, 6, ("6", "7", "8", "9", "10")),
            id="wildcard-straight-flush-claim",
        ),
    ),
)
def test_equivalent_cross_engine_interpretations_match(requested, actual):
    assert _matches_requested(requested, actual) is True


@pytest.mark.parametrize(
    ("requested", "actual"),
    (
        pytest.param(
            _project("Single", "B"),
            _fabledan(1, 14, ("BJ",)),
            id="small-joker-is-not-big-joker",
        ),
        pytest.param(
            _project("Single", "7"),
            _fabledan(1, 5, ("8",)),
            id="ordinary-single-wrong-rank-without-label",
        ),
        pytest.param(
            _project("Single", "8", label="7"),
            _fabledan(1, 4, ("7",)),
            id="ordinary-single-wrong-key-even-if-claim-matches",
        ),
        pytest.param(
            _project("Single", "6"),
            _fabledan(1, 4, ("7",)),
            id="level-card-is-not-ordinary-rank",
        ),
        pytest.param(
            _project("Pair", "7"),
            _fabledan(2, 5, ("8", "8")),
            id="pair-wrong-key",
        ),
        pytest.param(
            _project("Bomb", "7"),
            _fabledan(8, 5, ("8", "8", "8", "8")),
            id="bomb-wrong-key",
        ),
        pytest.param(
            _project("ThreeWithTwo", "2"),
            _fabledan(4, 8, ("J", "J", "J", "2", "2")),
            id="wildcard-full-house-same-type-different-key",
        ),
        pytest.param(
            _project("ThreeWithTwo", "2", label="JJJ22"),
            _fabledan(4, 8, ("J", "J", "J", "2", "2")),
            id="wildcard-full-house-wrong-key-even-if-claim-matches",
        ),
        pytest.param(
            _project("ThreeWithTwo", "J", label="JJJ33"),
            _fabledan(4, 8, ("J", "J", "J", "2", "2")),
            id="wildcard-full-house-same-key-different-claim",
        ),
        pytest.param(
            _project("StraightFlush", "6", label="6789J"),
            _fabledan(9, 6, ("6", "7", "8", "9", "10")),
            id="wildcard-straight-flush-wrong-claim",
        ),
        pytest.param(
            _project("Pair", "7", label="77"),
            _fabledan(8, 4, ("7", "7", "7", "7")),
            id="different-type-remains-rejected",
        ),
    ),
)
def test_non_equivalent_cross_engine_interpretations_are_rejected(requested, actual):
    assert _matches_requested(requested, actual) is False
