from __future__ import annotations

import pytest

from daguandan_bridge.live.turns import project_trick_turn


@pytest.mark.parametrize(
    ("leader", "receiver", "required_passes"),
    (
        ("self", "opposite", ("right", "opposite", "left")),
        ("right", "left", ("opposite", "left", "self")),
        ("opposite", "self", ("left", "self", "right")),
        ("left", "right", ("self", "right", "opposite")),
    ),
)
def test_finished_leader_wind_requires_every_active_player_to_pass(
    leader,
    receiver,
    required_passes,
):
    opening = project_trick_turn(leader, {leader})
    assert opening.wind_receiver == receiver
    assert opening.required_passers == set(required_passes)
    assert opening.expected_after(leader) == required_passes[0]

    for index, player in enumerate(required_passes[:-1], start=1):
        projection = project_trick_turn(leader, {leader}, required_passes[:index])
        assert not projection.is_complete
        assert projection.expected_after(player) == required_passes[index]

    closing = project_trick_turn(
        leader,
        {leader},
        required_passes,
    )
    assert closing.is_complete
    assert closing.expected_after(required_passes[-1]) == receiver


def test_legacy_projection_skips_future_wind_receiver_only_when_requested():
    opening = project_trick_turn(
        "right",
        {"right"},
        wind_receiver_must_pass=False,
    )
    assert opening.wind_receiver == "left"
    assert opening.required_passers == {"opposite", "self"}
    assert opening.expected_after("opposite") == "self"

    closing = project_trick_turn(
        "right",
        {"right"},
        {"opposite", "self"},
        wind_receiver_must_pass=False,
    )
    assert closing.is_complete
    assert closing.expected_after("self") == "left"


def test_unfinished_leader_remains_next_trick_leader_after_all_other_passes():
    closing = project_trick_turn("left", (), {"self", "right", "opposite"})

    assert closing.wind_receiver is None
    assert closing.is_complete
    assert closing.expected_after("opposite") == "left"
