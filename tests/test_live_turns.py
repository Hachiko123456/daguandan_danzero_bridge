from __future__ import annotations

import pytest

from daguandan_bridge.live.turns import project_trick_turn


@pytest.mark.parametrize(
    ("leader", "receiver", "first_opponent", "second_opponent"),
    (
        ("self", "opposite", "right", "left"),
        ("right", "left", "opposite", "self"),
        ("opposite", "self", "left", "right"),
        ("left", "right", "self", "opposite"),
    ),
)
def test_finished_leader_wind_skips_partner_until_both_opponents_pass(
    leader,
    receiver,
    first_opponent,
    second_opponent,
):
    opening = project_trick_turn(leader, {leader})
    assert opening.wind_receiver == receiver
    assert opening.required_passers == {first_opponent, second_opponent}
    assert opening.expected_after(leader) == first_opponent

    first_pass = project_trick_turn(leader, {leader}, {first_opponent})
    assert not first_pass.is_complete
    assert first_pass.expected_after(first_opponent) == second_opponent

    closing = project_trick_turn(
        leader,
        {leader},
        {first_opponent, second_opponent},
    )
    assert closing.is_complete
    assert closing.expected_after(second_opponent) == receiver


def test_unfinished_leader_remains_next_trick_leader_after_all_other_passes():
    closing = project_trick_turn("left", (), {"self", "right", "opposite"})

    assert closing.wind_receiver is None
    assert closing.is_complete
    assert closing.expected_after("opposite") == "left"
