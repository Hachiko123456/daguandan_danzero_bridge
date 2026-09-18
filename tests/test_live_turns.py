from __future__ import annotations

import pytest

from daguandan_bridge.live.turns import WindCatchPolicy, project_trick_turn


@pytest.mark.parametrize(
    ("leader", "receiver", "required_passes"),
    (
        ("self", "opposite", ("right", "opposite", "left")),
        ("right", "left", ("opposite", "left", "self")),
        ("opposite", "self", ("left", "self", "right")),
        ("left", "right", ("self", "right", "opposite")),
    ),
)
def test_explicit_pass_policy_requires_every_active_player_to_pass(
    leader,
    receiver,
    required_passes,
):
    opening = project_trick_turn(
        leader, {leader},
        wind_catch_policy=WindCatchPolicy.PARTNER_MUST_EXPLICITLY_PASS,
    )
    assert opening.wind_receiver == receiver
    assert opening.required_passers == set(required_passes)
    assert opening.expected_after(leader) == required_passes[0]

    for index, player in enumerate(required_passes[:-1], start=1):
        projection = project_trick_turn(
            leader, {leader}, required_passes[:index],
            wind_catch_policy=WindCatchPolicy.PARTNER_MUST_EXPLICITLY_PASS,
        )
        assert not projection.is_complete
        assert projection.expected_after(player) == required_passes[index]

    closing = project_trick_turn(
        leader,
        {leader},
        required_passes,
        wind_catch_policy=WindCatchPolicy.PARTNER_MUST_EXPLICITLY_PASS,
    )
    assert closing.is_complete
    assert closing.expected_after(required_passes[-1]) == receiver


def test_tencent_receiver_passes_only_when_a_later_opponent_still_must_respond():
    opening = project_trick_turn(
        "right", {"right"},
        wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    )
    assert opening.wind_receiver == "left"
    assert opening.required_passers == {"opposite", "self"}

    after_opposite = project_trick_turn(
        "right", {"right"}, {"opposite"},
        wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    )
    assert not after_opposite.is_complete
    assert after_opposite.expected_after("opposite") == "left"

    # left is the future wind receiver but must PASS once so self, the remaining
    # opponent, can respond.  That PASS is not itself required for completion.
    after_receiver = project_trick_turn(
        "right", {"right"}, {"opposite", "left"},
        wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    )
    assert not after_receiver.is_complete
    assert after_receiver.expected_after("left") == "self"

    closing = project_trick_turn(
        "right", {"right"}, {"opposite", "left", "self"},
        wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    )
    assert closing.is_complete
    assert closing.expected_after("self") == "left"


def test_unfinished_leader_remains_next_trick_leader_after_all_other_passes():
    closing = project_trick_turn("left", (), {"self", "right", "opposite"})

    assert closing.wind_receiver is None
    assert closing.is_complete
    assert closing.expected_after("opposite") == "left"


def test_tencent_wind_catch_with_one_finished_opponent_closes_after_only_opponent_pass():
    # Reproduces game_20260816_193531_f94cbf: self and left are already
    # finished; right is the only opponent that must pass before opposite leads.
    opening = project_trick_turn(
        "self", {"self", "left"},
        wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    )
    assert opening.wind_receiver == "opposite"
    assert opening.required_passers == {"right"}
    assert opening.expected_after("self") == "right"

    closing = project_trick_turn(
        "self", {"self", "left"}, {"right"},
        wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    )
    assert closing.is_complete
    assert closing.expected_after("right") == "opposite"
