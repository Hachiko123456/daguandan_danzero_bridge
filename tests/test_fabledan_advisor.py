from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from daguandan_bridge.advisor_strategy import (
    build_advisor,
    load_profile_advisor_strategy,
    load_profile_fabledan_debug,
    load_profile_fabledan_diagnostics,
    save_profile_advisor_strategy,
)
from daguandan_bridge.application.ports import AdvicePort
from daguandan_bridge.danzero.rules import (
    infer_best_action,
    logical_action_label,
    wildcard_substitutions,
)
from daguandan_bridge.danzero.state import GuanDanState
from daguandan_bridge.domain.advice import StrategyExecutionTrace
from daguandan_bridge.fabledan import (
    FableDanAdvisor,
    FableDanDecisionResult,
    FableDanStateError,
)
from daguandan_bridge.fabledan.advisor import (
    _PhysicalCards,
    _base_card_id,
    _card_code,
    _unique_observed_move,
)
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.truth_log import truth_log_from_dict


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _leading_state() -> GuanDanState:
    state = GuanDanState()
    state.set_context(
        round_level="8",
        wild_rank="8",
        current_player="self",
        lead_player="self",
    )
    state.confirm_hand(HAND)
    state.remaining_cards = {
        "self": 27,
        "right": 27,
        "opposite": 27,
        "left": 27,
    }
    return state


def _following_state(cards: tuple[str, ...], *, level: str = "8") -> GuanDanState:
    state = GuanDanState()
    state.set_context(
        round_level=level,
        wild_rank=level,
        current_player="self",
        lead_player="right",
    )
    state.confirm_hand(HAND)
    state.record_play("right", cards)
    state.record_pass("opposite")
    state.record_pass("left")
    state.remaining_cards = {
        "self": 27,
        "right": 27 - len(cards),
        "opposite": 27,
        "left": 27,
    }
    return state



REAL_OPENING_HAND = (
    "10D", "10H", "10H", "10S", "2C", "2D", "2H", "3D",
    "4C", "4D", "4H", "4S", "5H", "7H", "7S", "9C", "9D",
    "9H", "AC", "AC", "AD", "JD", "KD", "QC", "QC", "QH",
    "big_joker",
)

UNKNOWN_SUIT_HAND = tuple(
    f"{rank}{suit}"
    for rank in ("3", "4", "5")
    for suit in "HDCS"
    for _ in range(2)
) + ("7H", "7D", "7S")

NONUNIQUE_SUIT_HAND = tuple(
    f"{rank}{suit}"
    for rank in ("8", "9", "10")
    for suit in "HDCS"
    for _ in range(2)
) + ("AH", "AD", "AS")


def _unknown_suit_state(
    cards: tuple[str, ...],
    suit_options: tuple[tuple[str, ...], ...],
    *,
    level: str,
    hand: tuple[str, ...] = UNKNOWN_SUIT_HAND,
    action_metadata: dict[str, object] | None = None,
) -> GuanDanState:
    state = GuanDanState()
    state.set_context(
        round_level=level,
        wild_rank=level,
        current_player="self",
        lead_player="left",
    )
    state.confirm_hand(hand)
    state.record_play(
        "left",
        cards,
        suit_options=suit_options,
        action_metadata=action_metadata,
    )
    state.remaining_cards = {
        "self": 27,
        "right": 27,
        "opposite": 27,
        "left": 27 - len(cards),
    }
    return state


def _full_house_semantics() -> dict[str, object]:
    selected = {
        "type_id": 4,
        "key": 3,
        "claim_ranks": ["6", "6", "6", "2", "2"],
    }
    return {
        "interpretation_ambiguous": False,
        "candidate_interpretations": [selected],
        "selected_interpretation": selected,
        "selection_source": "rules_unique",
    }


LEFT_55_HAND = (
    "6S", "6H", "7S", "7H", "8S", "8H", "9S", "9H", "9D", "9C",
    "AS", "AH", "AD", "AC", "KS", "KH", "KD", "KC", "QS", "QH",
    "QD", "QC", "JS", "JH", "JD", "JC", "10S",
)


def _left_55_state() -> GuanDanState:
    state = GuanDanState()
    state.set_context(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="left",
    )
    state.confirm_hand(LEFT_55_HAND)
    state.record_play("left", ("5S", "5H"))
    state.remaining_cards = {
        "self": 27,
        "right": 27,
        "opposite": 27,
        "left": 25,
    }
    return state


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def q_values(self, _tokens, features):
        self.calls += 1
        return np.linspace(-0.5, 0.5, len(features), dtype=np.float32)


def _install_counting_runtime(advisor: FableDanAdvisor, tmp_path):
    from daguandan_bridge.fabledan import advisor as module

    model = _CountingModel()
    advisor._runtime = module._PolicyRuntime(
        module.NumpyAgent(model),
        "numpy",
        "loaded",
        tmp_path / "best.npz",
        "test-digest",
    )
    return model


def _realtime_action_metadata(
    cards: tuple[str, ...],
    *,
    level: str,
    preferred_play_type: str,
) -> dict[str, object]:
    inference = infer_best_action(
        cards,
        (),
        level,
        preferred_play_type=preferred_play_type,
    )
    assert inference.action is not None
    action = inference.action
    selected = {
        "move_type": str(action[0]),
        "key": str(action[1]),
        "logical_label": logical_action_label(action, level),
        "wildcard_assignments": [
            {"physical_card": card, "as_rank": rank}
            for card, rank in wildcard_substitutions(action, level)
        ],
    }
    return {
        "play_type": str(action[0]),
        "candidate_interpretations": [selected],
        "selected_interpretation": selected,
        "selection_source": "realtime_semantics",
    }


def test_fabledan_is_an_advice_port_and_missing_weights_use_rule_agent(tmp_path):
    advisor = FableDanAdvisor(tmp_path, "profile")

    assert isinstance(advisor, AdvicePort)
    advice = advisor.recommend(_leading_state(), request_id="fd-rule")

    assert advice.strategy == "fabledan-rule"
    assert advice.request_id == "fd-rule"
    assert advice.cards
    assert not (Counter(advice.cards) - Counter(HAND))
    assert advice.engine_input is not None
    assert advice.engine_input["backend"] == "rule"
    assert advice.engine_input["status"] == "missing"
    assert advice.engine_input["path"] == str(
        tmp_path / "profile" / "models" / "best.npz"
    )
    assert advice.engine_input["digest"] is None
    assert advice.engine_input["upstream_commit"] == (
        "7cc5e311b9860bc44f76c082d9c1b21fc8b2d3ec"
    )
    assert advice.engine_input["schema"] == "fabledan-adapter/v1"
    assert advice.engine_input["standard_no_tribute"] is True
    selected = advice.engine_input["selected_action"]
    assert selected in advice.engine_input["legal_actions"]


def test_fabledan_invalid_fixed_npz_falls_back_without_searching_other_paths(tmp_path):
    fixed = tmp_path / "profile" / "models" / "best.npz"
    fixed.parent.mkdir(parents=True)
    fixed.write_bytes(b"not an npz")
    alternate = tmp_path / "FableDan" / "best.npz"
    alternate.parent.mkdir()
    alternate.write_bytes(b"must not be read")

    audit = FableDanAdvisor(tmp_path, "profile").audit_info()

    assert audit["backend"] == "rule"
    assert audit["status"] == "invalid"
    assert audit["path"] == str(fixed)
    assert audit["digest"]
    assert "alternate" not in json.dumps(audit)


@pytest.mark.parametrize(
    ("code", "base"),
    (
        ("AH", 0),
        ("AD", 1),
        ("AS", 2),
        ("AC", 3),
        ("small_joker", 52),
        ("big_joker", 53),
    ),
)
def test_fabledan_card_map_round_trips_both_decks(code: str, base: int):
    assert _base_card_id(code) == base
    assert _card_code(base) == code
    assert _card_code(base + 54) == code


@pytest.mark.parametrize(
    ("project_type", "cards", "fabledan_type"),
    (
        ("Single", ("3S",), "SINGLE"),
        ("Pair", ("3S", "3H"), "PAIR"),
        ("Trips", ("3S", "3H", "3D"), "TRIPLE"),
        ("ThreeWithTwo", ("3S", "3H", "3D", "4S", "4H"), "FULL"),
        ("Straight", ("3S", "4H", "5D", "6C", "7S"), "STRAIGHT"),
        ("ThreePair", ("3S", "3H", "4S", "4H", "5D", "5C"), "PLATE"),
        ("TwoTrips", ("3S", "3H", "3D", "4S", "4H", "4D"), "TUBE"),
        ("Bomb", ("3S", "3H", "3D", "3C"), "BOMB"),
        ("StraightFlush", ("3S", "4S", "5S", "6S", "7S"), "SFLUSH"),
    ),
)
def test_realtime_action_semantics_match_fabledan_types_and_internal_keys(
    project_type: str,
    cards: tuple[str, ...],
    fabledan_type: str,
):
    allocation = _PhysicalCards()
    card_ids = tuple(allocation.allocate(card) for card in cards)

    move, resolution = _unique_observed_move(
        card_ids,
        "J",
        1,
        _realtime_action_metadata(
            cards,
            level="J",
            preferred_play_type=project_type,
        ),
    )

    assert resolution["selected_interpretation"]["type"] == fabledan_type
    assert resolution["selection_source"] == "realtime_semantics"
    assert "metadata_warning" not in resolution
    assert move.type == resolution["selected_interpretation"]["type_id"]


def test_unique_non_wildcard_move_ignores_bad_metadata_with_chinese_warning():
    allocation = _PhysicalCards()
    card_ids = tuple(allocation.allocate(card) for card in ("3S", "3H"))

    move, resolution = _unique_observed_move(
        card_ids,
        "J",
        1,
        {
            "selected_interpretation": {
                "move_type": "ThreePair",
                "key": "6",
            },
            "selection_source": "realtime_semantics",
        },
    )

    assert resolution["selected_interpretation"]["type"] == "PAIR"
    warning = resolution["metadata_warning"]
    assert warning["code"] == "unique_physical_move_metadata_mismatch"
    assert "第 1 条历史不含逢人配" in warning["message"]
    assert "记录牌型" in "；".join(warning["mismatch_reasons"])
    assert move.type == resolution["selected_interpretation"]["type_id"]


def test_latest_session_opening_three_pair_reaches_model_without_warning(tmp_path):
    state = GuanDanState()
    state.set_context(
        round_level="J",
        wild_rank="J",
        current_player="self",
        lead_player="self",
    )
    state.confirm_hand(
        (
            "10D", "10H", "10S", "3C", "3D", "3H", "3S", "3S",
            "5C", "5H", "6S", "AD", "AH", "JC", "KH", "KH", "KS",
            "QD", "QH", "QS", "big_joker",
        )
    )
    first_cards = ("6D", "6D", "7H", "7H", "8D", "8S")
    state.record_play(
        "self",
        first_cards,
        action_metadata=_realtime_action_metadata(
            first_cards,
            level="J",
            preferred_play_type="ThreePair",
        ),
    )
    state.record_pass("right")
    opposite_cards = ("7C", "7D", "8C", "8D", "9D", "9S")
    state.record_play(
        "opposite",
        opposite_cards,
        action_metadata=_realtime_action_metadata(
            opposite_cards,
            level="J",
            preferred_play_type="ThreePair",
        ),
    )
    state.record_play("left", ("10H", "10S", "8H", "8S", "9H", "9S"))
    state.remaining_cards = {
        "self": 21,
        "right": 27,
        "opposite": 21,
        "left": 21,
    }
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        diagnostics="full",
        write_decision_log=False,
    )
    model = _install_counting_runtime(advisor, tmp_path)

    result = advisor.recommend_detailed(state, request_id="latest-opening-turn-5")
    events = result.advice.engine_input["fabledan_trace"][
        "adapter_observation"
    ]["events"]

    assert model.calls == 1
    assert events[0]["move"]["type"] == "PLATE"
    assert events[0]["semantic_resolution"]["selection_source"] == "realtime_semantics"
    assert events[2]["move"]["type"] == "PLATE"
    assert result.warnings == ()


def test_fabledan_maps_complete_following_history_and_returns_a_legal_self_action(
    tmp_path,
):
    advice = FableDanAdvisor(tmp_path, "profile").recommend(
        _following_state(("3S",)),
        request_id="fd-follow",
    )

    assert advice.is_pass or not (Counter(advice.cards) - Counter(HAND))
    assert advice.engine_input is not None
    assert len(advice.engine_input["history"]) == 3
    assert advice.engine_input["project_snapshot"]["remaining_cards"] == {
        "self": 27,
        "right": 26,
        "opposite": 27,
        "left": 27,
    }
    assert advice.engine_input["selected_action"] in advice.engine_input["legal_actions"]


def test_fabledan_accepts_a_finished_leader_wind_with_partner_pass(tmp_path):
    """The adapter must use the live turn rule while reconstructing history."""

    state = GuanDanState()
    state.set_context(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="left",
    )
    state.confirm_hand(HAND)
    left_cards = tuple(
        f"{rank}{suit}"
        for rank in ("9", "10", "J", "Q", "K", "A")
        for suit in "SHCD"
    ) + ("8D", "small_joker", "big_joker")
    assert len(left_cards) == 27
    for card in left_cards[:-1]:
        state.record_play("left", (card,))
        state.record_pass("self")
        state.record_pass("right")
        state.record_pass("opposite")
    state.record_play("left", (left_cards[-1],))
    state.record_pass("self")
    state.record_pass("right")
    state.record_pass("opposite")
    # left has finished, so right catches wind only after its own PASS.
    state.record_play("right", ("9S",))
    state.record_play("opposite", ("10S",))
    state.trick_plays = state.play_history[-2:]
    state.current_player = "self"
    state.lead_player = "right"
    state.remaining_cards = {
        "self": 27,
        "right": 26,
        "opposite": 26,
        "left": 0,
    }

    advice = FableDanAdvisor(tmp_path, "profile").recommend(
        state,
        request_id="wind-with-partner-pass",
    )

    assert advice.engine_input is not None
    history = advice.engine_input["project_snapshot"]["play_history"]
    assert [event["player"] for event in history[-6:]] == [
        "left",
        "self",
        "right",
        "opposite",
        "right",
        "opposite",
    ]


def test_fabledan_accepts_exact_suit_options_from_live_reducer(tmp_path):
    """Concrete suit metadata from replay is not an ambiguous candidate."""

    reducer = LiveReducer("replay-history")
    reducer.confirm_initial_state(
        round_level="8",
        hand=HAND,
        lead_player="right",
        source="test",
    )
    reducer.record_play("right", ("3S",), source="test")
    reducer.record_pass("opposite", source="test")
    reducer.record_pass("left", source="test")

    snapshot = reducer.to_guandan_state().local_snapshot()
    assert snapshot.play_history[0].suit_options == (("S",),)
    advice = FableDanAdvisor(tmp_path, "profile").recommend(
        reducer.to_guandan_state(), request_id="exact-suit-options"
    )

    assert advice.engine_input["history"][0]["move"]["cards"] == ["3S"]


def test_fabledan_input_fingerprint_merges_only_identical_encoded_inputs(tmp_path):
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        runtime_policy="rule_only",
        write_decision_log=False,
    )

    assert advisor.decision_input_fingerprint(
        _following_state(("9S",)),
    ) == advisor.decision_input_fingerprint(
        _following_state(("9C",)),
    )
    assert advisor.decision_input_fingerprint(
        _following_state(("3S", "4S", "5S", "6S", "7S")),
    ) != advisor.decision_input_fingerprint(
        _following_state(("3S", "4H", "5D", "6C", "7S")),
    )


def test_unknown_suit_full_house_runs_both_candidates_and_returns_consensus(
    tmp_path,
    monkeypatch,
):
    state = _unknown_suit_state(
        ("2?", "2?", "6?", "6?", "6?"),
        (("H", "D"), ("H", "D"), ("H", "D"), ("H", "D"), ("S", "C")),
        level="5",
        hand=REAL_OPENING_HAND,
        action_metadata=_full_house_semantics(),
    )
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        runtime_policy="rule_only",
        write_decision_log=False,
    )
    calls: list[dict[str, object]] = []

    def choose_first(_runtime, observation):
        calls.append(observation)
        return 0, None, None, None

    monkeypatch.setattr(advisor, "_evaluate_policy", choose_first)

    result = advisor.recommend_detailed(state, request_id="unknown-suit-consensus")

    assert len(calls) == 6
    resolution = result.advice.engine_input["unknown_suit_resolution"]
    assert resolution["status"] == "consensus"
    assert resolution["candidate_count"] == 6
    assert result.advice.request_id == "unknown-suit-consensus"
    assert result.advice.engine_input["project_snapshot"]["play_history"][0][
        "cards"
    ] == ["2?", "2?", "6?", "6?", "6?"]
    assert "fabledan_training_input" not in result.advice.engine_input
    assert result.advice.engine_input["training_eligible"] is False


def test_unknown_suit_candidate_disagreement_refuses_to_guess(tmp_path, monkeypatch):
    state = _unknown_suit_state(
        ("2?", "2H", "6H", "6H", "6S"),
        (("H", "D"), ("H",), ("H",), ("H",), ("S",)),
        level="3",
        action_metadata=_full_house_semantics(),
    )
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        runtime_policy="rule_only",
        write_decision_log=False,
    )

    def choose_by_exact_candidate(_runtime, observation):
        move = observation["events"][0][2]
        exact_cards = tuple(_card_code(int(card)) for card in move.cards)
        return (0 if exact_cards.count("2H") == 2 else 1), None, None, None

    monkeypatch.setattr(advisor, "_evaluate_policy", choose_by_exact_candidate)

    with pytest.raises(FableDanStateError, match="不一致的 FableDan 推荐"):
        advisor.recommend(state, request_id="unknown-suit-disagreement")


def test_unknown_suit_candidate_rejects_illegal_third_physical_copy(tmp_path):
    state = _unknown_suit_state(
        ("2?", "2H", "2H"),
        (("H",), ("H",), ("H",)),
        level="3",
    )

    with pytest.raises(FableDanStateError, match="没有符合双副牌物理约束"):
        FableDanAdvisor(tmp_path, "profile").recommend(state)


def test_unknown_suit_candidate_requires_unique_type_and_size_semantics(tmp_path):
    state = _unknown_suit_state(
        ("3?", "4S", "5S", "6S", "7S"),
        (("S", "H"), ("S",), ("S",), ("S",), ("S",)),
        level="2",
        hand=NONUNIQUE_SUIT_HAND,
    )

    with pytest.raises(FableDanStateError, match="牌型或大小语义不唯一"):
        FableDanAdvisor(tmp_path, "profile").recommend(state)


def test_unknown_suit_is_audited_and_blocked_before_policy(tmp_path, monkeypatch):
    from daguandan_bridge.fabledan import advisor as module

    monkeypatch.setattr(
        module.RuleAgent,
        "act",
        lambda *_args: pytest.fail("policy must not run"),
    )
    trace = StrategyExecutionTrace("unknown")

    with pytest.raises(FableDanStateError, match="未知花色"):
        FableDanAdvisor(tmp_path, "profile").recommend(
            _following_state(("3?",)),
            request_id="unknown",
            trace=trace,
        )

    engine_input = trace.snapshot()["engine_input"]
    assert engine_input["validation_status"] == "blocked"
    assert "未知花色" in engine_input["validation_error"]
    assert engine_input["backend"] == "rule"


def test_ambiguous_wildcard_history_is_blocked_before_policy(tmp_path, monkeypatch):
    from daguandan_bridge.fabledan import advisor as module

    monkeypatch.setattr(
        module.RuleAgent,
        "act",
        lambda *_args: pytest.fail("policy must not run"),
    )
    trace = StrategyExecutionTrace("wildcard")
    state = _following_state(("JC", "JD", "9H", "2D", "2H"), level="9")

    with pytest.raises(FableDanStateError, match="wildcard.*不唯一"):
        FableDanAdvisor(tmp_path, "profile", diagnostics="full").recommend(
            state,
            request_id="wildcard",
            trace=trace,
        )

    engine_input = trace.snapshot()["engine_input"]
    assert engine_input["validation_status"] == "blocked"
    assert "wildcard" in engine_input["validation_error"]
    root = engine_input["fabledan_trace"]["diagnostics"]["root_cause"]
    assert root["code"] == "wildcard_ambiguity"
    assert root["source_turn_id"] == 1
    assert len(root["candidate_interpretations"]) >= 2
    assert root["selected_interpretation"] is None
    assert root["selection_source"] == "unresolved"


def test_wildcard_semantic_mismatch_reports_cards_candidates_and_differences(
    tmp_path,
    monkeypatch,
):
    from daguandan_bridge.fabledan import advisor as module

    monkeypatch.setattr(
        module.RuleAgent,
        "act",
        lambda *_args: pytest.fail("语义不一致时不应调用模型"),
    )
    state = _following_state(("JC", "JD", "9H", "2D", "2H"), level="9")
    original = state.play_history[0]
    invalid = replace(
        original,
        action_metadata={
            "selected_interpretation": {
                "move_type": "ThreePair",
                "key": "6",
                "wildcard_assignments": [
                    {"physical_card": "9H", "as_rank": "6"}
                ],
            },
            "selection_source": "realtime_semantics",
        },
    )
    state.play_history[0] = invalid
    state.trick_plays[0] = invalid
    trace = StrategyExecutionTrace("wildcard-mismatch")

    with pytest.raises(
        FableDanStateError,
        match="第 1 条历史动作语义不一致：实体牌",
    ):
        FableDanAdvisor(tmp_path, "profile", diagnostics="full").recommend(
            state,
            request_id="wildcard-mismatch",
            trace=trace,
        )

    root = trace.snapshot()["engine_input"]["fabledan_trace"]["diagnostics"][
        "root_cause"
    ]
    assert root["code"] == "action_semantics_mismatch"
    assert root["source_turn_id"] == 1
    assert root["physical_cards"] == ["2D", "2H", "9H", "JC", "JD"]
    assert root["wildcard_count"] == 1
    assert len(root["candidate_interpretations"]) >= 2
    assert all(
        item["mismatch_reasons"]
        for item in root["candidate_match_diagnostics"]
    )


def test_explicit_truth_semantics_resolves_wildcard_without_guessing(tmp_path):
    state = _following_state(("10C", "6C", "6H", "7C", "9C"), level="6")
    original = state.play_history[0]
    resolved = replace(
        original,
        action_metadata={
            "ambiguity": True,
            "selected_interpretation": {
                "type_id": 5,
                "key": 6,
                "claim_ranks": ["6", "7", "8", "9", "10"],
            },
            "selection_source": "exact_engine_state",
        },
    )
    state.play_history[0] = resolved
    state.trick_plays[0] = resolved

    result = FableDanAdvisor(
        tmp_path,
        "profile",
        runtime_policy="rule_only",
        diagnostics="full",
        write_decision_log=False,
    ).recommend_detailed(state, request_id="explicit-wildcard")

    resolution = result.advice.engine_input["fabledan_trace"][
        "adapter_observation"
    ]["events"][0]["semantic_resolution"]
    assert resolution["selection_source"] == "exact_engine_state"
    assert resolution["ambiguity"] is True
    assert resolution["selected_interpretation"]["type_id"] == 5


def test_incomplete_history_and_remaining_counts_are_blocked(tmp_path):
    state = _following_state(("3S",))
    state.remaining_cards["right"] = 25

    with pytest.raises(FableDanStateError, match="remaining_cards"):
        FableDanAdvisor(tmp_path, "profile").recommend(state)


@pytest.mark.parametrize("version", (1, 2, 3, 4))
def test_legacy_truth_schemas_with_unique_history_can_reach_fabledan(
    tmp_path,
    version: int,
):
    raw = {
        "source_session_id": f"legacy-{version}",
        "initial_state": {
            "round_level": "8",
            "lead_player": "right",
            "my_hand": list(HAND),
        },
        "turns": [
            {"turn_id": 1, "actor": "right", "is_pass": False, "cards": ["3S"]},
            {"turn_id": 2, "actor": "opposite", "is_pass": True, "cards": []},
            {"turn_id": 3, "actor": "left", "is_pass": True, "cards": []},
        ],
    }
    if version in {3, 4}:
        raw.update({"schema": f"guandan.truth/{version}", "schema_version": version})
    else:
        raw["schema_version"] = version
    truth = truth_log_from_dict(raw)
    reducer = LiveReducer(f"legacy-{version}")
    for event in truth.to_events(session_id=f"legacy-{version}"):
        reducer.apply(event)

    advice = FableDanAdvisor(tmp_path, "profile").recommend(
        reducer.to_guandan_state()
    )

    assert advice.strategy == "fabledan-rule"
    assert advice.engine_input["validation_status"] == "accepted"


def test_detailed_decision_uses_one_inference_and_appends_complete_jsonl(tmp_path):
    log_directory = tmp_path / "logs" / "fabledan_decisions"
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        debug=True,
        log_directory=log_directory,
    )
    model = _install_counting_runtime(advisor, tmp_path)

    first = advisor.recommend_detailed(_left_55_state(), request_id="left-55-a")
    second = advisor.recommend_detailed(_left_55_state(), request_id="left-55-b")

    assert isinstance(first, FableDanDecisionResult)
    assert model.calls == 2
    assert first.best_action == first.candidates[0].action
    assert first.best_q == first.candidates[0].q_value
    assert len(first.candidates) == first.legal_action_count
    assert [candidate.q_value for candidate in first.candidates] == sorted(
        (candidate.q_value for candidate in first.candidates),
        reverse=True,
    )
    assert first.second_q == first.candidates[1].q_value
    assert first.q_gap == pytest.approx(first.best_q - first.second_q)
    assert second.advice.cards
    assert "decision_log" not in first.advice.engine_input
    assert "decision_log_path" in first.advice.engine_input

    log_path = next(log_directory.glob("*.jsonl"))
    records = [json.loads(line) for line in log_path.read_text("utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["request_id"] == "left-55-a"
    assert records[1]["request_id"] == "left-55-b"
    assert records[0]["schema"] == "fabledan-trace/1"
    observation = records[0]["adapter_observation"]
    assert observation["lead"]["claim_ranks"] == ["5", "5"]
    assert observation["lead_owner_absolute"] == 3
    assert observation["lead_owner_relative"] == 3
    assert observation["player"]["seat_mapping"]["right"] == 1
    assert observation["player"]["seat_mapping"]["opposite"] == 2
    assert observation["player"]["seat_mapping"]["left"] == 3
    legal_count = records[0]["legal_actions"]["legal_count_after_cap"]
    assert len(records[0]["model_output"]["q_values"]) == legal_count
    selected = records[0]["model_output"]["selected_index"]
    assert records[0]["model_output"]["selected_action"] == records[0][
        "legal_actions"
    ]["actions"][selected]
    assert records[0]["model_output"]["model_path"].endswith("best.npz")
    assert records[0]["model_output"]["model_hash"] == "test-digest"
    assert records[0]["diagnostics"]["warnings"] == []


def test_debug_can_embed_complete_trace_without_appending_jsonl(tmp_path):
    log_directory = tmp_path / "logs" / "fabledan_decisions"
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        debug=True,
        write_decision_log=False,
        log_directory=log_directory,
    )
    model = _install_counting_runtime(advisor, tmp_path)

    result = advisor.recommend_detailed(_left_55_state(), request_id="embedded-log")

    trace = result.advice.engine_input["fabledan_trace"]
    assert model.calls == 1
    assert result.advice.engine_input["decision_log_mode"] == "trace_embedded"
    assert "decision_log" not in result.advice.engine_input
    assert "decision_log_path" not in result.advice.engine_input
    assert not log_directory.exists()
    assert trace["schema"] == "fabledan-trace/1"
    assert trace["request_id"] == "embedded-log"
    assert trace["diagnostics_mode"] == "full"
    assert trace["encoding"]["tokens"]["tokens"]
    assert trace["encoding"]["features"]["feats_shape"][1] == 80
    assert len(trace["model_output"]["q_values"]) == result.legal_action_count
    assert trace["model_output"]["selected_index"] < result.legal_action_count
    assert trace["diagnostics"]["errors"] == []


def test_debug_false_exposes_compact_top_three_without_writing_logs(tmp_path):
    log_directory = tmp_path / "logs" / "fabledan_decisions"
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        debug=False,
        log_directory=log_directory,
    )
    model = _install_counting_runtime(advisor, tmp_path)

    advice = advisor.recommend(_left_55_state(), request_id="debug-off")

    assert model.calls == 1
    assert advice.strategy == "fabledan-numpy"
    assert advice.engine_input["debug"] is False
    assert advice.engine_input["top_n"] == 3
    candidates = advice.engine_input["decision"]["candidates"]
    assert 1 <= len(candidates) <= 3
    assert [candidate["rank"] for candidate in candidates] == list(
        range(1, len(candidates) + 1)
    )
    training_input = advice.engine_input["fabledan_training_input"]
    assert training_input["feature_schema"] == "fabledan-token48-feat80/v1"
    assert training_input["tokens"]
    assert len(training_input["features"]) == advice.engine_input["decision"]["legal_action_count"]
    assert all(len(row) == 80 for row in training_input["features"])
    assert training_input["tokens_sha256"]
    assert training_input["features_sha256"]
    assert "fabledan_trace" not in advice.engine_input
    assert not log_directory.exists()


def test_basic_diagnostics_omits_raw_tokens_and_features_but_keeps_legal_q(tmp_path):
    advisor = FableDanAdvisor(
        tmp_path,
        "profile",
        diagnostics="basic",
        write_decision_log=False,
    )
    _install_counting_runtime(advisor, tmp_path)

    result = advisor.recommend_detailed(_left_55_state(), request_id="basic")
    trace = result.advice.engine_input["fabledan_trace"]

    assert trace["diagnostics_mode"] == "basic"
    assert "tokens" not in trace["encoding"]["tokens"]
    assert "feats" not in trace["encoding"]["features"]
    assert trace["encoding"]["tokens"]["tokens_sha256"]
    assert trace["encoding"]["features"]["feats_sha256"]
    assert len(trace["legal_actions"]["actions"]) == result.legal_action_count
    assert len(trace["model_output"]["q_values"]) == result.legal_action_count


def test_log_failure_warns_but_still_returns_model_recommendation(
    tmp_path,
    monkeypatch,
):
    advisor = FableDanAdvisor(tmp_path, "profile", debug=True)
    model = _install_counting_runtime(advisor, tmp_path)

    def fail_log(_timestamp, _payload):
        raise OSError("read-only test directory")

    monkeypatch.setattr(advisor, "_append_decision_log", fail_log)
    result = advisor.recommend_detailed(_left_55_state(), request_id="log-fail")

    assert model.calls == 1
    assert result.advice.cards
    assert any("日志写入失败" in warning for warning in result.warnings)
    assert any(
        "日志写入失败" in warning
        for warning in result.advice.engine_input["validation_warnings"]
    )


def test_real_left_55_state_exposes_all_legal_q_values_without_changing_choice(
    tmp_path,
):
    profiles_root = Path(__file__).parents[1] / "data" / "profiles"
    weights = profiles_root / "tencent_daguandan" / "models" / "best.npz"
    if not weights.is_file():
        pytest.skip("repository FableDan weights are unavailable")
    advisor = FableDanAdvisor(
        profiles_root,
        "tencent_daguandan",
        debug=True,
        log_directory=tmp_path / "fabledan_decisions",
    )

    result = advisor.recommend_detailed(_left_55_state(), request_id="real-left-55")
    action_texts = {candidate.action_text for candidate in result.candidates}

    assert {"66", "77", "88", "9999"} <= action_texts
    assert result.best_action == result.candidates[0].action
    assert result.advice.engine_input["lead_text"] == "55"
    assert result.advice.engine_input["lead_owner"] == 3
    assert result.advice.engine_input["lead_owner_seat"] == "left"
    assert not result.warnings


def test_profile_strategy_builds_fabledan_and_persists_default(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "profile.json").write_text(
        json.dumps({"name": "profile"}),
        encoding="utf-8",
    )

    assert load_profile_advisor_strategy(tmp_path, "profile") == "fabledan"
    assert save_profile_advisor_strategy(tmp_path, "profile", "FableDan") == "fabledan"
    assert load_profile_advisor_strategy(tmp_path, "profile") == "fabledan"
    advisor = build_advisor(
        "fabledan",
        profiles_root=tmp_path,
        profile_name="profile",
    )
    assert isinstance(advisor, FableDanAdvisor)
    assert advisor.runtime_policy == "model_required"
    assert advisor.audit_info()["backend"] == "numpy"
    assert advisor.audit_info()["status"] == "missing"

    profile_data = json.loads((profile / "profile.json").read_text("utf-8"))
    profile_data["fabledan_debug"] = True
    (profile / "profile.json").write_text(
        json.dumps(profile_data),
        encoding="utf-8",
    )
    assert load_profile_fabledan_debug(tmp_path, "profile") is True
    assert build_advisor(
        "fabledan",
        profiles_root=tmp_path,
        profile_name="profile",
    ).debug is True


def test_fabledan_diagnostics_environment_overrides_profile(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "profile.json").write_text(
        json.dumps({"fabledan_diagnostics": "basic"}), encoding="utf-8"
    )

    assert load_profile_fabledan_diagnostics(tmp_path, "profile") == "basic"
    monkeypatch.setenv("FABLEDAN_DIAGNOSTICS", "full")
    assert load_profile_fabledan_diagnostics(tmp_path, "profile") == "full"
    advisor = build_advisor("fabledan", profiles_root=tmp_path, profile_name="profile")
    assert advisor.diagnostics == "full"


def test_vendor_contains_only_approved_runtime_and_notice_files():
    vendor = (
        Path(__file__).parents[1]
        / "src"
        / "daguandan_bridge"
        / "fabledan"
        / "_vendor"
    )
    runtime = vendor / "fabledan"

    assert {path.name for path in runtime.iterdir() if path.is_file()} == {
        "__init__.py",
        "cards.py",
        "combos.py",
        "encode.py",
        "agents.py",
        "model_np.py",
        "engine.py",
    }
    assert {path.name for path in vendor.iterdir() if path.is_file()} == {
        "FABLEDAN_LICENSE",
        "FABLEDAN_REVISION",
    }
