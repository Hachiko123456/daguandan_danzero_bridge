from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from daguandan_bridge.advisor_strategy import (
    build_advisor,
    load_profile_advisor_strategy,
    save_profile_advisor_strategy,
)
from daguandan_bridge.application.ports import AdvicePort
from daguandan_bridge.danzero.state import GuanDanState
from daguandan_bridge.domain.advice import StrategyExecutionTrace
from daguandan_bridge.fabledan import FableDanAdvisor, FableDanStateError
from daguandan_bridge.fabledan.advisor import _base_card_id, _card_code
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
        tmp_path / "profile" / "models" / "fabledan_weights.npz"
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
    fixed = tmp_path / "profile" / "models" / "fabledan_weights.npz"
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
        FableDanAdvisor(tmp_path, "profile").recommend(
            state,
            request_id="wildcard",
            trace=trace,
        )

    engine_input = trace.snapshot()["engine_input"]
    assert engine_input["validation_status"] == "blocked"
    assert "wildcard" in engine_input["validation_error"]


def test_incomplete_history_and_remaining_counts_are_blocked(tmp_path):
    state = _following_state(("3S",))
    state.remaining_cards["right"] = 25

    with pytest.raises(FableDanStateError, match="remaining_cards"):
        FableDanAdvisor(tmp_path, "profile").recommend(state)


@pytest.mark.parametrize("version", (1, 2, 3))
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
    if version == 3:
        raw.update({"schema": "guandan.truth/3", "schema_version": 3})
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


def test_profile_strategy_builds_fabledan_and_persists_default(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "profile.json").write_text(
        json.dumps({"name": "profile"}),
        encoding="utf-8",
    )

    assert load_profile_advisor_strategy(tmp_path, "profile") == "danzero"
    assert save_profile_advisor_strategy(tmp_path, "profile", "FableDan") == "fabledan"
    assert load_profile_advisor_strategy(tmp_path, "profile") == "fabledan"
    assert isinstance(
        build_advisor("fabledan", profiles_root=tmp_path, profile_name="profile"),
        FableDanAdvisor,
    )


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
    }
    assert {path.name for path in vendor.iterdir() if path.is_file()} == {
        "FABLEDAN_LICENSE",
        "FABLEDAN_REVISION",
    }
