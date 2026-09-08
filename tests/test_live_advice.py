from __future__ import annotations

import threading
import time

import numpy as np

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.fabledan.advisor import FableDanStateError
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.recognition_service import FastSignalResult, PlayRegionResult


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")
OCCLUDED_FULL_HOUSE_HAND = (
    "10C",
    "10H",
    "10S",
    "2D",
    "3C",
    "5C",
    "5C",
    "5H",
    "6C",
    "6D",
    "6S",
    "6S",
    "8D",
    "8S",
    "AC",
    "AC",
    "AH",
    "AS",
    "JC",
    "JS",
    "JS",
    "KC",
    "KH",
    "KS",
    "QC",
    "QH",
    "big_joker",
)


class LeftPlayRecognition:
    def recognize_fast_signals(self, _image, expected_player):
        return FastSignalResult(
            expected_player=expected_player,
            active_player=expected_player,
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank):
        del wild_rank
        return PlayRegionResult(
            player=seat,
            cards=("7S", "7H"),
            is_pass=False,
            confidence=0.95,
            diagnostics=(),
            annotations=(),
            source="fake",
        )


class FakeAdvisor:
    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.calls = 0
        self.called = threading.Event()

    def recommend(self, state, *, request_id=""):
        self.calls += 1
        call = self.calls
        self.called.set()
        if call == 1 and self.gate is not None:
            assert self.gate.wait(2)
        return LocalAdvice(
            strategy="fake",
            cards=("2S",),
            play_type="Single",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.5,
            request_id=request_id,
            engine_input={"request_id": request_id, "legal_actions": [["Single"]]},
            timings={"agent_step": 1.0},
        )


class FailingAdvisor:
    def recommend(self, _state, *, request_id=""):
        raise RuntimeError(f"model failed: {request_id}")


class AbsentCardAdvisor:
    def recommend(self, state, *, request_id=""):
        return LocalAdvice(
            strategy="invalid-test",
            cards=("AS",),
            play_type="Single",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
            engine_input={"request_id": request_id},
            timings={},
        )


class SuitAwareAdvisor(FakeAdvisor):
    def recommend(self, state, *, request_id=""):
        self.calls += 1
        self.called.set()
        observed = state.play_history[-1].cards
        cards = ("2S",) if observed == ("8S",) else ("2H",)
        return LocalAdvice(
            strategy="suit-aware",
            cards=cards,
            play_type="Single",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
            engine_input={"request_id": request_id},
            timings={},
        )


class ExactSuitFableDanAdvisor(FakeAdvisor):
    strategy_id = "fabledan"
    display_name = "FableDan"
    requires_exact_history_suits = True

    def __init__(self):
        super().__init__()
        self.observed_histories = []

    def decision_input_fingerprint(self, _state, *, request_id=""):
        del request_id
        return ("same-fabledan-input",)

    def recommend(self, state, *, request_id=""):
        self.calls += 1
        self.called.set()
        history = tuple(tuple(event.cards) for event in state.play_history)
        assert all(not card.endswith("?") for cards in history for card in cards)
        self.observed_histories.append(history)
        return LocalAdvice(
            strategy="fabledan-numpy",
            cards=("2D",),
            play_type="Single",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
            engine_input={"request_id": request_id},
            timings={},
        )


class SemanticBranchAdvisor(FakeAdvisor):
    strategy_id = "fabledan"
    display_name = "FableDan"

    def __init__(self, *, conflicting: bool) -> None:
        super().__init__()
        self.conflicting = conflicting
        self.observed_choices: list[dict[str, object]] = []

    def recommend(self, state, *, request_id=""):
        self.calls += 1
        self.called.set()
        metadata = state.play_history[-1].action_metadata or {}
        selected = metadata.get("selected_interpretation")
        assert isinstance(selected, dict)
        self.observed_choices.append(dict(selected))
        cards = (
            ("2H",)
            if self.conflicting and str(selected.get("key")) == "2"
            else ("2S",)
        )
        return LocalAdvice(
            strategy="fabledan-numpy",
            cards=cards,
            play_type="SINGLE",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
            engine_input={"request_id": request_id},
            timings={},
        )


class HistoryConstrainedSemanticBranchAdvisor(SemanticBranchAdvisor):
    def __init__(self, *, reject_all: bool = False) -> None:
        super().__init__(conflicting=False)
        self.reject_all = reject_all

    def decision_input_fingerprint(self, state, *, request_id=""):
        del request_id
        metadata = state.play_history[-1].action_metadata or {}
        selected = metadata.get("selected_interpretation")
        assert isinstance(selected, dict)
        if str(selected.get("key")) == "2":
            raise FableDanStateError(
                "第 38 条历史动作 222JJ 不能压过右家的 55533",
                diagnostic={
                    "code": "history_move_does_not_beat_lead",
                    "source_turn_id": 38,
                    "physical_cards": ["2D", "2S", "6H", "JC", "JH"],
                },
            )
        if self.reject_all:
            raise FableDanStateError(
                "第 43 条历史不是合法牌型（实体牌 2H 3S）",
                diagnostic={
                    "code": "observed_move_invalid",
                    "source_turn_id": 43,
                    "physical_cards": ["2H", "3S"],
                },
            )
        return ("valid-j-branch",)


def _ambiguous_full_house_metadata() -> dict[str, object]:
    return {
        "interpretation_ambiguous": True,
        "candidate_interpretations": [
            {
                "move_type": "ThreeWithTwo",
                "key": "J",
                "logical_label": "JJJ22",
                "wildcard_assignments": [
                    {"physical_card": "9H", "as_rank": "J"}
                ],
            },
            {
                "move_type": "ThreeWithTwo",
                "key": "2",
                "logical_label": "222JJ",
                "wildcard_assignments": [
                    {"physical_card": "9H", "as_rank": "2"}
                ],
            },
        ],
        "selected_interpretation": None,
        "selection_source": "unresolved",
    }


def _build(
    tmp_path,
    advisor,
    *,
    on_update=None,
    hand=HAND,
    round_level="2",
):
    store = LiveSessionStore(tmp_path / "profiles", "tencent_daguandan", session_id="advice")
    store.start(
        {
            "application_version": "test",
            "configuration_hash": "config",
            "template_manifest_hash": "templates",
            "target_fps": 10,
            "codec": "MJPG",
        }
    )
    orchestrator = LiveOrchestrator(
        reducer=LiveReducer("advice"),
        store=store,
        recorder=SessionRecorder(store.directory, size=(64, 32), fps=10),
        recognition_service=LeftPlayRecognition(),
        advisor=advisor,
        settle_ms=100,
        burst_sample_interval_ms=50,
        minimum_free_bytes=0,
        on_update=on_update,
    )
    orchestrator.start(
        round_level=round_level,
        hand=hand,
        lead_player="left",
        monotonic_ms=0,
    )
    return orchestrator


def _commit_left_action(orchestrator):
    frame = np.zeros((32, 64, 3), np.uint8)
    for index in range(5):
        timestamp = 100 + index * 100
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"t{index}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.2 if index == 0 else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )
        if orchestrator.snapshot.current_player != "left":
            break


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_advice_starts_when_reducer_predicts_self_before_timer(tmp_path):
    advisor = FakeAdvisor()
    orchestrator = _build(tmp_path, advisor)

    _commit_left_action(orchestrator)
    assert advisor.called.wait(2)
    _wait_until(lambda: orchestrator.latest_advice is not None and orchestrator.latest_advice.status == "ready")

    assert advisor.calls == 1
    assert orchestrator.snapshot.current_player == "self"
    assert orchestrator.latest_advice.visible is False

    orchestrator.ingest_fast_signal(active_player="self")

    assert orchestrator.latest_advice.visible is True
    orchestrator.finish()


def test_ready_advice_notifies_ui_without_waiting_for_another_capture_frame(tmp_path):
    updates = []
    orchestrator = _build(tmp_path, FakeAdvisor(), on_update=updates.append)

    _commit_left_action(orchestrator)
    _wait_until(
        lambda: any(
            update.advice is not None and update.advice.status == "ready"
            for update in updates
        )
    )

    latest = updates[-1]
    assert latest.advice is not None
    assert latest.advice.status == "ready"
    assert latest.advice.visible is False
    orchestrator.finish()


def test_trusted_action_uses_live_turn_transition_and_waits_for_advice(tmp_path):
    advisor = FakeAdvisor()
    orchestrator = _build(tmp_path, advisor)

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("7S", "7H"),
        is_pass=False,
        monotonic_ms=100,
        evidence_refs=("TRUTH-000001",),
    )

    assert update.event is not None
    assert update.event.event_type == "player_played"
    assert update.event.source == "trusted_log_replay"
    assert update.event.evidence_refs == ("TRUTH-000001",)
    assert orchestrator.snapshot.current_player == "self"
    orchestrator.finish()


def test_self_decision_correlates_pre_state_advice_and_actual_action(tmp_path):
    advisor = FakeAdvisor()
    orchestrator = _build(tmp_path, advisor)
    _commit_left_action(orchestrator)
    assert advisor.called.wait(2)
    _wait_until(lambda: orchestrator.latest_advice is not None and orchestrator.latest_advice.status == "ready")

    update = orchestrator.commit_trusted_action(
        actor="self",
        cards=("2S", "2H"),
        is_pass=False,
        monotonic_ms=900,
    )
    records = read_json_lines(orchestrator.store.decisions_path)

    assert len(records) == 1
    assert records[0]["decision_id"].startswith("advice:turn_")
    assert records[0]["state_before"]["current_player"] == "self"
    assert records[0]["legal_actions"] == [["Single"]]
    assert records[0]["model_advice"]["cards"] == ["2S"]
    assert records[0]["actual_action"] == {"cards": ["2H", "2S"], "is_pass": False}
    assert records[0]["actual_action_event_id"] == update.event.event_id
    orchestrator.finish()
    assert update.advice is not None
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)
    assert advice is not None
    assert advice.status == "ready"
    assert advice.advice is not None
    assert advisor.calls == 1
    orchestrator.finish()


def test_advice_expands_occluded_suit_only_in_temporary_variants(tmp_path):
    advisor = SuitAwareAdvisor()
    orchestrator = _build(tmp_path, advisor)

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("8?",),
        suit_options=(("S", "C"),),
        is_pass=False,
        monotonic_ms=100,
    )

    assert update.event is not None
    assert update.event.payload["cards"] == ["8?"]
    assert update.event.payload["suit_options"] == [["S", "C"]]
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "ready"
    assert advice.suit_uncertain is True
    assert advice.variant_count == 2
    assert advice.advice_agrees_across_variants is False
    assert advisor.calls == 2
    assert orchestrator.snapshot.play_history[-1].cards == ("8?",)
    orchestrator.finish()


def test_advice_expands_an_occluded_initial_hand_before_model_inference(tmp_path):
    advisor = FakeAdvisor()
    hand = tuple(
        card for card in HAND if card not in {"3S", "3H", "8S"}
    ) + ("8?", "8H", "8C")
    orchestrator = _build(tmp_path, advisor, hand=hand)

    _commit_left_action(orchestrator)
    _wait_until(lambda: orchestrator.latest_advice is not None)
    advice = orchestrator.wait_for_advice(
        orchestrator.latest_advice.key,
        timeout=2.0,
    )

    assert advice is not None
    assert advice.status == "ready"
    assert advice.suit_uncertain is True
    assert advice.suit_variant_count == 2
    assert advisor.calls == 2
    assert orchestrator.snapshot.my_hand.count("8?") == 1
    orchestrator.finish()


def test_fabledan_evaluates_occluded_full_house_with_exact_temporary_suits(
    tmp_path,
):
    advisor = ExactSuitFableDanAdvisor()
    orchestrator = _build(
        tmp_path,
        advisor,
        hand=OCCLUDED_FULL_HOUSE_HAND,
        round_level="9",
    )

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("3C", "3D", "3H", "5?", "5D"),
        suit_options=(("C",), ("D",), ("H",), ("S", "C"), ("D",)),
        is_pass=False,
        monotonic_ms=100,
    )
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "ready"
    assert advice.suit_uncertain is True
    assert advice.variant_count == 1
    assert advice.advice_agrees_across_variants is True
    assert advisor.calls == 1
    assert {history[-1][3] for history in advisor.observed_histories} == {"5S"}
    assert orchestrator.snapshot.play_history[-1].cards == (
        "3C",
        "3D",
        "3H",
        "5?",
        "5D",
    )
    requested = next(
        event for event in orchestrator.events if event.event_type == "advice_requested"
    )
    assert requested.payload["advisor_strategy"] == "fabledan"
    assert requested.payload["advisor_name"] == "FableDan"
    orchestrator.finish()


def test_fabledan_deduplicates_suit_states_with_identical_model_semantics(tmp_path):
    advisor = ExactSuitFableDanAdvisor()
    orchestrator = _build(tmp_path, advisor)

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("8?",),
        suit_options=(("S", "C"),),
        is_pass=False,
        monotonic_ms=100,
    )
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "ready"
    assert advice.suit_uncertain is True
    assert advice.suit_variant_count == 2
    assert advice.suit_equivalence_class_count == 1
    assert advice.variant_count == 1
    assert advice.advice_agrees_across_variants is True
    assert advisor.calls == 1
    ready = next(
        item
        for item in read_json_lines(orchestrator.store.advice_path)
        if item.get("status") == "ready"
    )
    assert ready["suit_variant_count"] == 2
    assert ready["suit_equivalence_class_count"] == 1
    orchestrator.finish()


def test_wildcard_history_uses_one_strongest_model_interpretation(tmp_path):
    advisor = SemanticBranchAdvisor(conflicting=False)
    orchestrator = _build(tmp_path, advisor, round_level="9")

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("2D", "2H", "9H", "JC", "JD"),
        is_pass=False,
        monotonic_ms=100,
        action_metadata=_ambiguous_full_house_metadata(),
    )
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "ready"
    assert advice.semantic_uncertain is False
    assert advice.semantic_source_history_indices == ()
    assert advice.semantic_variant_count == 1
    assert advice.variant_count == 1
    assert advice.advice_agrees_across_variants is True
    assert advisor.calls == 1
    assert [str(item["key"]) for item in advisor.observed_choices] == ["J"]
    assert (
        orchestrator.snapshot.play_history[-1].action_metadata[
            "selected_interpretation"
        ]
        is None
    )
    ready = next(
        item
        for item in read_json_lines(orchestrator.store.advice_path)
        if item.get("status") == "ready"
    )
    assert ready["semantic_uncertain"] is False
    assert ready["semantic_source_history_indices"] == []
    assert ready["uncertainty_diagnostics"]["input_variant_count_before_validation"] == 1
    orchestrator.finish()


def test_complete_history_eliminates_impossible_semantic_branch(tmp_path):
    advisor = HistoryConstrainedSemanticBranchAdvisor()
    orchestrator = _build(tmp_path, advisor, round_level="9")

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("2D", "2H", "9H", "JC", "JD"),
        is_pass=False,
        monotonic_ms=100,
        action_metadata=_ambiguous_full_house_metadata(),
    )
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "ready"
    assert advisor.calls == 1
    assert [str(item["key"]) for item in advisor.observed_choices] == ["J"]
    ready = next(
        item
        for item in read_json_lines(orchestrator.store.advice_path)
        if item.get("status") == "ready"
    )
    diagnostics = ready["uncertainty_diagnostics"]
    assert diagnostics["input_variant_count_before_validation"] == 1
    assert diagnostics["valid_encoded_input_count"] == 1
    assert diagnostics["eliminated_variant_count"] == 0
    assert diagnostics["eliminated_variants"] == []
    orchestrator.finish()


def test_all_invalid_semantic_branches_report_deepest_source_error(tmp_path):
    advisor = HistoryConstrainedSemanticBranchAdvisor(reject_all=True)
    orchestrator = _build(tmp_path, advisor, round_level="9")

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("2D", "2H", "9H", "JC", "JD"),
        is_pass=False,
        monotonic_ms=100,
        action_metadata=_ambiguous_full_house_metadata(),
    )
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "failed"
    assert advisor.calls == 0
    assert "1 个候选状态全部被完整牌局历史排除" in advice.error
    assert "最深可达根因：第 43 条历史不是合法牌型" in advice.error
    assert "模型未被调用" in advice.error
    orchestrator.finish()


def test_wildcard_history_does_not_compare_weaker_interpretations(
    tmp_path,
):
    advisor = SemanticBranchAdvisor(conflicting=True)
    orchestrator = _build(tmp_path, advisor, round_level="9")

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("2D", "2H", "9H", "JC", "JD"),
        is_pass=False,
        monotonic_ms=100,
        action_metadata=_ambiguous_full_house_metadata(),
    )
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "ready"
    assert advice.semantic_uncertain is False
    assert advice.semantic_source_history_indices == ()
    assert advice.variant_count == 1
    assert advice.advice_agrees_across_variants is True
    assert advisor.calls == 1
    assert [str(item["key"]) for item in advisor.observed_choices] == ["J"]
    ready = next(
        item
        for item in read_json_lines(orchestrator.store.advice_path)
        if item.get("status") == "ready"
    )
    diagnostics = ready["uncertainty_diagnostics"]
    assert diagnostics["semantic_source_history_indices"] == []
    assert diagnostics["input_variant_count_before_validation"] == 1
    assert (
        orchestrator.snapshot.play_history[-1].action_metadata[
            "selected_interpretation"
        ]
        is None
    )
    orchestrator.finish()


def test_corrected_state_marks_inflight_advice_stale(tmp_path):
    gate = threading.Event()
    advisor = FakeAdvisor(gate)
    orchestrator = _build(tmp_path, advisor)
    _commit_left_action(orchestrator)
    assert advisor.called.wait(2)
    request = orchestrator.latest_advice.key

    orchestrator.correct_latest(cards=("8S",), is_pass=False)
    gate.set()

    _wait_until(
        lambda: any(
            event.event_type == "advice_stale"
            and event.payload.get("request_id") == request.request_id
            for event in orchestrator.events
        )
    )
    assert advisor.calls >= 1
    orchestrator.finish()


def test_advisor_failure_incident_contains_reproducible_engine_input(tmp_path):
    orchestrator = _build(tmp_path, FailingAdvisor())
    _commit_left_action(orchestrator)
    _wait_until(
        lambda: orchestrator.latest_advice is not None
        and orchestrator.latest_advice.status == "failed"
    )

    incident = next(orchestrator.store.incidents_directory.iterdir())
    engine_input = incident / "engine_input.json"

    assert engine_input.is_file()
    assert "project_snapshot" in engine_input.read_text("utf-8")
    orchestrator.finish()


def test_advice_containing_a_card_absent_from_current_hand_is_never_exposed(tmp_path):
    orchestrator = _build(tmp_path, AbsentCardAdvisor())
    _commit_left_action(orchestrator)
    _wait_until(
        lambda: orchestrator.latest_advice is not None
        and orchestrator.latest_advice.status == "failed"
    )

    assert orchestrator.latest_advice is not None
    assert orchestrator.latest_advice.advice is None
    assert "AS" in orchestrator.latest_advice.error
    assert any(
        event.event_type == "advice_failed"
        and event.payload.get("reason") == "cards_not_in_current_hand"
        for event in orchestrator.events
    )
    orchestrator.finish()
