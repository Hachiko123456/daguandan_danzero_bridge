from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from daguandan_bridge.advisor_strategy import build_evaluation_advisor
from daguandan_bridge.application.model_evaluation import ModelEvaluationService
from daguandan_bridge.danzero import DanzeroAdvisor
from daguandan_bridge.domain.advice import AdviceResult
from daguandan_bridge.fabledan import FableDanAdvisor
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    save_truth_log,
)


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _truth(session_id: str, *, future_card: str = "3S") -> TruthLog:
    return TruthLog(
        source_session_id=session_id,
        initial_state=TruthInitialState("8", "self", HAND),
        turns=(
            TruthTurn(1, "self", False, ("2S",), trick_id=1),
            TruthTurn(2, "right", False, (future_card,), trick_id=1),
            TruthTurn(3, "opposite", True, (), trick_id=1),
            TruthTurn(4, "left", True, (), trick_id=1),
            TruthTurn(5, "self", True, (), trick_id=1),
        ),
    )


def _session(tmp_path: Path, name: str, truth: TruthLog | None = None) -> Path:
    session = tmp_path / "profiles" / "profile" / "sessions" / name
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"session_id": name, "status": "sealed"}), encoding="utf-8"
    )
    if truth is not None:
        save_truth_log(session / "truth_log.json", truth)
    return session


class FakeAdvisor:
    def __init__(
        self,
        strategy_id: str,
        *,
        fail_call: int | None = None,
        drift: bool = False,
    ) -> None:
        self.strategy_id = strategy_id
        self.fail_call = fail_call
        self.drift = drift
        self.calls = []

    @property
    def binding(self) -> tuple[str, str]:
        return {
            "danzero_model": ("danzero", "danzero"),
            "fabledan_model": ("fabledan-numpy", "numpy"),
            "fabledan_rule": ("fabledan-rule", "rule"),
        }[self.strategy_id]

    def initialize(self) -> None:
        pass

    def audit_info(self):
        strategy, backend = self.binding
        del strategy
        return {
            "backend": backend,
            "status": "loaded",
            "path": "fake.ckpt",
            "digest": "f" * 64,
            "schema": "fake/v1",
        }

    def recommend(self, state, *, request_id="", trace=None):
        del trace
        self.calls.append(state.local_snapshot())
        call = len(self.calls)
        if self.fail_call == call:
            raise RuntimeError("deterministic failure")
        strategy, backend = self.binding
        if self.drift:
            backend = "rule" if backend != "rule" else "numpy"
        is_pass = call > 1
        return AdviceResult(
            strategy=strategy,
            cards=() if is_pass else ("2S",),
            play_type="PASS" if is_pass else "SINGLE",
            is_pass=is_pass,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
            engine_input={"backend": backend},
        )


@pytest.mark.parametrize(
    ("strategy_id", "backend"),
    (
        ("danzero_model", "danzero"),
        ("fabledan_model", "numpy"),
        ("fabledan_rule", "rule"),
    ),
)
def test_teacher_forced_run_binds_each_strategy_and_publishes_metrics(
    tmp_path, strategy_id, backend
):
    truth = _truth("game")
    session = _session(tmp_path, "game", truth)
    advisor = FakeAdvisor(strategy_id)
    service = ModelEvaluationService(advisor_factory=lambda _strategy: advisor)

    prepared = service.prepare(session, strategy_id=strategy_id)
    result = service.run(prepared)

    assert prepared.ready
    assert result.status == "completed"
    assert "evaluation_mode" not in result.summary
    assert "formal_eligible" not in result.summary
    assert result.summary["actual_backend"] == backend
    assert result.summary["evaluated_decisions"] == 2
    assert result.summary["metrics"]["coverage"] == {
        "numerator": 2,
        "denominator": 2,
        "rate": 1.0,
    }
    assert result.summary["metrics"]["latency_ms"]["count"] == 2
    assert [item["turn_id"] for item in result.decisions] == [1, 5]
    assert set(path.name for path in result.report_directory.iterdir()) == {
        "summary.json",
        "decisions.jsonl",
        "report.md",
    }
    assert "teacher-forced" in (result.report_directory / "report.md").read_text("utf-8")
    assert "不是闭环胜率" in (result.report_directory / "report.md").read_text("utf-8")


def test_prepare_accepts_saved_legacy_session_truth_without_source_id(tmp_path):
    session = _session(tmp_path, "game")
    raw = _truth("game").to_dict()
    raw.pop("schema")
    raw.pop("source_session_id")
    raw["schema_version"] = 1
    (session / "truth_log.json").write_text(
        json.dumps(raw, ensure_ascii=False), encoding="utf-8"
    )
    service = ModelEvaluationService(
        advisor_factory=lambda strategy: FakeAdvisor(strategy)
    )

    prepared = service.prepare(session, strategy_id="danzero_model")

    assert prepared.ready
    assert prepared.frozen_truth.source_session_id == "game"


def test_future_changes_do_not_change_first_model_input_or_prediction(tmp_path):
    session_a = _session(tmp_path, "a", _truth("a", future_card="3S"))
    session_b = _session(tmp_path, "b", _truth("b", future_card="4S"))
    advisor_a = FakeAdvisor("danzero_model")
    advisor_b = FakeAdvisor("danzero_model")
    result_a = ModelEvaluationService(advisor_factory=lambda _strategy: advisor_a).run(
        ModelEvaluationService(advisor_factory=lambda _strategy: advisor_a).prepare(
            session_a, strategy_id="danzero_model"
        )
    )
    service_b = ModelEvaluationService(advisor_factory=lambda _strategy: advisor_b)
    result_b = service_b.run(service_b.prepare(session_b, strategy_id="danzero_model"))

    first_a, first_b = result_a.decisions[0], result_b.decisions[0]
    assert first_a["state_before_sha256"] == first_b["state_before_sha256"]
    assert first_a["predicted_action"] == first_b["predicted_action"]


def test_truth_actions_not_predictions_advance_the_next_decision(tmp_path):
    session = _session(tmp_path, "game", _truth("game"))
    advisor = FakeAdvisor("danzero_model")
    service = ModelEvaluationService(advisor_factory=lambda _strategy: advisor)

    result = service.run(service.prepare(session, strategy_id="danzero_model"))

    assert result.status == "completed"
    second_state = advisor.calls[1]
    assert [event.player for event in second_state.play_history] == [
        "self",
        "right",
        "opposite",
        "left",
    ]
    assert second_state.my_hand.count("2S") == 0


def test_missing_fabledan_npz_is_blocked_before_any_decision(tmp_path):
    session = _session(tmp_path, "game", _truth("game"))
    service = ModelEvaluationService(
        profiles_root=tmp_path / "profiles", profile_name="profile"
    )

    prepared = service.prepare(session, strategy_id="fabledan_model")
    result = service.run(prepared)

    assert not prepared.ready
    assert any(
        issue.code == "STRATEGY_FABLEDAN_WEIGHTS_MISSING"
        for issue in prepared.issues
    )
    assert result.status == "blocked"
    assert result.summary["evaluated_decisions"] == 0
    assert result.decisions == ()

    weights = tmp_path / "profiles" / "profile" / "models" / "fabledan_weights.npz"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not-a-valid-npz")
    invalid = service.prepare(session, strategy_id="fabledan_model")
    invalid_result = service.run(invalid)
    assert any(
        issue.code == "STRATEGY_FABLEDAN_WEIGHTS_INVALID"
        for issue in invalid.issues
    )
    assert invalid_result.status == "blocked"
    assert invalid_result.summary["evaluated_decisions"] == 0


def test_statuses_cancel_error_and_backend_drift_are_auditable(tmp_path):
    session = _session(tmp_path, "game", _truth("game"))

    failing = FakeAdvisor("danzero_model", fail_call=2)
    service = ModelEvaluationService(advisor_factory=lambda _strategy: failing)
    partial = service.run(service.prepare(session, strategy_id="danzero_model"))
    assert partial.status == "completed_with_errors"
    assert "formal_eligible" not in partial.summary

    cancel_advisor = FakeAdvisor("danzero_model")
    cancel_service = ModelEvaluationService(advisor_factory=lambda _strategy: cancel_advisor)
    cancel_after_first = {"value": False}

    def progress(update):
        if update.phase == "running" and update.completed == 1:
            cancel_after_first["value"] = True

    cancelled = cancel_service.run(
        cancel_service.prepare(session, strategy_id="danzero_model"),
        progress=progress,
        is_cancelled=lambda: cancel_after_first["value"],
    )
    assert cancelled.status == "cancelled"
    assert len(cancelled.decisions) == 1

    drifting = FakeAdvisor("danzero_model", drift=True)
    drift_service = ModelEvaluationService(advisor_factory=lambda _strategy: drifting)
    failed = drift_service.run(
        drift_service.prepare(session, strategy_id="danzero_model")
    )
    assert failed.status == "failed"
    assert failed.decisions[0]["error_code"] == "BACKEND_IDENTITY_DRIFT"


def test_invalid_turn_order_blocks_during_preflight(tmp_path):
    truth = _truth("game")
    bad = replace(
        truth,
        turns=(TruthTurn(1, "right", False, ("3S",), trick_id=1),),
    )
    session = _session(tmp_path, "game", bad)
    advisor = FakeAdvisor("danzero_model")
    prepared = ModelEvaluationService(
        advisor_factory=lambda _strategy: advisor
    ).prepare(session, strategy_id="danzero_model")

    assert not prepared.ready
    assert prepared.eligible_self_decisions == 0
    assert any(issue.code == "INPUT_TURN_ORDER" for issue in prepared.issues)


def test_saved_unverified_log_does_not_require_formal_metadata_or_evidence(tmp_path):
    session = _session(tmp_path, "game", _truth("game"))
    advisor = FakeAdvisor("danzero_model")
    service = ModelEvaluationService(advisor_factory=lambda _strategy: advisor)

    prepared = service.prepare(session, strategy_id="danzero_model")
    result = service.run(prepared)

    assert prepared.ready
    assert result.status == "completed"
    assert not any(issue.code.startswith("FORMAL_") for issue in prepared.issues)
    assert prepared.frozen_truth.turns[1].cards == ("3S",)


def test_evaluation_contract_only_accepts_saved_truth_log(tmp_path):
    session = _session(tmp_path, "game", _truth("game"))
    service = ModelEvaluationService(
        advisor_factory=lambda strategy: FakeAdvisor(strategy)
    )

    with pytest.raises(TypeError):
        service.prepare(
            session,
            strategy_id="danzero_model",
            truth_log=_truth("game"),
        )
    with pytest.raises(TypeError):
        service.validate_input(session, mode="draft")


def test_two_representative_real_sessions_have_stable_preflight_outcomes():
    sessions = (
        Path(__file__).parents[1]
        / "data"
        / "profiles"
        / "tencent_daguandan"
        / "sessions"
    )
    first = sessions / "game_20260809_100241_aed6c1"
    second = sessions / "game_20260811_171243_d0ee3d"
    if not first.is_dir() or not second.is_dir():
        pytest.skip("representative local sessions are unavailable")
    service = ModelEvaluationService(
        advisor_factory=lambda strategy: FakeAdvisor(strategy)
    )

    valid = service.prepare(first, strategy_id="danzero_model")
    repaired_input = service.prepare(second, strategy_id="danzero_model")

    assert valid.ready
    assert valid.eligible_self_decisions == 14
    assert repaired_input.ready
    assert repaired_input.eligible_self_decisions == 17


def test_strategy_independent_validation_does_not_initialize_model(tmp_path):
    session = _session(tmp_path, "game", _truth("game"))
    advisor = FakeAdvisor("danzero_model")
    initialized = {"count": 0}
    original_initialize = advisor.initialize

    def initialize():
        initialized["count"] += 1
        original_initialize()

    advisor.initialize = initialize
    service = ModelEvaluationService(advisor_factory=lambda _strategy: advisor)

    validated = service.validate_input(session)

    assert validated.ready
    assert initialized["count"] == 0


def test_fabledan_strict_runtime_policies_never_implicitly_mix_backends(
    tmp_path, monkeypatch
):
    from daguandan_bridge.fabledan import advisor as module

    monkeypatch.setattr(
        module.RuleAgent,
        "act",
        lambda *_args: pytest.fail("model_required must not call RuleAgent"),
    )
    required = FableDanAdvisor(tmp_path, "profile", runtime_policy="model_required")
    with pytest.raises(RuntimeError, match="model_required"):
        required.initialize()
    assert required.audit_info()["backend"] == "numpy"
    assert required.audit_info()["status"] == "missing"

    weights = tmp_path / "profile" / "models" / "fabledan_weights.npz"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"broken")
    invalid = FableDanAdvisor(tmp_path, "profile", runtime_policy="model_required")
    with pytest.raises(RuntimeError, match="model_required"):
        invalid.initialize()
    assert invalid.audit_info()["status"] == "invalid"

    monkeypatch.setattr(module.RuleAgent, "act", lambda _self, observation: 0)
    rule = FableDanAdvisor(tmp_path, "profile", runtime_policy="rule_only")
    assert rule.audit_info()["backend"] == "rule"
    assert rule.audit_info()["status"] == "rule_only"


def test_evaluation_factory_is_independent_from_profile_default(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "profile.json").write_text(
        json.dumps({"advisor_strategy": "danzero"}), encoding="utf-8"
    )

    advisor = build_evaluation_advisor(
        "fabledan_rule", profiles_root=tmp_path, profile_name="profile"
    )

    assert isinstance(advisor, FableDanAdvisor)
    assert advisor.runtime_policy == "rule_only"
    assert json.loads((profile / "profile.json").read_text("utf-8"))[
        "advisor_strategy"
    ] == "danzero"


def test_danzero_audit_info_identifies_checkpoint_without_q_probe():
    audit = DanzeroAdvisor().audit_info()

    assert audit["backend"] == "danzero"
    assert audit["schema"] == "danzero-adapter/v1"
    assert len(str(audit["digest"])) == 64
    assert str(audit["path"]).endswith("q_network.ckpt")
