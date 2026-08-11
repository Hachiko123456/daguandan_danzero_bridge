from __future__ import annotations

import hashlib
import json

import pytest

from daguandan_bridge.application.dataset import DatasetUseCase, session_split
from daguandan_bridge.dataset_cli import main
from daguandan_bridge.domain.truth import LabelProvenance, TruthEvidence, TruthOutcome
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn, save_truth_log


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")


def _verified_session(tmp_path):
    session = tmp_path / "game_verified"
    (session / "video").mkdir(parents=True)
    (session / "video" / "game.avi").write_bytes(b"deterministic-video")
    provenance = LabelProvenance(source="human_review", annotator="tester", annotated_at="2026-08-11T12:00:00+08:00", confidence=1.0)
    truth = TruthLog(
        session.name,
        TruthInitialState("2", "self", HAND),
        (
            TruthTurn(
                1,
                "self",
                False,
                ("5H",),
                frame_index=12,
                monotonic_ms=100,
                label_status="verified",
                provenance=provenance,
                evidence=TruthEvidence((12,), 100, "my_play"),
            ),
            TruthTurn(2, "right", True, (), frame_index=13, label_status="draft"),
        ),
        label_status="verified",
        provenance=provenance,
        outcome=TruthOutcome(
            complete=True,
            finish_order=("self", "right", "opposite", "left"),
            team_result="win",
            reward=1.0,
            reward_scheme="team_win_v1",
        ),
    )
    save_truth_log(session / "truth_log.json", truth)
    decision = {
        "schema": "guandan.live-decision/1",
        "decision_id": f"{session.name}:turn_1:revision_1",
        "session_id": session.name,
        "actor": "self",
        "turn_id": 1,
        "state_before": {"my_hand": list(HAND), "revision": 1},
        "legal_actions": [["PASS", "PASS", -1], ["Single", 5, ["H5"]]],
        "feature_schema": "danzero-567/v1",
        "features_567": [[0] * 567, [1] * 567],
        "model_advice": {"cards": [], "is_pass": True},
        "actual_turn_id": 1,
        "actual_action_event_id": "EVT-000001",
        "actual_action": {"cards": ["5H"], "is_pass": False},
        "label_status": "draft",
    }
    (session / "decisions.jsonl").write_text(json.dumps(decision) + "\n", encoding="utf-8")
    return session


def test_validate_and_export_verified_samples_are_deterministic_and_non_mutating(tmp_path):
    session = _verified_session(tmp_path)
    truth_before = hashlib.sha256((session / "truth_log.json").read_bytes()).hexdigest()
    decisions_before = hashlib.sha256((session / "decisions.jsonl").read_bytes()).hexdigest()
    use_case = DatasetUseCase()

    report = use_case.validate_session(session)
    first = use_case.export_session(session)
    bytes_before = {path.name: path.read_bytes() for path in (session / "derived").iterdir()}
    second = use_case.export_session(session)

    assert report.valid
    assert first.sample_counts == second.sample_counts == {"vision": 1, "policy": 1}
    assert bytes_before == {path.name: path.read_bytes() for path in (session / "derived").iterdir()}
    assert hashlib.sha256((session / "truth_log.json").read_bytes()).hexdigest() == truth_before
    assert hashlib.sha256((session / "decisions.jsonl").read_bytes()).hexdigest() == decisions_before
    vision = json.loads((session / "derived" / "vision_samples.jsonl").read_text("utf-8"))
    policy = json.loads((session / "derived" / "policy_samples.jsonl").read_text("utf-8"))
    assert vision["schema"] == "guandan.vision-sample/1"
    assert vision["split_group"] == session.name
    assert policy["schema"] == "guandan.policy-decision/1"
    assert policy["feature_schema"] == "danzero-567/v1"
    assert policy["chosen_action"] == {"cards": ["5H"], "is_pass": False}
    assert policy["model_advice"] == {"cards": [], "is_pass": True}
    assert policy["split"] == session_split(session.name)


def test_validator_is_read_only_and_reports_machine_codes(tmp_path):
    session = tmp_path / "broken"
    session.mkdir()
    report = DatasetUseCase().validate_session(session)
    assert not report.valid
    assert report.findings[0].code == "TRUTH_MISSING"
    assert not (session / "derived").exists()


def test_cli_returns_nonzero_for_validation_errors(tmp_path, capsys):
    session = tmp_path / "broken"
    session.mkdir()
    assert main(("validate", str(session))) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is False
    assert output["findings"][0]["code"] == "TRUTH_MISSING"


def test_complete_outcome_integrity_rejects_conflicting_winner():
    with pytest.raises(ValueError, match="conflicts"):
        TruthOutcome(
            complete=True,
            finish_order=("right", "self", "left", "opposite"),
            team_result="win",
            reward=1,
            reward_scheme="team_win_v1",
        )
