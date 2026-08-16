from __future__ import annotations

import json

import pytest

from daguandan_bridge.application.fabledan_training_data import (
    FableDanTrainingDataService,
)


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _session(tmp_path, *, double_down: bool = True):
    session = tmp_path / "game_training"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps({"status": "sealed"}), encoding="utf-8"
    )
    features = [[float(index)] * 80 for index in range(2)]
    decision = {
        "schema": "guandan.live-decision/1",
        "session_id": session.name,
        "decision_id": "D-1",
        "request_id": "ADV-1",
        "status": "ready",
        "actor": "self",
        "feature_schema": "fabledan-token48-feat80/v1",
        "fabledan_training_input": {
            "schema": "fabledan-decision-input/1",
            "feature_schema": "fabledan-token48-feat80/v1",
            "tokens": [1, 7, 20],
            "features": features,
            "legal_action_count": 2,
            "model_hash": "model-hash",
            "adapter_schema": "fabledan-adapter/v1",
            "upstream_commit": "test-commit",
            "standard_no_tribute": True,
        },
        "legal_actions": [
            {"is_pass": True, "cards": [], "type": "PASS"},
            {"is_pass": False, "cards": ["7S"], "type": "SINGLE"},
        ],
        "actual_action_event_id": "EVT-1",
        "actual_turn_id": 1,
        "actual_trick_id": 1,
        "actual_action": {"cards": ["7S"], "is_pass": False},
        "model_advice": {"strategy": "fabledan-numpy"},
    }
    _write_jsonl(session / "decisions.jsonl", [decision])
    _write_jsonl(session / "advice.jsonl", [])
    events = [
        {
            "event_id": "EVT-1",
            "event_type": "player_played",
            "actor": "self",
            "payload": {
                "cards": ["7S"],
                "selected_interpretation": {"move_type": "Single"},
            },
        },
        {
            "event_id": "END-1",
            "event_type": "player_finished",
            "actor": "self",
            "payload": {"placement": "first"},
        },
        {
            "event_id": "END-2",
            "event_type": "player_finished",
            "actor": "opposite" if double_down else "right",
            "payload": {"placement": "second"},
        },
    ]
    if not double_down:
        events.extend(
            [
                {"event_id": "END-3", "event_type": "player_finished", "actor": "opposite", "payload": {"placement": "third"}},
                {"event_id": "END-4", "event_type": "player_finished", "actor": "left", "payload": {"placement": "last"}},
            ]
        )
    _write_jsonl(session / "timeline.jsonl", events)
    return session


def test_confirmed_double_down_exports_immutable_fabledan_sample(tmp_path):
    session = _session(tmp_path)
    service = FableDanTrainingDataService()

    review = service.inspect_session(session)

    assert review.status == "draft"
    assert review.can_confirm
    assert review.payload["outcome"]["raw_team_reward"] == 3
    exported = service.confirm_and_export(session)

    assert exported.sample_count == 1
    sample = json.loads(exported.samples_path.read_text("utf-8").splitlines()[0])
    assert sample["target"] == {
        "raw_team_reward": 3,
        "normalized_dmc_return": 1.0,
        "terminal_kind": "double_down",
        "finish_order": ["self", "opposite"],
    }
    assert sample["decision"]["chosen_legal_index"] == 1
    assert len(sample["encoding"]["chosen_feature"]) == 80
    assert "q_values" not in sample
    assert "model_replacement" not in sample
    assert json.loads((session / "fabledan_training_review.json").read_text("utf-8"))["status"] == "verified"


def test_confirmation_rejects_source_that_changed_after_review(tmp_path):
    session = _session(tmp_path)
    service = FableDanTrainingDataService()
    service.inspect_session(session)
    with (session / "timeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_id": "AUX", "event_type": "note"}) + "\n")

    with pytest.raises(ValueError, match="源数据已变化"):
        service.confirm_and_export(session)


def test_full_ranking_with_non_partner_second_place_is_exportable(tmp_path):
    session = _session(tmp_path, double_down=False)
    service = FableDanTrainingDataService()

    review = service.inspect_session(session)

    assert review.can_confirm
    assert review.payload["outcome"]["terminal_kind"] == "full_ranking"
