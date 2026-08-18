from __future__ import annotations

import json

import pytest

from daguandan_bridge.application.fabledan_training_data import (
    FableDanTrainingDataService,
    _infer_outcome,
)


def _write_jsonl(path, records) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _session(tmp_path, *, double_down: bool = True, incomplete: bool = False):
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
    if incomplete:
        events = events[:2]
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


def test_sealed_session_with_incomplete_outcome_writes_unconfirmable_draft(tmp_path):
    session = _session(tmp_path, incomplete=True)
    service = FableDanTrainingDataService()

    review = service.initialize_review(session)

    assert review.status == "draft"
    assert review.payload["outcome"]["status"] == "incomplete"
    assert not review.can_confirm
    assert review.payload["eligibility"]["can_confirm"] is False
    assert review.payload["candidates"]["eligible_decision_count"] == 0
    assert (session / "fabledan_training_review.json").is_file()


def _finished(actor, placement):
    return {
        "event_type": "player_finished",
        "actor": actor,
        "payload": {"placement": placement},
    }


def _terminal(counts):
    return {
        "event_type": "game_end_detected",
        "payload": {"remaining_cards": dict(counts)},
    }


def test_head_second_without_terminal_counts_keeps_legacy_double_down_outcome():
    outcome = _infer_outcome([_finished("right", "head"), _finished("left", "second")])

    assert outcome["status"] == "complete"
    assert outcome["terminal_kind"] == "double_down"
    assert outcome["finish_order"] == ["right", "left"]
    assert outcome["raw_team_reward"] == -3
    assert "placement_inference" not in outcome


def test_terminal_counts_infer_the_smaller_remaining_hand_as_third():
    outcome = _infer_outcome(
        [
            _finished("right", "head"),
            _finished("left", "second"),
            _terminal({"self": 1, "right": 0, "opposite": 10, "left": 0}),
        ]
    )

    assert outcome["finish_order"] == ["right", "left", "self", "opposite"]
    assert outcome["placement_inference"] == {
        "source": "game_end_detected.remaining_cards",
        "counts": {"self": 1, "right": 0, "opposite": 10, "left": 0},
        "inferred": {"third": "self", "last": "opposite"},
        "tie_breaker": None,
    }


def test_equal_terminal_counts_use_stable_turn_order_for_third_and_last():
    outcome = _infer_outcome(
        [
            _finished("right", "head"),
            _finished("left", "second"),
            _terminal({"self": 6, "right": 0, "opposite": 6, "left": 0}),
        ]
    )

    assert outcome["finish_order"] == ["right", "left", "self", "opposite"]
    assert outcome["placement_inference"]["tie_breaker"] == "TURN_ORDER:self,right,opposite,left"


def test_terminal_counts_infer_the_larger_remaining_hand_as_last():
    outcome = _infer_outcome(
        [
            _finished("right", "head"),
            _finished("left", "second"),
            _terminal({"self": 6, "right": 0, "opposite": 2, "left": 0}),
        ]
    )

    assert outcome["finish_order"] == ["right", "left", "opposite", "self"]
    assert outcome["placement_inference"]["inferred"] == {
        "third": "opposite",
        "last": "self",
    }


def test_explicit_ranking_is_not_replaced_by_conflicting_terminal_counts():
    outcome = _infer_outcome(
        [
            _finished("right", "head"),
            _finished("left", "second"),
            _finished("opposite", "third"),
            _finished("self", "last"),
            _terminal({"self": 1, "right": 0, "opposite": 10, "left": 0}),
        ]
    )

    assert outcome["terminal_kind"] == "full_ranking"
    assert outcome["finish_order"] == ["right", "left", "opposite", "self"]
    assert "placement_inference" not in outcome


def test_invalid_terminal_counts_leave_a_double_down_outcome_incomplete():
    outcome = _infer_outcome(
        [
            _finished("right", "head"),
            _finished("left", "second"),
            _terminal({"self": 1, "right": 0, "opposite": "10", "left": 0}),
        ]
    )

    assert outcome["status"] == "incomplete"
    assert "快照无效" in outcome["reason"]
