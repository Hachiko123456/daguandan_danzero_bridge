from __future__ import annotations

from daguandan_bridge.application.diagnostic_presentation import (
    build_detailed_diagnostic,
    build_user_view,
)
from daguandan_bridge.domain.recognition import LeadEvidence, RecognizedEvent, RecognitionResult


def test_user_view_is_chinese_and_copyable():
    view = build_user_view({"opening_readiness_inputs": {"readiness": {"status": "FAIL", "primary_reason": "WINDOW_MINIMIZED", "message": "窗口已最小化", "suggested_action": "恢复窗口"}}})
    payload = view.to_dict()
    assert payload["状态"] == "需要处理"
    assert "窗口已最小化" in payload["可复制错误说明"]
    assert "恢复窗口" in payload["可复制诊断摘要"]
    assert payload["技术详情"]


def test_user_view_maps_hand_count_wait():
    view = build_user_view({"opening_readiness_inputs": {"readiness": {"status": "WAIT", "primary_reason": "MID_GAME_HAND_COUNT", "message": "当前只有21张", "suggested_action": "等待下一局"}}, "recognition": {"result": {"my_hand": ["AS"] * 21}}})
    assert view.status == "等待处理"
    assert "21" in view.summary_copy


def _rich_result() -> RecognitionResult:
    return RecognitionResult(
        round_level="7",
        wild_rank="7",
        current_player="self",
        lead_player=None,
        my_hand=("3S", "4H"),
        events=(
            RecognizedEvent("right", ("9S", "9H"), False, 0.88, "template:right_play"),
            RecognizedEvent("opposite", (), True, 0.91, "template:passed"),
            RecognizedEvent("left", (), False, 0.21, "template:left_play"),
        ),
        field_confidences={"round_level": 0.94, "my_hand": 0.56, "events": 0.88},
        sources={"round_level": "template:level", "my_hand": "template:hand", "events": "template:play/status"},
        unresolved_fields=("lead_player", "my_hand"),
        diagnostics=("my_hand candidate below threshold",),
        buttons=("pass",),
        lead_evidence=(
            LeadEvidence("self", 0.61, 0.58, 0.30, "conflict", "candidate_conflict"),
            LeadEvidence("right", 0.59, 0.57, 0.31, "conflict", "candidate_conflict"),
        ),
    )


def test_detailed_report_preserves_result_events_lead_evidence_and_trace_reasons():
    result = _rich_result()
    report = build_detailed_diagnostic(
        result=result,
        trace={
            "schema": "guandan.recognition-trace/1",
            "threshold_policy": "production-unchanged",
            "candidates": [
                {"field": "level_rank", "label": "7", "score": 0.94, "threshold": 0.75, "accepted": True, "rejection_reason": None, "source": "template:level"},
                {"field": "my_hand", "label": "3S", "score": 0.56, "threshold": 0.75, "accepted": False, "rejection_reason": "below_threshold", "source": "template:hand"},
                {"field": "lead_player", "label": "right", "score": 0.59, "threshold": 0.75, "accepted": False, "rejection_reason": "candidate_conflict", "source": "template:first_play"},
            ],
        },
        readiness={"status": "WAIT", "primary_reason": "OPENING_UNRESOLVED"},
        gate={"reason": "opening_unresolved"},
        roi_validation={"status": "pass", "issues": []},
    )

    assert report["evidence_only"] is True
    summary = report["summary"]
    assert summary["round_level"] == "7"
    assert summary["wild_rank"] == "7"
    assert summary["hand_count"] == 2
    assert summary["expected_hand_count"] == 27
    assert summary["current_player"] == "self"
    assert summary["lead_player"] is None
    assert summary["buttons"] == ["pass"]

    actions = report["actions"]
    assert actions[0]["player"] == "right"
    assert actions[0]["cards"] == ["9S", "9H"]
    assert actions[0]["recognized_play"] is True
    assert actions[1]["is_pass"] is True
    assert actions[1]["status"] == "过牌"
    assert actions[2]["recognized_play"] is False
    assert actions[2]["status"].startswith("未识别")

    lead = report["lead_evidence"]
    assert lead[0]["first_play_score"] == 0.61
    assert lead[0]["candidate_score"] == 0.61
    assert lead[0]["rejection_reason"] == "candidate_conflict"

    findings = report["threshold_analysis"]["findings"]
    assert any(item["code"] == "BELOW_THRESHOLD" and "0.560" in item["message"] and "0.750" in item["message"] for item in findings)
    assert any(item["code"] == "FIELD_CONFIDENCE_BELOW_THRESHOLD" for item in findings)
    assert any(item["code"] == "CANDIDATE_CONFLICT" for item in (item for item in report["blockers"]))
    assert any(item["code"] == "HAND_COUNT_MISMATCH" for item in report["blockers"])
    assert "首出候选之间存在冲突" in report["user_text"]
    assert "不能把缺失牌补出来" in report["user_text"]


def test_detailed_report_never_guesses_when_evidence_is_absent():
    result = RecognitionResult(
        round_level=None,
        wild_rank=None,
        current_player=None,
        lead_player=None,
        my_hand=(),
        events=(),
        field_confidences={},
        sources={},
        unresolved_fields=("round_level", "wild_rank", "my_hand", "current_player", "lead_player"),
        diagnostics=(),
        buttons=(),
        lead_evidence=(),
    )
    report = build_detailed_diagnostic(result=result, trace={"candidates": []}, readiness={"status": "WAIT", "primary_reason": "LOBBY"})
    summary = report["summary"]
    assert summary["round_level"] is None
    assert summary["lead_player"] is None
    assert report["field_evidence"][0]["status"] == "unresolved"
    assert "不能推断首出玩家" in report["user_text"]
    assert "没有出牌" in report["user_text"]
    assert "self" not in report["user_text"]


def test_action_only_roi_overlap_warning_is_not_an_opening_blocker():
    result = _rich_result()
    report = build_detailed_diagnostic(
        result=result,
        readiness={"status": "PASS", "primary_reason": "READY_WAITING_FIRST_ACTION"},
        roi_validation={
            "status": "fail",
            "issues": [{
                "code": "roi.critical_play_overlap",
                "severity": "warning",
                "region": "right_play",
                "related_region": "my_play",
            }],
        },
    )

    assert not any(item["code"] == "ROI_INVALID" for item in report["blockers"])
    assert report["roi_validation"]["issues"][0]["code"] == "roi.critical_play_overlap"

    view = build_user_view({
        "opening_readiness_inputs": {
            "readiness": {"status": "PASS", "primary_reason": "READY_WAITING_FIRST_ACTION"}
        },
        "roi_validation": report["roi_validation"],
    })
    roi_group = next(group for group in view.groups if group["名称"] == "识别区域")
    assert roi_group["状态"] == "正常"


def test_mixed_action_only_roi_warning_and_fatal_roi_error_still_blocks():
    report = build_detailed_diagnostic(
        result=_rich_result(),
        readiness={"status": "PASS", "primary_reason": "READY_WAITING_FIRST_ACTION"},
        roi_validation={
            "status": "fail",
            "opening_blocking": False,
            "issues": [
                {
                    "code": "roi.critical_play_overlap",
                    "severity": "warning",
                    "action_only": True,
                },
                {
                    "code": "roi.out_of_bounds",
                    "severity": "fatal",
                    "region": "level_rank",
                },
            ],
        },
    )

    roi_blockers = [item for item in report["blockers"] if item["category"] == "roi"]
    assert len(roi_blockers) == 1
    assert roi_blockers[0]["code"] == "ROI_INVALID"
    assert roi_blockers[0]["evidence"]["issues"] == [{
        "code": "roi.out_of_bounds",
        "severity": "fatal",
        "region": "level_rank",
    }]


def test_ready_waiting_first_action_has_user_facing_copy():
    view = build_user_view({
        "opening_readiness_inputs": {
            "readiness": {
                "status": "PASS",
                "primary_reason": "READY_WAITING_FIRST_ACTION",
            }
        }
    })

    assert view.title == "已进入牌桌，等待自己首出"
    assert view.status == "已就绪"
