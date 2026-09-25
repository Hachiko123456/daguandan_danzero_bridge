from __future__ import annotations

from types import SimpleNamespace

import pytest

from daguandan_bridge.application.opening_readiness import (
    FAIL,
    PASS,
    WAIT,
    OpeningReadinessCode,
    OpeningReadinessStatus,
    coerce_report,
    report_for_error,
    report_for_page,
    report_for_phase,
)


@pytest.mark.parametrize(
    ("phase", "reason", "status"),
    [
        ("ready", OpeningReadinessCode.READY, PASS),
        ("ready_waiting_first_action", OpeningReadinessCode.READY_WAITING_FIRST_ACTION, PASS),
        ("lobby", OpeningReadinessCode.LOBBY, WAIT),
        ("hand_count_mismatch", OpeningReadinessCode.MID_GAME_HAND_COUNT, WAIT),
        ("hand_invalid", OpeningReadinessCode.HAND_UNSTABLE, WAIT),
        ("opening_seed_invalid", OpeningReadinessCode.OPENING_UNRESOLVED, WAIT),
        ("missed_opening", OpeningReadinessCode.DEAL_IN_PROGRESS, FAIL),
    ],
)
def test_phase_report_has_stable_code_and_safe_compact_policy(phase, reason, status):
    report = report_for_phase(phase, hand_count=19)

    assert report.status is status
    assert report.primary_reason is reason
    assert report.compact_allowed is (status is PASS)
    assert report.diagnostic_compact_allowed is (status is not FAIL)
    assert report.session_allowed is (status is PASS)
    assert report.to_dict()["error_code"] == reason.value
    assert report.message
    assert report.suggested_action


def test_ready_waiting_first_action_is_session_ready_but_waits_for_self_lead():
    report = report_for_phase("ready_waiting_first_action", hand_count=27)

    assert report.status is PASS
    assert report.primary_reason is OpeningReadinessCode.READY_WAITING_FIRST_ACTION
    assert report.session_allowed is True
    assert report.compact_allowed is True
    assert "已进入牌桌" in report.message
    assert "自己首出" in report.message
    assert report.to_dict()["phase"] == "ready_waiting_first_action"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (SimpleNamespace(code="WINDOW-NOT-FOUND"), OpeningReadinessCode.WINDOW_NOT_FOUND),
        (SimpleNamespace(code="WINDOW-AMBIGUOUS"), OpeningReadinessCode.MULTIPLE_WINDOWS),
        (SimpleNamespace(code="WINDOW-MINIMIZED"), OpeningReadinessCode.WINDOW_MINIMIZED),
        (SimpleNamespace(code="CAPTURE-BACKEND-FAILED"), OpeningReadinessCode.CAPTURE_FAILED),
        (SimpleNamespace(code="ROI_FATAL"), OpeningReadinessCode.ROI_FATAL),
        (SimpleNamespace(code="WORKER_FAULT"), OpeningReadinessCode.WORKER_FAULT),
    ],
)
def test_error_report_normalizes_window_capture_roi_and_worker_failures(error, reason):
    report = report_for_error(error)

    assert report.status is FAIL
    assert report.primary_reason is reason
    assert report.compact_allowed is False
    assert report.diagnostic_compact_allowed is False
    assert report.session_allowed is False
    assert report.hard_error is True
    assert report.to_dict()["primary_reason"] == reason.value


def test_minimized_window_can_be_reported_as_recovering_wait():
    report = report_for_error(
        SimpleNamespace(code="WINDOW-MINIMIZED", details={"hwnd": 7}),
        recovering=True,
    )

    assert report.status is WAIT
    assert report.primary_reason is OpeningReadinessCode.WINDOW_MINIMIZED
    assert report.recoverable is True
    assert report.compact_allowed is False
    assert report.diagnostic_compact_allowed is True
    assert report.details["hwnd"] == 7


@pytest.mark.parametrize("stage", ["unknown", "lobby", "settlement", "waiting_table"])
def test_non_table_pages_are_explicit_lobby_waits(stage):
    report = report_for_page(stage)

    assert report.status is WAIT
    assert report.primary_reason is OpeningReadinessCode.LOBBY
    assert "牌桌" in report.message
    assert report.suggested_action


def test_report_round_trips_from_signal_payload():
    original = report_for_phase("confirming_opening", hand_count=27)

    restored = coerce_report({"state": "opening", "report": original, "generation": 4})

    assert restored == original
    assert restored is not None
    assert restored.status is OpeningReadinessStatus.WAIT


def test_settlement_screen_is_waiting_for_next_game_not_opening_unresolved():
    report = report_for_phase("settlement_screen")
    assert report.status is WAIT
    assert report.primary_reason is OpeningReadinessCode.LOBBY
    assert "结算" in report.message
    assert "新一局" in report.suggested_action
