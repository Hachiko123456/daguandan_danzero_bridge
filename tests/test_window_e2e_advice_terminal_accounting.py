from __future__ import annotations

from daguandan_bridge.application import window_e2e_validation as validation


def _execution_gate(advice: dict[str, object]) -> dict[str, object]:
    advisor_terminal = validation._advice_lifecycle_terminal(advice)
    return validation._full_chain_execution_gate(
        development_fragment=False,
        common_checks={
            "no_errors": True,
            "simulator_eof": True,
            "captured_frames": True,
            "frame_target_reached": True,
            "analysis_drained": True,
            "advice_drained": True,
            "finish_completed": True,
            "runtime_audit": True,
            "opportunity_responses": True,
            "rule_engine_probe": True,
            "capture_worker_stopped": True,
            "analysis_worker_stopped": True,
        },
        fragment_prefix_passed=False,
        advisor_terminal=advisor_terminal,
        has_actions=True,
        business_health_passed=True,
        baseline_passed=True,
    )


def test_full_chain_counts_ready_and_local_pass_as_request_terminals() -> None:
    advice = {
        "requested": 23,
        "terminal_counts": {"ready": 14, "local_pass": 9},
    }

    assert validation._advice_lifecycle_terminal(advice) is True
    gate = _execution_gate(advice)
    assert gate["checks"]["advisor_terminal"] is True
    assert gate["passed"] is True


def test_full_chain_rejects_one_missing_advice_terminal() -> None:
    advice = {
        "requested": 23,
        "terminal_counts": {"ready": 14, "local_pass": 8},
    }

    assert validation._advice_lifecycle_terminal(advice) is False
    gate = _execution_gate(advice)
    assert gate["checks"]["advisor_terminal"] is False
    assert gate["passed"] is False
