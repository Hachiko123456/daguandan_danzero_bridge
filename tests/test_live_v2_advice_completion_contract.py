from dataclasses import replace
import json

import pytest

from daguandan_bridge.application.live_v2_advice_protocol import AdviceRequestIdentity
from daguandan_bridge.application.live_v2_advice_runtime import _same_opportunity
from daguandan_bridge.application.shadow_live_replay import summarize_advice_lifecycle
from daguandan_bridge.live_v2.identity import VersionIdentity


def identity(version, opportunity="advice:s:1:4"):
    return AdviceRequestIdentity(version, 1, opportunity)


def test_same_formal_opportunity_ignores_only_publication_sequence() -> None:
    old = identity(VersionIdentity("s", 1, 5, 1, 4))
    current = identity(replace(old.version, update_sequence=99))
    assert _same_opportunity(current, old)


@pytest.mark.parametrize(
    "changed",
    (
        VersionIdentity("s", 2, 5, 2, 4),
        VersionIdentity("s", 1, 6, 2, 4),
        VersionIdentity("s", 1, 5, 2, 5),
    ),
)
def test_formal_state_or_generation_change_invalidates_advice(changed) -> None:
    old = identity(VersionIdentity("s", 1, 5, 1, 4))
    assert not _same_opportunity(identity(changed), old)
    assert not _same_opportunity(identity(old.version, "another-opportunity"), old)


def test_requested_without_terminal_is_reported_not_silently_zero(tmp_path) -> None:
    (tmp_path / "advice.jsonl").write_text(
        json.dumps({"request_id": "r1", "status": "requested"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "timeline.jsonl").write_text("", encoding="utf-8")
    summary = summarize_advice_lifecycle(tmp_path, drained=True)
    assert summary["requested"] == 1
    assert summary["terminal_counts"] == {"timeout": 1}
    assert summary["requests"][0]["terminal_inferred"] is True


def test_local_pass_is_a_real_advice_terminal(tmp_path) -> None:
    records = (
        {"request_id": "r1", "status": "requested"},
        {"request_id": "r1", "status": "local_pass"},
    )
    (tmp_path / "advice.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (tmp_path / "timeline.jsonl").write_text("", encoding="utf-8")
    summary = summarize_advice_lifecycle(tmp_path, drained=True)
    assert summary["terminal_counts"] == {"local_pass": 1}
    assert summary["requests"][0]["terminal_inferred"] is False


def test_full_chain_summary_projects_and_aggregates_worker_timings(tmp_path) -> None:
    timing = {
        "worker_generation": 1,
        "worker_request_sequence": 7,
        "host_accepted": 100,
        "send_start": 101,
        "send_end": 102,
        "child_received": 103,
        "child_start": 104,
        "child_end": 140,
        "result_received": 141,
        "delta_ms": {
            "accepted_to_send_start": 1,
            "send": 1,
            "send_to_child_received": 1,
            "child_queue": 1,
            "child_execution": 36,
            "child_to_result_received": 1,
            "total": 41,
        },
    }
    records = (
        {"request_id": "r1", "status": "requested"},
        {"request_id": "r1", "status": "worker_started"},
        {"request_id": "r1", "status": "ready", "worker_timing": timing},
    )
    (tmp_path / "advice.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    (tmp_path / "timeline.jsonl").write_text("", encoding="utf-8")

    summary = summarize_advice_lifecycle(tmp_path, drained=True)

    assert summary["requests"][0]["worker_timing"] == timing
    assert summary["worker_timing"]["record_count"] == 1
    assert summary["worker_timing"]["complete_count"] == 1
    assert summary["worker_timing"]["delta_ms"]["total"]["max"] == 41
