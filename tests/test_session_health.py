from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
import hashlib
import json

from daguandan_bridge.danzero.state import PlayEvent
from daguandan_bridge.session_health import (
    ACTION_CHAIN_INCONSISTENT,
    FINISHED_SEATS_INCONSISTENT,
    PREMATURE_GAME_END,
    audit_session_health,
)
from daguandan_bridge.live.session_store import LiveSessionStore


def _terminal_event():
    return SimpleNamespace(event_type="game_end_detected", event_id="AUX-1")


def test_premature_terminal_is_a_fail_without_rewriting_timeline():
    events = (_terminal_event(),)
    snapshot = SimpleNamespace(
        remaining_cards={"self": 18, "right": 24, "opposite": 26, "left": 24},
        finished_seats=frozenset(),
        play_history=(),
    )

    report = audit_session_health(snapshot, events)

    assert report["status"] == "FAIL"
    assert [issue["code"] for issue in report["issues"]] == [PREMATURE_GAME_END]
    assert events[0].event_type == "game_end_detected"


def test_finished_and_action_chain_contradictions_are_separate_failures():
    play = PlayEvent(
        player="self",
        cards=("2S",),
        is_pass=False,
        observed_at=datetime.now().astimezone(),
    )
    snapshot = SimpleNamespace(
        remaining_cards={"self": 20, "right": 0, "opposite": 27, "left": 27},
        finished_seats=frozenset({"self"}),
        play_history=(play,),
    )

    report = audit_session_health(snapshot, (_terminal_event(),))
    codes = {issue["code"] for issue in report["issues"]}

    assert FINISHED_SEATS_INCONSISTENT in codes
    assert ACTION_CHAIN_INCONSISTENT in codes


def test_no_terminal_control_produces_no_premature_terminal_issue():
    snapshot = SimpleNamespace(
        remaining_cards={"self": 27, "right": 27, "opposite": 27, "left": 27},
        finished_seats=frozenset(),
        play_history=(),
    )

    report = audit_session_health(snapshot, ())

    assert report["status"] == "PASS"
    assert report["issues"] == []


def test_health_fail_is_appended_after_seal_without_timeline_mutation(tmp_path):
    store = LiveSessionStore(tmp_path, "profile", session_id="health_test")
    store.start({})
    timeline_before = hashlib.sha256(store.timeline_path.read_bytes()).hexdigest()
    store.seal(frame_count=0, dropped_frames=0)
    report = {
        "schema": "guandan.session-health/1",
        "status": "FAIL",
        "issues": [
            {
                "code": PREMATURE_GAME_END,
                "severity": "FAIL",
                "summary": "premature",
                "evidence": {"remaining_cards": {"self": 18}},
            }
        ],
    }

    store.append_post_seal_health_audit(report, state={"revision": 1}, monotonic_ms=123)

    assert hashlib.sha256(store.timeline_path.read_bytes()).hexdigest() == timeline_before
    incident = json.loads(
        next(store.incidents_directory.glob("*/incident.json")).read_text(encoding="utf-8")
    )
    assert incident["code"] == PREMATURE_GAME_END
    assert incident["post_seal"] is True
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "sealed"
    assert manifest["health_audit"]["status"] == "FAIL"
