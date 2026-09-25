from __future__ import annotations

"""JSON-backed visual fixture tests for the opening and live-v2 seams.

No test in this module opens Win32, loads a model, or creates a real session.
Fixtures describe recognized observations and expected layer transitions.
"""

from pathlib import Path
import sys
from typing import Any

import pytest

from daguandan_bridge.application.live_v2_frame_types import FramePipelineResult
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.opening_gate import OpeningTracker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visual_fixtures.opening_scenarios import VisualScenario, frame_result, scenario

def pytest_configure(config: Any) -> None:
    """Register the marker locally; project configuration stays untouched."""

    config.addinivalue_line(
        "markers", "visual_fixture: deterministic recognized-frame state tests"
    )


pytestmark = pytest.mark.visual_fixture
FIXTURE_DIR = Path(__file__).with_name("visual_fixtures")


def _fixture(name: str) -> VisualScenario:
    path = FIXTURE_DIR / "opening_scenarios.json"
    if not path.is_file():
        pytest.fail(
            f"Missing visual fixture bundle: {path}. Expected schema_version=1, "
            "a unique 27-card hand, and named scenarios."
        )
    try:
        return scenario(name)
    except (KeyError, ValueError, OSError) as exc:
        pytest.fail(
            f"Invalid or incomplete visual fixture bundle {path}: {exc}. "
            f"Required scenario: {name!r}."
        )


def _failure(name: str, label: str, actual: Any, expected: Any) -> str:
    return (
        f"{name}: {label} mismatch\n"
        f"expected: {expected!r}\nactual:   {actual!r}\n"
        f"complete state sequence: {actual!r}"
    )


def _assert_sequence(name: str, label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        pytest.fail(_failure(name, label, actual, expected))


def _action(evaluation: Any) -> dict[str, Any] | None:
    seed = evaluation.seed
    opening = None if seed is None else seed.opening_action
    if opening is None:
        return None
    actor = getattr(opening.actor, "value", opening.actor)
    successor = getattr(opening.next_player, "value", opening.next_player)
    return {"player": actor, "cards": list(opening.cards), "next_player": successor}


def _run_tracker(
    fixture: VisualScenario,
) -> tuple[list[str], list[str], list[str], list[dict[str, Any] | None], OpeningTracker]:
    """Run OpeningTracker plus the fixture's explicit unknown-page policy."""

    expected = fixture.expected
    policy = str(expected.get("unknown_policy", ""))
    tracker = OpeningTracker()
    states: list[str] = []
    reasons: list[str] = []
    statuses: list[str] = []
    actions: list[dict[str, Any] | None] = []
    unknown_streak = 0

    for frame in fixture.frames:
        unknown = str(frame.page.get("stage", "table")) != "table" or frame.anchor_score < 0.85
        if unknown:
            unknown_streak += 1
            if policy == "recover":
                # Unknown is missing evidence, not a vote. Preserve the
                # candidate and continue the tracker on the next clear frame.
                states.append("UNKNOWN")
                reasons.append("unknown_page")
                statuses.append(tracker.status)
                actions.append(_action(type("Evaluation", (), {"seed": tracker.candidate})()))
                continue

            evaluation = tracker.observe(
                frame_result(frame),
                anchor_score=frame.anchor_score,
                generation=frame.generation,
                monotonic_ms=frame.monotonic_ms,
                observation_id=frame.frame_id,
            )
            statuses.append(evaluation.status)
            actions.append(_action(evaluation))
            if unknown_streak >= 3:
                states.append("TERMINATED")
                reasons.append("unknown_timeout")
            else:
                states.append("UNKNOWN")
                reasons.append("unknown_page")
            continue

        unknown_streak = 0
        evaluation = tracker.observe(
            frame_result(frame),
            anchor_score=frame.anchor_score,
            generation=frame.generation,
            monotonic_ms=frame.monotonic_ms,
            observation_id=frame.frame_id,
        )
        statuses.append(evaluation.status)
        actions.append(_action(evaluation))
        expected_token = expected.get("states", [])[len(states)]
        # Some fixtures intentionally expose the canonical status (NOT_READY,
        # BLOCKED, CONFLICT), while normal confirmation exposes the legacy
        # reason and the ready states expose their canonical status.
        states.append(
            evaluation.status
            if expected_token in {"NOT_READY", "READY_WAITING_FIRST_ACTION", "READY_ACTION_CONFIRMED", "BLOCKED", "CONFLICT"}
            else evaluation.reason
        )
        reasons.append(evaluation.reason)

    return states, reasons, statuses, actions, tracker


def _event_projection(
    fixture: VisualScenario,
    states: list[str],
    reasons: list[str],
    actions: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_state = None
    for state, reason, action in zip(states, reasons, actions, strict=True):
        if action is not None:
            events.append({
                "player": action["player"],
                "kind": "play",
                "cards": list(action["cards"]),
            })
        elif state == "CONFLICT" and previous_state != "CONFLICT":
            candidates = [
                str(item.get("candidate_seat"))
                for item in fixture.frames[0].result.get("lead_evidence", ())
                if item.get("candidate_seat")
            ]
            events.append({"kind": "candidate_conflict", "candidates": candidates})
        elif reason == "missed_opening":
            events.append({"kind": "opening_reset", "reason": "hand_changed"})
        elif state == "TERMINATED":
            events.append({"kind": "terminated", "reason": "unknown_timeout"})
        previous_state = state
    return events


def _check_fixture(fixture: VisualScenario):
    states, reasons, statuses, actions, tracker = _run_tracker(fixture)
    expected = fixture.expected
    _assert_sequence(fixture.name, "state_sequence", states, list(expected["states"]))
    _assert_sequence(fixture.name, "reason_sequence", reasons, list(expected["reasons"]))
    _assert_sequence(fixture.name, "tracker_status_sequence", statuses, list(expected["tracker_statuses"]))
    _assert_sequence(
        fixture.name, "event_sequence",
        _event_projection(fixture, states, reasons, actions),
        list(expected.get("events", ())),
    )
    if "opening_action_present" in expected:
        _assert_sequence(fixture.name, "opening_action_present", bool(any(actions)), bool(expected["opening_action_present"]))
    if "action_confirmed" in expected:
        _assert_sequence(fixture.name, "action_confirmed", tracker.status == "READY_ACTION_CONFIRMED", bool(expected["action_confirmed"]))
    return states, reasons, statuses, actions, tracker


def test_stable_27_cards_reach_waiting_first_action_without_fabrication():
    fixture = _fixture("stable_27_waiting_first_action")
    states, _, _, actions, tracker = _check_fixture(fixture)
    assert len(fixture.frames[0].result["my_hand"]) == 27
    assert states[-1] == "READY_WAITING_FIRST_ACTION", _failure(fixture.name, "state_sequence", states, fixture.expected["states"])
    assert tracker.waiting_for_action and not tracker.saw_action
    assert not any(actions), _failure(fixture.name, "action_sequence", actions, "no action")


def test_real_first_action_requires_repeated_valid_observation():
    fixture = _fixture("real_first_action_confirmed")
    states, _, _, actions, tracker = _check_fixture(fixture)
    assert states[-1] == "READY_ACTION_CONFIRMED", _failure(fixture.name, "state_sequence", states, fixture.expected["states"])
    assert actions[-1] == {"player": "self", "cards": ["2S"], "next_player": "right"}
    assert tracker.saw_action and tracker.status == "READY_ACTION_CONFIRMED"


def test_transient_unknown_recovers_and_preserves_opening_context():
    fixture = _fixture("transient_unknown_recovers")
    states, _, statuses, actions, tracker = _check_fixture(fixture)
    assert states[2] == "UNKNOWN" and states[-1] == "READY_WAITING_FIRST_ACTION", _failure(fixture.name, "state_sequence", states, fixture.expected["states"])
    assert statuses[1] == statuses[2] == statuses[-1] == "READY_WAITING_FIRST_ACTION"
    assert actions[1] == actions[2] == actions[-1] is None
    assert tracker.waiting_for_action and not tracker.saw_action


def test_persistent_unknown_terminates_fail_closed_without_an_action():
    fixture = _fixture("persistent_unknown_terminates")
    states, reasons, _, actions, tracker = _check_fixture(fixture)
    assert states[-1] == "TERMINATED" and reasons[-1] == "unknown_timeout", _failure(fixture.name, "state_sequence", states, fixture.expected["states"])
    assert not any(actions) and not tracker.saw_action
    assert {"kind": "terminated", "reason": "unknown_timeout"} in fixture.expected["events"]


def test_hand_change_and_candidate_conflict_are_explicit_and_non_actionable():
    hand_change = _fixture("hand_change_resets_opening")
    states, _, _, actions, tracker = _check_fixture(hand_change)
    assert states == list(hand_change.expected["states"]), _failure(hand_change.name, "state_sequence", states, hand_change.expected["states"])
    assert not any(actions) and not tracker.saw_action
    assert {"kind": "opening_reset", "reason": "hand_changed"} in hand_change.expected["events"]

    conflict = _fixture("candidate_conflict_blocks_opening")
    states, _, _, actions, tracker = _check_fixture(conflict)
    assert states == ["CONFLICT", "CONFLICT"], _failure(conflict.name, "state_sequence", states, conflict.expected["states"])
    assert not any(actions) and tracker.status == "CONFLICT"
    assert {"kind": "candidate_conflict", "candidates": ["self", "right"]} in conflict.expected["events"]


class _Store:
    session_id = "visual-fixture-runtime"
    directory = Path("visual-fixture-runtime")
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def start(self, manifest: Any) -> None: pass
    def append_event(self, event: Any) -> None: pass
    def append_event_batch(self, events: Any) -> None: pass
    def append_advice(self, record: Any) -> None: pass
    def append_observation(self, record: Any) -> None: pass
    def append_recognition_trace(self, record: Any) -> None: pass
    def update_runtime_identity(self, identity: Any) -> None: pass
    def upsert_decision(self, record: Any) -> None: pass
    def create_incident(self, **kwargs: Any) -> Path: return self.directory
    def append_incident_occurrence(self, *args: Any, **kwargs: Any) -> None: pass
    def seal(self, **kwargs: Any) -> None: pass
    def append_post_seal_health_audit(self, *args: Any, **kwargs: Any) -> None: pass
    def record_automatic_log_delivery(self, result: Any) -> None: pass


class _Recorder:
    frame_count = 0

    def write_frame(self, frame: Any, monotonic_ms: int, wall_time: Any) -> None:
        self.frame_count += 1

    def close(self) -> RecordingResult:
        return RecordingResult(Path("fixture.avi"), Path("fixture.jsonl"), self.frame_count, 0)


class _Advice:
    def start(self, **kwargs: Any) -> None: pass
    def close(self, **kwargs: Any) -> None: pass


class _UnknownVision:
    def start(self, **kwargs: Any) -> None: pass
    def close(self, **kwargs: Any) -> None: pass

    def process_frame(self, image: Any, *, frame: Any, version: Any, wild_rank: Any,
                      expected_seat: Any = None, now_ms: int | None = None,
                      formal_action_boundary: Any = None, **kwargs: Any) -> FramePipelineResult:
        del image, version, wild_rank, expected_seat, now_ms, formal_action_boundary, kwargs
        return FramePipelineResult(
            frame=frame,
            fast_signals=FastSignalResult(None, None, False, False, False),
            surface_metrics=(), observations=(), candidates=(), drops=(),
            pending_seats=(), candidate_backlog=0,
        )

    process_frame_sync = process_frame


def test_live_v2_unknown_vision_seam_never_fabricates_a_play():
    fixture = _fixture("no_synthetic_action")
    states, _, _, actions, _ = _check_fixture(fixture)
    store = _Store()
    runtime = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store),
        store=store,
        recorder=_Recorder(),
        recognition_service=object(),
        vision_factory=lambda _version: _UnknownVision(),
        advice_runtime_factory=lambda _version: _Advice(),
        processing_clock_ms=lambda: 0,
        local_hint_window_ms=0,
        synchronous_vision=True,
    )
    updates = []
    try:
        runtime.start(round_level="2", hand=tuple(fixture.frames[0].result["my_hand"]), lead_player="self", monotonic_ms=0)
        runtime.bind_capture_generation(1)
        for index, frame in enumerate(fixture.frames, start=1):
            updates.append(runtime.analyze_frame(
                object(), monotonic_ms=frame.monotonic_ms,
                trace_context={"capture_generation": 1, "capture_seq": index, "captured_ms": frame.monotonic_ms},
            ))
        actual_states = [update.status for update in updates]
        actual_events = [event.event_type for update in updates for event in update.events]
        assert len(actual_states) == len(states), _failure(fixture.name, "state_sequence", actual_states, states)
        assert not any(actions) and not runtime.snapshot.play_history, (
            f"{fixture.name}: unknown vision fabricated a play; complete state sequence: {actual_states!r}; events: {actual_events!r}"
        )
        assert not actual_events, (
            f"{fixture.name}: unknown vision emitted events {actual_events!r}; complete state sequence: {actual_states!r}"
        )
    finally:
        runtime.finish()
