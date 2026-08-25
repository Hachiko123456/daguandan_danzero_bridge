from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from daguandan_bridge.application.shadow_live_replay import (
    AbsolutePacer,
    FaultProfile,
    ShadowLiveReplayConfig,
    ShadowLiveReplayRunner,
    apply_spatial_fault,
    build_fault_plan,
    compare_repeat_results,
    summarize_advice_lifecycle,
    summarize_fault_applications,
)
from daguandan_bridge.live.replay import FrameIndexRecord


def _records(count: int = 4) -> tuple[FrameIndexRecord, ...]:
    return tuple(
        FrameIndexRecord(index, 1_000 + index * 100, f"t{index}")
        for index in range(count)
    )


def test_fault_plan_is_seeded_hashed_and_captured_clock_never_regresses():
    profile = FaultProfile(
        jitter_ms=250,
        drop_probability=0.25,
        duplicate_probability=0.5,
        pause_probability=0.5,
        pause_ms=300,
        advisor_delay_ms=50,
        offset_x=3,
        offset_y=-2,
        scale=0.9,
    )

    first = build_fault_plan(_records(20), profile, seed=17, canvas_size=(64, 32))
    second = build_fault_plan(_records(20), profile, seed=17, canvas_size=(64, 32))
    different = build_fault_plan(_records(20), profile, seed=18, canvas_size=(64, 32))

    assert first.sha256 == second.sha256
    assert first.document == second.document
    assert first.sha256 != different.sha256
    deliveries = [delivery for entry in first.entries for delivery in entry["deliveries"]]
    assert [row["delivery_seq"] for row in deliveries] == list(range(1, len(deliveries) + 1))
    captured = [row["captured_monotonic_ms"] for row in deliveries]
    assert captured == sorted(captured)
    assert len(captured) == len(set(captured))
    assert first.document["transform"]["padding"] == "constant_bgr_black"


def test_each_fault_is_explicitly_planned_and_applied():
    record = _records(1)
    dropped = build_fault_plan(record, FaultProfile(drop_probability=1.0), seed=1)
    duplicated = build_fault_plan(record, FaultProfile(duplicate_probability=1.0), seed=1)
    paused = build_fault_plan(
        record,
        FaultProfile(pause_probability=1.0, pause_ms=500),
        seed=1,
    )

    assert dropped.entries[0]["faults"]["drop"] == {"planned": True, "applied": True}
    assert dropped.entries[0]["deliveries"] == []
    assert duplicated.entries[0]["faults"]["duplicate"]["applied"] is True
    assert len(duplicated.entries[0]["deliveries"]) == 2
    assert paused.entries[0]["faults"]["pause"]["value_ms"] == 500
    assert paused.entries[0]["deliveries"][0]["planned_delivery_ms"] == 500


def test_absolute_pacer_uses_fixed_epoch_without_relative_sleep_drift():
    clock = _VirtualClock()
    pacer = AbsolutePacer(clock=clock.now, sleep=clock.sleep)

    observed = [pacer.wait_until(target) for target in (0, 100, 250, 400)]

    assert observed == [0.0, 0.1, 0.25, 0.4]
    assert clock.sleeps == [0.1, 0.15, 0.15]


def test_repeat_comparison_marks_same_plan_different_business_as_nondeterminism():
    comparison = compare_repeat_results(
        {"fault_plan_sha256": "same", "business_signature": "previous"},
        "same",
        "current",
    )

    assert comparison["same_plan"] is True
    assert comparison["same_business_result"] is False
    assert comparison["concurrency_nondeterminism"] is True


def test_spatial_fault_keeps_fixed_uint8_bgr_canvas():
    gray = np.full((20, 30), 300, np.uint16)

    transformed, metadata = apply_spatial_fault(
        gray,
        FaultProfile(offset_x=4, offset_y=-3, scale=0.75),
    )

    assert transformed.shape == (20, 30, 3)
    assert transformed.dtype == np.uint8
    assert metadata["canvas_size"] == [30, 20]
    assert metadata["padding"] == "constant_bgr_black"
    assert metadata["matrix"] == [[0.75, 0.0, 7.75], [0.0, 0.75, -0.5]]


def test_shadow_runner_uses_async_worker_writes_outputs_and_preserves_source(tmp_path: Path):
    session = _make_session(tmp_path / "profile" / "sessions", frame_count=4)
    source_before = {path: path.read_bytes() for path in session.rglob("*") if path.is_file()}
    clock = _VirtualClock()
    holder: dict[str, _FakeOrchestrator] = {}

    def orchestrator_factory(**kwargs):
        value = _FakeOrchestrator(**kwargs)
        holder["value"] = value
        return value

    result = ShadowLiveReplayRunner(
        recognition_factory=lambda _session: object(),
        advisor_factory=lambda *_args: _Advisor(),
        orchestrator_factory=orchestrator_factory,
        clock=clock.now,
        sleep=clock.sleep,
    ).run(
        ShadowLiveReplayConfig(
            session=session,
            output=tmp_path / "reports",
            run_id="short",
            max_frames=4,
        )
    )

    runtime = Path(result.summary["runtime_directory"])
    assert result.execution_ok is True
    assert result.summary["plan"]["fully_consumed"] is True
    assert result.summary["source_integrity"]["unchanged"] is True
    assert result.summary["delivery"]["actual_deliveries"] == 4
    assert result.summary["drops"]["fault"] == 0
    assert result.summary["worker"]["pending_depth"] == 0
    assert result.summary["worker"]["priority_depth"] == 0
    assert result.summary["performance"]["analyze_ms"]["count"] >= 1
    assert result.summary["schedule"]["planned_duration_ms"] == 300
    assert result.summary["schedule"]["actual_duration_ms"] == 300
    assert result.summary["schedule"]["delivery_lag_p95_ms"] == 0
    assert result.summary["schedule"]["strict_no_fault_timing_pass"] is True
    assert (result.run_directory / "fault_plan.json").is_file()
    assert (result.run_directory / "deliveries.jsonl").is_file()
    assert (result.run_directory / "analysis.jsonl").is_file()
    assert (result.run_directory / "summary.md").is_file()
    assert (runtime / "timeline.jsonl").is_file()
    assert (runtime / "advice.jsonl").is_file()
    assert (runtime / "decisions.jsonl").is_file()
    assert (runtime / "recognition_trace.jsonl").is_file()
    assert all(
        row["worker_thread_id"] != row["producer_thread_id"]
        for row in _json_lines(result.run_directory / "analysis.jsonl")
    )
    assert holder["value"].all_records_precede_matching_submit is True
    assert {path: path.read_bytes() for path in session.rglob("*") if path.is_file()} == source_before


def test_runner_rejects_output_anywhere_below_source_sessions_root(tmp_path: Path):
    sessions = tmp_path / "profile" / "sessions"
    session = _make_session(sessions, frame_count=1)

    try:
        ShadowLiveReplayRunner().run(
            ShadowLiveReplayConfig(
                session=session,
                output=sessions / "shadow-sibling",
                run_id="forbidden",
                max_frames=1,
            )
        )
    except ValueError as exc:
        assert "outside the source sessions root" in str(exc)
    else:
        raise AssertionError("sessions-root output was accepted")
    assert not (sessions / "shadow-sibling").exists()


def test_strict_one_x_no_fault_timing_gate_fails_on_excess_delivery_lag(tmp_path: Path):
    session = _make_session(tmp_path / "profile" / "sessions", frame_count=3)
    clock = _OversleepClock()

    result = ShadowLiveReplayRunner(
        recognition_factory=lambda _session: object(),
        advisor_factory=lambda *_args: _Advisor(),
        orchestrator_factory=lambda **kwargs: _FakeOrchestrator(**kwargs),
        clock=clock.now,
        sleep=clock.sleep,
    ).run(
        ShadowLiveReplayConfig(
            session=session,
            output=tmp_path / "reports",
            run_id="late",
            max_frames=3,
        )
    )

    assert result.execution_ok is False
    assert result.summary["schedule"]["strict_no_fault_timing_required"] is True
    assert result.summary["schedule"]["strict_no_fault_timing_pass"] is False
    assert result.summary["schedule"]["delivery_lag_p95_ms"] > 200
    assert "strict 1x no-fault timing gate failed" in result.summary["errors"]


def test_advisor_delay_does_not_block_frame_delivery_and_request_gets_terminal(tmp_path: Path):
    session = _make_session(tmp_path / "profile" / "sessions", frame_count=5)
    delay_started = threading.Event()
    release_delay = threading.Event()
    holder: dict[str, object] = {}

    def advisor_sleep(_seconds: float) -> None:
        delay_started.set()
        assert release_delay.wait(2)

    def orchestrator_factory(**kwargs):
        value = _AdviceFakeOrchestrator(**kwargs)
        holder["orchestrator"] = value
        return value

    def run_shadow() -> None:
        holder["result"] = ShadowLiveReplayRunner(
            recognition_factory=lambda _session: object(),
            advisor_factory=lambda *_args: _Advisor(),
            orchestrator_factory=orchestrator_factory,
            advisor_sleep=advisor_sleep,
            clock=time.monotonic,
            sleep=time.sleep,
        ).run(
            ShadowLiveReplayConfig(
                session=session,
                output=tmp_path / "reports",
                run_id="advisor-delay",
                max_frames=5,
                time_scale=20.0,
                fault_profile=FaultProfile(advisor_delay_ms=500),
                drain_timeout_sec=2.0,
            )
        )

    thread = threading.Thread(target=run_shadow)
    thread.start()
    assert delay_started.wait(2)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        orchestrator = holder.get("orchestrator")
        if orchestrator is not None and orchestrator.recorder.frame_count == 5:
            break
        time.sleep(0.01)
    assert holder["orchestrator"].recorder.frame_count == 5
    assert thread.is_alive()
    release_delay.set()
    thread.join(3)
    assert not thread.is_alive()
    result = holder["result"]
    assert result.execution_ok is True
    assert result.summary["advice"]["requested"] == 1
    assert result.summary["advice"]["terminal_counts"] == {"ready": 1}
    assert result.summary["advice"]["withheld_without_request"] == 1
    assert result.summary["advice"]["non_request_terminal_counts"] == {"withheld": 1}
    assert result.summary["advice"]["requests"][0]["terminal_inferred"] is False
    assert result.summary["advice"]["requests"][0]["worker_start_wall_latency_ms"] is not None
    assert result.summary["faults"]["advisor_delay"] == {
        "planned": 1,
        "applied": 1,
        "skipped": 0,
    }


def test_prestart_discard_is_explained_and_advisor_delay_is_skipped(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "timeline.jsonl").write_text("", encoding="utf-8")
    (runtime / "advice.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "request_id": "ADV-PRESTART",
                    "status": "requested",
                    "turn_id": 3,
                    "state_revision": 7,
                    "wall_time": "2026-08-25T10:00:00+08:00",
                },
                {
                    "request_id": "ADV-PRESTART",
                    "status": "stale",
                    "turn_id": 3,
                    "state_revision": 7,
                    "discard_reason": "latest_replaced",
                    "wall_time": "2026-08-25T10:00:00.001000+08:00",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    lifecycle = summarize_advice_lifecycle(runtime, drained=True)
    plan = build_fault_plan(
        _records(1),
        FaultProfile(advisor_delay_ms=500),
        seed=3,
    )
    faults = summarize_fault_applications(plan, lifecycle)

    assert lifecycle["requested"] == 1
    assert lifecycle["worker_started"] == 0
    assert lifecycle["requests"][0]["terminal_status"] == "stale"
    assert lifecycle["requests"][0]["worker_start_explained"] is True
    assert lifecycle["requests"][0]["worker_start_skip_reason"] == "latest_replaced"
    assert lifecycle["requests"][0]["terminal_inferred"] is False
    assert faults["advisor_delay"] == {"planned": 1, "applied": 0, "skipped": 1}


def test_cli_returns_nonzero_for_sessions_root_output(tmp_path: Path):
    sessions = tmp_path / "profile" / "sessions"
    session = _make_session(sessions, frame_count=1)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_shadow_live_replay.py",
            "--session",
            str(session),
            "--output",
            str(sessions / "reports"),
            "--max-frames",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "outside the source sessions root" in completed.stderr


class _VirtualClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []
        self.lock = threading.Lock()

    def now(self) -> float:
        with self.lock:
            return self.value

    def sleep(self, seconds: float) -> None:
        with self.lock:
            self.sleeps.append(round(seconds, 10))
            self.value += seconds


class _OversleepClock(_VirtualClock):
    def sleep(self, seconds: float) -> None:
        with self.lock:
            self.sleeps.append(round(seconds, 10))
            self.value += seconds + 0.3


@dataclass
class _Snapshot:
    revision: int = 1

    def semantic_dict(self):
        return {"revision": self.revision, "current_player": "right"}


@dataclass
class _Update:
    snapshot: _Snapshot
    status: str = "running"
    events: tuple[object, ...] = ()


class _Advisor:
    def recommend(self, *_args, **_kwargs):
        return None


class _FakeOrchestrator:
    def __init__(self, *, store, recorder, recognition, advisor) -> None:
        del recognition, advisor
        self.store = store
        self.recorder = recorder
        self.snapshot = _Snapshot()
        self.calls: list[tuple[str, int]] = []
        self.recorded: set[int] = set()
        self.all_records_precede_matching_submit = True

    @property
    def needs_first_action_frames(self) -> bool:
        return False

    def start(self, **_kwargs):
        return _Update(self.snapshot)

    def record_frame(self, frame, *, monotonic_ms, wall_time):
        del frame, wall_time
        self.calls.append(("record", monotonic_ms))
        self.recorded.add(monotonic_ms)
        self.recorder.write_frame(np.zeros((1, 1, 3), np.uint8), monotonic_ms, "recorded")
        return None

    def analyze_frame(self, frame, *, monotonic_ms, trace_context):
        del frame
        self.calls.append(("analyze", monotonic_ms))
        self.all_records_precede_matching_submit &= monotonic_ms in self.recorded
        self.snapshot.revision += 1
        assert trace_context["delivery_seq"] > 0
        return _Update(_Snapshot(self.snapshot.revision))

    def wait_for_advice_idle(self, *, timeout):
        del timeout
        return True

    def finish(self):
        result = self.recorder.close()
        self.store.seal(frame_count=result.frame_count, dropped_frames=result.dropped_frames)
        return _Update(self.snapshot, status="sealed")


class _AdviceFakeOrchestrator(_FakeOrchestrator):
    def __init__(self, *, store, recorder, recognition, advisor) -> None:
        super().__init__(
            store=store,
            recorder=recorder,
            recognition=recognition,
            advisor=advisor,
        )
        self.advisor = advisor
        self.advice_thread: threading.Thread | None = None

    def analyze_frame(self, frame, *, monotonic_ms, trace_context):
        update = super().analyze_frame(
            frame,
            monotonic_ms=monotonic_ms,
            trace_context=trace_context,
        )
        if self.advice_thread is None:
            self.store.append_advice(
                {
                    "request_id": "WITHHELD-ONLY",
                    "status": "withheld",
                    "turn_id": 1,
                    "state_revision": self.snapshot.revision,
                }
            )
            self.store.append_advice(
                {
                    "request_id": "ADV-1",
                    "status": "requested",
                    "turn_id": 1,
                    "state_revision": self.snapshot.revision,
                }
            )

            def recommend():
                self.store.append_advice(
                    {
                        "request_id": "ADV-1",
                        "status": "worker_started",
                        "turn_id": 1,
                        "state_revision": self.snapshot.revision,
                    }
                )
                self.advisor.recommend(None, request_id="ADV-1")
                self.store.append_advice(
                    {
                        "request_id": "ADV-1",
                        "status": "ready",
                        "turn_id": 1,
                        "state_revision": self.snapshot.revision,
                    }
                )

            self.advice_thread = threading.Thread(target=recommend)
            self.advice_thread.start()
        return update

    def wait_for_advice_idle(self, *, timeout):
        if self.advice_thread is None:
            return True
        self.advice_thread.join(timeout)
        return not self.advice_thread.is_alive()


def _make_session(sessions: Path, *, frame_count: int) -> Path:
    session = sessions / "game-short"
    video = session / "video"
    video.mkdir(parents=True)
    (session / "manifest.json").write_text('{"session_id":"game-short"}', encoding="utf-8")
    (session / "timeline.jsonl").write_text(
        json.dumps(
            {
                "event_type": "initial_state_confirmed",
                "actor": "right",
                "payload": {
                    "round_level": "2",
                    "lead_player": "right",
                    "hand": ["2S"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (video / "frame_index.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": index,
                    "monotonic_ms": 1_000 + index * 100,
                    "wall_time": f"t{index}",
                    "dropped_before": index if index == 2 else 0,
                }
            )
            + "\n"
            for index in range(frame_count)
        ),
        encoding="utf-8",
    )
    writer = cv2.VideoWriter(
        str(video / "game.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (64, 32),
    )
    for index in range(frame_count):
        writer.write(np.full((32, 64, 3), index * 30, np.uint8))
    writer.release()
    return session


def _json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
