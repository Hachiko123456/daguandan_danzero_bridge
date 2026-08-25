"""Qt-free 1x Shadow Live replay with deterministic fault injection."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from inspect import signature
from typing import Any, Callable, Iterable
from uuid import uuid4

import cv2
import numpy as np

from ..advisor_strategy import build_advisor
from ..annotation_service import AnnotationService
from ..live.latest_worker import LatestOnlyWorker
from ..live.orchestrator import LiveOrchestrator
from ..live.recorder import InMemorySessionRecorder
from ..live.reducer import LiveReducer
from ..live.replay import FrameIndexRecord, VideoReplaySource
from ..live.session_store import LiveSessionStore, read_json_lines
from ..live.truth_log import load_truth_log
from ..recognition_service import ScreenshotRecognitionService
from ..storage import append_json_line, atomic_write_json
from ..template_service import TemplateService

_ACTION_TYPES = frozenset({"player_played", "player_passed", "manual_confirmed_event"})
_TERMINAL_ADVICE = frozenset({"ready", "failed", "stale", "withheld", "timeout", "cancelled"})


@dataclass(frozen=True)
class FaultProfile:
    jitter_ms: int = 0
    drop_probability: float = 0.0
    duplicate_probability: float = 0.0
    pause_probability: float = 0.0
    pause_ms: int = 0
    advisor_delay_ms: int = 0
    offset_x: int = 0
    offset_y: int = 0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.jitter_ms < 0 or self.pause_ms < 0 or self.advisor_delay_ms < 0:
            raise ValueError("fault delays must be non-negative")
        for name in ("drop_probability", "duplicate_probability", "pause_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be a positive finite number")

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "FaultProfile":
        names = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{name: raw[name] for name in names if name in raw})  # type: ignore[arg-type]


@dataclass(frozen=True)
class FaultPlan:
    document: dict[str, object]
    sha256: str

    @property
    def entries(self) -> tuple[dict[str, object], ...]:
        return tuple(self.document.get("entries", ()))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {**self.document, "sha256": self.sha256}


@dataclass(frozen=True)
class ShadowLiveReplayConfig:
    session: Path
    output: Path
    fault_profile: FaultProfile = FaultProfile()
    seed: int = 0
    time_scale: float = 1.0
    start_frame: int | None = None
    end_frame: int | None = None
    max_frames: int | None = None
    drain_timeout_sec: float = 30.0
    run_id: str | None = None
    baseline: Path | None = None
    compare_summary: Path | None = None
    expected_plan_sha256: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_scale) or self.time_scale <= 0:
            raise ValueError("time_scale must be a positive finite number")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.end_frame is not None and self.start_frame is not None and self.end_frame < self.start_frame:
            raise ValueError("end_frame must not precede start_frame")
        if self.drain_timeout_sec < 0:
            raise ValueError("drain_timeout_sec must be non-negative")


@dataclass(frozen=True)
class ShadowLiveReplayResult:
    run_directory: Path
    summary_path: Path
    fault_plan_path: Path
    execution_ok: bool
    summary: dict[str, object]


@dataclass(frozen=True)
class _AnalysisTask:
    delivery_seq: int
    source_frame_index: int
    source_monotonic_ms: int
    planned_delivery_ms: float
    captured_monotonic_ms: int
    submitted_wall_monotonic_ms: float
    delivery_wall_monotonic_ms: float
    frame: np.ndarray


class _TaskFailure(RuntimeError):
    def __init__(self, task: _AnalysisTask, cause: Exception) -> None:
        super().__init__(f"delivery {task.delivery_seq}: {type(cause).__name__}: {cause}")
        self.task = task
        self.cause = cause


class AbsolutePacer:
    """Wait against one fixed epoch so per-frame delays never accumulate."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clock = clock
        self.sleep = sleep
        self.started_at: float | None = None

    def start(self) -> float:
        if self.started_at is None:
            self.started_at = self.clock()
        return self.started_at

    def wait_until(self, target_ms: float) -> float:
        epoch = self.start()
        deadline = epoch + max(0.0, float(target_ms)) / 1000.0
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                return self.clock()
            self.sleep(remaining)


class DelayedAdvisor:
    """Transparent AdvicePort decorator used only for advisor-delay faults."""

    def __init__(
        self,
        advisor: Any,
        delay_ms: int,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._advisor = advisor
        self.delay_ms = max(0, int(delay_ms))
        self._sleep = sleep

    def recommend(
        self,
        state: Any,
        *,
        request_id: str = "",
        trace: Any | None = None,
    ) -> Any:
        if self.delay_ms:
            self._sleep(self.delay_ms / 1000.0)
        kwargs: dict[str, object] = {"request_id": request_id}
        if trace is not None and "trace" in signature(self._advisor.recommend).parameters:
            kwargs["trace"] = trace
        return self._advisor.recommend(state, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._advisor, name)


def build_fault_plan(
    records: Iterable[FrameIndexRecord | dict[str, object]],
    profile: FaultProfile,
    *,
    seed: int,
    time_scale: float = 1.0,
    canvas_size: tuple[int, int] | None = None,
) -> FaultPlan:
    """Build every random decision before replay; thread timing cannot affect it."""

    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be positive")
    normalized = tuple(
        row if isinstance(row, FrameIndexRecord) else FrameIndexRecord.from_dict(row)
        for row in records
    )
    if not normalized:
        raise ValueError("fault plan requires at least one indexed frame")
    rng = random.Random(int(seed))
    source_origin = normalized[0].monotonic_ms
    previous_virtual = 0
    previous_captured = source_origin - 1
    accumulated_pause = 0
    delivery_seq = 0
    entries: list[dict[str, object]] = []
    for record in normalized:
        jitter = rng.randint(-profile.jitter_ms, profile.jitter_ms) if profile.jitter_ms else 0
        pause_applied = bool(profile.pause_ms and rng.random() < profile.pause_probability)
        if pause_applied:
            accumulated_pause += profile.pause_ms
        raw_virtual = record.monotonic_ms - source_origin + jitter + accumulated_pause
        virtual_ms = max(previous_virtual, raw_virtual, 0)
        captured_ms = max(previous_captured + 1, source_origin + virtual_ms)
        previous_virtual = captured_ms - source_origin
        previous_captured = captured_ms
        dropped = rng.random() < profile.drop_probability
        duplicated = (not dropped) and rng.random() < profile.duplicate_probability
        deliveries: list[dict[str, object]] = []
        if not dropped:
            count = 2 if duplicated else 1
            for duplicate_index in range(count):
                delivery_seq += 1
                delivery_captured = captured_ms + duplicate_index
                previous_captured = max(previous_captured, delivery_captured)
                deliveries.append(
                    {
                        "delivery_seq": delivery_seq,
                        "duplicate_index": duplicate_index,
                        "captured_monotonic_ms": delivery_captured,
                        "planned_delivery_ms": (previous_virtual + duplicate_index) / time_scale,
                    }
                )
        entries.append(
            {
                "frame_index": record.frame_index,
                "source_monotonic_ms": record.monotonic_ms,
                "source_wall_time": record.wall_time,
                "indexed_dropped_before": record.dropped_before,
                "faults": {
                    "jitter": {"planned": profile.jitter_ms > 0, "applied": jitter != 0, "value_ms": jitter},
                    "drop": {"planned": profile.drop_probability > 0, "applied": dropped},
                    "duplicate": {"planned": profile.duplicate_probability > 0, "applied": duplicated},
                    "pause": {"planned": profile.pause_probability > 0 and profile.pause_ms > 0, "applied": pause_applied, "value_ms": profile.pause_ms if pause_applied else 0},
                },
                "deliveries": deliveries,
            }
        )
    transform = _transform_spec(profile, canvas_size)
    document: dict[str, object] = {
        "schema": "guandan.shadow-fault-plan/1",
        "seed": int(seed),
        "time_scale": float(time_scale),
        "profile": asdict(profile),
        "source_origin_monotonic_ms": source_origin,
        "source_frame_count": len(entries),
        "planned_delivery_count": delivery_seq,
        "planned_duration_ms": max(
            (float(delivery["planned_delivery_ms"]) for entry in entries for delivery in entry["deliveries"]),
            default=0.0,
        ),
        "transform": transform,
        "advisor_delay": {
            "planned": profile.advisor_delay_ms > 0,
            "delay_ms": profile.advisor_delay_ms,
        },
        "entries": entries,
    }
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FaultPlan(document, hashlib.sha256(payload).hexdigest())


def apply_spatial_fault(
    frame: np.ndarray,
    profile: FaultProfile,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply scale/offset on a fixed BGR uint8 canvas with constant padding."""

    image = np.asarray(frame)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("shadow frame must be grayscale, BGR, or BGRA")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    tx = (1.0 - profile.scale) * width / 2.0 + profile.offset_x
    ty = (1.0 - profile.scale) * height / 2.0 + profile.offset_y
    matrix = np.array([[profile.scale, 0.0, tx], [0.0, profile.scale, ty]], dtype=np.float64)
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return transformed, {
        "matrix": matrix.tolist(),
        "padding": "constant_bgr_black",
        "canvas_size": [width, height],
        "dtype": str(transformed.dtype),
        "channels": int(transformed.shape[2]),
    }


class ShadowLiveReplayRunner:
    def __init__(
        self,
        *,
        recognition_factory: Callable[[Path], Any] | None = None,
        advisor_factory: Callable[[Path, str], Any] | None = None,
        orchestrator_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        advisor_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.recognition_factory = recognition_factory or _recognizer_for_session
        self.advisor_factory = advisor_factory or _advisor_for_profile
        self.orchestrator_factory = orchestrator_factory or _make_orchestrator
        self.clock = clock
        self.sleep = sleep
        self.advisor_sleep = advisor_sleep

    def run(self, config: ShadowLiveReplayConfig) -> ShadowLiveReplayResult:
        session = Path(config.session).resolve()
        if not (session / "manifest.json").is_file():
            raise ValueError(f"not a recorded session: {session}")
        output_root = Path(config.output).resolve()
        sessions_root = session.parent
        if _is_relative_to(output_root, sessions_root):
            raise ValueError("shadow output must be outside the source sessions root")
        run_id = _safe_name(config.run_id or _new_run_id())
        run_directory = output_root / run_id
        if _is_relative_to(run_directory, sessions_root):
            raise ValueError("shadow output must be outside the source sessions root")
        run_directory.mkdir(parents=True, exist_ok=False)
        deliveries_path = run_directory / "deliveries.jsonl"
        analysis_path = run_directory / "analysis.jsonl"
        plan_path = run_directory / "fault_plan.json"
        before_path = run_directory / "source_snapshot_before.json"
        after_path = run_directory / "source_snapshot_after.json"
        deliveries_path.touch()
        analysis_path.touch()
        before = _source_snapshot(session)
        atomic_write_json(before_path, before)

        records = _selected_records(session, config)
        canvas_size = _video_canvas_size(session / "video" / "game.avi")
        plan = build_fault_plan(
            records,
            config.fault_profile,
            seed=config.seed,
            time_scale=config.time_scale,
            canvas_size=canvas_size,
        )
        atomic_write_json(plan_path, plan.to_dict())

        store = LiveSessionStore(run_directory, "runtime", session_id="shadow-live")
        store.start(
            {
                "mode": "shadow_live",
                "source_session": str(session),
                "fault_plan_sha256": plan.sha256,
                "time_scale": config.time_scale,
            }
        )
        recorder = InMemorySessionRecorder(store.directory)
        recognition = self.recognition_factory(session)
        advisor = self.advisor_factory(session.parent.parent.parent, session.parent.parent.name)
        if config.fault_profile.advisor_delay_ms:
            advisor = DelayedAdvisor(
                advisor,
                config.fault_profile.advisor_delay_ms,
                sleep=self.advisor_sleep,
            )
        orchestrator = self.orchestrator_factory(
            store=store,
            recorder=recorder,
            recognition=recognition,
            advisor=advisor,
        )
        initial = _initial_state(session)
        orchestrator.start(
            round_level=initial["round_level"],
            hand=initial["hand"],
            lead_player=initial["lead_player"],
            monotonic_ms=records[0].monotonic_ms,
        )

        pacer = AbsolutePacer(clock=self.clock, sleep=self.sleep)
        delivery_rows: dict[int, dict[str, object]] = {}
        plan_drop_rows: list[dict[str, object]] = []
        analysis_rows: list[dict[str, object]] = []
        errors: list[str] = []
        lock = threading.Lock()
        recorder_drops = 0
        source_consumed = 0
        producer_thread_id = threading.get_ident()

        def analyze(task: _AnalysisTask) -> dict[str, object]:
            started = self.clock()
            try:
                update = orchestrator.analyze_frame(
                    task.frame,
                    monotonic_ms=task.captured_monotonic_ms,
                    trace_context={
                        "shadow_live": True,
                        "delivery_seq": task.delivery_seq,
                        "source_frame_index": task.source_frame_index,
                        "source_monotonic_ms": task.source_monotonic_ms,
                        "planned_delivery_ms": task.planned_delivery_ms,
                        "captured_ms": task.captured_monotonic_ms,
                    },
                )
            except Exception as exc:
                raise _TaskFailure(task, exc) from exc
            ended = self.clock()
            events = tuple(getattr(update, "events", ()) or ())
            row = {
                "delivery_seq": task.delivery_seq,
                "source_frame_index": task.source_frame_index,
                "worker_thread_id": threading.get_ident(),
                "producer_thread_id": producer_thread_id,
                "queue_wait_ms": (started - task.submitted_wall_monotonic_ms) * 1000.0,
                "analyze_ms": (ended - started) * 1000.0,
                "end_to_end_ms": (ended - task.delivery_wall_monotonic_ms) * 1000.0,
                "analysis_started_wall_monotonic_ms": started * 1000.0,
                "analysis_ended_wall_monotonic_ms": ended * 1000.0,
                "state_revision": getattr(getattr(update, "snapshot", None), "revision", None),
                "status": getattr(update, "status", None),
                "event_ids": [getattr(event, "event_id", None) for event in events],
            }
            with lock:
                analysis_rows.append(row)
                delivery_rows[task.delivery_seq]["status"] = "analyzed"
            return row

        def failed(exc: Exception) -> None:
            with lock:
                errors.append(str(exc))
                if isinstance(exc, _TaskFailure):
                    delivery_rows[exc.task.delivery_seq]["status"] = "analysis_failed"
                    delivery_rows[exc.task.delivery_seq]["error"] = str(exc.cause)

        def discarded(task: object, reason: str) -> None:
            if not isinstance(task, _AnalysisTask):
                return
            with lock:
                row = delivery_rows[task.delivery_seq]
                row["status"] = reason
                row["discard_reason"] = reason

        worker = LatestOnlyWorker(analyze, on_error=failed, on_discard=discarded)
        worker.start()
        source = VideoReplaySource(
            session / "video" / "game.avi",
            session / "video" / "frame_index.jsonl",
        )
        frame_iterator = iter(source.frames(start_frame=records[0].frame_index))
        delivery_finished = False
        try:
            pacer.start()
            for entry in plan.entries:
                decode_started = self.clock()
                decoded_record, frame = next(frame_iterator)
                decode_ended = self.clock()
                if decoded_record.frame_index != int(entry["frame_index"]):
                    raise RuntimeError(
                        f"source frame mismatch: plan={entry['frame_index']}, decoded={decoded_record.frame_index}"
                    )
                source_consumed += 1
                deliveries = entry.get("deliveries", ())
                if not deliveries:
                    plan_drop_rows.append(
                        {
                            "kind": "source_frame",
                            "frame_index": decoded_record.frame_index,
                            "source_monotonic_ms": decoded_record.monotonic_ms,
                            "status": "fault_drop",
                            "faults": entry.get("faults", {}),
                            "decode_ms": (decode_ended - decode_started) * 1000.0,
                        }
                    )
                    continue
                for delivery in deliveries:
                    transform_started = self.clock()
                    transformed, transform = apply_spatial_fault(frame, config.fault_profile)
                    transform_ended = self.clock()
                    planned_ms = float(delivery["planned_delivery_ms"])
                    actual = pacer.wait_until(planned_ms)
                    pacer_epoch = (
                        pacer.started_at
                        if pacer.started_at is not None
                        else actual
                    )
                    delivery_seq = int(delivery["delivery_seq"])
                    captured_ms = int(delivery["captured_monotonic_ms"])
                    delivery_row: dict[str, object] = {
                        "kind": "delivery",
                        "delivery_seq": delivery_seq,
                        "duplicate_index": delivery["duplicate_index"],
                        "source_frame_index": decoded_record.frame_index,
                        "source_monotonic_ms": decoded_record.monotonic_ms,
                        "planned_delivery_ms": planned_ms,
                        "actual_wall_monotonic_ms": actual * 1000.0,
                        "actual_elapsed_ms": (actual - pacer_epoch) * 1000.0,
                        "delivery_lag_ms": (actual - pacer_epoch) * 1000.0 - planned_ms,
                        "captured_monotonic_ms": captured_ms,
                        "decode_ms": (decode_ended - decode_started) * 1000.0,
                        "transform_ms": (transform_ended - transform_started) * 1000.0,
                        "transform": transform,
                        "faults": entry.get("faults", {}),
                        "status": "recorded",
                        "recorded_before_submit": True,
                    }
                    warning = orchestrator.record_frame(
                        transformed,
                        monotonic_ms=captured_ms,
                        wall_time=datetime.now().astimezone().isoformat(),
                    )
                    if warning is not None:
                        recorder_drops += 1
                        delivery_row["recorder_warning"] = str(getattr(warning, "reason", warning))
                    submitted = self.clock()
                    delivery_row["submitted_wall_monotonic_ms"] = submitted * 1000.0
                    task = _AnalysisTask(
                        delivery_seq,
                        decoded_record.frame_index,
                        decoded_record.monotonic_ms,
                        planned_ms,
                        captured_ms,
                        submitted,
                        actual,
                        transformed,
                    )
                    with lock:
                        delivery_rows[delivery_seq] = delivery_row
                    preserve = bool(orchestrator.needs_first_action_frames)
                    delivery_row["preserved"] = preserve
                    worker.submit(task, preserve=preserve, max_preserved=8)
            delivery_finished = True
        except StopIteration:
            errors.append("video ended before the fault plan was consumed")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            close = getattr(frame_iterator, "close", None)
            if callable(close):
                close()

        analysis_drained = worker.wait_idle(config.drain_timeout_sec)
        worker_stopped = worker.stop(timeout=config.drain_timeout_sec)
        if not analysis_drained:
            errors.append("analysis worker drain timeout")
        if not worker_stopped:
            errors.append("analysis worker stop timeout")
        advice_drained = False
        finish_error: str | None = None
        final_snapshot: dict[str, object] = {}
        if worker_stopped:
            advice_drained = bool(
                orchestrator.wait_for_advice_idle(timeout=config.drain_timeout_sec)
            )
            if not advice_drained:
                errors.append("advice worker drain timeout")
            try:
                final_snapshot = dict(orchestrator.snapshot.semantic_dict())
                orchestrator.finish()
            except Exception as exc:
                finish_error = f"{type(exc).__name__}: {exc}"
                errors.append(f"finish failed: {finish_error}")

        worker_stats = worker.stats
        for row in plan_drop_rows:
            append_json_line(deliveries_path, row)
        for delivery_seq in sorted(delivery_rows):
            append_json_line(deliveries_path, delivery_rows[delivery_seq])
        for row in sorted(analysis_rows, key=lambda item: int(item["delivery_seq"])):
            append_json_line(analysis_path, row)

        runtime = store.directory
        advice_lifecycle = summarize_advice_lifecycle(runtime, advice_drained)
        inferred_advice_terminals = sum(
            bool(row.get("terminal_inferred"))
            or not bool(row.get("worker_start_explained"))
            for row in advice_lifecycle.get("requests", ())
        )
        if inferred_advice_terminals:
            errors.append(
                f"{inferred_advice_terminals} advice request(s) lack an explained worker start or persisted terminal status"
            )
        action_latency = _action_latency(session, runtime, records)
        baseline = _baseline_comparison(config.baseline, session, runtime, final_snapshot)
        business_signature = _business_signature(runtime, final_snapshot, advice_lifecycle)
        repeat = _repeat_comparison(config.compare_summary, plan.sha256, business_signature)
        after = _source_snapshot(session)
        atomic_write_json(after_path, after)
        planned_deliveries = int(plan.document["planned_delivery_count"])
        final_statuses = Counter(str(row.get("status", "unknown")) for row in delivery_rows.values())
        unattributed = sum(status in {"recorded", "submitted"} for status in final_statuses.elements())
        actual_elapsed = [float(row["actual_elapsed_ms"]) for row in delivery_rows.values()]
        delivery_lag = summarize_samples(
            [float(row["delivery_lag_ms"]) for row in delivery_rows.values()]
        )
        planned_duration = float(plan.document["planned_duration_ms"])
        actual_duration = (
            max(actual_elapsed) - min(actual_elapsed) if actual_elapsed else 0.0
        )
        duration_tolerance = max(1_000.0, planned_duration * 0.02)
        temporal_profile = config.fault_profile
        strict_no_fault_timing_required = bool(
            config.time_scale == 1.0
            and temporal_profile.jitter_ms == 0
            and temporal_profile.drop_probability == 0
            and temporal_profile.duplicate_probability == 0
            and temporal_profile.pause_probability == 0
        )
        duration_within_tolerance = abs(actual_duration - planned_duration) <= duration_tolerance
        lag_p95 = delivery_lag.get("p95")
        strict_no_fault_timing_pass = bool(
            duration_within_tolerance
            and lag_p95 is not None
            and float(lag_p95) <= 200.0
        )
        if strict_no_fault_timing_required and not strict_no_fault_timing_pass:
            errors.append("strict 1x no-fault timing gate failed")
        expected_hash_matches = (
            config.expected_plan_sha256 is None
            or config.expected_plan_sha256 == plan.sha256
        )
        summary: dict[str, object] = {
            "schema": "guandan.shadow-live-summary/1",
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "source_session": str(session),
            "session_id": _session_id(session),
            "fault_plan_sha256": plan.sha256,
            "expected_plan_sha256": config.expected_plan_sha256,
            "expected_plan_hash_matches": expected_hash_matches,
            "time_scale": config.time_scale,
            "seed": config.seed,
            "fragment": {
                "start_frame": config.start_frame,
                "end_frame": config.end_frame,
                "max_frames": config.max_frames,
            },
            "plan": {
                "source_frames": len(plan.entries),
                "source_frames_consumed": source_consumed,
                "fully_consumed": delivery_finished and source_consumed == len(plan.entries),
                "planned_deliveries": planned_deliveries,
                "planned_duration_ms": plan.document["planned_duration_ms"],
            },
            "clocks": {
                "source": "frame_index.monotonic_ms",
                "planned": "absolute_epoch_plus_planned_delivery_ms",
                "actual_wall": "monotonic_clock",
                "captured": "seeded_monotonic_fault_stream",
            },
            "delivery": {
                "actual_deliveries": len(delivery_rows),
                "statuses": dict(sorted(final_statuses.items())),
                "unattributed": unattributed,
                "lag_ms": delivery_lag,
            },
            "schedule": {
                "planned_duration_ms": planned_duration,
                "actual_duration_ms": actual_duration,
                "deviation_ms": actual_duration - planned_duration,
                "duration_error_ms": actual_duration - planned_duration,
                "duration_tolerance_ms": duration_tolerance,
                "duration_within_tolerance": duration_within_tolerance,
                "delivery_lag_p95_ms": lag_p95,
                "strict_no_fault_timing_required": strict_no_fault_timing_required,
                "strict_no_fault_timing_pass": (
                    strict_no_fault_timing_pass
                    if strict_no_fault_timing_required
                    else None
                ),
            },
            "drops": {
                "source_indexed": sum(record.dropped_before for record in records),
                "fault": len(plan_drop_rows),
                "worker_latest_replaced": worker_stats["latest_replaced"],
                "worker_preserved_evicted": worker_stats["preserved_evicted"],
                "worker_stop_discarded": worker_stats["stop_discarded"],
                "recorder": recorder_drops,
            },
            "faults": summarize_fault_applications(plan, advice_lifecycle),
            "worker": {**worker_stats, "analysis_drained": analysis_drained, "stopped": worker_stopped},
            "advice_drained": advice_drained,
            "advice": advice_lifecycle,
            "performance": {
                "decode_ms": summarize_samples([float(row.get("decode_ms", 0.0)) for row in delivery_rows.values()]),
                "transform_ms": summarize_samples([float(row.get("transform_ms", 0.0)) for row in delivery_rows.values()]),
                "queue_wait_ms": summarize_samples([float(row.get("queue_wait_ms", 0.0)) for row in analysis_rows]),
                "analyze_ms": summarize_samples([float(row.get("analyze_ms", 0.0)) for row in analysis_rows]),
                "end_to_end_ms": summarize_samples([float(row.get("end_to_end_ms", 0.0)) for row in analysis_rows]),
            },
            "action_confirmation_latency": action_latency,
            "baseline_comparison": baseline,
            "repeat_comparison": repeat,
            "business_signature": business_signature,
            "final_snapshot": final_snapshot,
            "runtime_directory": str(runtime),
            "source_integrity": {"unchanged": before == after, "before": str(before_path), "after": str(after_path)},
            "video_warnings": [{"reason": warning.reason, "details": warning.details} for warning in source.warnings],
            "errors": errors,
            "finish_error": finish_error,
        }
        summary_path = run_directory / "summary.json"
        summary_md = run_directory / "summary.md"
        atomic_write_json(summary_path, summary)
        _write_summary_markdown(summary_md, summary)
        observations_path = (
            runtime / "observations.jsonl.gz"
            if (runtime / "observations.jsonl.gz").is_file()
            else runtime / "observations.jsonl.part"
        )
        required = (
            summary_path,
            plan_path,
            deliveries_path,
            analysis_path,
            summary_md,
            runtime / "manifest.json",
            runtime / "timeline.jsonl",
            runtime / "advice.jsonl",
            runtime / "decisions.jsonl",
            runtime / "recognition_trace.jsonl",
            observations_path,
        )
        execution_ok = bool(
            not errors
            and summary["plan"]["fully_consumed"]
            and analysis_drained
            and worker_stopped
            and advice_drained
            and inferred_advice_terminals == 0
            and not unattributed
            and expected_hash_matches
            and before == after
            and all(path.is_file() for path in required)
        )
        summary["execution_status"] = "completed" if execution_ok else "error"
        summary["required_artifacts"] = {str(path): path.is_file() for path in required}
        atomic_write_json(summary_path, summary)
        _write_summary_markdown(summary_md, summary)
        return ShadowLiveReplayResult(run_directory, summary_path, plan_path, execution_ok, summary)


def summarize_samples(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not samples:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(samples),
        "min": samples[0],
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "max": samples[-1],
    }


def summarize_fault_applications(
    plan: FaultPlan,
    advice: dict[str, object],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in ("jitter", "drop", "duplicate", "pause"):
        decisions = [entry.get("faults", {}).get(name, {}) for entry in plan.entries]
        planned = sum(bool(item.get("planned")) for item in decisions)
        applied = sum(bool(item.get("applied")) for item in decisions)
        result[name] = {"planned": planned, "applied": applied, "skipped": max(0, planned - applied)}
    delivery_count = int(plan.document.get("planned_delivery_count", 0))
    transform = plan.document.get("transform", {})
    transform_planned = delivery_count if transform.get("planned") else 0
    result["offset_scale"] = {
        "planned": transform_planned,
        "applied": transform_planned,
        "skipped": 0,
    }
    delay = plan.document.get("advisor_delay", {})
    request_count = int(advice.get("requested", 0))
    delay_planned = request_count if delay.get("planned") else 0
    delay_applied = (
        min(delay_planned, int(advice.get("worker_started", 0)))
        if delay.get("planned")
        else 0
    )
    result["advisor_delay"] = {
        "planned": delay_planned,
        "applied": delay_applied,
        "skipped": max(0, delay_planned - delay_applied),
    }
    return result


def _percentile(samples: list[float], quantile: float) -> float:
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def _selected_records(session: Path, config: ShadowLiveReplayConfig) -> tuple[FrameIndexRecord, ...]:
    records = tuple(
        FrameIndexRecord.from_dict(raw)
        for raw in read_json_lines(session / "video" / "frame_index.jsonl")
    )
    selected = tuple(
        record
        for record in records
        if (config.start_frame is None or record.frame_index >= config.start_frame)
        and (config.end_frame is None or record.frame_index <= config.end_frame)
    )
    if config.max_frames is not None:
        selected = selected[: config.max_frames]
    if not selected:
        raise ValueError("selected shadow fragment contains no indexed frames")
    return selected


def _transform_spec(profile: FaultProfile, canvas_size: tuple[int, int] | None) -> dict[str, object]:
    width, height = canvas_size or (0, 0)
    tx = (1.0 - profile.scale) * width / 2.0 + profile.offset_x
    ty = (1.0 - profile.scale) * height / 2.0 + profile.offset_y
    return {
        "planned": profile.scale != 1.0 or profile.offset_x != 0 or profile.offset_y != 0,
        "scale": profile.scale,
        "offset": [profile.offset_x, profile.offset_y],
        "matrix": [[profile.scale, 0.0, tx], [0.0, profile.scale, ty]],
        "canvas_size": [width, height],
        "padding": "constant_bgr_black",
        "output_dtype": "uint8",
        "output_channels": 3,
    }


def _video_canvas_size(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open source video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("source video has invalid dimensions")
    return width, height


def _initial_state(session: Path) -> dict[str, object]:
    for event in read_json_lines(session / "timeline.jsonl"):
        if event.get("event_type") != "initial_state_confirmed":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        lead = payload.get("lead_player", event.get("actor"))
        return {
            "round_level": str(payload.get("round_level", "")),
            "hand": tuple(str(card) for card in payload.get("hand", ())),
            "lead_player": lead if lead in _SEAT_SET else None,
        }
    raise ValueError("source timeline lacks initial_state_confirmed")


_SEAT_SET = frozenset({"self", "right", "opposite", "left"})


def _recognizer_for_session(session: Path) -> ScreenshotRecognitionService:
    profile = session.parent.parent
    return ScreenshotRecognitionService(
        AnnotationService(profile.parent, profile.name),
        TemplateService(profile.parent, profile.name),
    )


def _advisor_for_profile(profiles_root: Path, profile_name: str) -> Any:
    return build_advisor(
        "fabledan",
        profiles_root=profiles_root,
        profile_name=profile_name,
        fabledan_diagnostics="full",
    )


def _make_orchestrator(*, store: Any, recorder: Any, recognition: Any, advisor: Any) -> LiveOrchestrator:
    return LiveOrchestrator(
        reducer=LiveReducer(store.session_id),
        store=store,
        recorder=recorder,
        recognition_service=recognition,
        advisor=advisor,
        recognition_strategy="two_valid_streak",
        minimum_free_bytes=0,
    )


def _source_snapshot(session: Path) -> dict[str, dict[str, object]]:
    paths = [
        session / "manifest.json",
        session / "timeline.jsonl",
        session / "truth_log.json",
        session / "recognition_trace.jsonl",
        session / "video" / "game.avi",
        session / "video" / "frame_index.jsonl",
    ]
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        if not path.is_file():
            continue
        stat = path.stat()
        result[path.relative_to(session).as_posix()] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(path),
        }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_advice_lifecycle(runtime: Path, drained: bool) -> dict[str, object]:
    records = read_json_lines(runtime / "advice.jsonl")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in records:
        request_id = str(row.get("request_id", ""))
        if request_id:
            grouped.setdefault(request_id, []).append(row)
    timeline = read_json_lines(runtime / "timeline.jsonl")
    virtual_by_request: dict[str, list[dict[str, object]]] = {}
    for event in timeline:
        if not str(event.get("event_type", "")).startswith("advice_"):
            continue
        payload = event.get("payload", {})
        payload = payload if isinstance(payload, dict) else {}
        request_id = str(payload.get("request_id", ""))
        if request_id:
            virtual_by_request.setdefault(request_id, []).append(event)
    requests: list[dict[str, object]] = []
    terminal_counts = Counter()
    non_request_terminal_counts = Counter()
    non_request_groups: list[dict[str, object]] = []
    for request_id, rows in sorted(grouped.items()):
        requested = next((row for row in rows if row.get("status") == "requested"), None)
        if requested is None:
            statuses = [str(row.get("status", "unknown")) for row in rows]
            non_request_terminal_counts.update(
                status for status in statuses if status in _TERMINAL_ADVICE
            )
            non_request_groups.append(
                {
                    "request_id": request_id,
                    "statuses": statuses,
                    "reason": "no persisted status=requested record",
                }
            )
            continue
        worker_started = next((row for row in rows if row.get("status") == "worker_started"), None)
        terminal = next((row for row in reversed(rows) if row.get("status") in _TERMINAL_ADVICE), None)
        terminal_status = str(terminal.get("status")) if terminal else ("cancelled" if not drained else "timeout")
        prestart_discard = bool(
            worker_started is None
            and terminal is not None
            and terminal_status in {"stale", "cancelled"}
            and str(terminal.get("discard_reason", ""))
            in {"latest_replaced", "preserved_evicted", "stop_discarded"}
        )
        worker_start_explained = worker_started is not None or prestart_discard
        terminal_counts[terminal_status] += 1
        wall_latency = _wall_latency_ms(requested, terminal)
        request_to_worker_ms = _wall_latency_ms(requested, worker_started)
        worker_duration_ms = _wall_latency_ms(worker_started, terminal) if worker_started else None
        virtual_events = virtual_by_request.get(request_id, ())
        virtual_latency = None
        if virtual_events:
            first_ms = _int_or_none(virtual_events[0].get("monotonic_ms"))
            last_ms = _int_or_none(virtual_events[-1].get("monotonic_ms"))
            if first_ms is not None and last_ms is not None:
                virtual_latency = last_ms - first_ms
        requests.append(
            {
                "request_id": request_id,
                "turn_id": requested.get("turn_id"),
                "state_revision": requested.get("state_revision"),
                "terminal_status": terminal_status,
                "wall_latency_ms": wall_latency,
                "request_to_worker_start_ms": request_to_worker_ms,
                "worker_start_wall_latency_ms": request_to_worker_ms,
                "worker_duration_ms": worker_duration_ms,
                "worker_started": worker_started is not None,
                "worker_start_explained": worker_start_explained,
                "worker_start_skip_reason": (
                    str(terminal.get("discard_reason"))
                    if prestart_discard and terminal is not None
                    else None
                ),
                "virtual_latency_ms": virtual_latency,
                "visible": any(
                    event.get("event_type") == "advice_visible"
                    for event in virtual_events
                ),
                "terminal_inferred": terminal is None,
                "terminal_reason": None if terminal else "advice drain incomplete" if not drained else "terminal record missing after drain",
            }
        )
    return {
        "requested": len(requests),
        "worker_started": sum(bool(row["worker_started"]) for row in requests),
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "withheld_without_request": int(non_request_terminal_counts.get("withheld", 0)),
        "non_request_terminal_counts": dict(sorted(non_request_terminal_counts.items())),
        "non_request_groups": non_request_groups,
        "wall_latency_ms": summarize_samples(row["wall_latency_ms"] for row in requests if row["wall_latency_ms"] is not None),
        "request_to_worker_start_ms": summarize_samples(row["request_to_worker_start_ms"] for row in requests if row["request_to_worker_start_ms"] is not None),
        "worker_start_wall_latency_ms": summarize_samples(row["worker_start_wall_latency_ms"] for row in requests if row["worker_start_wall_latency_ms"] is not None),
        "worker_duration_ms": summarize_samples(row["worker_duration_ms"] for row in requests if row["worker_duration_ms"] is not None),
        "virtual_latency_ms": summarize_samples(row["virtual_latency_ms"] for row in requests if row["virtual_latency_ms"] is not None),
        "requests": requests,
    }


def _wall_latency_ms(start: dict[str, object], end: dict[str, object] | None) -> float | None:
    if end is None:
        return None
    try:
        left = datetime.fromisoformat(str(start.get("wall_time")))
        right = datetime.fromisoformat(str(end.get("wall_time")))
    except (TypeError, ValueError):
        return None
    return (right - left).total_seconds() * 1000.0


def _action_latency(
    session: Path,
    runtime: Path,
    records: tuple[FrameIndexRecord, ...],
) -> dict[str, object]:
    truth_path = session / "truth_log.json"
    if not truth_path.is_file():
        return {"available": False, "reason": "canonical truth_log.json is absent", "rows": [], "latency_ms": summarize_samples(())}
    try:
        truth = load_truth_log(truth_path, session_id=_session_id(session))
    except Exception as exc:
        return {"available": False, "reason": f"truth load failed: {exc}", "rows": [], "latency_ms": summarize_samples(())}
    source_ms = {record.frame_index: record.monotonic_ms for record in records}
    actual = [row for row in read_json_lines(runtime / "timeline.jsonl") if row.get("event_type") in _ACTION_TYPES]
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    for index, turn in enumerate(truth.turns):
        reference_ms = turn.monotonic_ms
        source = "truth_monotonic_ms"
        if reference_ms is None and turn.frame_index is not None:
            reference_ms = source_ms.get(turn.frame_index)
            source = "truth_frame_index_to_source_monotonic"
        action = actual[index] if index < len(actual) else None
        actual_ms = _int_or_none(action.get("monotonic_ms")) if action else None
        latency = actual_ms - reference_ms if actual_ms is not None and reference_ms is not None else None
        if latency is not None:
            latencies.append(float(latency))
        rows.append(
            {
                "turn_id": turn.index,
                "reference_monotonic_ms": reference_ms,
                "reference_source": source if reference_ms is not None else "N/A",
                "actual_event_id": action.get("event_id") if action else None,
                "actual_monotonic_ms": actual_ms,
                "latency_ms": latency,
                "status": "measured" if latency is not None else "N/A",
            }
        )
    return {"available": True, "rows": rows, "latency_ms": summarize_samples(latencies)}


def _baseline_comparison(
    baseline_path: Path | None,
    session: Path,
    runtime: Path,
    final_snapshot: dict[str, object],
) -> dict[str, object]:
    if baseline_path is None:
        return {"available": False, "reason": "no phase-one baseline supplied"}
    try:
        raw = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "reason": f"baseline read failed: {exc}"}
    candidates = raw.get("sessions", ()) if isinstance(raw, dict) else ()
    session_id = _session_id(session)
    baseline = next((row for row in candidates if str(row.get("session_id")) == session_id), None)
    if baseline is None and isinstance(raw, dict) and str(raw.get("session_id")) == session_id:
        baseline = raw
    if not isinstance(baseline, dict):
        return {"available": False, "reason": "session not found in phase-one baseline"}
    expected_actions = baseline.get("visual", {}).get("actions", {}).get("rows", ())
    actual_actions = [
        {
            "actor": row.get("actor"),
            "is_pass": bool(row.get("payload", {}).get("is_pass", row.get("event_type") == "player_passed")),
            "cards": sorted(str(card) for card in row.get("payload", {}).get("cards", ())),
            "event_id": row.get("event_id"),
            "turn_id": row.get("turn_id"),
        }
        for row in read_json_lines(runtime / "timeline.jsonl")
        if row.get("event_type") in _ACTION_TYPES and isinstance(row.get("payload", {}), dict)
    ]
    first = None
    for index in range(max(len(expected_actions), len(actual_actions))):
        expected = expected_actions[index] if index < len(expected_actions) else None
        actual = actual_actions[index] if index < len(actual_actions) else None
        if expected is None or actual is None or (
            expected.get("actor"), bool(expected.get("is_pass")), sorted(expected.get("cards", ()))
        ) != (actual.get("actor"), bool(actual.get("is_pass")), sorted(actual.get("cards", ()))):
            first = {"position": index + 1, "expected": expected, "actual": actual}
            break
    return {
        "available": True,
        "expected_action_count": len(expected_actions),
        "actual_action_count": len(actual_actions),
        "actions_identical": first is None,
        "first_divergence": first,
        "baseline_final_snapshot": baseline.get("final_snapshot"),
        "actual_final_snapshot": final_snapshot,
        "fabledan_baseline": baseline.get("fabledan"),
    }


def _business_signature(runtime: Path, snapshot: dict[str, object], advice: dict[str, object]) -> str:
    actions = [
        row for row in read_json_lines(runtime / "timeline.jsonl")
        if row.get("event_type") in _ACTION_TYPES
    ]
    payload = {"actions": actions, "final_snapshot": snapshot, "advice_terminal_counts": advice.get("terminal_counts", {})}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _repeat_comparison(path: Path | None, plan_sha256: str, business_signature: str) -> dict[str, object]:
    if path is None:
        return {"available": False, "reason": "no prior summary supplied"}
    try:
        previous = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "reason": f"prior summary read failed: {exc}"}
    return {
        **compare_repeat_results(previous, plan_sha256, business_signature),
        "previous_summary": str(path),
    }


def compare_repeat_results(
    previous_summary: dict[str, object],
    plan_sha256: str,
    business_signature: str,
) -> dict[str, object]:
    same_plan = previous_summary.get("fault_plan_sha256") == plan_sha256
    same_business = previous_summary.get("business_signature") == business_signature
    return {
        "available": True,
        "same_plan": same_plan,
        "same_business_result": same_business,
        "concurrency_nondeterminism": bool(same_plan and not same_business),
    }


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    drops = summary.get("drops", {})
    lines = [
        "# Shadow Live 1× 报告",
        "",
        f"- 执行状态：{summary.get('execution_status', 'pending')}",
        f"- 源会话：`{summary.get('session_id')}`",
        f"- 时间倍率：{summary.get('time_scale')}×",
        f"- Fault plan：`{summary.get('fault_plan_sha256')}`",
        f"- 计划帧消费：{summary.get('plan', {}).get('source_frames_consumed')}/{summary.get('plan', {}).get('source_frames')}",
        f"- 丢帧分类：{json.dumps(drops, ensure_ascii=False)}",
        f"- 识别 worker drain：{summary.get('worker', {}).get('analysis_drained')}",
        f"- FableDan drain：{summary.get('advice_drained')}",
        "",
        "## 错误",
        "",
    ]
    errors = summary.get("errors", ())
    lines.extend(f"- {error}" for error in errors)
    if not errors:
        lines.append("- 无")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _session_id(session: Path) -> str:
    try:
        raw = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    return str(raw.get("session_id", session.name))


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._")
    if not cleaned:
        raise ValueError("run_id is invalid")
    return cleaned


def _new_run_id() -> str:
    stamp = datetime.now().astimezone().strftime("shadow_%Y%m%dT%H%M%S")
    return f"{stamp}_{uuid4().hex[:10]}"


__all__ = [
    "AbsolutePacer",
    "DelayedAdvisor",
    "FaultPlan",
    "FaultProfile",
    "ShadowLiveReplayConfig",
    "ShadowLiveReplayResult",
    "ShadowLiveReplayRunner",
    "apply_spatial_fault",
    "build_fault_plan",
    "compare_repeat_results",
    "summarize_advice_lifecycle",
    "summarize_fault_applications",
    "summarize_samples",
]
