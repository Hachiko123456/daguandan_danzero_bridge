"""Read-only recorded replay through the production live-v2 composition."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from ..advisor_strategy import load_profile_advisor_strategy
from ..domain.frame import FrameEnvelope
from ..infrastructure.live_session import build_session_manifest
from ..infrastructure.live_v2_composition import build_production_live_v2_runtime
from ..live.replay import (
    ReplayComparison,
    ReplayWarning,
    VisualPipelineReplayResult,
    VideoReplaySource,
)
from ..live.frame_pipeline import analyze_frame_envelope
from ..live.recorder import InMemorySessionRecorder
from ..live.session_store import LiveSessionStore, read_json_lines
from ..opening_gate import (
    OpeningActionSeed, OpeningSessionSeed, OpeningTracker, opening_semantic_key,
)
from ..live.turns import TURN_ORDER, next_active_seat
from ..storage import append_json_line, atomic_write_json


RuntimeBuilder = Callable[..., Any]


def _wait_for_recorded_deadline(
    *,
    captured_monotonic_ms: int,
    source_started_ms: int,
    playback_started_at: float,
    stop_requested: Callable[[], bool] | None,
) -> bool:
    """Pace one frame against the recording clock with cancellable waits."""

    source_elapsed = (
        max(0, int(captured_monotonic_ms) - int(source_started_ms)) / 1000
    )
    deadline = float(playback_started_at) + source_elapsed
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        if stop_requested is not None and stop_requested():
            return False
        time.sleep(min(0.05, remaining))


def replay_video_through_production_live_v2(
    session: Path,
    recognition_service: Any,
    *,
    profile_root: Path,
    output_root: Path,
    on_progress: Callable[[int, int, int], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    runtime_builder: RuntimeBuilder | None = None,
    advisor_backend: str | None = None,
    advisor: Any | None = None,
    vision_delivery: Literal["latest", "synchronous"] = "synchronous",
    pace_to_recording_timestamps: bool = False,
) -> VisualPipelineReplayResult:
    """Replay AVI frames through the production LiveV2 runtime.

    TruthLog is deliberately absent from this function.  The opening level,
    hand and first player come from the same recognition service and
    ``OpeningTracker`` used by the live controller.  Callers compare the
    resulting artifacts with TruthLog only after this function returns.
    When ``pace_to_recording_timestamps`` is enabled, delivery waits against
    the recorded capture clock so latest-only scheduling is tested at live
    cadence instead of being flooded by a zero-delay replay loop.
    """

    del advisor
    session = Path(session)
    profile_root = Path(profile_root).resolve()
    runtime_builder = runtime_builder or build_production_live_v2_runtime
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "visual_replay.jsonl"
    comparison_path = output_root / "visual_replay_comparison.json"
    report_path = output_root / "visual_replay_report.txt"
    for path in (output, comparison_path, report_path):
        path.unlink(missing_ok=True)

    source = VideoReplaySource(
        session / "video" / "game.avi",
        session / "video" / "frame_index.jsonl",
    )
    total = source.indexed_frame_count
    tracker = OpeningTracker()
    opening_reads: list[dict[str, object]] = []
    runtime = None
    store = None
    recorder = None
    runtime_directory: Path | None = None
    temp_runtime_root: tempfile.TemporaryDirectory[str] | None = None
    runtime_started = False
    runtime_finished = False
    replay_cancelled = False
    frames = source.envelopes(capture_generation=1)
    source_started_ms: int | None = None
    playback_started_at: float | None = None
    runtime_start_frame: int | None = None
    runtime_start_reason = "waiting_table"
    replay_opening_candidate: OpeningSessionSeed | None = None
    replay_opening_candidate_count = 0
    frame_count = 0
    last_envelope: FrameEnvelope | None = None
    advice_statuses: Counter[str] = Counter()
    warnings: list[ReplayWarning] = []
    actual_events: tuple[Any, ...] = ()
    final_current_player: Any = None

    try:
        for envelope in frames:
            if stop_requested is not None and stop_requested():
                warnings.append(ReplayWarning("stopped", "recorded replay was cancelled"))
                replay_cancelled = True
                break
            if pace_to_recording_timestamps:
                if source_started_ms is None:
                    source_started_ms = int(envelope.captured_monotonic_ms)
                    playback_started_at = time.monotonic()
                assert playback_started_at is not None
                if not _wait_for_recorded_deadline(
                    captured_monotonic_ms=int(envelope.captured_monotonic_ms),
                    source_started_ms=source_started_ms,
                    playback_started_at=playback_started_at,
                    stop_requested=stop_requested,
                ):
                    warnings.append(ReplayWarning("stopped", "recorded replay was cancelled"))
                    replay_cancelled = True
                    break

            last_envelope = envelope
            frame_count += 1
            frame_index = int(envelope.frame_index or frame_count - 1)
            if on_progress is not None:
                on_progress(frame_count, total, frame_index)

            if not runtime_started:
                try:
                    page = recognition_service.recognize_listening_page(envelope.image)
                except Exception as exc:
                    runtime_start_reason = f"page_probe_failed:{type(exc).__name__}"
                    opening_reads.append({
                        "frame_index": frame_index,
                        "stage": "error",
                        "reason": runtime_start_reason,
                    })
                    continue
                page_row = {
                    "frame_index": frame_index,
                    "stage": str(getattr(page, "stage", "unknown")),
                    "anchor_score": float(getattr(page, "anchor_score", 0.0) or 0.0),
                    "table_anchor_1_score": getattr(page, "table_anchor_1_score", None),
                    "table_anchor_2_score": getattr(page, "table_anchor_2_score", None),
                    "game_logo_anchor_score": getattr(page, "game_logo_anchor_score", None),
                    "buttons": list(getattr(page, "buttons", ()) or ()),
                }
                if str(getattr(page, "stage", "unknown")) != "table":
                    opening_reads.append(page_row)
                    if str(getattr(page, "stage", "unknown")) == "settlement":
                        tracker.reset()
                        runtime_start_reason = "settlement_screen"
                    else:
                        tracker.discard_candidates()
                        runtime_start_reason = "waiting_table"
                    _append_frame(output, envelope, None, runtime_start_reason)
                    continue
                try:
                    recognized = recognition_service.recognize(
                        envelope.image, allow_unknown_suit=True
                    )
                    evaluation = tracker.observe(
                        recognized,
                        anchor_score=float(getattr(page, "anchor_score", 0.0) or 0.0),
                        generation=1,
                        monotonic_ms=envelope.captured_monotonic_ms,
                        observation_id=envelope.evidence_frame_id or frame_index,
                    )
                except Exception as exc:
                    runtime_start_reason = f"opening_probe_failed:{type(exc).__name__}"
                    page_row["reason"] = runtime_start_reason
                    opening_reads.append(page_row)
                    _append_frame(output, envelope, None, runtime_start_reason)
                    continue
                seed = evaluation.seed
                recognized_level, recognized_hand = _opening_recognition_fields(recognized)
                replay_candidate = _replay_confirmed_opening_seed(
                    recognized, round_level=recognized_level, hand=recognized_hand
                )
                if replay_candidate is not None:
                    candidate_key = opening_semantic_key(replay_candidate)
                    prior_key = (
                        opening_semantic_key(replay_opening_candidate)
                        if replay_opening_candidate is not None else None
                    )
                    replay_opening_candidate_count = (
                        replay_opening_candidate_count + 1
                        if candidate_key == prior_key else 1
                    )
                    replay_opening_candidate = replay_candidate
                else:
                    replay_opening_candidate = None
                    replay_opening_candidate_count = 0
                if (
                    (not evaluation.ready or seed is None)
                    and replay_opening_candidate is not None
                    and replay_opening_candidate_count >= 2
                ):
                    # Recorded replay may carry a real opening event with an
                    # unresolved suit (for example ``7?`` plus suit_options).
                    # The generic gate intentionally remains conservative for
                    # live capture, but replay must not discard this confirmed
                    # event: two identical frames are the replay stability
                    # proof and the seed remains auditable.
                    seed = replay_opening_candidate
                    runtime_start_reason = "ready"
                    evaluation_ready = True
                else:
                    evaluation_ready = bool(evaluation.ready)
                page_row.update({
                    "reason": evaluation.reason,
                    # Persist the actual production recognition result.  The
                    # post-replay TruthLog comparison needs the complete hand,
                    # not only a count, and TruthLog is intentionally not
                    # involved in this opening read.
                    "round_level": recognized_level,
                    "hand": recognized_hand,
                    "hand_count": len(recognized_hand),
                    "lead_player": getattr(recognized, "lead_player", None),
                    "candidate_ready": bool(evaluation_ready and seed is not None),
                })
                opening_reads.append(page_row)
                runtime_start_reason = evaluation.reason
                if not evaluation_ready or seed is None:
                    _append_frame(output, envelope, None, runtime_start_reason)
                    continue
                if not seed.round_level or len(seed.hand) != 27:
                    runtime_start_reason = "opening_not_observable"
                    _append_frame(output, envelope, None, runtime_start_reason)
                    continue
                temp_runtime_root = tempfile.TemporaryDirectory(prefix="daguandan-live-v2-replay-")
                root = Path(temp_runtime_root.name)
                store = LiveSessionStore(root, "replay", session_id="visual-replay")
                manifest = build_session_manifest(
                    profile_root / "profile.json",
                    profile_root / "templates_config.json",
                )
                manifest.update({
                    "runtime": "live_v2",
                    "mode": "recorded_visual_replay",
                    "source_session": str(session),
                    "source_video": str(session / "video" / "game.avi"),
                    "source_frame_index": str(session / "video" / "frame_index.jsonl"),
                    "visual_truth_source": "recognition_only_until_replay_end",
                    "legacy_orchestrator_used": False,
                })
                store.start(manifest)
                # The AVI is already the immutable input.  Use the production
                # recorder port without opening a second video writer; this
                # keeps the replay read-only and bounds resource usage while
                # still exercising the runtime's record_frame lifecycle.
                recorder = InMemorySessionRecorder(store.directory)
                backend = advisor_backend or load_profile_advisor_strategy(
                    profile_root.parent, profile_root.name
                )
                runtime = runtime_builder(
                    store=store,
                    recorder=recorder,
                    recognizer=recognition_service,
                    profiles_root=profile_root.parent,
                    profile_name=profile_root.name,
                    advisor_backend=backend,
                    on_update=None,
                    vision_delivery=vision_delivery,
                )
                update = _start_runtime_from_opening_seed(
                    runtime,
                    seed,
                    envelope=envelope,
                    capture_generation=1,
                )
                runtime_started = True
                runtime_start_frame = frame_index
                runtime_start_reason = "ready"
                _append_frame(output, envelope, runtime, "visual", update)
                continue

            update = _record_and_analyze_frame(
                runtime,
                envelope,
                trace_context={"replay_mode": "sequential_every_frame"},
            )
            _append_frame(output, envelope, runtime, "visual", update)

        if runtime is None:
            status = "blocked"
            reason = "opening_not_observable"
        else:
            try:
                runtime.wait_for_advice_idle(timeout=60.0)
            except Exception as exc:
                warnings.append(ReplayWarning("advice_wait_failed", f"{type(exc).__name__}: {exc}"))
            status_before_finish = str(runtime.status)
            final_snapshot = None
            try:
                final_snapshot = runtime.snapshot
            except Exception:
                final_snapshot = None
            runtime.finish()
            runtime_finished = str(runtime.status) == "sealed"
            current_player = getattr(final_snapshot, "current_player", None)
            final_current_player = current_player
            listener_terminal = current_player is None
            if replay_cancelled:
                status = "incomplete"
                reason = "replay_cancelled"
            elif status_before_finish == "review_required":
                status = "review_required"
                reason = "runtime_review_required"
            elif not frame_count == total:
                status = "incomplete"
                reason = f"frame_replay_incomplete:{frame_count}/{total}"
            elif not listener_terminal:
                status = "incomplete"
                reason = f"listener_not_terminal:current_player={current_player}"
            elif runtime.status == "sealed":
                status = "complete"
                # Sealing is storage lifecycle only; this explicit reason is
                # emitted only after the pre-finish snapshot proved terminal.
                reason = "listener_terminal"
            else:
                status = "incomplete"
                reason = status_before_finish
            actual_events = _timeline_events(store.timeline_path if store else None)
            if store is not None:
                advice_statuses.update(
                    str(item.get("status", "unknown"))
                    for item in read_json_lines(store.advice_path)
                )
            if store is not None:
                runtime_directory = output_root / "runtime"
                runtime_directory.mkdir(parents=True, exist_ok=True)
                for name in (
                    "manifest.json", "timeline.jsonl", "timeline.md", "advice.jsonl",
                    "decisions.jsonl", "recognition_trace.jsonl",
                    "observations.jsonl.part", "observations.jsonl.gz",
                ):
                    source_path = store.directory / name
                    if source_path.is_file():
                        target_path = runtime_directory / name
                        shutil.copy2(source_path, target_path)
    except Exception as exc:
        warnings.append(ReplayWarning("runtime_error", f"{type(exc).__name__}: {exc}"))
        status, reason = "blocked", f"runtime_error:{type(exc).__name__}"
        if runtime is not None and str(runtime.status) != "sealed":
            try:
                runtime.finish()
            except Exception:
                pass
        if store is not None:
            actual_events = _timeline_events(store.timeline_path)
    finally:
        try:
            frames.close()
        except Exception:
            pass
        if last_envelope is not None and frame_count != total:
            warnings.append(ReplayWarning("partial_replay", f"processed {frame_count}/{total} frames"))
        if runtime is not None and not runtime_finished and str(getattr(runtime, "status", "")) != "sealed":
            try:
                runtime.finish()
            except Exception:
                pass
        if temp_runtime_root is not None:
            temp_runtime_root.cleanup()

    confirmed_opening = next(
        (row for row in reversed(opening_reads)
         if isinstance(row, dict) and row.get("candidate_ready")),
        None,
    )
    opening = {
        "status": "recognized" if runtime_started else "opening_not_observable",
        "reason": runtime_start_reason,
        "start_frame": runtime_start_frame,
        "reads": opening_reads,
        "confirmed": confirmed_opening,
        "truth_log_used_before_replay": False,
    }
    _append_json_line_final(output, {
        "phase": "replay_summary",
        "status": status,
        "status_reason": reason,
        "frame_count": frame_count,
        "indexed_frame_count": total,
        "opening": opening,
        "runtime": {
            "runtime": "live_v2",
            "legacy_orchestrator_used": False,
            "runtime_type": type(runtime).__name__ if runtime is not None else None,
            "runtime_started": runtime_started,
            "listener_status": status if runtime_started else "not_started",
            "listener_status_reason": reason,
            "listener_terminal": bool(runtime_started and status == "complete"),
            "current_player": final_current_player,
        },
    })
    atomic_write_json(comparison_path, {
        "schema_version": 1,
        "expected_source": "post_replay_truth_log_comparison",
        "actual_event_count": len(actual_events),
        "identical_turn_ids": [],
        "missing": [], "added": [], "changed": [], "metric_deltas": [],
    })
    report_path.write_text(
        f"runtime=live_v2\nlegacy_orchestrator_used=false\n"
        f"status={status}\nreason={reason}\nframes={frame_count}/{total}\n",
        encoding="utf-8",
    )
    return VisualPipelineReplayResult(
        output_path=output,
        comparison_path=comparison_path,
        frame_count=frame_count,
        warnings=tuple((*source.warnings, *warnings)),
        comparison=ReplayComparison((), (), (), (), ()),
        run_directory=runtime_directory,
        processed_turn_count=sum(
            str(event.event_type) in {"player_played", "player_passed", "manual_confirmed_event"}
            for event in actual_events
        ),
        completed=status == "complete",
        advice_requested=int(advice_statuses.get("requested", 0)),
        advice_ready=int(advice_statuses.get("ready", 0)),
        advice_failed=int(advice_statuses.get("failed", 0)),
        advice_stale=int(advice_statuses.get("stale", 0)),
        advice_timeouts=int(advice_statuses.get("timeout", 0)),
        advice_withheld=int(advice_statuses.get("withheld", 0)),
        advice_statuses=dict(sorted(advice_statuses.items())),
        artifact_paths={
            name: runtime_directory / name
            for name in (
                "manifest.json", "timeline.jsonl", "timeline.md", "advice.jsonl",
                "decisions.jsonl", "recognition_trace.jsonl",
                "observations.jsonl.part", "observations.jsonl.gz",
            )
            if runtime_directory is not None and (runtime_directory / name).is_file()
        },
        status=status,
        status_reason=reason,
        opening=opening,
        runtime_identity={
            "runtime": "live_v2",
            "legacy_orchestrator_used": False,
            "runtime_type": type(runtime).__name__ if runtime is not None else None,
            "listener_status": status if runtime_started else "not_started",
            "listener_status_reason": reason,
            "listener_terminal": bool(runtime_started and status == "complete"),
            "current_player": final_current_player,
        },
    )


def _opening_recognition_fields(recognized: Any) -> tuple[str | None, list[str]]:
    """Serialize the opening fields produced by visual recognition.

    This helper deliberately has no TruthLog dependency.  Keeping the full
    hand here makes the opening evidence auditable and prevents downstream
    comparisons from falling back to ``hand_count`` only.
    """

    raw_level = getattr(recognized, "round_level", None)
    level = None if raw_level is None else str(raw_level)
    raw_hand = getattr(recognized, "my_hand", ()) or ()
    hand = [str(card) for card in tuple(raw_hand)]
    return level, hand


def _seat_text(value: object | None) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw)


def _replay_confirmed_opening_seed(
    result: object, *, round_level: str, hand: tuple[str, ...]
) -> OpeningSessionSeed | None:
    """Recover a confirmed replay opening event without weakening live gating.

    Recorded evidence can preserve an unknown suit (``7?``) together with
    concrete ``suit_options``.  The live gate rejects that incomplete glyph,
    correctly, but replay must retain the real event rather than silently
    downgrade it to a no-action seed.
    """

    if not round_level or len(hand) != 27:
        return None
    lead = _seat_text(getattr(result, "lead_player", None))
    current = _seat_text(getattr(result, "current_player", None))
    events = tuple(getattr(result, "events", ()) or ())
    if len(events) != 1 or lead not in TURN_ORDER or current is None:
        return None
    event = events[0]

    def value(name: str, default: object = None) -> object:
        if isinstance(event, Mapping):
            return event.get(name, default)
        return getattr(event, name, default)

    actor = _seat_text(value("player", value("actor")))
    cards = tuple(str(card) for card in (value("cards", ()) or ()))
    options = tuple(
        tuple(str(option) for option in choices)
        for choices in (value("suit_options", ()) or ())
    )
    if actor != lead or not cards or bool(value("is_pass", False)):
        return None
    if current != next_active_seat(actor, frozenset()):
        return None
    if len(options) != len(cards) or any(not choices for choices in options):
        return None
    try:
        confidence = float(value("confidence", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not 0.80 <= confidence <= 1.0:
        return None
    return OpeningSessionSeed(
        round_level=round_level,
        hand=tuple(hand),
        lead_player=actor,
        opening_action=OpeningActionSeed(
            actor=actor,
            cards=cards,
            next_player=current,
            confidence=confidence,
            source=str(value("source", "recorded_replay_opening")),
            suit_options=options,
        ),
    )


def _start_runtime_from_opening_seed(
    runtime: Any,
    seed: OpeningSessionSeed,
    *,
    envelope: FrameEnvelope,
    capture_generation: int,
) -> Any:
    """Start, bind, and optionally bootstrap from one confirmed opening seed.

    This mirrors the live controller boundary. The confirming frame is first
    recorded as evidence, but is not sent through visual recognition again:
    ``OpeningTracker`` has already consumed it and ``opening_action`` is the
    authoritative handoff for that frame.
    """

    start_kwargs = {
        "round_level": str(seed.round_level),
        "hand": tuple(seed.hand),
        "lead_player": _seat_text(seed.lead_player),
        "monotonic_ms": envelope.captured_monotonic_ms,
        "wall_time": envelope.wall_time,
        # The confirmed action is handed to bootstrap after the exact frame is
        # recorded; start itself must not commit it twice.
        "opening_action": None,
    }
    runtime.start(**start_kwargs)
    update = runtime.bind_capture_generation(capture_generation)
    runtime.record_frame(
        envelope.image,
        monotonic_ms=envelope.captured_monotonic_ms,
        wall_time=envelope.wall_time,
    )
    action = seed.opening_action
    if action is None:
        return update
    return runtime.bootstrap_opening_action(
        actor=_seat_text(action.actor),
        cards=tuple(action.cards),
        expected_next_player=_seat_text(action.next_player),
        monotonic_ms=envelope.captured_monotonic_ms,
        confidence=float(action.confidence),
        source=str(action.source),
        suit_options=tuple(tuple(value) for value in action.suit_options),
    )


def _record_and_analyze_frame(
    runtime: Any,
    envelope: FrameEnvelope,
    *,
    trace_context: dict[str, object],
) -> Any:
    """Record one replay frame, then enter the runtime through the canonical port."""

    runtime.record_frame(
        envelope.image,
        monotonic_ms=envelope.captured_monotonic_ms,
        wall_time=envelope.wall_time,
    )
    return analyze_frame_envelope(
        runtime,
        envelope,
        trace_context=trace_context,
    )


def _append_frame(
    path: Path,
    envelope: FrameEnvelope,
    runtime: Any,
    phase: str,
    live_update: Any | None = None,
) -> None:
    update = getattr(live_update, "snapshot", None)
    if update is None and runtime is not None:
        try:
            update = runtime.snapshot
        except Exception:
            update = None
    payload: dict[str, object] = {
        "frame_index": int(envelope.frame_index or 0),
        "monotonic_ms": int(envelope.captured_monotonic_ms),
        "wall_time": envelope.wall_time,
        "phase": phase,
        "status": str(getattr(runtime, "status", "opening")),
        "runtime": "live_v2",
        "legacy_orchestrator_used": False,
    }
    if update is not None:
        payload.update({
            "current_player": getattr(update, "current_player", None),
            "state_revision": getattr(update, "revision", None),
        })
    if live_update is not None:
        events = tuple(getattr(live_update, "events", ()) or ())
        if not events and getattr(live_update, "event", None) is not None:
            events = (live_update.event,)
        payload["event"] = events[-1].to_dict() if events else None
        payload["events"] = [event.to_dict() for event in events]
    _append_json_line_final(path, payload)


def _append_json_line_final(path: Path, payload: dict[str, object]) -> None:
    append_json_line(path, payload)


def _timeline_events(path: Path | None) -> tuple[Any, ...]:
    if path is None or not path.is_file():
        return ()
    from ..domain.live import LiveEvent
    result = []
    for raw in read_json_lines(path):
        try:
            result.append(LiveEvent.from_dict(raw))
        except Exception:
            continue
    return tuple(result)


__all__ = ["replay_video_through_production_live_v2"]
