from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from ..storage import append_json_line, atomic_write_json
from .models import LiveEvent, LiveSnapshot
from .orchestrator import LiveOrchestrator
from .recorder import SessionRecorder
from .reducer import LiveReducer
from .session_store import LiveSessionStore, read_json_lines


@dataclass(frozen=True)
class EventReplayResult:
    final_snapshot: LiveSnapshot
    snapshot_hashes: tuple[str, ...]
    ordered_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrameIndexRecord:
    frame_index: int
    monotonic_ms: int
    wall_time: str
    dropped_before: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "FrameIndexRecord":
        return cls(
            frame_index=int(raw["frame_index"]),
            monotonic_ms=int(raw["monotonic_ms"]),
            wall_time=str(raw["wall_time"]),
            dropped_before=int(raw.get("dropped_before", 0)),
        )


@dataclass(frozen=True)
class ReplayWarning:
    reason: str
    details: str


@dataclass(frozen=True)
class ChangedTurn:
    turn_id: int
    expected: LiveEvent
    actual: LiveEvent


@dataclass(frozen=True)
class ReplayMetricDelta:
    turn_id: int
    confidence_delta: float
    latency_delta_ms: int


@dataclass(frozen=True)
class ReplayComparison:
    identical_turn_ids: tuple[int, ...]
    missing: tuple[LiveEvent, ...]
    added: tuple[LiveEvent, ...]
    changed: tuple[ChangedTurn, ...]
    metric_deltas: tuple[ReplayMetricDelta, ...]


@dataclass(frozen=True)
class VisualPipelineReplayResult:
    output_path: Path
    comparison_path: Path
    frame_count: int
    warnings: tuple[ReplayWarning, ...]
    comparison: ReplayComparison


_ACTION_TYPES = {"player_played", "player_passed", "manual_confirmed_event"}


def _action_semantics(event: LiveEvent) -> tuple[object, ...]:
    is_pass = event.event_type == "player_passed" or bool(
        event.payload.get("is_pass", False)
    )
    cards = () if is_pass else tuple(
        sorted(str(card) for card in event.payload.get("cards", ()))
    )
    return event.actor, is_pass, cards


def compare_timelines(
    expected: list[LiveEvent] | tuple[LiveEvent, ...],
    actual: list[LiveEvent] | tuple[LiveEvent, ...],
) -> ReplayComparison:
    """Compare formal actions by turn without treating score drift as card drift."""

    expected_by_turn = {
        event.turn_id: event for event in _effective_actions(expected)
    }
    actual_by_turn = {event.turn_id: event for event in _effective_actions(actual)}
    expected_turns = set(expected_by_turn)
    actual_turns = set(actual_by_turn)
    missing = tuple(expected_by_turn[key] for key in sorted(expected_turns - actual_turns))
    added = tuple(actual_by_turn[key] for key in sorted(actual_turns - expected_turns))
    identical: list[int] = []
    changed: list[ChangedTurn] = []
    deltas: list[ReplayMetricDelta] = []
    for turn_id in sorted(expected_turns & actual_turns):
        left = expected_by_turn[turn_id]
        right = actual_by_turn[turn_id]
        if _action_semantics(left) == _action_semantics(right):
            identical.append(turn_id)
        else:
            changed.append(ChangedTurn(turn_id, left, right))
        deltas.append(
            ReplayMetricDelta(
                turn_id=turn_id,
                confidence_delta=right.confidence - left.confidence,
                latency_delta_ms=right.monotonic_ms - left.monotonic_ms,
            )
        )
    return ReplayComparison(
        identical_turn_ids=tuple(identical),
        missing=missing,
        added=added,
        changed=tuple(changed),
        metric_deltas=tuple(deltas),
    )


def _effective_actions(
    events: list[LiveEvent] | tuple[LiveEvent, ...],
) -> tuple[LiveEvent, ...]:
    corrections = {
        str(event.payload.get("target_event_id")): event
        for event in events
        if event.event_type == "event_correction"
    }
    effective: list[LiveEvent] = []
    for event in events:
        if event.event_type not in _ACTION_TYPES:
            continue
        correction = corrections.get(event.event_id)
        if correction is None:
            effective.append(event)
            continue
        is_pass = bool(correction.payload.get("is_pass", False))
        effective.append(
            replace(
                event,
                event_type="player_passed" if is_pass else "player_played",
                payload={
                    "cards": list(correction.payload.get("cards", ())),
                    "is_pass": is_pass,
                },
                confidence=correction.confidence,
                source=correction.source,
                evidence_refs=correction.evidence_refs or event.evidence_refs,
            )
        )
    return tuple(effective)


class EventReplayer:
    """Replay confirmed events on a virtual monotonic clock with no waiting."""

    def __init__(self, reducer_factory: Callable[[], LiveReducer]) -> None:
        self._reducer_factory = reducer_factory

    def replay(self, events: tuple[LiveEvent, ...] | list[LiveEvent]) -> EventReplayResult:
        ordered = tuple(sorted(events, key=lambda event: (event.monotonic_ms, event.seq)))
        reducer = self._reducer_factory()
        hashes: list[str] = []
        for event in ordered:
            reducer.apply(event)
            snapshot_payload = reducer.snapshot().semantic_dict()
            canonical = json.dumps(
                snapshot_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            hashes.append(hashlib.sha256(canonical).hexdigest())
        return EventReplayResult(
            final_snapshot=reducer.snapshot(),
            snapshot_hashes=tuple(hashes),
            ordered_event_ids=tuple(event.event_id for event in ordered),
        )


class VideoReplaySource:
    """Decode recorded frames without replacing their explicit capture timestamps."""

    def __init__(self, video_path: Path, index_path: Path) -> None:
        self.video_path = Path(video_path)
        self.index_path = Path(index_path)
        self._warnings: tuple[ReplayWarning, ...] = ()

    @property
    def warnings(self) -> tuple[ReplayWarning, ...]:
        return self._warnings

    def frames(self) -> Iterator[tuple[FrameIndexRecord, np.ndarray]]:
        records = tuple(
            FrameIndexRecord.from_dict(raw) for raw in read_json_lines(self.index_path)
        )
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开回放视频：{self.video_path}")
        warnings: list[ReplayWarning] = []
        decoded_count = 0
        try:
            for record in records:
                ok, frame = capture.read()
                if not ok:
                    warnings.append(
                        ReplayWarning(
                            "missing_video_frames",
                            f"索引有 {len(records)} 帧，视频仅解码出 {decoded_count} 帧",
                        )
                    )
                    break
                decoded_count += 1
                yield record, frame
            else:
                ok, _ = capture.read()
                if ok:
                    extra = 1
                    while capture.read()[0]:
                        extra += 1
                    warnings.append(
                        ReplayWarning(
                            "extra_video_frames",
                            f"视频比索引多 {extra} 帧",
                        )
                    )
        finally:
            capture.release()
            self._warnings = tuple(warnings)


def replay_video_through_live_pipeline(
    session: Path,
    recognition_service: Any,
    *,
    stop_requested: Callable[[], bool] | None = None,
) -> VisualPipelineReplayResult:
    """Run timestamped recorded frames through the production live pipeline."""

    session = Path(session)
    output = session / "visual_replay.jsonl"
    comparison_path = session / "visual_replay_comparison.json"
    output.unlink(missing_ok=True)
    comparison_path.unlink(missing_ok=True)
    expected_events = tuple(
        LiveEvent.from_dict(raw)
        for raw in read_json_lines(session / "timeline.jsonl")
    )
    initial = next(
        (event for event in expected_events if event.event_type == "initial_state_confirmed"),
        None,
    )
    if initial is None:
        raise ValueError("对局时间线缺少 initial_state_confirmed")
    hand = tuple(str(card) for card in initial.payload.get("hand", ()))
    round_level = str(initial.payload.get("round_level", ""))
    lead_player = initial.payload.get("lead_player")
    if lead_player not in {"self", "right", "opposite", "left"}:
        raise ValueError("对局时间线中的首发座位无效")

    video_source = VideoReplaySource(
        session / "video" / "game.avi",
        session / "video" / "frame_index.jsonl",
    )
    frames = iter(video_source.frames())
    first = next(frames, None)
    if first is None:
        raise ValueError("录像没有可回放帧")
    first_record, first_frame = first
    actual_events: tuple[LiveEvent, ...] = ()
    frame_count = 0

    with tempfile.TemporaryDirectory(prefix="daguandan-visual-replay-") as temp:
        root = Path(temp)
        store = LiveSessionStore(root, "replay", session_id="visual-replay")
        store.start({"source_session": str(session), "mode": "visual_pipeline"})
        height, width = first_frame.shape[:2]
        recorder = SessionRecorder(store.directory, size=(width, height), fps=10)
        runner = LiveOrchestrator(
            reducer=LiveReducer("visual-replay"),
            store=store,
            recorder=recorder,
            recognition_service=recognition_service,
            minimum_free_bytes=0,
        )
        runner.start(
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            monotonic_ms=int(initial.monotonic_ms),
        )
        try:
            for record, frame in chain(((first_record, first_frame),), frames):
                if stop_requested is not None and stop_requested():
                    break
                update = runner.analyze_frame(
                    frame,
                    monotonic_ms=record.monotonic_ms,
                )
                append_json_line(
                    output,
                    {
                        "frame_index": record.frame_index,
                        "monotonic_ms": record.monotonic_ms,
                        "status": update.status,
                        "current_player": update.snapshot.current_player,
                        "state_revision": update.snapshot.revision,
                        "event": update.event.to_dict() if update.event else None,
                        "review_reason": (
                            update.review.reason if update.review is not None else None
                        ),
                    },
                )
                frame_count += 1
            actual_events = runner.events
        finally:
            close_frames = getattr(frames, "close", None)
            if close_frames is not None:
                close_frames()
            runner.finish()

    comparison = compare_timelines(expected_events, actual_events)
    atomic_write_json(
        comparison_path,
        {
            "schema_version": 1,
            "identical_turn_ids": list(comparison.identical_turn_ids),
            "missing": [event.to_dict() for event in comparison.missing],
            "added": [event.to_dict() for event in comparison.added],
            "changed": [
                {
                    "turn_id": item.turn_id,
                    "expected": item.expected.to_dict(),
                    "actual": item.actual.to_dict(),
                }
                for item in comparison.changed
            ],
            "metric_deltas": [
                {
                    "turn_id": item.turn_id,
                    "confidence_delta": item.confidence_delta,
                    "latency_delta_ms": item.latency_delta_ms,
                }
                for item in comparison.metric_deltas
            ],
        },
    )
    return VisualPipelineReplayResult(
        output_path=output,
        comparison_path=comparison_path,
        frame_count=frame_count,
        warnings=video_source.warnings,
        comparison=comparison,
    )
