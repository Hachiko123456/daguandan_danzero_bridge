from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .models import LiveEvent, LiveSnapshot
from .reducer import LiveReducer
from .session_store import read_json_lines


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
        event.turn_id: event for event in expected if event.event_type in _ACTION_TYPES
    }
    actual_by_turn = {
        event.turn_id: event for event in actual if event.event_type in _ACTION_TYPES
    }
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
