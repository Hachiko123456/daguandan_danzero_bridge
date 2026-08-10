from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import chain, product
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

import cv2
import numpy as np

from ..storage import append_json_line, atomic_write_json
from ..danzero.state import GuanDanState
from .models import LiveEvent, LiveSnapshot
from .orchestrator import LiveOrchestrator
from .recorder import SessionRecorder
from .reducer import LiveReducer
from .session_store import LiveSessionStore, read_json_lines
from .truth_log import TruthLog, load_truth_log


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


@dataclass(frozen=True)
class TrustedAdviceReplayResult:
    output_path: Path
    summary_path: Path
    run_directory: Path
    source_session_id: str
    turn_count: int
    processed_turn_count: int
    completed: bool
    advice_requested: int
    advice_ready: int
    advice_failed: int
    advice_stale: int
    advice_timeouts: int
    unknown_card_resolutions: int


_ACTION_TYPES = {"player_played", "player_passed", "manual_confirmed_event"}
# 逐帧扫描时单条出牌的最大搜索窗口（帧数，10fps 约 50 秒）。
_TURN_SEARCH_WINDOW = 500
_SEAT_LABELS = {"self": "自己", "right": "右家", "opposite": "对家", "left": "左家"}


class _ReplayAdvisorAdapter:
    """Preserve the orchestrator's temporary unknown-suit branching in replay."""

    def __init__(self, advisor: Any) -> None:
        self._advisor = advisor

    def recommend(
        self,
        state: GuanDanState,
        *,
        request_id: str = "",
        trace: Any | None = None,
    ) -> Any:
        kwargs: dict[str, object] = {"request_id": request_id}
        if trace is not None and "trace" in signature(self._advisor.recommend).parameters:
            kwargs["trace"] = trace
        return self._advisor.recommend(state, **kwargs)


def replay_truth_through_live_advisor(
    session: Path,
    advisor: Any,
    *,
    truth_log: TruthLog | Path | None = None,
    output_root: Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_advice: Callable[[dict[str, object]], None] | None = None,
    advice_timeout_sec: float = 60.0,
) -> TrustedAdviceReplayResult:
    """Drive the production live state/advisor path from trusted actions.

    This mode intentionally does not decode or recognize the source video. A
    trusted ``TruthLog`` supplies already-confirmed actions, while
    ``LiveOrchestrator`` still owns state advancement, self-turn detection,
    DanZero scheduling, result staleness, and advice visibility.
    """

    session = Path(session)
    if isinstance(truth_log, Path):
        manifest_path = session / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text("utf-8"))
            if manifest_path.is_file()
            else {}
        )
        truth_log = load_truth_log(
            truth_log,
            session_id=str(manifest.get("session_id", session.name)),
        )
    elif truth_log is None:
        manifest_path = session / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text("utf-8"))
            if manifest_path.is_file()
            else {}
        )
        truth_log = load_truth_log(
            session / "truth_log.json",
            session_id=str(manifest.get("session_id", session.name)),
        )

    assert truth_log is not None
    source_session_id = truth_log.source_session_id
    output_root = Path(output_root or (session / "replay_runs"))
    run_id = (
        f"trusted_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid4().hex[:6]}"
    )
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    advice_timeout_count = 0
    processed_turn_count = 0
    started = False
    orchestrator: LiveOrchestrator | None = None
    advisor_adapter = _ReplayAdvisorAdapter(advisor)

    def consume_advice(update: object, store: LiveSessionStore) -> None:
        nonlocal advice_timeout_count
        raw = getattr(update, "advice", None)
        if raw is None or getattr(raw, "key", None) is None:
            return
        key = raw.key
        if getattr(raw, "status", "") == "requested":
            assert orchestrator is not None
            if orchestrator.snapshot.current_player == "self":
                orchestrator.ingest_fast_signal(active_player="self")
            result = orchestrator.wait_for_advice(
                key,
                timeout=advice_timeout_sec,
            )
            if result is None:
                advice_timeout_count += 1
                store.append_advice(
                    {
                        "request_id": key.request_id,
                        "status": "timeout",
                        "turn_id": key.turn_id,
                        "state_revision": key.state_revision,
                    }
                )
                record = {
                    "request_id": key.request_id,
                    "status": "timeout",
                    "turn_id": key.turn_id,
                    "state_revision": key.state_revision,
                }
                if on_advice is not None:
                    on_advice(record)
                return
            raw = result
        elif getattr(raw, "status", "") == "ready":
            assert orchestrator is not None
            if orchestrator.snapshot.current_player == "self":
                orchestrator.ingest_fast_signal(active_player="self")
                raw = orchestrator.latest_advice or raw

        advice = getattr(raw, "advice", None)
        record: dict[str, object] = {
            "request_id": key.request_id,
            "status": str(getattr(raw, "status", "unknown")),
            "turn_id": key.turn_id,
            "state_revision": key.state_revision,
            "visible": bool(getattr(raw, "visible", False)),
            "error": str(getattr(raw, "error", "") or ""),
        }
        if advice is not None:
            record.update(
                {
                    "cards": list(advice.cards),
                    "is_pass": bool(advice.is_pass),
                    "play_type": advice.play_type,
                    "elapsed_ms": float(advice.elapsed_ms),
                }
            )
        if on_advice is not None:
            on_advice(record)

    with tempfile.TemporaryDirectory(prefix="daguandan-trusted-advisor-") as temp:
        root = Path(temp)
        store = LiveSessionStore(root, "replay", session_id="trusted-live")
        store.start(
            {
                "source_session": str(session),
                "source_truth_log": source_session_id,
                "mode": "trusted_live_advisor",
            }
        )
        recorder = SessionRecorder(store.directory, size=(64, 32), fps=10)
        orchestrator = LiveOrchestrator(
            reducer=LiveReducer("trusted-live"),
            store=store,
            recorder=recorder,
            recognition_service=object(),
            advisor=advisor_adapter,
            minimum_free_bytes=0,
        )
        try:
            update = orchestrator.start(
                round_level=truth_log.initial_state.round_level,
                hand=truth_log.initial_state.my_hand,
                lead_player=truth_log.initial_state.lead_player,
                monotonic_ms=0,
            )
            started = True
            consume_advice(update, store)
            for turn in truth_log.turns:
                if stop_requested is not None and stop_requested():
                    break
                update = orchestrator.commit_trusted_action(
                    actor=turn.actor,
                    cards=turn.cards,
                    is_pass=turn.is_pass,
                    monotonic_ms=turn.monotonic_ms or turn.index,
                    evidence_refs=(f"TRUTH-{turn.index:06d}",),
                )
                processed_turn_count += 1
                consume_advice(update, store)
        finally:
            if started:
                orchestrator.finish()

        for name in (
            "manifest.json",
            "timeline.jsonl",
            "timeline.md",
            "advice.jsonl",
            "observations.jsonl.part",
        ):
            source = store.directory / name
            if source.is_file():
                shutil.copy2(source, run_directory / name)

        advice_records = read_json_lines(store.advice_path)
        statuses = Counter(str(item.get("status", "unknown")) for item in advice_records)
        completed = processed_turn_count == len(truth_log.turns)
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "source_session_id": source_session_id,
            "source_session": str(session),
            "mode": "trusted_live_advisor",
            "turn_count": len(truth_log.turns),
            "processed_turn_count": processed_turn_count,
            "completed": completed,
            "advice_requested": statuses.get("requested", 0),
            "advice_ready": statuses.get("ready", 0),
            "advice_failed": statuses.get("failed", 0),
            "advice_stale": statuses.get("stale", 0),
            "advice_timeouts": advice_timeout_count,
            "advice_statuses": dict(statuses),
            # Compatibility field: no irreversible replacement is made.
            "unknown_card_resolutions": [],
            "unknown_card_policy": "temporary_suit_variants",
        }
        summary_path = run_directory / "summary.json"
        atomic_write_json(summary_path, summary)

    return TrustedAdviceReplayResult(
        output_path=run_directory / "advice.jsonl",
        summary_path=summary_path,
        run_directory=run_directory,
        source_session_id=source_session_id,
        turn_count=len(truth_log.turns),
        processed_turn_count=processed_turn_count,
        completed=completed,
        advice_requested=int(statuses.get("requested", 0)),
        advice_ready=int(statuses.get("ready", 0)),
        advice_failed=int(statuses.get("failed", 0)),
        advice_stale=int(statuses.get("stale", 0)),
        advice_timeouts=advice_timeout_count,
        unknown_card_resolutions=0,
    )


def _cards_match(expected: tuple[str, ...], recognized: tuple[str, ...]) -> bool:
    """比较两组牌面，未知花色（如 8?）与任意同点数花色等价。"""
    if Counter(expected) == Counter(recognized):
        return True
    question_indices = [
        index
        for index, card in enumerate(expected)
        if str(card).endswith("?")
    ]
    if not question_indices:
        return False
    recognized_counter = Counter(recognized)
    for combination in product(
        ("S", "H", "C", "D"), repeat=len(question_indices)
    ):
        trial = list(expected)
        for index, suit in zip(question_indices, combination):
            trial[index] = str(trial[index])[:-1] + suit
        if Counter(trial) == recognized_counter:
            return True
    return False


def _write_truth_replay_report(
    report_path: Path,
    truth_log: TruthLog,
    round_level: str,
    output_lines: list[str],
    identical: list[int],
    missing: list[LiveEvent],
) -> None:
    """Write a human-readable per-turn comparison report (Chinese)."""

    by_id = {
        int(json.loads(raw)["turn_id"]): json.loads(raw) for raw in output_lines
    }
    missing_ids = {event.turn_id for event in missing}
    identical_ids = set(identical)
    lines = [
        f"对局复测报告：{truth_log.source_session_id}",
        f"级牌：{round_level}　首出玩家：{_SEAT_LABELS.get(truth_log.initial_state.lead_player, truth_log.initial_state.lead_player)}"
        f"　出牌日志：{len(truth_log.turns)} 条",
        "",
    ]
    for turn in truth_log.turns:
        seat = _SEAT_LABELS.get(turn.actor, turn.actor)
        expected = "不出" if turn.is_pass else "出牌 " + " ".join(turn.cards)
        data = by_id.get(turn.index)
        if data is None or turn.index in missing_ids:
            result = "匹配异常：录像未识别到该回合（录像可能未覆盖）"
        elif turn.index in identical_ids:
            result = f"匹配正确（置信度 {float(data['confidence']):.2f}）"
        else:
            got = (
                "不出"
                if data.get("recognized_pass")
                else (" ".join(data.get("recognized_cards") or []) or "未识别到")
            )
            result = f"匹配异常：录像识别为「{got}」"
        lines.append(
            f"[第 {turn.index:>2} 条] {seat} {expected} ｜ 日志：{expected} ｜ {result}"
        )
    lines.append("")
    lines.append(
        f"汇总：一致 {len(identical_ids)} 条，缺失 {len(missing)} 条，"
        f"新增 0 条，变化 0 条"
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _event_to_turn_data(
    event: LiveEvent,
    frame_index: int,
    truth_log: TruthLog | None,
) -> dict[str, object]:
    """把状态机提交的动作事件转成复测输出用的逐条数据。"""
    is_pass = event.event_type == "player_passed" or bool(
        event.payload.get("is_pass", False)
    )
    cards = tuple(str(card) for card in event.payload.get("cards", ()))
    expected = None
    if truth_log is not None:
        expected = next(
            (turn for turn in truth_log.turns if turn.index == event.turn_id),
            None,
        )
    matched = bool(
        expected
        and expected.actor == event.actor
        and expected.is_pass == is_pass
        and _cards_match(expected.cards, cards)
    )
    return {
        "turn_id": event.turn_id,
        "frame_index": frame_index,
        "actor": event.actor,
        "expected_cards": list(expected.cards) if expected else [],
        "expected_pass": expected.is_pass if expected else False,
        "recognized_cards": list(cards),
        "recognized_pass": is_pass,
        "confidence": round(float(event.confidence), 4),
        "matched": matched,
    }


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

    def frames(
        self,
        *,
        start_frame: int | None = None,
    ) -> Iterator[tuple[FrameIndexRecord, np.ndarray]]:
        records = tuple(
            FrameIndexRecord.from_dict(raw) for raw in read_json_lines(self.index_path)
        )
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开回放视频：{self.video_path}")
        seeked = False
        if start_frame is not None:
            seeked = bool(
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            )
        warnings: list[ReplayWarning] = []
        decoded_count = 0
        try:
            for record in records:
                if start_frame is not None and seeked and record.frame_index < start_frame:
                    continue
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
                if start_frame is not None and record.frame_index < start_frame:
                    continue
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
    truth_log: TruthLog | Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_turn: Callable[[dict[str, object]], None] | None = None,
    wait_for_position: Callable[[int], bool] | None = None,
    use_live_pipeline: bool = False,
    sample_every_frame: bool = False,
    recognition_strategy: str = "two_valid_streak",
) -> VisualPipelineReplayResult:
    """Run timestamped recorded frames through the production live pipeline.

    ``on_turn`` receives each per-turn comparison dict as soon as it is
    produced, so callers can stream readable lines during the replay.
    ``wait_for_position(target_frame)`` gates the scan so recognition only
    advances as fast as the playback position (pause stops the scan).
    ``use_live_pipeline=True`` runs the replay through the real
    ``LiveOrchestrator`` core (current-player gate + consensus + reducer)
    instead of the log-driven scan. ``sample_every_frame`` is retained as a
    compatibility keyword but is intentionally ignored; the action window is
    always exercised.
    """

    del sample_every_frame
    session = Path(session)
    using_truth_log = truth_log is not None
    output = session / ("truth_replay.jsonl" if using_truth_log else "visual_replay.jsonl")
    comparison_path = session / (
        "truth_replay_comparison.json"
        if using_truth_log
        else "visual_replay_comparison.json"
    )
    report_path = session / (
        "truth_replay_report.txt" if using_truth_log else "visual_replay_report.txt"
    )
    output.unlink(missing_ok=True)
    comparison_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    if isinstance(truth_log, Path):
        manifest = json.loads((session / "manifest.json").read_text("utf-8"))
        truth_log = load_truth_log(
            truth_log,
            session_id=str(manifest.get("session_id", session.name)),
        )
    if truth_log is not None:
        expected_events = truth_log.to_events(session_id="truth-log")
        initial = expected_events[0]
        hand = truth_log.initial_state.my_hand
        round_level = truth_log.initial_state.round_level
        lead_player = truth_log.initial_state.lead_player
        video_path = session / truth_log.source_video
        frame_index_path = session / truth_log.frame_index_path
        if truth_log.turns and not use_live_pipeline:
            # 出牌日志优先走逐帧校验：有帧号定点比对，无帧号按序扫描。
            return _replay_truth_by_frame(
                session,
                recognition_service,
                truth_log,
                round_level=round_level,
                video_path=video_path,
                frame_index_path=frame_index_path,
                output=output,
                comparison_path=comparison_path,
                report_path=report_path,
                on_turn=on_turn,
                wait_for_position=wait_for_position,
            )
    else:
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
        video_path = session / "video" / "game.avi"
        frame_index_path = session / "video" / "frame_index.jsonl"
    if not use_live_pipeline and lead_player not in {"self", "right", "opposite", "left"}:
        raise ValueError("复测基线中的首出座位无效")

    video_source = VideoReplaySource(video_path, frame_index_path)
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
            advisor=None,
            minimum_free_bytes=0,
            recognition_strategy=recognition_strategy,
            # 复测无人值守：识别落空不等人确认，自动重置继续。
        )
        runner.start(
            round_level=round_level,
            hand=hand,
            # State-machine replay must exercise the same visual opening
            # phase as a new live game.  The saved timeline remains the
            # comparison baseline only; it is not permitted to pre-fill who
            # leads this replay.
            lead_player=None if use_live_pipeline else lead_player,
            # 必须与录像时间线对齐，保证区域生命周期从首帧开始计时。
            monotonic_ms=int(first_record.monotonic_ms),
        )
        try:
            for record, frame in chain(((first_record, first_frame),), frames):
                if stop_requested is not None and stop_requested():
                    break
                if wait_for_position is not None and not wait_for_position(
                    record.frame_index
                ):
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
                if (
                    on_turn is not None
                    and update.event is not None
                    and update.event.event_type in _ACTION_TYPES
                ):
                    on_turn(
                        _event_to_turn_data(
                            update.event,
                            record.frame_index,
                            truth_log,
                        )
                    )
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


def _replay_truth_by_frame(
    session: Path,
    recognition_service: Any,
    truth_log: TruthLog,
    *,
    round_level: str,
    video_path: Path,
    frame_index_path: Path,
    output: Path,
    comparison_path: Path,
    report_path: Path,
    on_turn: Callable[[dict[str, object]], None] | None = None,
    wait_for_position: Callable[[int], bool] | None = None,
) -> VisualPipelineReplayResult:
    """Deterministic per-turn verification against the recorded video.

    When turns carry a saved ``frame_index`` (the editor records where each
    action was recognized), the recognizer re-runs on exactly those frames.
    Otherwise the video is scanned forward once and each expected turn is
    matched as soon as the expected player's zone shows that action.
    ``wait_for_position`` gates the scan to the playback position.
    """

    records = tuple(
        FrameIndexRecord.from_dict(raw) for raw in read_json_lines(frame_index_path)
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开回放视频：{video_path}")
    expected_by_turn = {
        event.turn_id: event
        for event in _effective_actions(truth_log.to_events(session_id="truth-log"))
    }
    identical: list[int] = []
    changed: list[ChangedTurn] = []
    missing: list[LiveEvent] = []
    output_lines: list[str] = []
    exact_frames = all(turn.frame_index is not None for turn in truth_log.turns)
    frame_count = 0
    try:
        if exact_frames:
            for turn in truth_log.turns:
                record = next(
                    (item for item in records if item.frame_index >= int(turn.frame_index)),
                    None,
                )
                if record is None:
                    expected = expected_by_turn.get(turn.index)
                    if expected is not None:
                        missing.append(expected)
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, record.frame_index)
                ok, frame = capture.read()
                if not ok:
                    expected = expected_by_turn.get(turn.index)
                    if expected is not None:
                        missing.append(expected)
                    continue
                if wait_for_position is not None and not wait_for_position(
                    record.frame_index
                ):
                    break
                result = recognition_service.recognize_play_region(
                    frame,
                    turn.actor,
                    wild_rank=round_level,
                )
                _record_frame_match(
                    turn,
                    record,
                    result,
                    output_lines,
                    identical,
                    changed,
                    missing,
                    expected_by_turn,
                    on_turn=on_turn,
                )
        else:
            expected_turns = list(truth_log.turns)
            turn_index = 0
            window_begin: int | None = None
            # 单条动作的搜索窗口（帧数）。识别不到的牌型（如未知花色）
            # 不能卡住整局扫描：超时记为缺失，并回退到窗口起点重新找下一条，
            # 避免视频在等待期间已经跑过下一条的内容。
            search_window = max(100, int(_TURN_SEARCH_WINDOW))

            def make_iterator(start_frame: int | None):
                return iter(
                    VideoReplaySource(video_path, frame_index_path).frames(
                        start_frame=start_frame
                    )
                )

            iterator = make_iterator(None)
            while turn_index < len(expected_turns):
                try:
                    record, frame = next(iterator)
                except StopIteration:
                    break
                frame_count += 1
                if wait_for_position is not None and not wait_for_position(
                    record.frame_index
                ):
                    break
                turn = expected_turns[turn_index]
                if window_begin is None:
                    window_begin = record.frame_index
                elif record.frame_index - window_begin > search_window:
                    expected = expected_by_turn.get(turn.index)
                    if expected is not None:
                        missing.append(expected)
                    turn_data: dict[str, object] = {
                        "turn_id": turn.index,
                        "frame_index": record.frame_index,
                        "actor": turn.actor,
                        "expected_cards": list(turn.cards),
                        "expected_pass": turn.is_pass,
                        "recognized_cards": [],
                        "recognized_pass": False,
                        "confidence": 0.0,
                        "matched": False,
                        "reason": "search_window_exceeded",
                    }
                    output_lines.append(
                        json.dumps(turn_data, ensure_ascii=False)
                    )
                    if on_turn is not None:
                        on_turn(turn_data)
                    turn_index += 1
                    iterator = make_iterator(window_begin)
                    window_begin = None
                    continue
                result = recognition_service.recognize_play_region(
                    frame,
                    turn.actor,
                    wild_rank=round_level,
                )
                matched = (
                    result.is_pass == turn.is_pass
                    and _cards_match(turn.cards, tuple(result.cards))
                )
                if not matched:
                    continue
                _record_frame_match(
                    turn,
                    record,
                    result,
                    output_lines,
                    identical,
                    changed,
                    missing,
                    expected_by_turn,
                    on_turn=on_turn,
                )
                turn_index += 1
                window_begin = None
            for turn in expected_turns[turn_index:]:
                expected = expected_by_turn.get(turn.index)
                if expected is not None:
                    missing.append(expected)
                if on_turn is not None:
                    on_turn(
                        {
                            "turn_id": turn.index,
                            "frame_index": None,
                            "actor": turn.actor,
                            "expected_cards": list(turn.cards),
                            "expected_pass": turn.is_pass,
                            "recognized_cards": [],
                            "recognized_pass": False,
                            "confidence": 0.0,
                            "matched": False,
                            "reason": "video_end",
                        }
                    )
    finally:
        capture.release()
    output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    comparison = ReplayComparison(
        identical_turn_ids=tuple(identical),
        missing=tuple(missing),
        added=(),
        changed=tuple(changed),
        metric_deltas=(),
    )
    _write_truth_replay_report(
        report_path,
        truth_log,
        round_level,
        output_lines,
        identical,
        missing,
    )
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
            "metric_deltas": [],
        },
    )
    return VisualPipelineReplayResult(
        output_path=output,
        comparison_path=comparison_path,
        frame_count=max(frame_count, len(truth_log.turns)),
        warnings=(),
        comparison=comparison,
    )


def _record_frame_match(
    turn: TruthTurn,
    record: FrameIndexRecord,
    result: Any,
    output_lines: list[str],
    identical: list[int],
    changed: list[ChangedTurn],
    missing: list[LiveEvent],
    expected_by_turn: dict[int, LiveEvent],
    *,
    on_turn: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Append one per-turn verification line and update the comparison."""

    recognized_cards = tuple(result.cards)
    matched = (
        result.is_pass == turn.is_pass
        and _cards_match(turn.cards, recognized_cards)
    )
    turn_data: dict[str, object] = {
        "turn_id": turn.index,
        "frame_index": record.frame_index,
        "actor": turn.actor,
        "expected_cards": list(turn.cards),
        "expected_pass": turn.is_pass,
        "recognized_cards": list(recognized_cards),
        "recognized_pass": result.is_pass,
        "confidence": round(float(result.confidence), 4),
        "matched": matched,
        "boxes": [
            {
                "label": annotation.label,
                "box": list(annotation.box),
                "confidence": round(float(annotation.confidence), 4),
            }
            for annotation in result.annotations
        ],
    }
    output_lines.append(json.dumps(turn_data, ensure_ascii=False))
    if on_turn is not None:
        on_turn(turn_data)
    expected = expected_by_turn.get(turn.index)
    if expected is None:
        return
    if matched:
        identical.append(turn.index)
    else:
        changed.append(
            ChangedTurn(
                turn_id=turn.index,
                expected=expected,
                actual=LiveEvent(
                    event_id=f"TRUTH-{turn.index:06d}",
                    event_type="player_passed" if result.is_pass else "player_played",
                    session_id="truth-log",
                    seq=turn.index,
                    monotonic_ms=record.monotonic_ms,
                    wall_time=record.wall_time,
                    trick_id=1,
                    turn_id=turn.index,
                    actor=turn.actor,
                    payload={
                        "cards": list(recognized_cards),
                        "is_pass": result.is_pass,
                    },
                    confidence=float(result.confidence),
                    source="frame_replay",
                    state_revision_before=turn.index,
                    state_revision_after=turn.index + 1,
                ),
            )
        )
