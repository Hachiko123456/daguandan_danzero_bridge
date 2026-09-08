from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
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
    # Defaults keep the historical five-argument construction API intact.
    run_directory: Path | None = None
    processed_turn_count: int = 0
    completed: bool = False
    advice_requested: int = 0
    advice_ready: int = 0
    advice_failed: int = 0
    advice_stale: int = 0
    advice_timeouts: int = 0
    advice_withheld: int = 0
    advice_statuses: dict[str, int] = field(default_factory=dict)
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    # ``completed`` is retained for API compatibility. ``status`` carries
    # the safety decision for scan consumers.
    status: str = "complete"
    status_reason: str = ""
    untrusted_pass_count: int = 0


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
                    action_metadata=turn.move_semantics,
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
            "decisions.jsonl",
            "recognition_trace.jsonl",
            "observations.jsonl.part",
            "observations.jsonl.gz",
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
    integrity_warnings = tuple(
        str(item)
        for item in event.payload.get("integrity_warnings", ())
        if str(item)
    )
    if not is_pass:
        pass_audit = "not_applicable"
    elif "derived_pass" in event.source or any(
        "derived_pass" in warning for warning in integrity_warnings
    ):
        pass_audit = "inferred"
    elif "recovery" in event.source or "marker" in event.source:
        pass_audit = "recovered_with_marker"
    else:
        pass_audit = "direct_visual"
    return {
        "kind": "action",
        "event_id": event.event_id,
        "turn_id": event.turn_id,
        "trick_id": event.trick_id,
        "frame_index": frame_index,
        "actor": event.actor,
        "expected_cards": list(expected.cards) if expected else [],
        "expected_pass": expected.is_pass if expected else False,
        "recognized_cards": list(cards),
        "recognized_pass": is_pass,
        "confidence": round(float(event.confidence), 4),
        "matched": matched,
        "source": event.source,
        "evidence_refs": list(event.evidence_refs),
        "integrity_warnings": list(integrity_warnings),
        "pass_audit": pass_audit,
        "reliable": not is_pass or pass_audit != "inferred",
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

    @property
    def indexed_frame_count(self) -> int:
        """Return the stable number of frames advertised by the index."""

        return sum(1 for _ in read_json_lines(self.index_path))

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
                    missing_count = max(0, len(records) - decoded_count)
                    warnings.append(
                        ReplayWarning(
                            "missing_video_frames",
                            f"索引有 {len(records)} 帧，视频仅解码出 {decoded_count} 帧；"
                            f"缺失尾部 {missing_count} 帧（不使用恢复帧伪造完整回放）",
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


def _stable_visual_initial_state(
    recognition_service: Any,
    frames: tuple[np.ndarray, ...],
) -> tuple[str, tuple[str, ...]] | None:
    """Return the same valid 27-card state from two recorded opening frames."""

    recognize = getattr(recognition_service, "recognize", None)
    if not callable(recognize) or len(frames) < 2:
        return None
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for frame in frames[:2]:
        result = recognize(frame)
        level = str(getattr(result, "round_level", ""))
        cards = tuple(str(card) for card in getattr(result, "my_hand", ()))
        try:
            state = GuanDanState()
            state.set_round_level(level)
            state.confirm_hand(cards)
        except Exception:
            return None
        if len(state.my_hand) != 27:
            return None
        candidates.append((level, state.my_hand))
    return candidates[0] if candidates[0] == candidates[1] else None


def replay_video_through_live_pipeline(
    session: Path,
    recognition_service: Any,
    *,
    truth_log: TruthLog | Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_turn: Callable[[dict[str, object]], None] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
    wait_for_position: Callable[[int], bool] | None = None,
    use_live_pipeline: bool = False,
    sample_every_frame: bool = False,
    recognition_strategy: str = "two_valid_streak",
    output_root: Path | None = None,
    persist_frame_log: bool = True,
    advisor: Any | None = None,
    advice_timeout_sec: float = 60.0,
    use_saved_baseline: bool = False,
) -> VisualPipelineReplayResult:
    """Run timestamped recorded frames through the production live pipeline.

    ``on_turn`` receives each per-turn comparison dict as soon as it is
    produced, so callers can stream readable lines during the replay.
    ``on_progress`` receives ``(processed, total, frame_index)`` for every
    processed frame.  ``total`` comes from the immutable frame index so UI
    callers can expose a stable scan percentage without inspecting the video.
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
    explicit_output_root = output_root is not None
    artifact_root = Path(output_root) if output_root is not None else session
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = artifact_root / (
        "truth_replay.jsonl" if using_truth_log else "visual_replay.jsonl"
    )
    comparison_path = artifact_root / (
        "truth_replay_comparison.json"
        if using_truth_log
        else "visual_replay_comparison.json"
    )
    report_path = artifact_root / (
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
                on_progress=on_progress,
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
        if lead_player not in {"self", "right", "opposite", "left"}:
            confirmed_lead = next(
                (
                    event.payload.get("lead_player", event.actor)
                    for event in expected_events
                    if event.event_type == "lead_player_confirmed"
                ),
                None,
            )
            if confirmed_lead in {"self", "right", "opposite", "left"}:
                lead_player = confirmed_lead
        video_path = session / "video" / "game.avi"
        frame_index_path = session / "video" / "frame_index.jsonl"
    if not use_live_pipeline and lead_player not in {"self", "right", "opposite", "left"}:
        raise ValueError("复测基线中的首出座位无效")

    video_source = VideoReplaySource(video_path, frame_index_path)
    indexed_frame_count = video_source.indexed_frame_count
    frames = iter(video_source.frames())
    first = next(frames, None)
    if first is None:
        raise ValueError("录像没有可回放帧")
    first_record, first_frame = first
    if on_progress is not None:
        on_progress(0, indexed_frame_count, first_record.frame_index)
    prefetched_frames = [(first_record, first_frame)]
    initial_state_warnings: list[ReplayWarning] = []
    if use_live_pipeline:
        second = next(frames, None)
        if second is not None:
            prefetched_frames.append(second)
        visual_initial = _stable_visual_initial_state(
            recognition_service,
            tuple(frame for _record, frame in prefetched_frames),
        )
        if visual_initial is not None:
            visual_level, visual_hand = visual_initial
            stored_hand = tuple(sorted(str(card) for card in hand))
            if visual_level != round_level or visual_hand != stored_hand:
                initial_state_warnings.append(
                    ReplayWarning(
                        "initial_state_mismatch",
                        "Recorded opening frames disagree with the stored initial "
                        f"state (level {round_level!r} -> {visual_level!r}); "
                        "the visual state was used for live-pipeline replay.",
                    )
                )
            round_level = visual_level
            hand = visual_hand
    actual_events: tuple[LiveEvent, ...] = ()
    frame_count = 0
    last_record = first_record
    runtime_directory: Path | None = None
    advice_timeout_count = 0
    waited_advice_requests: set[str] = set()
    advice_statuses: Counter[str] = Counter()
    runtime_artifacts: dict[str, Path] = {}
    untrusted_pass_count = 0
    pipeline_status_before_finish = "initializing"
    terminal_detected = False

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
            advisor=advisor,
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
            lead_player=lead_player if use_saved_baseline else (
                None if use_live_pipeline else lead_player
            ),
            # 必须与录像时间线对齐，保证区域生命周期从首帧开始计时。
            monotonic_ms=int(first_record.monotonic_ms),
            wall_time=first_record.wall_time,
            historical_scan=use_saved_baseline,
        )
        try:
            for record, frame in chain(prefetched_frames, frames):
                if stop_requested is not None and stop_requested():
                    break
                if wait_for_position is not None and not wait_for_position(
                    record.frame_index
                ):
                    break
                update = runner.analyze_frame(
                    frame,
                    monotonic_ms=record.monotonic_ms,
                    trace_context={
                        "source_wall_time": record.wall_time,
                        "historical_scan": use_saved_baseline,
                    },
                )
                raw_advice = getattr(update, "advice", None)
                advice_key = getattr(raw_advice, "key", None)
                if (
                    advisor is not None
                    and advice_key is not None
                    and getattr(raw_advice, "status", "") == "requested"
                    and advice_key.request_id not in waited_advice_requests
                ):
                    waited_advice_requests.add(advice_key.request_id)
                    resolved_advice = runner.wait_for_advice(
                        advice_key,
                        timeout=advice_timeout_sec,
                    )
                    if resolved_advice is None:
                        advice_timeout_count += 1
                        store.append_advice(
                            {
                                "request_id": advice_key.request_id,
                                "status": "timeout",
                                "turn_id": advice_key.turn_id,
                                "state_revision": advice_key.state_revision,
                            }
                        )
                # A committed action can carry lifecycle events produced in
                # the same frame (finish placement, wind catch, next turn).
                # Retain the full batch for diagnostics while keeping the
                # legacy ``event`` field for existing replay readers.
                update_events = tuple(update.events) or (
                    (update.event,) if update.event is not None else ()
                )
                if persist_frame_log:
                    append_json_line(
                        output,
                        {
                        "frame_index": record.frame_index,
                        "monotonic_ms": record.monotonic_ms,
                        "status": update.status,
                        "current_player": update.snapshot.current_player,
                        "state_revision": update.snapshot.revision,
                        "event": update.event.to_dict() if update.event else None,
                        "events": [event.to_dict() for event in update_events],
                        "review_reason": (
                            update.review.reason if update.review is not None else None
                        ),
                        },
                    )
                last_record = record
                frame_count += 1
                if on_progress is not None:
                    on_progress(
                        frame_count,
                        indexed_frame_count,
                        record.frame_index,
                    )
                if on_turn is not None:
                    turn_id_by_event_id: dict[str, int] = {
                        event.event_id: event.turn_id
                        for event in runner.events
                        if event.event_type in _ACTION_TYPES
                    }
                    for event in update_events:
                        if event.event_type in _ACTION_TYPES:
                            turn_data = _event_to_turn_data(
                                event,
                                record.frame_index,
                                truth_log,
                            )
                            if turn_data["pass_audit"] == "inferred":
                                untrusted_pass_count += 1
                            on_turn(turn_data)
                            continue
                        if event.event_type not in {
                            "suit_corrected",
                            "event_correction",
                        }:
                            continue
                        target_event_id = str(
                            event.payload.get("target_event_id", "")
                        )
                        target_turn_id = turn_id_by_event_id.get(target_event_id)
                        if target_turn_id is None:
                            # A correction is auxiliary evidence.  Never turn
                            # an unresolvable target into a new action.
                            continue
                        on_turn(
                            {
                                # Stream the reducer's effective action
                                # corrections to the draft assembler instead
                                # of treating them as additional turns.
                                "kind": event.event_type,
                                "target_turn_id": target_turn_id,
                                "actor": event.actor,
                                "recognized_cards": list(
                                    event.payload.get("cards", ())
                                ),
                                "recognized_pass": bool(
                                    event.payload.get("is_pass", False)
                                ),
                                "trick_id": event.trick_id,
                                "confidence": round(float(event.confidence), 4),
                                "frame_index": record.frame_index,
                            }
                        )
        finally:
            close_frames = getattr(frames, "close", None)
            if close_frames is not None:
                close_frames()
            pipeline_status_before_finish = str(runner.status)
            terminal_detected = bool(
                getattr(runner, "_game_end_detected", False)
                or runner.snapshot.current_player is None
            )
            event_count_before_finish = len(runner.events)
            runner.finish()
            final_events = runner.events[event_count_before_finish:]
            if final_events and persist_frame_log:
                snapshot = runner.snapshot
                append_json_line(
                    output,
                    {
                        "frame_index": last_record.frame_index,
                        "monotonic_ms": last_record.monotonic_ms,
                        "status": runner.status,
                        "current_player": snapshot.current_player,
                        "state_revision": snapshot.revision,
                        "event": final_events[-1].to_dict(),
                        "events": [event.to_dict() for event in final_events],
                        "review_reason": None,
                        "phase": "finalize",
                    },
                )
            actual_events = runner.events

        advice_records = tuple(read_json_lines(store.advice_path))
        advice_statuses.update(
            str(item.get("status", "unknown")) for item in advice_records
        )
        if explicit_output_root:
            runtime_directory = artifact_root / "runtime"
            runtime_directory.mkdir(parents=True, exist_ok=True)
            for name in (
                "manifest.json",
                "timeline.jsonl",
                "timeline.md",
                "advice.jsonl",
                "decisions.jsonl",
                "recognition_trace.jsonl",
                "observations.jsonl.part",
                "observations.jsonl.gz",
            ):
                source = store.directory / name
                if source.is_file():
                    target = runtime_directory / name
                    shutil.copy2(source, target)
                    runtime_artifacts[name] = target

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
    frame_integrity_issue = any(
        warning.reason in {"missing_video_frames", "extra_video_frames"}
        for warning in video_source.warnings
    )
    if frame_integrity_issue:
        scan_status = "blocked"
        scan_reason = "video frame index does not match decoded video"
    elif pipeline_status_before_finish in {"review_required", "waiting_lead"}:
        scan_status = "blocked"
        scan_reason = f"pipeline ended in {pipeline_status_before_finish}"
    elif untrusted_pass_count:
        scan_status = "partial"
        scan_reason = f"{untrusted_pass_count} pass actions lack sufficient evidence"
    elif frame_count != indexed_frame_count:
        scan_status = "partial"
        scan_reason = f"processed {frame_count}/{indexed_frame_count} frames"
    elif use_saved_baseline and not terminal_detected:
        scan_status = "partial"
        scan_reason = "all frames processed but action chain did not reach a terminal state"
    else:
        scan_status = "complete"
        scan_reason = "all frames processed and action chain reached a terminal state"
    return VisualPipelineReplayResult(
        output_path=output,
        comparison_path=comparison_path,
        frame_count=frame_count,
        warnings=(*video_source.warnings, *initial_state_warnings),
        comparison=comparison,
        run_directory=runtime_directory,
        processed_turn_count=sum(
            event.event_type in _ACTION_TYPES for event in actual_events
        ),
        completed=scan_status == "complete",
        advice_requested=int(advice_statuses.get("requested", 0)),
        advice_ready=int(advice_statuses.get("ready", 0)),
        advice_failed=int(advice_statuses.get("failed", 0)),
        advice_stale=int(advice_statuses.get("stale", 0)),
        advice_timeouts=advice_timeout_count,
        advice_withheld=int(advice_statuses.get("withheld", 0)),
        advice_statuses=dict(sorted(advice_statuses.items())),
        artifact_paths=runtime_artifacts,
        status=scan_status,
        status_reason=scan_reason,
        untrusted_pass_count=untrusted_pass_count,
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
    on_progress: Callable[[int, int, int], None] | None = None,
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
    if on_progress is not None:
        on_progress(0, len(records), records[0].frame_index if records else 0)
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
                frame_count += 1
                if on_progress is not None:
                    on_progress(frame_count, len(records), record.frame_index)
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
                if on_progress is not None:
                    on_progress(frame_count, len(records), record.frame_index)
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
                    trick_id=turn.trick_id,
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
