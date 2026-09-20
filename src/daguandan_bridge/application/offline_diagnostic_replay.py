from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
import zipfile
from typing import Callable

from ..annotation_service import AnnotationService
from ..config import DIAGNOSTICS_ROOT, PROFILES_ROOT
from ..recognition_service import ScreenshotRecognitionService
from ..template_service import TemplateService
from .live_v2_recorded_replay import replay_video_through_production_live_v2


@dataclass(frozen=True)
class OfflineDiagnosticReplayResult:
    input_path: Path
    output_directory: Path
    report_path: Path
    status: str
    status_reason: str
    frame_count: int
    processed_turn_count: int
    advice_statuses: dict[str, int]
    modes: dict[str, dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe result, including every nested replay artifact.

        The mode values are built from ``VisualPipelineReplayResult`` objects.
        Those values contain nested dataclasses, ``Path`` instances, tuples and
        artifact-path mappings, so a shallow ``dict(self.modes)`` is not enough
        for the CLI's final ``json.dumps`` call.
        """

        value = _json_safe({
            "schema": "guandan.offline-diagnostic-replay/1",
            "input_path": self.input_path,
            "output_directory": self.output_directory,
            "report_path": self.report_path,
            "status": self.status,
            "status_reason": self.status_reason,
            "frame_count": self.frame_count,
            "processed_turn_count": self.processed_turn_count,
            "advice_statuses": self.advice_statuses,
            "modes": self.modes,
        })
        # The shape above is fixed; keeping this guard makes an accidental
        # future change fail close rather than returning a non-object to the
        # command-line caller.
        if not isinstance(value, dict):
            raise TypeError("offline diagnostic result must serialize to an object")
        return value


class OfflineDiagnosticReplayService:
    """Run a copied diagnostic ZIP through the production live-v2 replay path."""

    def __init__(
        self,
        *,
        profiles_root: Path | str = PROFILES_ROOT,
        profile_name: str = "tencent_daguandan",
        output_root: Path | str = DIAGNOSTICS_ROOT / "offline-replays",
    ) -> None:
        self.profiles_root = Path(profiles_root).expanduser().resolve()
        self.profile_name = str(profile_name)
        self.output_root = Path(output_root).expanduser().resolve()

    def run(
        self,
        input_path: Path | str,
        *,
        on_progress: Callable[[dict[str, object]], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        environment: dict[str, object] | None = None,
    ) -> OfflineDiagnosticReplayResult:
        source = Path(input_path).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        output = self.output_root / (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{source.stem}"
        )
        output.mkdir(parents=True, exist_ok=False)
        def emit_progress(processed: int, total: int, frame_index: int) -> None:
            if on_progress is not None:
                on_progress({
                    "phase": "replay",
                    "mode": current_mode,
                    "processed": int(processed),
                    "total": int(total),
                    "frame_index": int(frame_index),
                    "message": f"正在回放 {processed}/{total} 帧",
                })

        with tempfile.TemporaryDirectory(prefix="guandan-offline-replay-") as temporary:
            session = self._prepare_session(source, Path(temporary))
            recognition = ScreenshotRecognitionService(
                AnnotationService(self.profiles_root, self.profile_name),
                TemplateService(self.profiles_root, self.profile_name),
            )
            mode_results: dict[str, object] = {}
            cancelled = False
            for mode in ("latest", "synchronous"):
                if stop_requested is not None and stop_requested():
                    cancelled = True
                    break
                current_mode = mode
                emit_progress(0, 0, -1)
                result = replay_video_through_production_live_v2(
                    session,
                    recognition,
                    profile_root=self.profiles_root / self.profile_name,
                    output_root=output / mode,
                    on_progress=emit_progress,
                    stop_requested=stop_requested,
                    vision_delivery=mode,
                )
                mode_results[mode] = result
                if str(getattr(result, "status", "")) == "incomplete" and str(getattr(result, "status_reason", "")) == "replay_cancelled":
                    cancelled = True
                    break
        summary = self._build_summary(source, mode_results, environment=environment or {}, cancelled=cancelled)
        json_path = output / "offline_diagnostic_summary.json"
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path = output / "offline_diagnostic_report.md"
        report_path.write_text(self._markdown_report(summary), encoding="utf-8")
        statuses = {
            mode: str(getattr(value, "status", "unknown"))
            for mode, value in mode_results.items()
        }
        advice_statuses: dict[str, int] = {}
        processed_turn_count = 0
        frame_count = 0
        for value in mode_results.values():
            frame_count = max(frame_count, int(getattr(value, "frame_count", 0)))
            processed_turn_count = max(processed_turn_count, int(getattr(value, "processed_turn_count", 0)))
            for key, count in getattr(value, "advice_statuses", {}).items():
                advice_statuses[f"{key}"] = advice_statuses.get(f"{key}", 0) + int(count)
        return OfflineDiagnosticReplayResult(
            input_path=source,
            output_directory=output,
            report_path=report_path,
            status=("cancelled" if cancelled else "complete" if len(mode_results) == 2 else "incomplete"),
            status_reason=json.dumps(statuses, ensure_ascii=False),
            frame_count=frame_count,
            processed_turn_count=processed_turn_count,
            advice_statuses=advice_statuses,
            modes={mode: self._result_dict(value) for mode, value in mode_results.items()},
        )

    @staticmethod
    def _prepare_session(source: Path, temporary: Path) -> Path:
        if source.is_dir():
            if (source / "video" / "game.avi").is_file():
                return source
            archives = sorted(source.glob("*.zip"))
            if archives:
                source = archives[0]
            else:
                raise ValueError("目录中没有可回放的 game.avi 或诊断 ZIP")
        if source.suffix.casefold() != ".zip" or not source.is_file():
            raise ValueError("请选择完整诊断 ZIP 或包含完整诊断 ZIP 的目录")
        session = temporary / "session"
        session.mkdir(parents=True, exist_ok=True)
        root = session.resolve()
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                name = info.filename.replace("\\", "/")
                if not name.startswith("session/") or name.endswith("/"):
                    continue
                relative = Path(name).relative_to("session")
                target = (session / relative).resolve()
                if target != root and root not in target.parents:
                    raise ValueError("诊断 ZIP 含有越界路径")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source_stream, target.open("wb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream)
        if not (session / "video" / "game.avi").is_file():
            raise ValueError("诊断 ZIP 缺少 session/video/game.avi")
        if not (session / "video" / "frame_index.jsonl").is_file():
            raise ValueError("诊断 ZIP 缺少 session/video/frame_index.jsonl")
        return session

    @classmethod
    def _result_dict(cls, result: object) -> dict[str, object]:
        """Normalize one replay result and add report-facing key diagnostics."""

        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        elif isinstance(result, Mapping):
            value = result
        else:
            value = vars(result)
        normalized = _json_safe(value)
        if not isinstance(normalized, dict):
            normalized = {"status": str(normalized)}
        return cls._add_mode_diagnostics(result, normalized)

    @classmethod
    def _add_mode_diagnostics(
        cls,
        result: object,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Expose the few fields needed to compare the two replay modes.

        The production replay result intentionally stays independent of this
        diagnostic feature.  We therefore derive these fields from its copied
        timeline and advice artifacts instead of changing the production result
        contract:

        * ``lead_player`` comes from the confirmed opening evidence;
        * ``first_action`` is the first formal play/pass after opening;
        * ``worker_failures`` and ``advice_failures`` come from advice.jsonl;
        * ``advice_summary`` is a compact terminal-status sequence used by the
          cross-mode divergence report.
        """

        artifact_paths = getattr(result, "artifact_paths", None)
        if not isinstance(artifact_paths, Mapping):
            raw_paths = payload.get("artifact_paths", {})
            artifact_paths = raw_paths if isinstance(raw_paths, Mapping) else {}
        timeline_path = _path_from_mapping(artifact_paths, "timeline.jsonl")
        advice_path = _path_from_mapping(artifact_paths, "advice.jsonl")
        timeline_events = _read_json_lines(timeline_path)
        advice_events = _read_json_lines(advice_path)

        opening = payload.get("opening")
        opening = opening if isinstance(opening, dict) else {}
        confirmed = opening.get("confirmed")
        confirmed = confirmed if isinstance(confirmed, dict) else {}
        lead_player = confirmed.get("lead_player")
        if lead_player is None:
            lead_player = opening.get("lead_player")
        if lead_player is None:
            initial = next(
                (event for event in timeline_events
                 if event.get("event_type") == "initial_state_confirmed"),
                None,
            )
            initial_payload = initial.get("payload") if isinstance(initial, dict) else {}
            if isinstance(initial_payload, dict):
                lead_player = initial_payload.get("lead_player") or initial.get("actor")

        first_action = next(
            (
                _json_safe(event)
                for event in timeline_events
                if event.get("event_type") in {"player_played", "player_passed"}
            ),
            None,
        )
        advice_summary = _advice_summary(advice_events)
        worker_failures = _failure_events(advice_events, worker_only=True)
        advice_failures = _failure_events(advice_events, worker_only=False)
        payload.update({
            "lead_player": lead_player,
            "first_action": first_action,
            "worker_failures": worker_failures,
            "advice_failures": advice_failures,
            "advice_summary": advice_summary,
        })
        return _json_safe(payload)

    @classmethod
    def _build_summary(cls, source: Path, results: dict[str, object], *, environment: dict[str, object], cancelled: bool) -> dict[str, object]:
        modes = {mode: cls._result_dict(result) for mode, result in results.items()}
        return _json_safe({
            "schema": "guandan.offline-diagnostic-replay/1",
            "input_path": source,
            "runtime": "live_v2",
            "legacy_orchestrator_used": False,
            "cancelled": bool(cancelled),
            "environment": environment,
            "modes": modes,
            "comparison": {
                "synchronous_status": str(getattr(results.get("synchronous"), "status", "missing")),
                "latest_status": str(getattr(results.get("latest"), "status", "missing")),
                "advice_divergence": _compare_advice_summaries(modes),
                "purpose": "compare deterministic synchronous recognition with production latest-only scheduling",
            },
        })


    @staticmethod
    def _markdown_report(summary: dict[str, object]) -> str:
        lines = [
            "# 离线对局诊断报告",
            "",
            f"- 输入：`{summary.get('input_path')}`",
            f"- 运行时：`{summary.get('runtime')}`",
            f"- 是否取消：`{summary.get('cancelled')}`",
            "",
            "## 两种回放模式",
            "",
        ]
        modes = summary.get("modes", {})
        if isinstance(modes, dict):
            for mode, value in modes.items():
                item = value if isinstance(value, dict) else {}
                lines.extend((
                    f"### {mode}",
                    f"- 状态：`{item.get('status', 'unknown')}`",
                    f"- 原因：`{item.get('status_reason', '')}`",
                    f"- 帧数：`{item.get('frame_count', 0)}`",
                    f"- 处理回合：`{item.get('processed_turn_count', 0)}`",
                    f"- 首出方：`{item.get('lead_player', 'unknown')}`",
                    f"- 首个 action：`{_markdown_value(item.get('first_action'))}`",
                    f"- 建议状态：`{item.get('advice_statuses', {})}`",
                    f"- worker failures：`{_markdown_value(item.get('worker_failures', []))}`",
                    f"- advice failures：`{_markdown_value(item.get('advice_failures', []))}`",
                    f"- advice summary：`{_markdown_value(item.get('advice_summary', {}))}`",
                    "",
                ))
        lines.extend((
            "## 解释",
            "",
            "- `latest`：模拟生产环境的异步 latest-only 识别链路。",
            "- `synchronous`：同一录像的同步识别对照链路。",
            "- 两者差异可用于区分识别问题与异步调度/丢帧问题。",
            f"- advice divergence：`{_markdown_value((summary.get('comparison') or {}).get('advice_divergence', {}))}`",
        ))
        return "\n".join(lines) + "\n"


_FORMAL_ACTION_TYPES = {"player_played", "player_passed"}
_ADVICE_NONTERMINAL_STATUSES = {"requested", "worker_started"}
_ADVICE_FAILURE_STATUSES = {"failed", "timeout", "stale", "worker_failed", "worker_crashed"}
_WORKER_FAILURE_STATUSES = {"worker_failed", "worker_crashed"}


def _json_safe(value: object) -> object:
    """Recursively convert replay values to strict ``json.dumps`` values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (Path, os.PathLike)):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            safe_key = _json_safe(key)
            if not isinstance(safe_key, (str, int, float, bool)) and safe_key is not None:
                safe_key = str(safe_key)
            converted[str(safe_key)] = _json_safe(item)
        return converted
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        custom_to_dict = getattr(value, "to_dict", None)
        if callable(custom_to_dict):
            return _json_safe(custom_to_dict())
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    custom_to_dict = getattr(value, "to_dict", None)
    if callable(custom_to_dict):
        return _json_safe(custom_to_dict())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _path_from_mapping(mapping: Mapping[object, object], name: str) -> Path | None:
    value = mapping.get(name)
    if value is None:
        return None
    try:
        return Path(value)
    except TypeError:
        return None


def _read_json_lines(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def _failure_events(events: list[dict[str, object]], *, worker_only: bool) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for event in events:
        status = str(event.get("status", ""))
        failure_code = str(event.get("failure_code", ""))
        failure_type = str(event.get("failure_type", ""))
        is_worker_failure = (
            status in _WORKER_FAILURE_STATUSES
            or failure_code.startswith("worker_")
            or "worker" in failure_type.casefold()
        )
        if worker_only and not is_worker_failure:
            continue
        if not worker_only and status not in _ADVICE_FAILURE_STATUSES and not failure_code and not failure_type:
            continue
        result.append({
            key: _json_safe(event[key])
            for key in (
                "request_id", "opportunity_id", "status", "turn_id",
                "failure_code", "failure_type", "message", "worker_generation",
                "worker_pid",
            )
            if key in event
        })
    return result


def _advice_summary(events: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[str]] = {}
    for event in events:
        request_id = str(event.get("request_id") or event.get("opportunity_id") or "")
        if not request_id:
            continue
        grouped.setdefault(request_id, []).append(str(event.get("status", "unknown")))
    terminal_statuses: list[str] = []
    request_sequences: list[dict[str, object]] = []
    for request_id, statuses in grouped.items():
        terminal = next(
            (status for status in reversed(statuses) if status not in _ADVICE_NONTERMINAL_STATUSES),
            statuses[-1] if statuses else "unknown",
        )
        terminal_statuses.append(terminal)
        request_sequences.append({
            "request_id": request_id,
            "statuses": statuses,
            "terminal_status": terminal,
        })
    return {
        "request_count": len(grouped),
        "event_count": len(events),
        "terminal_statuses": terminal_statuses,
        "terminal_status_counts": {
            status: terminal_statuses.count(status)
            for status in sorted(set(terminal_statuses))
        },
        "requests": request_sequences,
    }


def _compare_advice_summaries(modes: Mapping[object, object]) -> dict[str, object]:
    latest = modes.get("latest") if isinstance(modes, Mapping) else None
    synchronous = modes.get("synchronous") if isinstance(modes, Mapping) else None
    latest_summary = latest.get("advice_summary", {}) if isinstance(latest, Mapping) else {}
    synchronous_summary = synchronous.get("advice_summary", {}) if isinstance(synchronous, Mapping) else {}
    latest_statuses = list(latest_summary.get("terminal_statuses", [])) if isinstance(latest_summary, Mapping) else []
    synchronous_statuses = list(synchronous_summary.get("terminal_statuses", [])) if isinstance(synchronous_summary, Mapping) else []
    differences = [
        {
            "request_index": index,
            "latest": latest_statuses[index] if index < len(latest_statuses) else None,
            "synchronous": synchronous_statuses[index] if index < len(synchronous_statuses) else None,
        }
        for index in range(max(len(latest_statuses), len(synchronous_statuses)))
        if (latest_statuses[index] if index < len(latest_statuses) else None)
        != (synchronous_statuses[index] if index < len(synchronous_statuses) else None)
    ]
    return {
        "status": "divergent" if differences else "same",
        "latest_terminal_statuses": latest_statuses,
        "synchronous_terminal_statuses": synchronous_statuses,
        "differences": differences,
    }


def _markdown_value(value: object) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))


__all__ = ["OfflineDiagnosticReplayResult", "OfflineDiagnosticReplayService"]

