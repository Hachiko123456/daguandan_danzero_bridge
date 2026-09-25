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
from .session_replay_audit import prepared_replay_input


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
            with prepared_replay_input(
                source,
                fallback_profile=self.profiles_root / self.profile_name,
                temporary_root=Path(temporary),
            ) as replay_input:
                session = replay_input.session
                profile_root = replay_input.profile_path
                configuration = {
                    "source_health": replay_input.source_health,
                    "profile_resource_match": replay_input.profile_resource_match,
                    "input_kind": replay_input.input_kind,
                    "embedded_snapshot": replay_input.embedded_snapshot,
                }
                recognition = ScreenshotRecognitionService(
                    AnnotationService(profile_root.parent, profile_root.name),
                    TemplateService(profile_root.parent, profile_root.name),
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
                        profile_root=profile_root,
                        output_root=output / mode,
                        on_progress=emit_progress,
                        stop_requested=stop_requested,
                        vision_delivery=mode,
                        pace_to_recording_timestamps=mode == "latest",
                    )
                    mode_results[mode] = result
                    if str(getattr(result, "status", "")) == "incomplete" and str(getattr(result, "status_reason", "")) == "replay_cancelled":
                        cancelled = True
                        break
        summary = self._build_summary(
            source, mode_results, environment=environment or {},
            cancelled=cancelled, configuration=configuration,
        )
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
    def _prepare_session(
        source: Path, temporary: Path,
    ) -> tuple[Path, Path, str, dict[str, object]]:
        """Compatibility adapter for callers of the old private helper.

        New execution goes through ``prepared_replay_input`` directly; this
        adapter keeps older integrations working without maintaining a second
        ZIP/profile extraction implementation.
        """
        with prepared_replay_input(
            source,
            fallback_profile=PROFILES_ROOT / "tencent_daguandan",
            temporary_root=temporary,
        ) as replay_input:
            profile = replay_input.profile_path
            return (
                replay_input.session,
                profile.parent,
                profile.name,
                {
                    "source_health": replay_input.source_health,
                    "profile_resource_match": replay_input.profile_resource_match,
                    "input_kind": replay_input.input_kind,
                    "embedded_snapshot": replay_input.embedded_snapshot,
                },
            )

    @staticmethod
    def _configuration_status(
        session: Path, profiles_root: Path, profile_name: str,
        *, embedded: bool = False,
    ) -> dict[str, object]:
        """Legacy compatibility helper; execution uses the shared input policy."""
        manifest_path = session / "manifest.json"
        source_manifest: dict[str, object] = {}
        if manifest_path.is_file():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    source_manifest = value
            except (OSError, json.JSONDecodeError):
                pass
        current = recognition_resource_identity(profiles_root, profile_name)
        source_identity = source_manifest.get("recognition_resource_identity")
        source_config = source_manifest.get("configuration_hash")
        current_config = None
        profile_path = Path(profiles_root) / profile_name / "profile.json"
        if profile_path.is_file():
            import hashlib
            current_config = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if isinstance(source_identity, dict) and isinstance(current, dict):
            match = source_identity.get("sha256") == current.get("sha256")
        elif source_config and current_config:
            match = str(source_config) == str(current_config)
        else:
            match = None
        return {
            "status": "embedded_match" if embedded and match is True else
                "embedded_snapshot" if embedded else
                "match" if match is True else
                "mismatch" if match is False else "unavailable",
            "embedded_snapshot": bool(embedded),
            "source_configuration_hash": source_config,
            "current_configuration_hash": current_config,
            "source_resource_identity": source_identity,
            "current_resource_identity": current,
            "profile_name": profile_name,
        }

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
                lead_player = initial_payload.get("lead_player") or (initial or {}).get("actor")

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
    def _build_summary(
        cls, source: Path, results: dict[str, object], *,
        environment: dict[str, object], cancelled: bool,
        configuration: dict[str, object],
    ) -> dict[str, object]:
        modes = {mode: cls._result_dict(result) for mode, result in results.items()}
        return _json_safe({
            "schema": "guandan.offline-diagnostic-replay/1",
            "input_path": source,
            "runtime": "live_v2",
            "legacy_orchestrator_used": False,
            "cancelled": bool(cancelled),
            "environment": environment,
            "configuration": _json_safe(configuration),
            "source_health": _json_safe(configuration.get("source_health", {})),
            "profile_resource_match": _json_safe(configuration.get("profile_resource_match", {})),
            "visual_replay": {
                "status": "completed" if len(results) == 2 and not cancelled else "incomplete",
                "modes": {mode: item.get("status", "unknown") for mode, item in modes.items()},
            },
            "truth_comparison": {
                "status": "not_available",
                "strict_regression": False,
                "reason": "diagnostic ZIP replay has no TruthLog comparison input",
            },
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
        raw_configuration = summary.get("configuration")
        configuration = raw_configuration if isinstance(raw_configuration, dict) else {}
        source_identity = configuration.get("source_resource_identity")
        current_identity = configuration.get("current_resource_identity")
        compact_configuration = {
            key: configuration.get(key)
            for key in (
                "status", "embedded_snapshot", "profile_name",
                "source_configuration_hash", "current_configuration_hash",
            )
        }
        if isinstance(source_identity, dict):
            compact_configuration["source_resource_sha256"] = source_identity.get("sha256")
        if isinstance(current_identity, dict):
            compact_configuration["current_resource_sha256"] = current_identity.get("sha256")
        lines = [
            "# 离线对局诊断报告",
            "",
            f"- 输入：`{summary.get('input_path')}`",
            f"- 运行时：`{summary.get('runtime')}`",
            f"- 是否取消：`{summary.get('cancelled')}`",
            f"- 源数据健康：`{_markdown_value(summary.get('source_health', {}))}`",
            f"- 配置/资源匹配：`{_markdown_value(summary.get('profile_resource_match', compact_configuration))}`",
            f"- 视觉回放：`{_markdown_value(summary.get('visual_replay', {}))}`",
            f"- TruthLog 严格对比：`{_markdown_value(summary.get('truth_comparison', {}))}`",
            f"- 回放配置：`{_markdown_value(compact_configuration)}`",
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
            "- `latest`：按录像原始时间戳模拟生产环境的异步 latest-only 识别链路。",
            "- `synchronous`：同一录像的同步逐帧识别对照链路（不等待原始帧间隔）。",
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

