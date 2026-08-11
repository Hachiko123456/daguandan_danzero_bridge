from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterable
from uuid import uuid4

from ..profiles import normalize_profile_name
from ..storage import atomic_write_json
from .display_text import event_action_text, event_prefix, reasons_text
from .models import LiveEvent


SCHEMA_VERSION = 1
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_SEAT_LABELS = {
    "self": "我方",
    "right": "右家",
    "opposite": "对家",
    "left": "左家",
}


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _new_session_id() -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"game_{stamp}_{uuid4().hex[:6]}"


def _validate_session_id(session_id: str) -> str:
    value = session_id.strip()
    if not value or _SESSION_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("session_id 只能包含英文字母、数字、下划线和连字符")
    return value


def _append_json_line(path: Path, record: dict[str, object], *, durable: bool) -> None:
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())


def read_json_lines(path: Path) -> list[dict[str, object]]:
    """Read JSONL while tolerating only an incomplete final process-crash write."""

    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonblank = [(index, line) for index, line in enumerate(lines) if line.strip()]
    records: list[dict[str, object]] = []
    for position, (_, line) in enumerate(nonblank):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if position == len(nonblank) - 1:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError("JSONL 的每一行都必须是 JSON 对象")
        records.append(value)
    return records


class LiveSessionStore:
    """Own all append-only diagnostics for exactly one game session."""

    @classmethod
    def recover_incomplete_sessions(
        cls,
        profiles_root: Path,
        profile_name: str,
    ) -> tuple[Path, ...]:
        """Mark sessions left running by a previous process as aborted.

        Recovery deliberately preserves ``*.part`` and all append-only files so
        the last readable observations remain available for diagnosis.
        """

        profile = normalize_profile_name(profile_name)
        sessions_root = Path(profiles_root) / profile / "sessions"
        if not sessions_root.is_dir():
            return ()

        recovered: list[Path] = []
        for manifest_path in sorted(sessions_root.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict) or manifest.get("status") != "running":
                continue
            try:
                owner_pid = int(manifest.get("owner_pid", 0))
            except (TypeError, ValueError):
                owner_pid = 0
            if owner_pid > 0 and cls._process_is_alive(owner_pid):
                continue
            manifest.update(
                {
                    "status": "aborted",
                    "recovered_at": _now_text(),
                    "recovery_reason": "previous_process_did_not_seal",
                }
            )
            atomic_write_json(manifest_path, manifest)
            recovered.append(manifest_path.parent)
        return tuple(recovered)

    @staticmethod
    def _process_is_alive(process_id: int) -> bool:
        if int(process_id) == os.getpid():
            return True
        try:
            os.kill(int(process_id), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except SystemError:
            return False
        except OSError:
            return False
        return True

    def __init__(
        self,
        profiles_root: Path,
        profile_name: str,
        *,
        session_id: str | None = None,
    ) -> None:
        self.profile_name = normalize_profile_name(profile_name)
        self.session_id = _validate_session_id(session_id or _new_session_id())
        self.directory = (
            Path(profiles_root)
            / self.profile_name
            / "sessions"
            / self.session_id
        )
        self.manifest_path = self.directory / "manifest.json"
        self.timeline_path = self.directory / "timeline.jsonl"
        self.timeline_markdown_path = self.directory / "timeline.md"
        self.advice_path = self.directory / "advice.jsonl"
        self.decisions_path = self.directory / "decisions.jsonl"
        self.observations_part_path = self.directory / "observations.jsonl.part"
        self.observations_gzip_path = self.directory / "observations.jsonl.gz"
        self.incidents_directory = self.directory / "incidents"
        self._lock = RLock()
        self._started = False
        self._sealed = False
        self._incident_ids: list[str] = []
        self._decisions: dict[str, dict[str, object]] = {}

    def start(self, manifest: dict[str, object]) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("对局存储已经启动")
            if self.directory.exists():
                raise FileExistsError(f"对局目录已经存在：{self.directory}")
            self.directory.mkdir(parents=True)
            self.incidents_directory.mkdir()
            for path in (
                self.timeline_path,
                self.advice_path,
                self.decisions_path,
                self.observations_part_path,
            ):
                path.touch()
            self.timeline_markdown_path.write_text(
                f"# 对局时间线：{self.session_id}\n\n",
                encoding="utf-8",
            )
            document = dict(manifest)
            document.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": self.session_id,
                    "profile": self.profile_name,
                    "status": "running",
                    "started_at": _now_text(),
                    "incidents": [],
                }
            )
            atomic_write_json(self.manifest_path, document)
            self._started = True

    def append_event(self, event: LiveEvent) -> None:
        with self._lock:
            self._ensure_writable()
            if event.session_id != self.session_id:
                raise ValueError("事件 session_id 与当前对局不一致")
            record = event.to_dict()
            record["schema_version"] = SCHEMA_VERSION
            _append_json_line(self.timeline_path, record, durable=True)
            with self.timeline_markdown_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(self._format_timeline_event(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def append_advice(self, record: dict[str, object]) -> None:
        with self._lock:
            self._ensure_writable()
            payload = dict(record)
            payload.setdefault("schema_version", SCHEMA_VERSION)
            payload.setdefault("session_id", self.session_id)
            payload.setdefault("wall_time", _now_text())
            _append_json_line(self.advice_path, payload, durable=False)

    def append_observation(self, record: dict[str, object]) -> None:
        with self._lock:
            self._ensure_writable()
            payload = dict(record)
            payload.setdefault("schema_version", SCHEMA_VERSION)
            payload.setdefault("session_id", self.session_id)
            _append_json_line(self.observations_part_path, payload, durable=False)

    def upsert_decision(self, record: dict[str, object]) -> None:
        """Atomically maintain one correlated training record per self decision."""

        with self._lock:
            self._ensure_writable()
            decision_id = str(record.get("decision_id", "")).strip()
            if not decision_id:
                raise ValueError("decision_id is required")
            current = dict(self._decisions.get(decision_id, {}))
            current.update(record)
            current.setdefault("schema", "guandan.live-decision/1")
            current.setdefault("session_id", self.session_id)
            self._decisions[decision_id] = current
            payload = "".join(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for _, value in sorted(self._decisions.items())
            )
            temp_path = self.decisions_path.with_name(
                f".{self.decisions_path.name}.{uuid4().hex}.tmp"
            )
            try:
                with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._replace_decisions_with_retry(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

    def _replace_decisions_with_retry(self, temp_path: Path) -> None:
        """Publish on Windows despite short-lived reader/antivirus locks."""

        for attempt in range(8):
            try:
                os.replace(temp_path, self.decisions_path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.01 * (attempt + 1))

    def create_incident(
        self,
        *,
        reason: str,
        state_before: dict[str, object],
        state_after: dict[str, object],
        observations: list[dict[str, object]],
        frame_paths: Iterable[Path] = (),
        engine_input: dict[str, object] | None = None,
        trigger_ms: int | None = None,
    ) -> Path:
        with self._lock:
            self._ensure_writable()
            incident_number = len(self._incident_ids) + 1
            while (self.incidents_directory / f"INC-{incident_number:04d}").exists():
                incident_number += 1
            incident_id = f"INC-{incident_number:04d}"
            path = self.incidents_directory / incident_id
            # Build the complete incident in a hidden sibling directory and
            # publish it with one rename.  Readers (including
            # the UI and diagnostic scripts) must never observe an incident
            # directory before engine_input.json and the other evidence files
            # have been written.
            staging_path = self.directory / f".{incident_id}.{uuid4().hex}.tmp"
            staging_path.mkdir()
            try:
                copied_frames = self._copy_incident_frames(staging_path, frame_paths)
                incident = {
                    "schema_version": SCHEMA_VERSION,
                    "incident_id": incident_id,
                    "session_id": self.session_id,
                    "reason": reason,
                    "wall_time": _now_text(),
                    "observation_ids": [
                        item.get("id") for item in observations if item.get("id")
                    ],
                    "frames": copied_frames,
                    "state_advanced": state_before != state_after,
                    "media_manifest": "media.json",
                    "media_error": "media_error.json",
                }
                if trigger_ms is not None:
                    incident["trigger_ms"] = int(trigger_ms)
                atomic_write_json(staging_path / "incident.json", incident)
                atomic_write_json(staging_path / "state_before.json", state_before)
                atomic_write_json(staging_path / "state_after.json", state_after)
                atomic_write_json(staging_path / "observations.json", observations)
                _append_json_line(
                    staging_path / "occurrences.jsonl",
                    {
                        "monotonic_ms": None,
                        "wall_time": incident["wall_time"],
                        "reason": reason,
                        "coalesced": False,
                    },
                    durable=False,
                )
                if engine_input is not None:
                    atomic_write_json(staging_path / "engine_input.json", engine_input)
                (staging_path / "llm_report.md").write_text(
                    self._format_incident_report(
                        incident_id,
                        reason,
                        state_before,
                        state_after,
                        observations,
                        copied_frames,
                        engine_input is not None,
                    ),
                    encoding="utf-8",
                )
                staging_path.replace(path)
            except BaseException:
                shutil.rmtree(staging_path, ignore_errors=True)
                raise
            self._incident_ids.append(incident_id)
            self._update_manifest({"incidents": list(self._incident_ids)})
            return path

    def append_incident_occurrence(
        self,
        incident_directory: Path,
        *,
        monotonic_ms: int,
        reason: str,
    ) -> None:
        with self._lock:
            self._ensure_writable()
            path = Path(incident_directory)
            if path.parent != self.incidents_directory or not path.is_dir():
                raise ValueError("事故目录不属于当前对局")
            _append_json_line(
                path / "occurrences.jsonl",
                {
                    "monotonic_ms": int(monotonic_ms),
                    "wall_time": _now_text(),
                    "reason": str(reason),
                    "coalesced": True,
                },
                durable=False,
            )

    def seal(
        self,
        *,
        frame_count: int,
        dropped_frames: int,
        metrics: dict[str, object] | None = None,
        incident_media_failures: Iterable[dict[str, object]] = (),
    ) -> None:
        with self._lock:
            self._ensure_writable()
            with self.observations_part_path.open("rb") as source:
                with gzip.open(self.observations_gzip_path, "wb") as target:
                    shutil.copyfileobj(source, target)
            self.observations_part_path.unlink()
            changes: dict[str, object] = {
                    "status": "sealed",
                    "finished_at": _now_text(),
                    "frame_count": frame_count,
                    "dropped_frames": dropped_frames,
                    "incidents": list(self._incident_ids),
                }
            if metrics is not None:
                changes["performance_metrics"] = dict(metrics)
            failures = [dict(item) for item in incident_media_failures]
            if failures:
                changes["incident_media_failures"] = failures
            self._update_manifest(changes)
            self._sealed = True

    def _ensure_writable(self) -> None:
        if not self._started:
            raise RuntimeError("请先启动对局存储")
        if self._sealed:
            raise RuntimeError("对局已经封存")

    def _update_manifest(self, changes: dict[str, object]) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.update(changes)
        atomic_write_json(self.manifest_path, manifest)

    @staticmethod
    def _copy_incident_frames(path: Path, frame_paths: Iterable[Path]) -> list[str]:
        frames_directory = path / "frames"
        copied: list[str] = []
        for index, raw_source in enumerate(frame_paths, start=1):
            source = Path(raw_source)
            if not source.is_file():
                continue
            frames_directory.mkdir(exist_ok=True)
            destination = frames_directory / f"{index:04d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            copied.append(destination.relative_to(path).as_posix())
        return copied

    @staticmethod
    def _format_incident_report(
        incident_id: str,
        reason: str,
        state_before: dict[str, object],
        state_after: dict[str, object],
        observations: list[dict[str, object]],
        frames: list[str],
        has_engine_input: bool,
    ) -> str:
        observation_ids = [
            str(item.get("id")) for item in observations if item.get("id")
        ]
        changed_keys = sorted(
            key
            for key in state_before.keys() | state_after.keys()
            if state_before.get(key) != state_after.get(key)
        )
        files = [
            "incident.json",
            "state_before.json",
            "state_after.json",
            "observations.json",
            "occurrences.jsonl",
            "media.json（媒体成功后原子生成）",
            "media_error.json（仅媒体失败时生成）",
            "clip.avi（媒体成功时）",
            "contact_sheet.png（媒体成功时）",
            "frames/trigger.png（媒体成功时）",
        ]
        if has_engine_input:
            files.append("engine_input.json")
        files.extend(frames)
        return (
            f"# 对局异常报告 {incident_id}\n\n"
            f"- 异常原因：{reasons_text(reason)}\n"
            f"- 相关观察：{', '.join(observation_ids) or '无'}\n"
            f"- 状态变化字段：{', '.join(changed_keys) or '无（状态未推进）'}\n\n"
            "- 说明：识别不确定时状态机不会推进，因此状态前后相同是预期的安全行为。\n\n"
            "## 建议排查顺序\n\n"
            "1. 查看 `observations.json` 中的候选、置信度和采用/拒绝原因。\n"
            "2. 比较 `state_before.json` 与 `state_after.json`。\n"
            "3. 对照关键帧确认是动画遮挡、模板误识别还是状态机约束问题。\n\n"
            "## 文件索引\n\n"
            + "".join(f"- `{name}`\n" for name in files)
        )

    @staticmethod
    def _format_timeline_event(event: LiveEvent) -> str:
        elapsed = max(event.monotonic_ms, 0)
        minutes, remainder = divmod(elapsed, 60_000)
        seconds, millis = divmod(remainder, 1_000)
        prefix = f"[{minutes:02d}:{seconds:02d}.{millis:03d}]{event_prefix(event)}"
        details = (
            f"置信度={event.confidence:.0%}，"
            f"证据={', '.join(event.evidence_refs) or '无'}"
        )
        return f"{prefix} {event_action_text(event)}，{details}"
