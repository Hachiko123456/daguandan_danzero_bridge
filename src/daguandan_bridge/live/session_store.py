from __future__ import annotations

import gzip
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterable
from uuid import uuid4

from ..profiles import normalize_profile_name
from ..storage import atomic_write_json
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
        self.observations_part_path = self.directory / "observations.jsonl.part"
        self.observations_gzip_path = self.directory / "observations.jsonl.gz"
        self.incidents_directory = self.directory / "incidents"
        self._lock = RLock()
        self._started = False
        self._sealed = False
        self._incident_ids: list[str] = []

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

    def create_incident(
        self,
        *,
        reason: str,
        state_before: dict[str, object],
        state_after: dict[str, object],
        observations: list[dict[str, object]],
        frame_paths: Iterable[Path] = (),
        engine_input: dict[str, object] | None = None,
    ) -> Path:
        with self._lock:
            self._ensure_writable()
            incident_id = f"INC-{len(self._incident_ids) + 1:04d}"
            path = self.incidents_directory / incident_id
            path.mkdir()
            copied_frames = self._copy_incident_frames(path, frame_paths)
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
            }
            atomic_write_json(path / "incident.json", incident)
            atomic_write_json(path / "state_before.json", state_before)
            atomic_write_json(path / "state_after.json", state_after)
            atomic_write_json(path / "observations.json", observations)
            if engine_input is not None:
                atomic_write_json(path / "engine_input.json", engine_input)
            (path / "llm_report.md").write_text(
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
            self._incident_ids.append(incident_id)
            self._update_manifest({"incidents": list(self._incident_ids)})
            return path

    def seal(self, *, frame_count: int, dropped_frames: int) -> None:
        with self._lock:
            self._ensure_writable()
            with self.observations_part_path.open("rb") as source:
                with gzip.open(self.observations_gzip_path, "wb") as target:
                    shutil.copyfileobj(source, target)
            self.observations_part_path.unlink()
            self._update_manifest(
                {
                    "status": "sealed",
                    "finished_at": _now_text(),
                    "frame_count": frame_count,
                    "dropped_frames": dropped_frames,
                    "incidents": list(self._incident_ids),
                }
            )
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
        ]
        if has_engine_input:
            files.append("engine_input.json")
        files.extend(frames)
        return (
            f"# 对局异常报告 {incident_id}\n\n"
            f"- 异常原因：`{reason}`\n"
            f"- 相关观察：{', '.join(observation_ids) or '无'}\n"
            f"- 状态变化字段：{', '.join(changed_keys) or '无（状态未推进）'}\n\n"
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
        prefix = (
            f"[{minutes:02d}:{seconds:02d}.{millis:03d}]"
            f"[墩 T{event.trick_id:02d}]"
            f"[回合 R{event.turn_id:03d}]"
        )
        seat = _SEAT_LABELS.get(event.actor or "", event.actor or "系统")
        cards = event.payload.get("cards", [])
        if isinstance(cards, (list, tuple)):
            cards_text = ", ".join(str(card) for card in cards)
        else:
            cards_text = str(cards)
        details = (
            f"置信度={event.confidence:.0%}，"
            f"证据={', '.join(event.evidence_refs) or '无'}"
        )
        if event.event_type == "player_played":
            action = f"{seat}出牌：[{cards_text}]，{details}"
        elif event.event_type == "player_passed":
            action = f"{seat}不出，{details}"
        elif event.event_type == "turn_started":
            action = f"轮到{seat}"
        elif event.event_type == "initial_state_confirmed":
            starter = _SEAT_LABELS.get(
                str(event.payload.get("starter", event.actor or "")), seat
            )
            action = f"初始状态确认，首发：{starter}，{details}"
        elif event.event_type == "advice_ready":
            request_id = event.payload.get("request_id", "未知")
            action = f"DanZero 建议：[{cards_text}]，请求={request_id}，{details}"
        elif event.event_type == "event_correction":
            target = event.payload.get("target_event_id", "未知事件")
            action = f"纠正事件 {target}：{json.dumps(event.payload, ensure_ascii=False)}"
        else:
            action = (
                f"{event.event_type}："
                f"{json.dumps(event.payload, ensure_ascii=False, separators=(',', ':'))}，"
                f"{details}"
            )
        return f"{prefix} {action}"
