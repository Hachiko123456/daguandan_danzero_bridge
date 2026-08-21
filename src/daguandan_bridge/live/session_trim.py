from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2

from ..storage import atomic_write_json


@dataclass(frozen=True)
class SessionTrimResult:
    session_directory: Path
    source_frame_count: int
    retained_frame_count: int
    first_retained_source_frame_index: int


def trim_session_after_frame(
    session_directory: Path,
    *,
    after_frame_index: int,
) -> SessionTrimResult:
    """Physically retain only video and waiting traces after a sealed-frame boundary."""

    directory = Path(session_directory)
    manifest_path = directory / "manifest.json"
    video_path = directory / "video" / "game.avi"
    index_path = directory / "video" / "frame_index.jsonl"
    trace_path = directory / "recognition_trace.jsonl"
    manifest = _read_json(manifest_path)
    if str(manifest.get("status", "")) != "sealed":
        raise RuntimeError("对局录像仍在写入，封存后才能裁剪")
    source_records = _read_json_lines(index_path)
    retained_records = [
        record
        for record in source_records
        if int(record.get("frame_index", -1)) > int(after_frame_index)
    ]
    if not retained_records:
        raise RuntimeError("指定边界之后没有可保留的录像帧")

    _rewrite_video(video_path, source_records, retained_records)
    rebased_records = [
        {
            **record,
            "frame_index": index,
            "source_frame_index": int(record["frame_index"]),
        }
        for index, record in enumerate(retained_records)
    ]
    _atomic_write_json_lines(index_path, rebased_records)
    first_wall_time = str(retained_records[0].get("wall_time", ""))
    if trace_path.exists():
        traces = _read_json_lines(trace_path)
        _atomic_write_json_lines(
            trace_path,
            [
                record
                for record in traces
                if _trace_is_not_before(record, first_wall_time)
            ],
        )
    manifest["frame_count"] = len(rebased_records)
    manifest["retention"] = {
        "source_frame_count": len(source_records),
        "discarded_through_source_frame_index": int(after_frame_index),
        "first_retained_source_frame_index": int(retained_records[0]["frame_index"]),
        "first_retained_wall_time": first_wall_time,
    }
    atomic_write_json(manifest_path, manifest)
    return SessionTrimResult(
        session_directory=directory,
        source_frame_count=len(source_records),
        retained_frame_count=len(rebased_records),
        first_retained_source_frame_index=int(retained_records[0]["frame_index"]),
    )


def _rewrite_video(
    video_path: Path,
    source_records: list[dict[str, object]],
    retained_records: list[dict[str, object]],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法读取录像：{video_path}")
    temporary = video_path.with_name(f".{video_path.name}.{uuid4().hex}.tmp.avi")
    writer = None
    retained_source_indexes = {
        int(record["frame_index"]) for record in retained_records
    }
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError("录像元数据无效，无法安全裁剪")
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建裁剪后的录像：{temporary}")
        for record in source_records:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("录像帧数量与索引不一致，已放弃裁剪")
            if int(record["frame_index"]) in retained_source_indexes:
                writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    try:
        os.replace(temporary, video_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取对局元数据：{path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"对局元数据格式无效：{path}")
    return data


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"无法读取对局索引：{path}") from exc
    records: list[dict[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"对局索引包含损坏记录：{path}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"对局索引包含非对象记录：{path}")
        records.append(record)
    return records


def _atomic_write_json_lines(path: Path, records: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _trace_is_not_before(record: dict[str, object], first_wall_time: str) -> bool:
    timestamp = str(record.get("captured_at") or record.get("wall_time") or "")
    return bool(timestamp) and timestamp >= first_wall_time
