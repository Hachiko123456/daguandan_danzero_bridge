from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from daguandan_bridge.live.session_trim import trim_session_after_frame


def _write_session(directory, *, status: str = "sealed"):
    video_directory = directory / "video"
    video_directory.mkdir(parents=True)
    video_path = video_directory / "game.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (32, 32),
    )
    assert writer.isOpened()
    for index in range(4):
        writer.write(np.full((32, 32, 3), index * 40, dtype=np.uint8))
    writer.release()
    index_lines = [
        {
            "frame_index": index,
            "monotonic_ms": index * 100,
            "wall_time": f"2026-08-21T00:00:0{index}+08:00",
            "dropped_before": 0,
        }
        for index in range(4)
    ]
    (video_directory / "frame_index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in index_lines),
        encoding="utf-8",
    )
    (directory / "recognition_trace.jsonl").write_text(
        "".join(
            json.dumps(
                {"captured_at": record["wall_time"], "frame": record["frame_index"]}
            )
            + "\n"
            for record in index_lines
        ),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps({"status": status, "frame_count": 4}), encoding="utf-8"
    )


def test_trim_session_keeps_only_records_after_the_requested_frame(tmp_path):
    session = tmp_path / "session"
    _write_session(session)

    result = trim_session_after_frame(session, after_frame_index=1)

    assert result.source_frame_count == 4
    assert result.retained_frame_count == 2
    records = [
        json.loads(line)
        for line in (session / "video" / "frame_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert records == [
        {
            "frame_index": 0,
            "monotonic_ms": 200,
            "wall_time": "2026-08-21T00:00:02+08:00",
            "dropped_before": 0,
            "source_frame_index": 2,
        },
        {
            "frame_index": 1,
            "monotonic_ms": 300,
            "wall_time": "2026-08-21T00:00:03+08:00",
            "dropped_before": 0,
            "source_frame_index": 3,
        },
    ]
    trace = [
        json.loads(line)
        for line in (session / "recognition_trace.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["frame"] for record in trace] == [2, 3]
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 2
    assert manifest["retention"]["discarded_through_source_frame_index"] == 1
    capture = cv2.VideoCapture(str(session / "video" / "game.avi"))
    frames = [capture.read()[1] for _ in range(2)]
    capture.release()
    assert all(frame is not None for frame in frames)
    assert int(frames[0].mean()) > 60
    assert int(frames[1].mean()) > 100


def test_trim_session_rejects_an_active_recording(tmp_path):
    session = tmp_path / "session"
    _write_session(session, status="running")

    with pytest.raises(RuntimeError, match="仍在写入"):
        trim_session_after_frame(session, after_frame_index=1)
