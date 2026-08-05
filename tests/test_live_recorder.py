from __future__ import annotations

import json

import cv2
import numpy as np

from daguandan_bridge.live.recorder import SessionRecorder


def _read_json_lines(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_recorder_writes_playable_video_and_explicit_timestamp_index(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(320, 180), fps=10)
    for index in range(5):
        recorder.write_frame(
            np.full((180, 320, 3), index * 20, np.uint8),
            captured_monotonic_ms=index * 100,
            wall_time=f"t{index}",
        )

    result = recorder.close()

    assert result.frame_count == 5
    assert result.dropped_frames == 0
    capture = cv2.VideoCapture(str(result.video_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    finally:
        capture.release()
    index = _read_json_lines(result.index_path)
    assert [row["monotonic_ms"] for row in index] == [0, 100, 200, 300, 400]
    assert [row["frame_index"] for row in index] == list(range(5))


def test_bad_frame_is_reported_and_next_index_records_drop(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10)

    warning = recorder.write_frame(
        np.zeros((10, 10, 3), np.uint8),
        captured_monotonic_ms=0,
        wall_time="bad",
    )
    recorder.write_frame(
        np.zeros((32, 64, 3), np.uint8),
        captured_monotonic_ms=100,
        wall_time="ok",
    )
    result = recorder.close()

    assert warning is not None
    assert warning.reason == "frame_size_mismatch"
    assert result.dropped_frames == 1
    assert _read_json_lines(result.index_path)[0]["dropped_before"] == 1


def test_incident_media_uses_ring_buffer_and_saves_png_evidence(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, buffer_seconds=2)
    for index in range(11):
        recorder.write_frame(
            np.full((32, 64, 3), index * 10, np.uint8),
            captured_monotonic_ms=index * 100,
            wall_time=f"t{index}",
        )

    incident_dir = tmp_path / "incidents" / "INC-0001"
    media = recorder.save_incident_media(
        incident_dir,
        trigger_ms=500,
        before_ms=300,
        after_ms=300,
    )
    evidence = recorder.save_evidence_frame(
        incident_dir / "frames" / "confirmed.png",
        np.full((32, 64, 3), 127, np.uint8),
    )
    recorder.close()

    assert media.clip_path.is_file()
    assert media.contact_sheet_path.is_file()
    assert media.trigger_frame_path.is_file()
    assert evidence.is_file()
    assert cv2.imread(str(media.contact_sheet_path)) is not None
