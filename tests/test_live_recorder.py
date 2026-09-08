from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np

from daguandan_bridge.live.recorder import SessionRecorder, audit_recording_integrity


def _read_json_lines(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_recording_quota_stops_media_once_without_losing_existing_video(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=100000)
    random = np.random.default_rng(17)
    warnings = []
    for index in range(200):
        warning = recorder.write_frame(
            random.integers(0, 256, (32, 64, 3), dtype=np.uint8), index * 100, str(index)
        )
        if warning:
            warnings.append(warning)
    assert len(warnings) == 1
    assert warnings[0].reason == "recording_capacity_reached"
    assert 0 < recorder.frame_count < 200
    result = recorder.close()
    assert result.video_path.stat().st_size <= 100000
    assert result.integrity["status"] == "PARTIAL"
    assert result.integrity["stop_reason"] == "recording_capacity_reached"
    assert result.integrity["decodable_frame_count"] == result.frame_count
    assert len(_read_json_lines(result.index_path)) == result.frame_count


def test_already_full_recording_quota_creates_no_new_video(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=0)
    frame = np.zeros((32, 64, 3), dtype=np.uint8)
    assert recorder.write_frame(frame, 100, "first").reason == "recording_capacity_reached"
    assert recorder.write_frame(frame, 200, "second") is None
    result = recorder.close()
    assert result.frame_count == 0
    assert not result.video_path.exists()
    assert result.integrity["recording_state"] == "partial"
    assert result.integrity["stop_reason"] == "recording_capacity_reached"


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
    assert result.integrity["status"] == "PASS"
    assert result.integrity["indexed_frame_count"] == 5
    assert result.integrity["decodable_frame_count"] == 5


def test_integrity_audit_reports_indexed_tail_that_cannot_be_decoded(
    tmp_path,
    monkeypatch,
):
    video = tmp_path / "game.avi"
    video.write_bytes(b"placeholder")
    index = tmp_path / "frame_index.jsonl"
    index.write_text("".join("{}\n" for _ in range(5)), encoding="utf-8")

    class TruncatedCapture:
        def __init__(self, _path):
            self.remaining = 3

        def isOpened(self):
            return True

        def read(self):
            if self.remaining <= 0:
                return False, None
            self.remaining -= 1
            return True, np.zeros((32, 64, 3), np.uint8)

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", TruncatedCapture)

    report = audit_recording_integrity(video, index, writer_frame_count=5)

    assert report["status"] == "FAIL"
    assert report["writer_frame_count"] == 5
    assert report["indexed_frame_count"] == 5
    assert report["decodable_frame_count"] == 3
    assert report["last_decodable_frame_index"] == 2
    assert "indexed_decodable_count_mismatch" in report["issues"]
    json.dumps(report)


def test_close_atomically_exports_missing_tail_from_ring_buffer(tmp_path, monkeypatch):
    recorder = SessionRecorder(
        tmp_path,
        size=(64, 32),
        fps=10,
        buffer_seconds=2,
    )
    frames = [np.full((32, 64, 3), index * 10, np.uint8) for index in range(5)]
    for index, frame in enumerate(frames):
        recorder.write_frame(frame, index * 100, f"t{index}")

    from daguandan_bridge.live import recorder as recorder_module

    monkeypatch.setattr(
        recorder_module,
        "audit_recording_integrity",
        lambda *_args, **_kwargs: {
            "schema": recorder_module.RECORDING_INTEGRITY_SCHEMA,
            "status": "FAIL",
            "writer_frame_count": 5,
            "indexed_frame_count": 5,
            "decodable_frame_count": 3,
            "last_decodable_frame_index": 2,
            "last_decodable_frame_sha256": recorder_module._frame_sha256(frames[2]),
            "issues": ["indexed_decodable_count_mismatch"],
            "index_error": "",
        },
    )

    result = recorder.close()

    recovery = tmp_path / "video" / "tail_recovery"
    assert result.integrity["status"] == "PARTIAL"
    assert result.integrity["recording_state"] == "partial"
    assert result.integrity["tail_recovery"]["status"] == "PARTIAL"
    assert result.integrity["tail_recovery"]["candidate"] is True
    assert result.integrity["tail_recovery"]["path"] == str(recovery)
    assert sorted(path.name for path in (recovery / "frames").glob("*.png")) == [
        "frame_00000003.png",
        "frame_00000004.png",
    ]
    assert len((recovery / "index.jsonl").read_text("utf-8").splitlines()) == 2
    assert len((recovery / "hashes.jsonl").read_text("utf-8").splitlines()) == 2
    assert (recovery / "manifest.json").is_file()
    assert not list((tmp_path / "video").glob(".tail_recovery.*.tmp"))


def test_close_accepts_recovered_only_with_authoritative_tail_proof(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, buffer_seconds=2)
    frames = [np.full((32, 64, 3), index * 10, np.uint8) for index in range(5)]
    for index, frame in enumerate(frames):
        recorder.write_frame(frame, index * 100, f"t{index}")

    from daguandan_bridge.live import recorder as recorder_module

    monkeypatch.setattr(
        recorder_module,
        "audit_recording_integrity",
        lambda *_args, **_kwargs: {
            "schema": recorder_module.RECORDING_INTEGRITY_SCHEMA,
            "status": "FAIL",
            "writer_frame_count": 5,
            "indexed_frame_count": 5,
            "decodable_frame_count": 3,
            "last_decodable_frame_index": 2,
            "last_decodable_frame_sha256": recorder_module._frame_sha256(frames[2]),
            "tail_proof": {
                "status": "verified",
                "kind": "authoritative_packet_sequence",
                "all_prior_frames_verified": True,
                "first_missing_frame_index": 3,
                "last_decodable_frame_index": 2,
                "indexed_frame_count": 5,
            },
            "issues": ["indexed_decodable_count_mismatch"],
            "index_error": "",
        },
    )

    result = recorder.close()

    assert result.integrity["status"] == "RECOVERED"
    assert result.integrity["recording_state"] == "recovered"
    assert result.integrity["recovered_frame_count"] == 2
    assert result.integrity["tail_recovery"]["candidate"] is False
    manifest = json.loads(
        (tmp_path / "video" / "tail_recovery" / "manifest.json").read_text("utf-8")
    )
    assert manifest["status"] == "RECOVERED"
    assert manifest["candidate"] is False


def test_close_marks_ambiguous_or_over_buffer_loss_partial(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, buffer_seconds=0.2)
    frames = [np.full((32, 64, 3), index * 10, np.uint8) for index in range(8)]
    for index, frame in enumerate(frames):
        recorder.write_frame(frame, index * 100, f"t{index}")

    from daguandan_bridge.live import recorder as recorder_module

    monkeypatch.setattr(
        recorder_module,
        "audit_recording_integrity",
        lambda *_args, **_kwargs: {
            "schema": recorder_module.RECORDING_INTEGRITY_SCHEMA,
            "status": "FAIL",
            "writer_frame_count": 8,
            "indexed_frame_count": 8,
            "decodable_frame_count": 3,
            "last_decodable_frame_index": 2,
            "last_decodable_frame_sha256": recorder_module._frame_sha256(frames[2]),
            "issues": ["indexed_decodable_count_mismatch"],
            "index_error": "",
        },
    )

    result = recorder.close()

    assert result.integrity["status"] == "PARTIAL"
    assert result.integrity["tail_recovery"]["status"] == "unavailable"
    assert not (tmp_path / "video" / "tail_recovery").exists()


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


def test_scheduled_incident_media_collects_frames_after_trigger(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, buffer_seconds=2)
    for index in range(6):
        recorder.write_frame(
            np.full((32, 64, 3), index * 10, np.uint8),
            captured_monotonic_ms=index * 100,
            wall_time=f"t{index}",
        )
    incident_dir = tmp_path / "incidents" / "INC-0002"

    recorder.schedule_incident_media(
        incident_dir,
        trigger_ms=500,
        before_ms=300,
        after_ms=300,
    )
    assert not (incident_dir / "clip.avi").exists()
    for index in range(6, 9):
        recorder.write_frame(
            np.full((32, 64, 3), index * 10, np.uint8),
            captured_monotonic_ms=index * 100,
            wall_time=f"t{index}",
        )
    recorder.close()

    capture = cv2.VideoCapture(str(incident_dir / "clip.avi"))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
    finally:
        capture.release()
    assert (incident_dir / "contact_sheet.png").is_file()


def test_close_releases_main_recording_and_reports_incident_media_failure(
    tmp_path,
    monkeypatch,
):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, buffer_seconds=2)
    recorder.write_frame(
        np.zeros((32, 64, 3), np.uint8),
        captured_monotonic_ms=100,
        wall_time="trigger",
    )
    incident_dir = tmp_path / "incidents" / "INC-0003"
    recorder.schedule_incident_media(
        incident_dir,
        trigger_ms=100,
        after_ms=5_000,
    )
    monkeypatch.setattr(
        recorder,
        "_write_incident_media",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("codec full")),
    )

    result = recorder.close()

    assert recorder._closed
    assert result.incident_media_failures[0].reason == "incident_media_finalize_failed"
    assert result.incident_media_failures[0].trigger_ms == 100
    assert (incident_dir / "media_error.json").is_file()
    capture = cv2.VideoCapture(str(result.video_path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
    finally:
        capture.release()


def test_incident_encoding_does_not_block_capture_path(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, buffer_seconds=2)
    recorder.write_frame(
        np.zeros((32, 64, 3), np.uint8),
        captured_monotonic_ms=100,
        wall_time="trigger",
    )
    recorder.schedule_incident_media(
        tmp_path / "incidents" / "INC-async",
        trigger_ms=100,
        after_ms=100,
    )
    started = threading.Event()
    release = threading.Event()

    def slow_media(*_args, **_kwargs):
        started.set()
        assert release.wait(2)
        return None

    monkeypatch.setattr(recorder, "_write_incident_media", slow_media)

    started_at = time.perf_counter()
    recorder.write_frame(
        np.zeros((32, 64, 3), np.uint8),
        captured_monotonic_ms=200,
        wall_time="deadline",
    )
    elapsed = time.perf_counter() - started_at

    assert started.wait(1)
    assert elapsed < 0.2
    assert recorder.frame_count == 2
    release.set()
    recorder.close()
