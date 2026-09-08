from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.infrastructure.live_session import DefaultLiveSessionFactory
from daguandan_bridge.live.recorder import SessionRecorder


def _factory(tmp_path, limit):
    root = tmp_path / "profiles"
    profile = root / "tencent_daguandan"
    profile.mkdir(parents=True)
    config = profile / "profile.json"
    config.write_text(json.dumps({"recording_max_total_bytes": limit, "recording_mode": "game"}), encoding="utf-8")
    loaded = SimpleNamespace(
        paths=SimpleNamespace(profile_config_path=config, templates_config_path=profile / "templates.json"),
        config=SimpleNamespace(base_size=(64, 32)),
    )
    capture = SimpleNamespace(
        profiles_root=root, load_profile=lambda _name: loaded,
        open_live_source=lambda _name: SimpleNamespace(close=lambda: None),
    )
    return DefaultLiveSessionFactory(capture, object(), None, profile_name=profile.name), profile, config


def _media_size(path):
    return sum(item.stat().st_size for item in path.rglob("*")
               if item.is_file() and item.suffix.lower() in {".avi", ".mp4", ".png", ".jpg"})


def test_factory_counts_old_partial_clips_and_pngs_without_deleting_them(tmp_path):
    factory, profile, config = _factory(tmp_path, 80000)
    files = [profile / "sessions" / "old" / name for name in
             ("video/game.avi", "video/game.partial.avi", "incidents/INC-1/clip.avi", "incidents/INC-1/frames/proof.png")]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 5000)
    before = {path: path.read_bytes() for path in files}
    assert factory._remaining_video_allowance(config) == 60000
    recording = factory._create_recording("two_valid_streak")
    warning = recording.recorder.write_frame(np.zeros((32, 64, 3), np.uint8), 100, "full")
    assert warning.reason == "recording_capacity_reached"
    recording.close_start_failed()
    assert not recording.recorder.video_path.exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_factory_quota_is_shared_across_successive_sessions(tmp_path):
    factory, profile, config = _factory(tmp_path, 120000)
    rng = np.random.default_rng(78)
    for session in range(4):
        recording = factory._create_recording("two_valid_streak")
        for index in range(100):
            recording.recorder.write_frame(rng.integers(0, 256, (32, 64, 3), dtype=np.uint8), index, str(session))
        recording.close_start_failed()
        assert _media_size(profile / "sessions") <= 120000
    assert factory._remaining_video_allowance(config) == 120000 - _media_size(profile / "sessions")


@pytest.mark.parametrize("limit", [True, False, -1, 0.5, "100000", None])
def test_factory_rejects_invalid_recording_limit(tmp_path, limit):
    factory, _profile, config = _factory(tmp_path, limit)
    with pytest.raises(ValueError, match="non-negative integer"):
        factory._remaining_video_allowance(config)


@pytest.mark.parametrize("limit", [True, -1, 0.5, "100000"])
def test_recorder_rejects_invalid_limit_before_creating_media(tmp_path, limit):
    with pytest.raises(ValueError):
        SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=limit)
    assert not list(tmp_path.rglob("*.avi"))
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.parametrize("property_result", [0, float("nan"), "raise"])
def test_missing_framebytes_property_uses_conservative_accounting(tmp_path, property_result):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=100000)
    writer = recorder._writer

    class WriterProxy:
        def write(self, frame):
            writer.write(frame)

        def get(self, _prop):
            if property_result == "raise":
                raise cv2.error("unsupported FRAMEBYTES")
            return property_result

        def release(self):
            writer.release()

    recorder._writer = WriterProxy()
    for index in range(100):
        recorder.write_frame(np.full((32, 64, 3), index, np.uint8), index, str(index))
    result = recorder.close()
    assert result.frame_count == 2
    assert result.integrity["decodable_frame_count"] == 2
    assert result.video_path.stat().st_size <= 100000
    assert result.integrity["stop_reason"] == "recording_capacity_reached"


def test_standalone_evidence_png_cannot_bypass_media_quota(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=66000)
    frame = np.random.default_rng(91).integers(0, 256, (32, 64, 3), dtype=np.uint8)
    destination = tmp_path / "incidents" / "proof.png"
    with pytest.raises(RuntimeError, match="recording_capacity_reached"):
        recorder.save_evidence_frame(destination, frame)
    assert not destination.exists()
    assert _media_size(tmp_path) <= 66000
    recorder.close()


def test_pending_clip_reservation_closes_without_lock_cycle_and_preserves_exact_media(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=400000)
    entered, release = Event(), Event()
    original = recorder._write_incident_media_payload

    def delayed(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(recorder, "_write_incident_media_payload", delayed)
    frame = np.full((32, 64, 3), 75, np.uint8)
    recorder.write_frame(frame, 0, "before")
    directory = tmp_path / "incidents" / "INC-1"
    recorder.schedule_incident_media(directory, trigger_ms=0, after_ms=100)
    recorder.write_frame(frame, 100, "after")
    assert entered.wait(1)
    result = []
    closer = Thread(target=lambda: result.append(recorder.close()), daemon=True)
    closer.start()
    release.set()
    closer.join(2)
    assert not closer.is_alive()
    assert len(result) == 1
    assert result[0].incident_media_failures == ()
    assert result[0].integrity["decodable_frame_count"] == 2
    manifest = json.loads((directory / "media.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 2
    assert cv2.imread(str(directory / "frames" / "trigger.png")).shape == (32, 64, 3)
    capture = cv2.VideoCapture(str(directory / "clip.avi"))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    finally:
        capture.release()
    assert _media_size(tmp_path) <= 400000


def test_clip_and_standalone_pngs_share_the_main_recording_allowance(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=400000)
    frame = np.zeros((32, 64, 3), np.uint8)
    recorder.write_frame(frame, 0, "first")
    first = recorder.save_incident_media(tmp_path / "incidents" / "first", trigger_ms=0)
    before = _media_size(tmp_path)
    assert first.clip_path.exists()
    # Existing clip media has consumed the same counter used by evidence PNGs.
    recorder.max_video_bytes = recorder._video_accounted_bytes + recorder._incident_budget_used
    with pytest.raises(RuntimeError, match="recording_capacity_reached"):
        recorder.save_evidence_frame(tmp_path / "incidents" / "extra.png", frame)
    with pytest.raises(RuntimeError, match="recording_capacity_reached"):
        recorder.save_incident_media(tmp_path / "incidents" / "second", trigger_ms=0)
    assert _media_size(tmp_path) == before
    recorder.close()


def test_full_recording_quota_does_not_disable_durable_session_state(tmp_path):
    factory, _profile, _config = _factory(tmp_path, 0)

    class Advisor:
        def recommend(self, state, *, request_id):
            return LocalAdvice(strategy="test", cards=("2S",), play_type="Single", is_pass=False,
                state_revision=state.revision, elapsed_ms=0, request_id=request_id)

    factory.advisor = Advisor()
    hand = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")
    session = factory.start_session(round_level="2", hand=hand, lead_player="self", recognition_strategy="two_valid_streak")
    orchestrator = session.orchestrator
    warning = orchestrator.record_frame(np.zeros((32, 64, 3), np.uint8), monotonic_ms=100, wall_time="full")
    assert warning.reason == "recording_capacity_reached"
    assert orchestrator.snapshot.initialized
    assert orchestrator.status == "running"
    assert not orchestrator.recorder.video_path.exists()
    orchestrator.finish()


def test_tail_recovery_cannot_create_unbudgeted_media_at_close(tmp_path, monkeypatch):
    from daguandan_bridge.live import recorder as recorder_module

    recorder = SessionRecorder(tmp_path, size=(64, 32), fps=10, max_video_bytes=100000)
    for index in range(3):
        recorder.write_frame(np.full((32, 64, 3), index * 30, np.uint8), index, str(index))
    monkeypatch.setattr(recorder_module, "audit_recording_integrity", lambda *_args, **_kwargs: {
        "status": "FAIL", "writer_frame_count": 3,
        "indexed_frame_count": 3, "decodable_frame_count": 2,
        "issues": ["indexed_tail_not_decodable"],
    })

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unbudgeted recovery media must not be created")

    monkeypatch.setattr(recorder, "_codec_roundtrip", forbidden)
    monkeypatch.setattr(recorder, "_write_tail_recovery", forbidden)
    result = recorder.close()
    assert result.integrity["status"] == "PARTIAL"
    assert result.integrity["tail_recovery"]["reason"] == "recording_capacity_reached"
    assert not list(tmp_path.rglob("*.png"))
    assert result.video_path.exists()
    assert _media_size(tmp_path) <= 100000
