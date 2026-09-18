from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from daguandan_bridge.advisor_strategy import (
    load_profile_automatic_log_include_media,
    load_profile_recording_max_total_bytes,
    load_profile_recording_mode,
    load_profile_session_data_recording_enabled,
    save_profile_automatic_log_include_media,
    save_profile_recording_max_total_bytes,
    save_profile_recording_mode,
    recording_media_usage_bytes,
    recording_storage_summary,
    save_profile_session_data_recording_enabled,
)
from daguandan_bridge.infrastructure.live_session import (
    DefaultLiveSessionFactory,
    build_session_manifest,
)
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


class _Source:
    def close(self):
        pass


class _Capture:
    def __init__(self, root: Path):
        self.profiles_root = root
        self._profile = root / "tencent_daguandan"

    def load_profile(self, _profile_name):
        return SimpleNamespace(
            paths=SimpleNamespace(
                profile_config_path=self._profile / "profile.json",
                templates_config_path=self._profile / "templates_config.json",
            ),
            config=SimpleNamespace(base_size=(64, 32)),
        )

    def open_live_source(self, _profile_name):
        return _Source()


def _profile(
    root: Path,
    *,
    save_session_data: bool,
    recording_mode: str | None = None,
) -> Path:
    path = root / "tencent_daguandan"
    path.mkdir(parents=True)
    raw = {"save_session_data": save_session_data}
    if recording_mode is not None:
        raw["recording_mode"] = recording_mode
    (path / "profile.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )
    return path


def test_session_data_preference_defaults_on_and_persists_false(tmp_path):
    profile = _profile(tmp_path, save_session_data=True)

    assert load_profile_session_data_recording_enabled(tmp_path, profile.name) is True
    assert save_profile_session_data_recording_enabled(
        tmp_path, profile.name, False
    ) is False
    assert load_profile_session_data_recording_enabled(tmp_path, profile.name) is False
    assert json.loads((profile / "profile.json").read_text("utf-8"))["save_session_data"] is False


def test_recording_capacity_and_automatic_media_preferences_persist(tmp_path):
    profile = _profile(tmp_path, save_session_data=True, recording_mode="all")

    assert load_profile_recording_max_total_bytes(tmp_path, profile.name) > 0
    assert save_profile_recording_max_total_bytes(
        tmp_path, profile.name, 20 * 1024 ** 3
    ) == 20 * 1024 ** 3
    assert save_profile_automatic_log_include_media(
        tmp_path, profile.name, True
    ) is True
    saved = json.loads((profile / "profile.json").read_text("utf-8"))
    assert saved["recording_max_total_bytes"] == 20 * 1024 ** 3
    assert saved["automatic_log_include_media"] is True
    assert load_profile_automatic_log_include_media(tmp_path, profile.name) is True
    summary = recording_storage_summary(tmp_path, profile.name)
    assert summary["limit_bytes"] == 20 * 1024 ** 3
    assert summary["used_bytes"] == 0
    assert summary["capacity_exhausted"] is False


def test_session_manifest_contains_runtime_identity_on_first_write(tmp_path):
    config = tmp_path / "profile.json"
    templates = tmp_path / "templates_config.json"
    config.write_text("{}", encoding="utf-8")
    templates.write_text("{}", encoding="utf-8")

    manifest = build_session_manifest(config, templates)

    identity = manifest["runtime_identity"]
    assert identity["schema"] == "guandan.runtime-identity/1"
    assert identity["run_id"]
    assert identity["implementation_fingerprint"]
    assert Path(str(identity["executable_path"])).name == identity["executable_path"]


def test_recording_mode_upgrades_the_legacy_boolean_and_persists_all(tmp_path):
    profile = _profile(tmp_path, save_session_data=True)

    assert load_profile_recording_mode(tmp_path, profile.name) == "game"
    assert save_profile_recording_mode(tmp_path, profile.name, "all") == "all"
    saved = json.loads((profile / "profile.json").read_text("utf-8"))
    assert saved["recording_mode"] == "all"
    assert saved["save_session_data"] is True


def test_full_recording_mode_persists_listener_frames_without_an_initial_hand(tmp_path):
    profile = _profile(tmp_path, save_session_data=True, recording_mode="all")
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )

    recording = factory.start_listener_recording(
        recognition_strategy="two_valid_streak",
    )

    assert recording is not None
    recording.record_frame(
        np.ones((32, 64, 3), dtype=np.uint8),
        monotonic_ms=10,
        wall_time="t1",
    )
    recording.record_recognition(
        SimpleNamespace(round_level=None, my_hand=(), diagnostics=("no hand",))
    )
    recording.close(reason="listener_stopped")

    assert recording.store.directory.parent.name == ".preopening"
    assert recording.store.directory.name.startswith("opening_")
    assert not tuple((profile / "sessions").glob("game_*"))
    manifest = json.loads((recording.store.directory / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "sealed"
    assert manifest["recording_mode"] == "all"
    assert manifest["recording_phase"] == "ended_without_initial_state"
    assert manifest["initial_state_status"] == "unconfirmed"
    assert manifest["termination_reason"] == "listener_stopped"
    assert manifest["frame_count"] == 1
    assert len(
        read_json_lines(recording.store.directory / "video" / "frame_index.jsonl")
    ) == 1



def test_confirmed_opening_is_archived_under_the_only_formal_game_directory(tmp_path):
    profile = _profile(tmp_path, save_session_data=True, recording_mode="all")
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )

    opening = factory.start_listener_recording(
        recognition_strategy="two_valid_streak",
    )
    assert opening is not None
    opening.record_frame(
        np.full((32, 64, 3), 127, dtype=np.uint8),
        monotonic_ms=10,
        wall_time="opening-frame",
    )
    opening.record_recognition(
        SimpleNamespace(round_level="2", my_hand=HAND, diagnostics=())
    )
    opening.close(reason="initial_state_confirmed")

    sessions_root = profile / "sessions"
    preopening = opening.store.directory
    assert preopening.is_dir()
    assert not tuple(sessions_root.glob("game_*"))

    constructed = factory.start_session(
        round_level="2",
        hand=HAND,
        lead_player=None,
        recognition_strategy="two_valid_streak",
    )
    formal = constructed.orchestrator.store.directory
    opening_archive = formal / "opening"

    assert len(tuple(sessions_root.glob("game_*"))) == 1
    assert formal.name.startswith("game_")
    assert opening_archive.is_dir()
    assert not preopening.exists()
    assert not (sessions_root / ".preopening").exists()
    opening_manifest = json.loads(
        (opening_archive / "manifest.json").read_text(encoding="utf-8")
    )
    assert opening_manifest["status"] == "sealed"
    assert opening_manifest["recording_phase"] == "opening_confirmed"
    assert opening_manifest["initial_state_status"] == "confirmed"
    assert opening_manifest["frame_count"] == 1
    assert len(
        read_json_lines(opening_archive / "video" / "frame_index.jsonl")
    ) == 1
    receipt = json.loads(
        (opening_archive / "archive_receipt.json").read_text(encoding="utf-8")
    )
    formal_manifest = json.loads(
        (formal / "manifest.json").read_text(encoding="utf-8")
    )
    assert receipt["formal_session_id"] == formal.name
    assert receipt["relative_path"] == "opening"
    assert receipt["recording_integrity"]["writer_frame_count"] == 1
    assert formal_manifest["opening_evidence"] == receipt

    constructed.orchestrator.finish()



def test_opening_archive_move_does_not_double_count_recording_capacity(tmp_path):
    profile = _profile(tmp_path, save_session_data=True, recording_mode="all")
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )
    opening = factory.start_listener_recording(
        recognition_strategy="two_valid_streak",
    )
    assert opening is not None
    opening.record_frame(
        np.full((32, 64, 3), 31, dtype=np.uint8),
        monotonic_ms=20,
        wall_time="opening-frame",
    )
    opening.close(reason="initial_state_confirmed")
    used_before = recording_media_usage_bytes(tmp_path, profile.name)
    assert used_before > 0

    formal = LiveSessionStore(tmp_path, profile.name)
    formal.start({"runtime": "archive-capacity-test"})
    opening.archive_into(formal)

    assert recording_media_usage_bytes(tmp_path, profile.name) == used_before
    formal.seal(frame_count=0, dropped_frames=0)



def test_completed_opening_move_survives_formal_manifest_annotation_failure(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path, save_session_data=True, recording_mode="all")
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )
    opening = factory.start_listener_recording(
        recognition_strategy="two_valid_streak",
    )
    assert opening is not None
    opening.record_frame(
        np.full((32, 64, 3), 47, dtype=np.uint8),
        monotonic_ms=25,
        wall_time="opening-frame",
    )
    opening.close(reason="initial_state_confirmed")
    source = opening.store.directory

    formal = LiveSessionStore(tmp_path, profile.name)
    formal.start({"runtime": "archive-metadata-failure-test"})

    def fail_metadata(_metadata):
        raise OSError("synthetic manifest update failure")

    monkeypatch.setattr(formal, "update_session_metadata", fail_metadata)
    destination = opening.archive_into(formal)

    assert destination == formal.directory / "opening"
    assert destination.is_dir()
    assert not source.exists()
    error = json.loads(
        (destination / "archive_metadata_error.json").read_text(encoding="utf-8")
    )
    assert error["status"] == "archived_manifest_update_failed"
    assert error["error_type"] == "OSError"
    formal.seal(frame_count=0, dropped_frames=0)


def test_live_start_failure_keeps_confirmed_opening_evidence_for_diagnosis(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path, save_session_data=True, recording_mode="all")
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )
    opening = factory.start_listener_recording(
        recognition_strategy="two_valid_streak",
    )
    assert opening is not None
    opening.record_frame(
        np.full((32, 64, 3), 63, dtype=np.uint8),
        monotonic_ms=30,
        wall_time="opening-frame",
    )
    opening.close(reason="initial_state_confirmed")
    opening_directory = opening.store.directory

    def fail_recording(_strategy):
        raise RuntimeError("synthetic live recording failure")

    monkeypatch.setattr(factory, "_create_recording", fail_recording)
    with pytest.raises(RuntimeError, match="synthetic live recording failure"):
        factory.start_session(
            round_level="2",
            hand=HAND,
            lead_player=None,
            recognition_strategy="two_valid_streak",
        )

    assert opening_directory.is_dir()
    assert len(
        read_json_lines(opening_directory / "video" / "frame_index.jsonl")
    ) == 1
    manifest = json.loads(
        (opening_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "sealed"
    assert manifest["opening_archive_status"] == "not_archived"
    assert manifest["opening_archive_reason"] == "live_recording_create_failed"
    assert not tuple((profile / "sessions").glob("game_*"))


def test_disabled_video_keeps_durable_rule_events_without_creating_media(tmp_path):
    profile = _profile(tmp_path, save_session_data=False)
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )

    constructed = factory.start_session(
        round_level="2",
        hand=HAND,
        lead_player="self",
        recognition_strategy="two_valid_streak",
    )

    assert constructed.orchestrator.store.persistence_enabled is True
    assert isinstance(constructed.orchestrator.recorder, InMemorySessionRecorder)
    constructed.orchestrator.record_frame(
        np.zeros((32, 64, 3), dtype=np.uint8),
        monotonic_ms=1,
        wall_time="t1",
    )
    constructed.orchestrator.finish()

    sessions = tuple((profile / "sessions").iterdir())
    assert len(sessions) == 1
    assert read_json_lines(sessions[0] / "timeline.jsonl")[0]["event_type"] == "initial_state_confirmed"
    assert not tuple(profile.rglob("*.avi"))


def test_live_session_is_created_only_after_initial_state_is_confirmed(tmp_path):
    profile = _profile(tmp_path, save_session_data=True)
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )
    assert not (profile / "sessions").exists()
    constructed = factory.start_session(
        round_level="2",
        hand=HAND,
        lead_player=None,
        recognition_strategy="two_valid_streak",
    )
    constructed.orchestrator.record_frame(
        np.ones((32, 64, 3), dtype=np.uint8),
        monotonic_ms=10,
        wall_time="t1",
    )
    constructed.orchestrator.finish()

    manifest = json.loads(
        (constructed.orchestrator.store.directory / "manifest.json").read_text("utf-8")
    )
    assert manifest["status"] == "sealed"
    assert manifest["recording_phase"] == "live"
    assert manifest["initial_state_status"] == "confirmed"
    assert manifest["frame_count"] == 1
    assert len(
        read_json_lines(constructed.orchestrator.store.directory / "video" / "frame_index.jsonl")
    ) == 1
