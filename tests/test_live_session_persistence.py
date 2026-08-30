from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from daguandan_bridge.advisor_strategy import (
    load_profile_recording_mode,
    load_profile_session_data_recording_enabled,
    save_profile_recording_mode,
    save_profile_session_data_recording_enabled,
)
from daguandan_bridge.infrastructure.live_session import (
    DefaultLiveSessionFactory,
    build_session_manifest,
)
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore, read_json_lines


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


def test_disabled_recording_uses_memory_only_without_creating_sessions(tmp_path):
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

    assert isinstance(constructed.orchestrator.store, InMemoryLiveSessionStore)
    assert isinstance(constructed.orchestrator.recorder, InMemorySessionRecorder)
    constructed.orchestrator.record_frame(
        np.zeros((32, 64, 3), dtype=np.uint8),
        monotonic_ms=1,
        wall_time="t1",
    )
    constructed.orchestrator.finish()

    assert not (profile / "sessions").exists()
    assert not tuple(profile.rglob("*.avi"))
    assert not tuple(profile.rglob("*.jsonl"))


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
