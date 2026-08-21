from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from daguandan_bridge.advisor_strategy import (
    load_profile_session_data_recording_enabled,
    save_profile_session_data_recording_enabled,
)
from daguandan_bridge.infrastructure.live_session import DefaultLiveSessionFactory
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


def _profile(root: Path, *, save_session_data: bool) -> Path:
    path = root / "tencent_daguandan"
    path.mkdir(parents=True)
    (path / "profile.json").write_text(
        json.dumps({"save_session_data": save_session_data}), encoding="utf-8"
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


def test_listener_persists_unconfirmed_initial_state_before_a_live_session_exists(tmp_path):
    profile = _profile(tmp_path, save_session_data=True)
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )

    recording = factory.begin_listening_recording(
        recognition_strategy="two_valid_streak",
    )

    assert recording is not None
    recording.record_frame(
        np.zeros((32, 64, 3), dtype=np.uint8),
        monotonic_ms=10,
        wall_time="t0",
    )
    recording.record_recognition(
        SimpleNamespace(
            round_level=None,
            wild_rank=None,
            current_player="self",
            lead_player=None,
            my_hand=HAND,
            field_confidences={},
            diagnostics=("未识别到当前级牌",),
            unresolved_fields=("round_level",),
        ),
        captured_at="t0",
        acceptance_reason="round_level_unrecognized",
    )
    recording.close_unconfirmed("listening_stopped_before_initial_state")

    manifest = json.loads((recording.store.directory / "manifest.json").read_text("utf-8"))
    trace = read_json_lines(recording.store.directory / "recognition_trace.jsonl")
    assert manifest["status"] == "sealed"
    assert manifest["initial_state_status"] == "unconfirmed"
    assert manifest["termination_reason"] == "listening_stopped_before_initial_state"
    assert manifest["frame_count"] == 1
    assert any(
        item.get("initial_state_acceptance") == "round_level_unrecognized"
        for item in trace
    )
    initial_read = next(item for item in trace if item.get("kind") == "initial_recognition")
    assert initial_read["initial_state_summary"] == {
        "recognized_round_level": "unrecognized",
        "recognized_wild_rank": "unrecognized",
        "hand_count": 27,
        "current_player": "self",
        "lead_player": "unrecognized",
        "acceptance_reason": "round_level_unrecognized",
    }
    assert (recording.store.directory / "video" / "game.avi").is_file()


def test_listener_recording_is_promoted_into_the_same_live_session(tmp_path):
    profile = _profile(tmp_path, save_session_data=True)
    factory = DefaultLiveSessionFactory(
        _Capture(tmp_path),
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )
    recording = factory.begin_listening_recording(
        recognition_strategy="two_valid_streak",
    )

    assert recording is not None
    recording.record_frame(
        np.zeros((32, 64, 3), dtype=np.uint8),
        monotonic_ms=10,
        wall_time="t0",
    )
    constructed = factory.start_session_from_listening_recording(
        recording,
        round_level="2",
        hand=HAND,
        lead_player=None,
        recognition_strategy="two_valid_streak",
    )
    constructed.orchestrator.record_frame(
        np.ones((32, 64, 3), dtype=np.uint8),
        monotonic_ms=20,
        wall_time="t1",
    )
    constructed.orchestrator.finish()

    assert constructed.orchestrator.store.directory == recording.store.directory
    assert constructed.orchestrator.recorder is recording.recorder
    manifest = json.loads((recording.store.directory / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "sealed"
    assert manifest["recording_phase"] == "live"
    assert manifest["initial_state_status"] == "confirmed"
    assert manifest["frame_count"] == 2
    assert len(read_json_lines(recording.store.directory / "video" / "frame_index.jsonl")) == 2
