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
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore


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
