from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from daguandan_bridge.application.session_diagnostic_frames import SessionDiagnosticFrameStore
from daguandan_bridge.infrastructure.live_session import DefaultLiveSessionFactory
from daguandan_bridge.live.session_store import LiveSessionStore


class _Capture:
    def __init__(self, root: Path):
        self.profiles_root = root

    def load_profile(self, _profile_name):
        profile = self.profiles_root / "profile"
        return SimpleNamespace(
            paths=SimpleNamespace(
                profile_config_path=profile / "profile.json",
                templates_config_path=profile / "templates.json",
            ),
            config=SimpleNamespace(base_size=(8, 8)),
        )


def _profile(tmp_path: Path) -> Path:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "profile.json").write_text("{}", encoding="utf-8")
    return profile


def test_episode_owns_manifest_trace_and_diagnostic_frames_in_one_directory(tmp_path):
    profile = _profile(tmp_path)
    store = LiveSessionStore.for_episode(tmp_path, profile.name)
    store.start_episode({"runtime": "test"})

    assert store.directory.parent.name == ".preopening"
    assert store.directory.name.startswith("episode_")
    assert (store.directory / "manifest.json").is_file()
    assert (store.directory / "recognition_trace.jsonl").is_file()
    assert (store.directory / "diagnostic_frames").is_dir()
    manifest = json.loads((store.directory / "manifest.json").read_text("utf-8"))
    assert manifest["lifecycle"] == "opening"
    assert manifest["lifecycle_status"] == "opening"


def test_diagnostic_frame_path_is_the_episode_path(tmp_path):
    profile = _profile(tmp_path)
    store = LiveSessionStore.for_episode(tmp_path, profile.name)
    store.start_episode({})
    frames = SessionDiagnosticFrameStore()
    record = frames.save_snapshot(
        store.directory,
        SimpleNamespace(image=np.zeros((4, 4, 3), dtype=np.uint8)),
        session_id=store.session_id,
        capture_generation=1,
        capture_seq=2,
    )
    assert record.session_directory == store.directory
    assert record.image_path.parent == store.directory / "diagnostic_frames"


def test_episode_promotion_puts_diagnostics_in_formal_root_without_nested_opening(
    tmp_path,
):
    profile = _profile(tmp_path)
    episode = LiveSessionStore.for_episode(tmp_path, profile.name)
    episode.start_episode({})
    SessionDiagnosticFrameStore().save_snapshot(
        episode.directory,
        SimpleNamespace(image=np.full((4, 4, 3), 7, dtype=np.uint8)),
        session_id=episode.session_id,
        capture_generation=1,
        capture_seq=1,
    )
    formal = LiveSessionStore(tmp_path, profile.name, session_id="game_formal")
    formal.start({})

    destination = episode.promote_episode_into(formal)

    assert destination == formal.directory
    assert (formal.directory / "diagnostic_frames" / "000001.png").is_file()
    assert not (formal.directory / "opening").exists()
    assert not episode.directory.exists()
    manifest = json.loads((formal.directory / "manifest.json").read_text("utf-8"))
    assert manifest["episode_id"] == episode.session_id
    assert manifest["lifecycle"] == "running"


def test_episode_promotion_failure_keeps_source_directory(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    episode = LiveSessionStore.for_episode(tmp_path, profile.name)
    episode.start_episode({})
    formal = LiveSessionStore(tmp_path, profile.name, session_id="game_formal")
    formal.start({})

    def fail(_changes):
        raise OSError("manifest write failed")

    monkeypatch.setattr(formal, "_update_manifest", fail)
    with pytest.raises(OSError, match="manifest write failed"):
        episode.promote_episode_into(formal)
    assert episode.directory.is_dir()
    assert (episode.directory / "manifest.json").is_file()


def test_legacy_opening_directory_remains_readable(tmp_path):
    profile = _profile(tmp_path)
    legacy = LiveSessionStore.for_opening_evidence(tmp_path, profile.name)
    legacy.start({"runtime": "legacy"})
    assert legacy.directory.name.startswith("opening_")
    assert legacy.directory.is_dir()


def _episode_with_frame(tmp_path: Path):
    profile = _profile(tmp_path)
    episode = LiveSessionStore.for_episode(tmp_path, profile.name)
    episode.start_episode({})
    SessionDiagnosticFrameStore().save_snapshot(
        episode.directory,
        SimpleNamespace(image=np.full((4, 4, 3), 7, dtype=np.uint8)),
        session_id=episode.session_id,
        capture_generation=1,
        capture_seq=1,
    )
    episode.append_recognition_trace({"phase": "episode", "value": 1})
    formal = LiveSessionStore(tmp_path, profile.name, session_id="game_formal")
    formal.start({})
    return episode, formal


def test_promotion_rolls_back_exact_trace_and_media_after_trace_failure(tmp_path, monkeypatch):
    episode, formal = _episode_with_frame(tmp_path)
    before_trace = (formal.directory / "recognition_trace.jsonl").read_bytes()

    monkeypatch.setattr(episode, "_promotion_after_trace", lambda: (_ for _ in ()).throw(RuntimeError("trace fail")))
    with pytest.raises(RuntimeError, match="trace fail"):
        episode.promote_episode_into(formal)

    assert episode.directory.is_dir()
    assert (formal.directory / "recognition_trace.jsonl").read_bytes() == before_trace
    assert not (formal.directory / "episode_promotion.json").exists()
    assert not (formal.directory / "diagnostic_frames" / "000001.png").exists()


def test_promotion_rolls_back_media_after_media_failure(tmp_path, monkeypatch):
    episode, formal = _episode_with_frame(tmp_path)
    before_trace = (formal.directory / "recognition_trace.jsonl").read_bytes()

    monkeypatch.setattr(episode, "_promotion_after_media", lambda: (_ for _ in ()).throw(RuntimeError("media fail")))
    with pytest.raises(RuntimeError, match="media fail"):
        episode.promote_episode_into(formal)

    assert episode.directory.is_dir()
    assert (formal.directory / "recognition_trace.jsonl").read_bytes() == before_trace
    assert not (formal.directory / "episode_promotion.json").exists()
    assert not (formal.directory / "diagnostic_frames" / "000001.png").exists()


def test_promotion_rolls_back_after_manifest_failure_and_retry_is_clean(tmp_path, monkeypatch):
    episode, formal = _episode_with_frame(tmp_path)
    before_trace = (formal.directory / "recognition_trace.jsonl").read_bytes()
    original_update = formal._update_manifest

    def fail_once(_changes):
        raise OSError("manifest fail")

    monkeypatch.setattr(formal, "_update_manifest", fail_once)
    with pytest.raises(OSError, match="manifest fail"):
        episode.promote_episode_into(formal)

    assert episode.directory.is_dir()
    assert (formal.directory / "recognition_trace.jsonl").read_bytes() == before_trace
    assert not (formal.directory / "episode_promotion.json").exists()
    assert not (formal.directory / "diagnostic_frames" / "000001.png").exists()

    monkeypatch.setattr(formal, "_update_manifest", original_update)
    assert episode.promote_episode_into(formal) == formal.directory
    assert not episode.directory.exists()
    assert (formal.directory / "diagnostic_frames" / "000001.png").is_file()
    trace = (formal.directory / "recognition_trace.jsonl").read_bytes()
    assert trace.count(b'"phase":"episode"') == 1
