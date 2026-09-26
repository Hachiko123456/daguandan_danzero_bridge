from __future__ import annotations

import json
import os
from pathlib import Path
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


# Exercise the real controller writer and frame store; only desktop capture is fake.
def _migration_snapshot(value=7):
    from daguandan_bridge.capture_service import FrameSnapshot
    from daguandan_bridge.image_io import StandardizationResult
    from daguandan_bridge.models import Box, ClientRect
    from daguandan_bridge.window_capture import CapturedStandardizedFrame

    image = np.full((4, 4, 3), value, dtype=np.uint8)
    return FrameSnapshot(
        frame=CapturedStandardizedFrame(
            standardization=StandardizationResult(
                image=image, source_size=(4, 4), source_viewport=Box(0, 0, 4, 4),
                content_box=Box(0, 0, 4, 4), scale=1.0, padding=(0, 0, 0, 0),
                aspect_error=0.0, aspect_compatible=True,
            ),
            rect=ClientRect(10, 20, 4, 4), backend="screen", dpi=96,
            window_title="migration test", raw_image=image.copy(),
        ),
        evidence_frame_id=f"migration-frame-{value}",
    )


@pytest.fixture
def diagnostic_controller(tmp_path, monkeypatch):
    monkeypatch.delenv("DAGUANDAN_SESSIONS_ROOT", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from daguandan_bridge.gui.live_controller import LiveAssistantController

    app = QApplication.instance() or QApplication([])
    profile = _profile(tmp_path)
    capture = SimpleNamespace(
        profiles_root=tmp_path, open_calls=[], snapshot=_migration_snapshot(),
    )

    def open_source(profile_name):
        capture.open_calls.append(profile_name)
        return SimpleNamespace(capture=lambda: capture.snapshot, close=lambda: None)

    capture.open_live_source = open_source
    controller = LiveAssistantController(
        capture, profile_name=profile.name,
        recognition_service=SimpleNamespace(), advisor=object(),
        session_factory=SimpleNamespace(), opening_evidence_monitor=SimpleNamespace(),
    )
    try:
        yield controller
    finally:
        controller.shutdown()
        app.processEvents()


def _diagnostic_target(controller, phase):
    root = controller.capture_service.profiles_root
    if phase == "manual":
        # Diagnostic-only preopening stores have no recorder manifest to retain.
        return LiveSessionStore.for_episode(root, controller.profile_name)
    formal = LiveSessionStore(root, controller.profile_name, session_id="game_formal")
    formal.start({})
    return formal


def _save_controller_diagnostic(controller, phase, *, retain=False):
    if phase == "preopening":
        episode = _diagnostic_target(controller, "manual")
        controller._bind_preopening_diagnostic_to_recording(SimpleNamespace(store=episode))
    if phase == "preopening" or retain:
        controller._listening_enabled = True
        controller._accept_waiting_frame(
            controller.capture_service.snapshot,
            generation=controller._waiting_generation, capture_seq=1,
        )
    result = controller.save_latest_live_frame_to_session()
    assert controller._listener_evidence.writer.wait_idle(5)
    if retain:
        controller.stop_listening()
        assert controller._retained_diagnostic_frame is not None
        assert controller._retained_diagnostic_frame.session_directory == Path(result["session_directory"])
    return result


def _migrate_diagnostics(controller, phase, target):
    if phase == "manual":
        controller._bind_preopening_diagnostic_to_recording(SimpleNamespace(store=target))
    else:
        controller._migrate_preopening_diagnostic_frames(target)


def _make_directory_link(link, destination, kind):
    link.parent.mkdir(parents=True, exist_ok=True)
    if kind == "junction":
        if os.name != "nt":
            pytest.skip("junction creation requires Windows")
        import _winapi
        _winapi.CreateJunction(str(destination), str(link))
        assert link.is_junction()
    else:
        try:
            link.symlink_to(destination, target_is_directory=True)
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege is unavailable; junction cases still run")
            raise
        assert link.is_symlink()


@pytest.mark.parametrize("phase", ["manual", "preopening"])
@pytest.mark.parametrize("link_kind", ["symlink", "junction"])
@pytest.mark.parametrize("linked_component", ["session", "diagnostic_frames"])
def test_diagnostic_migration_rejects_link_destination_before_external_write(
    diagnostic_controller, tmp_path, phase, link_kind, linked_component,
):
    controller = diagnostic_controller
    saved = _save_controller_diagnostic(controller, phase)
    source = Path(saved["session_directory"])
    source_bytes = {p.name: p.read_bytes() for p in (source / "diagnostic_frames").iterdir()}
    target = _diagnostic_target(controller, phase)
    # Formal start makes an empty diagnostic directory; remove only that empty
    # directory when replacing it with a link. Never traverse/delete the target.
    if linked_component == "diagnostic_frames":
        link = target.directory / "diagnostic_frames"
        if link.exists():
            link.rmdir()
    else:
        target = SimpleNamespace(
            directory=target.directory.parent / "linked_session",
            session_id="linked_session",
        )
        link = target.directory
    outside = tmp_path / "outside_sessions"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_bytes(b"must not change")
    _make_directory_link(link, outside, link_kind)
    before = {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()}
    statuses = []
    controller.diagnostic_frame_status.connect(statuses.append)

    _migrate_diagnostics(controller, phase, target)

    assert statuses and statuses[-1]["status"] == "MIGRATION_SKIPPED"
    assert {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()} == before
    assert sorted(p.name for p in outside.iterdir()) == ["keep.txt"]
    assert {p.name: p.read_bytes() for p in (source / "diagnostic_frames").iterdir()} == source_bytes
    assert controller.diagnostic_frame_directory() == source / "diagnostic_frames"


@pytest.mark.parametrize("phase", ["manual", "preopening"])
def test_controller_saved_diagnostics_migrate_to_formal_session_and_update_directory(
    diagnostic_controller, phase,
):
    controller = diagnostic_controller
    saved = _save_controller_diagnostic(controller, phase)
    original = Path(saved["session_directory"])
    original_png = Path(saved["image_path"]).read_bytes()
    store = SessionDiagnosticFrameStore()
    if phase == "manual":
        episode = _diagnostic_target(controller, "manual")
        _migrate_diagnostics(controller, "manual", episode)
        assert not original.exists()
        assert len(store.list_frames(episode.directory)) == 1
        assert controller.diagnostic_frame_directory() == episode.directory / "diagnostic_frames"
        preopening = episode.directory
    else:
        preopening = original
    formal = _diagnostic_target(controller, "preopening")
    existing = store.save_snapshot(
        formal.directory, _migration_snapshot(19), session_id=formal.session_id,
        capture_generation=3, capture_seq=9,
    )
    existing_png, existing_json = existing.image_path.read_bytes(), existing.metadata_path.read_bytes()

    _migrate_diagnostics(controller, "preopening", formal)

    records = store.list_frames(formal.directory)
    assert len(records) == 2, "a real controller save must not permanently pin ordinary diagnostics"
    migrated = records[1]
    assert existing.image_path.read_bytes() == existing_png
    assert existing.metadata_path.read_bytes() == existing_json
    assert migrated.sequence == 2
    assert migrated.image_path.read_bytes() == original_png
    assert migrated.metadata["session_id"] == formal.session_id
    assert migrated.metadata["evidence_frame_id"] == saved["evidence_frame_id"]
    assert migrated.metadata["capture_seq"] == saved["capture_seq"]
    assert migrated.metadata["capture_generation"] == saved["capture_generation"]
    np.testing.assert_array_equal(store.load_image(migrated), controller.capture_service.snapshot.image)
    assert not preopening.exists()
    assert controller._last_diagnostic_frame_directory == formal.directory / "diagnostic_frames"
    assert controller.diagnostic_frame_directory() == formal.directory / "diagnostic_frames"


@pytest.mark.parametrize("phase", ["manual", "preopening"])
def test_retained_listener_frame_rebinds_after_migration_without_recreating_source(
    diagnostic_controller, phase,
):
    controller = diagnostic_controller
    saved = _save_controller_diagnostic(controller, phase, retain=True)
    original = Path(saved["session_directory"])
    retained = controller._retained_diagnostic_frame
    assert retained.session_directory == original
    assert retained.source_phase == "listener_stopped"
    assert controller.capture_service.open_calls == []
    if phase == "manual":
        episode = _diagnostic_target(controller, "manual")
        _migrate_diagnostics(controller, "manual", episode)
        assert not original.exists()
        assert controller._retained_diagnostic_frame.session_directory == episode.directory
        preopening = episode.directory
    else:
        preopening = original
    formal = _diagnostic_target(controller, "preopening")

    _migrate_diagnostics(controller, "preopening", formal)

    assert not preopening.exists()
    rebound = controller._retained_diagnostic_frame
    assert rebound.session_directory == formal.directory
    assert rebound.session_id == formal.session_id
    assert rebound.identity == retained.identity
    cached = controller._listener_evidence.find(
        retained.snapshot, retained.capture_generation, retained.capture_scope[0],
        scope_id=retained.capture_scope[2],
    )
    assert cached is not None
    # Fallback captures may remain storage-unbound in the ring (None), unlike
    # the retained save context. Neither cache may name a deleted directory.
    if phase == "manual" and cached.session_directory is None:
        assert cached.session_id == ""
    else:
        assert cached.session_directory == formal.directory
        assert cached.session_id == formal.session_id

    again = controller.save_latest_live_frame_to_session()
    assert controller._listener_evidence.writer.wait_idle(5)

    assert not original.exists(), "manual save must not recreate the deleted source directory"
    assert not preopening.exists()
    assert Path(again["session_directory"]) == formal.directory
    assert Path(again["image_path"]).parent == formal.directory / "diagnostic_frames"
    assert again["session_id"] == formal.session_id
    assert again["evidence_frame_id"] == saved["evidence_frame_id"]
    assert again["source_phase"] == "listener_stopped"
    assert controller.capture_service.open_calls == [], "save must reuse retained evidence, not recapture"
    records = SessionDiagnosticFrameStore().list_frames(formal.directory)
    assert len(records) == 2
    assert records[0].image_path.read_bytes() == records[1].image_path.read_bytes()
    assert controller.diagnostic_frame_directory() == formal.directory / "diagnostic_frames"


@pytest.mark.parametrize("phase", ["manual", "preopening"])
def test_diagnostic_migration_preserves_orphan_json_and_does_not_reuse_its_sequence(
    diagnostic_controller, phase,
):
    controller = diagnostic_controller
    saved = _save_controller_diagnostic(controller, phase)
    original_png = Path(saved["image_path"]).read_bytes()
    target = _diagnostic_target(controller, phase)
    directory = SessionDiagnosticFrameStore.diagnostic_directory(target.directory, create=True)
    orphan = directory / "000001.json"
    orphan_bytes = b'{"interrupted_write": true, "preserve": "exact bytes"}\n'
    orphan.write_bytes(orphan_bytes)
    assert not orphan.with_suffix(".png").exists()

    _migrate_diagnostics(controller, phase, target)

    assert orphan.read_bytes() == orphan_bytes
    assert not orphan.with_suffix(".png").exists()
    records = SessionDiagnosticFrameStore().list_frames(target.directory)
    assert len(records) == 1
    assert records[0].sequence == 2
    assert records[0].image_path.read_bytes() == original_png
    assert records[0].metadata["session_id"] == target.session_id
    assert not Path(saved["image_path"]).exists()
    assert not Path(saved["metadata_path"]).exists()
    assert controller.diagnostic_frame_directory() == directory
