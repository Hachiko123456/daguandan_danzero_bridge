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
def _case_snapshot(value=7):
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
            window_title="case test", raw_image=image.copy(),
        ),
        evidence_frame_id=f"case-frame-{value}",
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
        profiles_root=tmp_path, open_calls=[], snapshot=_case_snapshot(),
    )

    def open_source(profile_name):
        capture.open_calls.append(profile_name)
        return SimpleNamespace(capture=lambda: capture.snapshot, close=lambda: None)

    capture.open_live_source = open_source
    controller = LiveAssistantController(
        capture, profile_name=profile.name,
        recognition_service=SimpleNamespace(), advisor=object(),
        session_factory=SimpleNamespace(), opening_evidence_monitor=SimpleNamespace(),
        diagnostics_root=tmp_path / "diagnostics",
    )
    try:
        yield controller
    finally:
        # Formal-frame tests use only a runtime identity stub, not a running
        # orchestrator whose finish() should start a background shutdown job.
        controller._active_live_token = None
        controller.orchestrator = None
        controller.shutdown()
        app.processEvents()


def _case_session(controller, *, formal=False):
    root = controller.capture_service.profiles_root
    if formal:
        store = LiveSessionStore(root, controller.profile_name, session_id="game_formal")
        store.start({})
    else:
        store = LiveSessionStore.for_episode(root, controller.profile_name)
        store.start_episode({})
    return store


def _save_case_frame(controller):
    result = controller.save_latest_live_frame_to_session()
    assert controller._listener_evidence.writer.wait_idle(5)
    return result


def _accept_waiting_case_frame(controller, *, value=7, capture_seq=1):
    controller._listening_enabled = True
    controller._accept_waiting_frame(
        _case_snapshot(value), generation=controller._waiting_generation,
        capture_seq=capture_seq,
    )


def _assert_case_path(result, tmp_path):
    case = Path(result["session_directory"])
    assert case.parent == tmp_path / "diagnostics" / "cases"
    assert case.name.startswith("case_")
    assert Path(result["image_path"]).parent == case / "frames"
    assert Path(result["metadata_path"]).parent == case / "frames"
    assert (case / "case.json").is_file()
    assert not (case / "diagnostic_frames").exists()
    return case


def _assert_case_session_links(case, episode, formal):
    manifest = json.loads((case / "case.json").read_text(encoding="utf-8"))
    assert manifest["case_id"] == case.name
    assert manifest["frames_directory"] == "frames"
    assert manifest["session_id"] == formal.session_id
    assert Path(manifest["session_directory"]) == formal.directory
    assert {
        (link["kind"], link["session_id"], Path(link["directory"]))
        for link in manifest["session_links"]
    } == {
        ("preopening", episode.session_id, episode.directory),
        ("session", formal.session_id, formal.directory),
    }
    assert list(case.parent.glob("*/case.json")) == [case / "case.json"]
    for store in (episode, formal):
        assert not tuple(store.directory.rglob("*.png"))
        assert not (store.directory / "case.json").exists()
        assert not (store.directory / "frames").exists()


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


@pytest.mark.parametrize("link_kind", ["symlink", "junction"])
@pytest.mark.parametrize("linked_component", ["diagnostics_root", "cases_root", "case", "frames"])
def test_case_save_rejects_link_on_actual_write_path_before_external_write(
    diagnostic_controller, tmp_path, link_kind, linked_component,
):
    from daguandan_bridge.application.session_diagnostic_frames import SessionDiagnosticFrameError

    controller = diagnostic_controller
    # Allocate only the identity, then substitute a link before the real writer
    # creates anything. This tests the write path, not an unrelated session path.
    case = controller._diagnostic_cases.allocate()
    assert case.parent == tmp_path / "diagnostics" / "cases"
    link = {
        "diagnostics_root": tmp_path / "diagnostics",
        "cases_root": case.parent,
        "case": case,
        "frames": case / "frames",
    }[linked_component]
    assert not link.exists()
    outside = tmp_path / "outside_diagnostics"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"must not change")
    _make_directory_link(link, outside, link_kind)

    with pytest.raises((ValueError, SessionDiagnosticFrameError), match="链接|重解析点"):
        controller.save_latest_live_frame_to_session()
    assert controller._listener_evidence.writer.wait_idle(5)

    assert sorted(path.name for path in outside.iterdir()) == ["keep.txt"]
    assert (outside / "keep.txt").read_bytes() == b"must not change"
    assert not tuple(outside.rglob("*.png"))
    assert not tuple(outside.rglob("*.json"))


@pytest.mark.parametrize("initial_source", ["manual", "listener"])
def test_controller_frames_stay_in_one_case_across_preopening_and_formal_session(
    diagnostic_controller, tmp_path, initial_source,
):
    controller = diagnostic_controller
    if initial_source == "listener":
        _accept_waiting_case_frame(controller)
    first = _save_case_frame(controller)
    case = _assert_case_path(first, tmp_path)
    original = {
        Path(first[key]): Path(first[key]).read_bytes()
        for key in ("image_path", "metadata_path")
    }
    first_manifest = json.loads((case / "case.json").read_text(encoding="utf-8"))
    assert first_manifest["session_id"] is None
    assert first_manifest["session_links"] == []

    episode = _case_session(controller)
    controller._bind_preopening_diagnostic_to_recording(SimpleNamespace(store=episode))
    assert controller._preopening_diagnostic_directory == case
    assert controller.diagnostic_frame_directory() == case / "frames"
    _accept_waiting_case_frame(controller, value=19, capture_seq=2)
    waiting = _save_case_frame(controller)
    assert _assert_case_path(waiting, tmp_path) == case
    original.update({
        Path(waiting[key]): Path(waiting[key]).read_bytes()
        for key in ("image_path", "metadata_path")
    })

    formal = _case_session(controller, formal=True)
    controller._migrate_preopening_diagnostic_frames(formal)
    assert controller._preopening_diagnostic_directory == case
    assert all(path.read_bytes() == contents for path, contents in original.items())
    runtime = SimpleNamespace(store=formal, snapshot=SimpleNamespace(session_id=formal.session_id))
    controller.orchestrator = runtime
    token = controller._activate_live_token(runtime)
    controller._retain_listener_snapshot(_case_snapshot(31), token.generation, 1, token=token)
    live = _save_case_frame(controller)

    assert _assert_case_path(live, tmp_path) == case
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert [first["sequence"], waiting["sequence"], live["sequence"]] == [1, 2, 3]
    assert first["source_phase"] == (
        "manual_window_capture" if initial_source == "manual" else "preopening_listener"
    )
    assert waiting["source_phase"] == "preopening_listener"
    assert live["source_phase"] == "live_session"
    assert live["session_id"] == formal.session_id
    records = SessionDiagnosticFrameStore().list_frames(case)
    assert len(records) == 3
    for record, value in zip(records, (7, 19, 31), strict=True):
        assert record.image_path.parent == case / "frames"
        np.testing.assert_array_equal(
            SessionDiagnosticFrameStore().load_image(record), _case_snapshot(value).image,
        )
    assert controller._last_diagnostic_frame_directory == case / "frames"
    assert controller.diagnostic_frame_directory() == case / "frames"
    assert controller.capture_service.open_calls == (
        [controller.profile_name] if initial_source == "manual" else []
    )
    _assert_case_session_links(case, episode, formal)


@pytest.mark.parametrize("initial_binding", ["unbound", "preopening"])
def test_retained_listener_frame_keeps_case_directory_and_saves_without_recapture(
    diagnostic_controller, tmp_path, initial_binding,
):
    controller = diagnostic_controller
    episode = _case_session(controller)
    if initial_binding == "preopening":
        controller._bind_preopening_diagnostic_to_recording(SimpleNamespace(store=episode))
    _accept_waiting_case_frame(controller)
    first = _save_case_frame(controller)
    case = _assert_case_path(first, tmp_path)
    original = {
        Path(first[key]): Path(first[key]).read_bytes()
        for key in ("image_path", "metadata_path")
    }
    controller.stop_listening()
    retained = controller._retained_diagnostic_frame
    assert retained is not None
    assert retained.session_directory == case
    assert retained.source_phase == "listener_stopped"
    assert controller.capture_service.open_calls == []

    if initial_binding == "unbound":
        controller._bind_preopening_diagnostic_to_recording(SimpleNamespace(store=episode))
    formal = _case_session(controller, formal=True)
    controller._migrate_preopening_diagnostic_frames(formal)

    assert controller._retained_diagnostic_frame.session_directory == case
    assert controller._retained_diagnostic_frame.identity == retained.identity
    cached = controller._listener_evidence.find(
        retained.snapshot, retained.capture_generation, retained.capture_scope[0],
        scope_id=retained.capture_scope[2],
    )
    assert cached is not None
    assert cached.session_directory == case
    assert controller.diagnostic_frame_directory() == case / "frames"
    again = _save_case_frame(controller)

    assert _assert_case_path(again, tmp_path) == case
    assert again["sequence"] == 2
    assert again["evidence_frame_id"] == first["evidence_frame_id"]
    assert again["capture_seq"] == first["capture_seq"]
    assert again["capture_generation"] == first["capture_generation"]
    assert again["source_phase"] == "listener_stopped"
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert Path(again["image_path"]).read_bytes() == Path(first["image_path"]).read_bytes()
    assert controller.capture_service.open_calls == [], "retained save must not recapture the desktop"
    assert len(SessionDiagnosticFrameStore().list_frames(case)) == 2
    _assert_case_session_links(case, episode, formal)


@pytest.mark.parametrize("source", ["manual", "listener"])
def test_case_save_preserves_orphan_json_and_does_not_reuse_its_sequence(
    diagnostic_controller, tmp_path, source,
):
    controller = diagnostic_controller
    case = controller._diagnostic_cases.allocate()
    assert case.parent == tmp_path / "diagnostics" / "cases"
    directory = case / "frames"
    directory.mkdir(parents=True)
    orphan = directory / "000001.json"
    orphan_bytes = b'{"interrupted_write": true, "preserve": "exact bytes"}\n'
    orphan.write_bytes(orphan_bytes)
    assert not orphan.with_suffix(".png").exists()
    if source == "listener":
        _accept_waiting_case_frame(controller)

    saved = _save_case_frame(controller)

    assert _assert_case_path(saved, tmp_path) == case
    assert saved["sequence"] == 2
    assert orphan.read_bytes() == orphan_bytes
    assert not orphan.with_suffix(".png").exists()
    records = SessionDiagnosticFrameStore().list_frames(case)
    assert len(records) == 1
    assert records[0].sequence == 2
    assert records[0].image_path == directory / "000002.png"
    assert records[0].metadata_path == directory / "000002.json"
    np.testing.assert_array_equal(
        SessionDiagnosticFrameStore().load_image(records[0]), controller.capture_service.snapshot.image,
    )
    assert controller.diagnostic_frame_directory() == directory
