from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from daguandan_bridge.application.diagnostic_cases import DiagnosticCases
from daguandan_bridge.application.session_diagnostic_frames import SessionDiagnosticFrameStore
from daguandan_bridge.gui.live_controller import LiveAssistantController, _LiveRunToken
from test_live_listener_diagnostic_capture import _CaptureServiceStub, _snapshot

pytestmark = pytest.mark.integration


@pytest.fixture
def controller(tmp_path):
    app = QApplication.instance() or QApplication([])
    capture = _CaptureServiceStub(tmp_path / "profiles", manual_snapshot=_snapshot(41))
    profile = capture.profiles_root / "tencent_daguandan"
    profile.mkdir(parents=True)
    for name in ("profile.json", "regions_config.json", "templates_config.json"):
        (profile / name).write_text("{}", encoding="utf8")
    value = LiveAssistantController(capture, recognition_service=SimpleNamespace(),
        advisor=object(), session_factory=SimpleNamespace(),
        opening_evidence_monitor=SimpleNamespace(), diagnostics_root=tmp_path / "diagnostics")
    yield value
    if value._problem_export_thread is not None:
        value._problem_export_thread.wait(20_000)
        app.processEvents()
    value.orchestrator = None
    value._active_live_token = None
    value.shutdown()
    app.processEvents()


def test_all_capture_sources_share_stable_case_and_exact_pixels(controller, tmp_path):
    manual = controller.save_latest_live_frame_to_session()
    case = Path(manual["session_directory"])
    assert case.parent == tmp_path / "diagnostics/cases"
    assert Path(manual["image_path"]).parent == case / "frames"
    original = Path(manual["image_path"]).read_bytes()
    controller._listening_enabled = True
    controller._waiting_generation = 2
    episode = SimpleNamespace(directory=tmp_path / "external-sessions/.preopening/episode", session_id="episode")
    controller._bind_preopening_diagnostic_to_recording(SimpleNamespace(store=episode))
    failure = controller._retain_listener_snapshot(_snapshot(51), 2, 1)
    controller._queue_listener_incident("page_unknown", context=failure, error="unknown page")
    assert controller._listener_evidence.writer.wait_idle(5)
    formal = SimpleNamespace(store=SimpleNamespace(directory=tmp_path / "external-sessions/game",session_id="game"),
                             snapshot=SimpleNamespace(session_id="game"))
    controller.orchestrator = formal
    controller._capture_generation = 4
    token = _LiveRunToken(formal,"game",1,4)
    controller._active_live_token = token
    controller._retain_listener_snapshot(_snapshot(61),4,1,token=token)
    live = controller.save_latest_live_frame_to_session()
    assert Path(live["session_directory"]) == case
    assert Path(manual["image_path"]).read_bytes() == original
    records = SessionDiagnosticFrameStore().list_frames(case)
    assert len(records) == 3
    assert [r.sequence for r in records] == [1, 2, 3]
    assert {r.metadata["source"] for r in records} == {"manual_window_capture", "live_listener_frame"}
    assert {r.metadata["case_id"] for r in records} == {case.name}
    assert any(r.metadata.get("source_phase") == "failed_listener_frame" for r in records)
    assert np.array_equal(SessionDiagnosticFrameStore().load_image(records[0]), controller.capture_service.manual_snapshot.image)
    assert len(list((case / "frames").glob("incident_*.json"))) == 1
    assert len(list((tmp_path / "diagnostics/cases").iterdir())) == 1
    assert not episode.directory.exists()
    assert not formal.store.directory.exists()
    assert controller.diagnostic_frame_directory() == case / "frames"
    metadata = json.loads((case / "case.json").read_text("utf8"))
    assert metadata["session_id"] == "game"
    assert {x["kind"] for x in metadata["session_links"]} == {"session", "preopening"}


def test_case_assignment_is_lazy_and_new_formal_games_are_separate(tmp_path):
    repo = DiagnosticCases(tmp_path / "diagnostics", profile_name="tencent_daguandan")
    before = repo.allocate()
    assert not before.exists()
    assert repo.bind_session("episode", tmp_path / "episode", formal=False) == before
    assert repo.bind_session("first", tmp_path / "first") == before
    repo.finish_session("first")
    after = repo.bind_session("second", tmp_path / "second")
    assert after != before
    repo.materialize(before)
    repo.materialize(after)
    assert repo.describe(before)["session_id"] == "first"
    assert repo.describe(before)["status"] == "finished"
    assert repo.describe(after)["session_id"] == "second"


def test_config_snapshot_does_not_change_when_export_happens_later(controller):
    saved = controller.save_latest_live_frame_to_session()
    case = Path(saved["session_directory"])
    profile = controller.capture_service.profiles_root / controller.profile_name
    (profile / "profile.json").write_text('{"changed":true}',encoding="utf8")
    controller._diagnostic_cases.materialize(case, profile_directory=profile)
    assert json.loads((case / "config/profile.json").read_text("utf8")) == {}


def test_export_without_game_or_images_does_not_stop_listener(controller):
    from daguandan_bridge import problem_bundle
    statuses = []
    controller.problem_export_status.connect(statuses.append)
    controller._listening_enabled = True
    assert controller.request_problem_export(include_images=False)
    assert not controller.request_problem_export(include_images=False)
    assert controller._problem_export_thread.wait(20_000)
    QApplication.instance().processEvents()
    assert controller._listening_enabled
    assert statuses[0]["status"] == "RUNNING"
    assert statuses[-1]["status"] in {"SUCCESS", "PARTIAL"}, statuses
    assert Path(statuses[-1]["archive_path"]).is_file()
    assert controller._problem_export_thread is None


def test_slow_case_disk_write_does_not_block_capture_metadata(tmp_path, monkeypatch):
    from threading import Event, Thread
    from daguandan_bridge.application import diagnostic_cases as module
    entered, release, captured = Event(), Event(), Event()
    real_write = module.atomic_json
    def slow_write(path, value):
        entered.set()
        assert release.wait(5)
        real_write(path, value)
    monkeypatch.setattr(module, "atomic_json", slow_write)
    repo = DiagnosticCases(tmp_path / "diagnostics", profile_name="test")
    directory = repo.allocate()
    writer = Thread(target=lambda: repo.materialize(directory))
    writer.start()
    try:
        assert entered.wait(2)
        capture = Thread(target=lambda: (repo.bind_session("new", tmp_path / "new"), captured.set()))
        capture.start()
        assert captured.wait(1), "capture waited on evidence disk IO"
        capture.join(2)
    finally:
        release.set()
        writer.join(5)


def test_real_problem_zip_contains_manual_failure_and_active_session_frames(controller, tmp_path):
    import zipfile
    manual = controller.save_latest_live_frame_to_session()
    case = Path(manual["session_directory"])
    controller._listening_enabled = True
    controller._waiting_generation = 2
    failed = controller._retain_listener_snapshot(_snapshot(52), 2, 1)
    controller._queue_listener_incident("page_unknown", context=failed)
    assert controller._listener_evidence.writer.wait_idle(5)
    controller._listener_evidence.recover(controller._retain_listener_snapshot(_snapshot(62), 2, 2))
    assert controller._listener_evidence.writer.wait_idle(5)
    statuses = []
    controller.problem_export_status.connect(statuses.append)
    assert controller.request_problem_export()
    assert controller._problem_export_thread.wait(20_000)
    QApplication.instance().processEvents()
    assert statuses[-1]["status"] in {"SUCCESS", "PARTIAL"}, statuses
    with zipfile.ZipFile(statuses[-1]["archive_path"]) as archive:
        pngs = [name for name in archive.namelist() if name.endswith(".png")]
        assert len(pngs) == 3
        assert any("incident_" in name for name in archive.namelist())
        for name in pngs:
            assert archive.read(name) == (case / "frames" / Path(name).name).read_bytes()
    assert controller._listening_enabled
    assert controller.capture_service.sources[0].capture_calls == 1


def test_mixed_legacy_directory_does_not_split_new_case(controller):
    first = controller.save_latest_live_frame_to_session()
    case = Path(first["session_directory"])
    (case / "diagnostic_frames").mkdir()
    second = controller.save_latest_live_frame_to_session()
    assert Path(second["image_path"]).parent == case / "frames"
    assert controller.diagnostic_frame_directory() == case / "frames"
    assert len(SessionDiagnosticFrameStore().list_frames(case)) == 2
    assert not list((case / "diagnostic_frames").iterdir())


def test_shutdown_final_case_state_is_persisted_without_processing_qt_callbacks(controller, tmp_path):
    case = controller._diagnostic_cases.bind_session("shutdown-session", tmp_path / "session")
    controller._diagnostic_cases.materialize(case)
    class Runtime:
        snapshot = SimpleNamespace(session_id="shutdown-session")
        def begin_finalizing(self): pass
        def finish(self): return SimpleNamespace(snapshot=self.snapshot)
    controller.orchestrator = Runtime()
    # shutdown waits the real finish QThread; deliberately do not processEvents.
    controller.shutdown()
    assert json.loads((case / "case.json").read_text("utf8"))["status"] == "finished"
