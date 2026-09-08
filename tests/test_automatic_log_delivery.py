from __future__ import annotations

import json
import gzip
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

import daguandan_bridge.automatic_log_delivery as automatic_log_delivery
from daguandan_bridge.automatic_log_delivery import AutomaticLogDeliveryService
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.orchestrator import LiveOrchestrator


def _sealed_session(tmp_path: Path) -> Path:
    store = LiveSessionStore(
        tmp_path / "profiles",
        "profile",
        session_id="game_20260902_120000_abcdef",
    )
    store.start({"recording_mode": "game"})
    store.append_recognition_trace({"phase": "test", "cards": ["10C"]})
    store.seal(frame_count=2, dropped_frames=0)
    store.append_post_seal_health_audit(
        {
            "schema": "guandan.session-health/1",
            "status": "PASS",
            "issues": [],
        },
        state={"revision": 1},
        monotonic_ms=100,
    )
    video = store.directory / "video"
    video.mkdir()
    (video / "game.avi").write_bytes(b"sensitive-video")
    (video / "frame_index.jsonl").write_text("{}\n", encoding="utf-8")
    incident = store.incidents_directory / "INC-9999"
    incident.mkdir()
    (incident / "incident.json").write_text("{}", encoding="utf-8")
    (incident / "frame.png").write_bytes(b"sensitive-image")
    return store.directory


def _zip_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def test_automatic_delivery_is_idempotent_and_excludes_sensitive_media(tmp_path):
    session = _sealed_session(tmp_path)
    service = AutomaticLogDeliveryService(
        documents_root=tmp_path / "Documents",
        fallback_root=tmp_path / "fallback",
    )

    first = service.export(session)
    second = service.export(session)

    assert first.output_directory == second.output_directory
    assert first.diagnostic_zip_path == second.diagnostic_zip_path
    assert first.summary_path is not None and first.summary_path.is_file()
    assert first.machine_summary_path is not None
    summary = json.loads(first.machine_summary_path.read_text(encoding="utf-8"))
    assert summary["includes_sensitive_media"] is False
    assert summary["retention_policy"].startswith("user-managed")
    names = _zip_names(first.diagnostic_zip_path)
    assert "automatic_delivery_manifest.json" in names
    assert "session/manifest.json" in names
    assert "session/timeline.jsonl" in names
    assert "session/advice.jsonl" in names
    assert "session/health_audit.json" in names
    assert "session/recognition_trace.jsonl" in names
    assert "session/video/frame_index.jsonl" in names
    assert "session/incidents/INC-9999/incident.json" in names
    assert not any(name.endswith((".avi", ".png")) for name in names)
    assert not list(first.output_directory.glob("*.tmp"))
    with zipfile.ZipFile(first.diagnostic_zip_path) as archive:
        assert archive.getinfo("automatic_delivery_manifest.json").date_time == (
            1980,
            1,
            1,
            0,
            0,
            0,
        )
        delivery = json.loads(
            archive.read("automatic_delivery_manifest.json").decode("utf-8")
        )
    assert delivery["includes_sensitive_media"] is False
    assert all(len(item["sha256"]) == 64 for item in delivery["files"])
    assert len(first.diagnostic_zip_sha256) == 64
    summary = json.loads(first.machine_summary_path.read_text(encoding="utf-8"))
    assert summary["diagnostic_zip_path"] == str(first.diagnostic_zip_path)
    assert summary["diagnostic_zip_sha256"] == first.diagnostic_zip_sha256


def test_explicit_full_diagnostic_includes_video_and_images(tmp_path):
    session = _sealed_session(tmp_path)
    result = AutomaticLogDeliveryService(
        documents_root=tmp_path / "Documents",
        fallback_root=tmp_path / "fallback",
    ).export(session, include_media=True)

    names = _zip_names(result.diagnostic_zip_path)
    assert "session/video/game.avi" in names
    assert "session/video/frame_index.jsonl" in names
    assert "session/incidents/INC-9999/frame.png" in names


def test_changed_retry_is_versioned_instead_of_silently_overwritten(tmp_path):
    session = _sealed_session(tmp_path)
    service = AutomaticLogDeliveryService(
        documents_root=tmp_path / "Documents",
        fallback_root=tmp_path / "fallback",
    )
    first = service.export(session)
    first.diagnostic_zip_path.write_bytes(b"foreign-content")

    retried = service.export(session)

    assert retried.diagnostic_zip_path != first.diagnostic_zip_path
    assert retried.diagnostic_zip_path.name.endswith("-retry-02.zip")
    assert first.diagnostic_zip_path.read_bytes() == b"foreign-content"


def test_unwritable_documents_falls_back_and_records_actual_path(tmp_path):
    session = _sealed_session(tmp_path)
    blocked = tmp_path / "blocked-documents"
    blocked.write_text("not a directory", encoding="utf-8")
    fallback = tmp_path / "safe-fallback"

    result = AutomaticLogDeliveryService(
        documents_root=blocked,
        fallback_root=fallback,
    ).export(session)

    assert result.used_fallback is True
    assert result.output_directory.is_relative_to(fallback)
    summary = json.loads(result.machine_summary_path.read_text(encoding="utf-8"))
    assert summary["used_fallback_directory"] is True
    assert summary["output_directory"] == str(result.output_directory)


def test_explicit_log_root_configuration_is_used(tmp_path, monkeypatch):
    session = _sealed_session(tmp_path)
    configured = (tmp_path / "configured").resolve()
    monkeypatch.setenv("DAGUANDAN_LOG_ROOT", str(configured))

    result = AutomaticLogDeliveryService(
        fallback_root=tmp_path / "fallback"
    ).export(session)

    assert result.used_fallback is False
    assert result.output_directory.is_relative_to(configured / "掼蛋助手日志")


def test_orchestrator_finish_auto_delivers_without_affecting_seal(tmp_path, monkeypatch):
    configured = (tmp_path / "visible").resolve()
    monkeypatch.setenv("DAGUANDAN_LOG_ROOT", str(configured))
    store = LiveSessionStore(
        tmp_path / "profiles",
        "profile",
        session_id="game_20260902_130000_auto01",
        automatic_log_delivery_enabled=True,
    )
    store.start({})
    recorder = SessionRecorder(store.directory, size=(64, 32), fps=10)
    orchestrator = LiveOrchestrator(
        reducer=LiveReducer(store.session_id),
        store=store,
        recorder=recorder,
        recognition_service=object(),
        advisor=None,
        minimum_free_bytes=0,
    )
    hand = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")
    orchestrator.start(
        round_level="2",
        hand=hand,
        lead_player="right",
        monotonic_ms=0,
    )
    orchestrator.record_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="2026-09-02T13:00:00+08:00",
    )

    update = orchestrator.finish()

    assert update.status == "sealed"
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    delivery = manifest["automatic_log_delivery"]
    assert delivery["status"] == "PASS"
    assert Path(delivery["diagnostic_zip_path"]).is_file()
    assert Path(delivery["output_directory"]).is_relative_to(
        configured / "掼蛋助手日志"
    )
    assert (store.directory / "automatic_log_delivery.json").is_file()


def test_automatic_delivery_failure_is_recorded_but_session_still_seals(
    tmp_path,
    monkeypatch,
):
    from daguandan_bridge import automatic_log_delivery

    monkeypatch.setattr(
        automatic_log_delivery,
        "export_automatic_session_log",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("documents full")),
    )
    store = LiveSessionStore(
        tmp_path / "profiles",
        "profile",
        session_id="game_20260902_130100_fail01",
        automatic_log_delivery_enabled=True,
    )
    store.start({})
    orchestrator = LiveOrchestrator(
        reducer=LiveReducer(store.session_id),
        store=store,
        recorder=SessionRecorder(store.directory, size=(64, 32), fps=10),
        recognition_service=object(),
        advisor=None,
        minimum_free_bytes=0,
    )
    hand = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")
    orchestrator.start(
        round_level="2",
        hand=hand,
        lead_player="right",
        monotonic_ms=0,
    )

    update = orchestrator.finish()

    assert update.status == "sealed"
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "sealed"
    assert manifest["automatic_log_delivery"]["status"] == "FAIL"
    assert "documents full" in manifest["automatic_log_delivery"]["error"]
    evidence = json.loads(
        store.directory.joinpath("automatic_log_delivery.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "FAIL"


def test_recorded_delivery_metadata_does_not_change_repeat_zip(tmp_path):
    store = LiveSessionStore(
        tmp_path / "profiles",
        "profile",
        session_id="game_20260902_140000_repeat",
    )
    store.start({"runtime_identity": {"executable_path": r"C:\Users\Alice\app.exe"}})
    store.seal(frame_count=0, dropped_frames=0)
    service = AutomaticLogDeliveryService(
        documents_root=tmp_path / "Documents",
        fallback_root=tmp_path / "fallback",
    )

    first = service.export(store.directory)
    store.record_automatic_log_delivery(first.to_dict())
    second = service.export(store.directory)

    assert second.diagnostic_zip_path == first.diagnostic_zip_path
    assert second.diagnostic_zip_sha256 == first.diagnostic_zip_sha256
    assert not list(first.output_directory.glob("*-retry-*.zip"))


def test_all_text_evidence_is_sanitized_without_deleting_business_fields(tmp_path):
    session = _sealed_session(tmp_path)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_identity"] = {
        "executable_path": r"C:\Users\yang yang\Desktop\DaguandanAssistant.exe",
        "model_path": r"C:\Users\yang yang\AppData\Local\model.npz",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (session / "advice.jsonl").write_text(
        json.dumps(
            {
                "decision_log_path": r"C:\Users\yhx\Documents\private\decision.jsonl",
                "status": "ready",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session / "recognition_trace.jsonl").write_text(
        json.dumps({"source": r"C:\Users\yang yang\Pictures\frame.png"}) + "\n",
        encoding="utf-8",
    )
    with gzip.open(session / "observations.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"cache": r"C:\Users\yhx\AppData\Local\cache.bin"})
            + "\n"
        )

    result = AutomaticLogDeliveryService(
        documents_root=tmp_path / "Documents",
        fallback_root=tmp_path / "fallback",
    ).export(session, include_media=True)

    text_parts: list[str] = []
    with zipfile.ZipFile(result.diagnostic_zip_path) as archive:
        delivery = json.loads(
            archive.read("automatic_delivery_manifest.json").decode("utf-8")
        )
        for entry in archive.infolist():
            if entry.filename.endswith((".avi", ".png", ".jpg", ".jpeg", ".bmp")):
                continue
            content = archive.read(entry)
            if entry.filename.endswith(".gz"):
                content = gzip.decompress(content)
            text_parts.append(content.decode("utf-8", errors="replace"))
    combined = "\n".join(text_parts)
    assert "C:\\Users\\" not in combined
    assert "yang yang" not in combined
    assert "yhx" not in combined
    assert "decision_log_path" in combined
    assert "status" in combined
    assert delivery["text_sanitized"] is True
    assert delivery["redaction_count"] >= 4
    assert result.redaction_count == delivery["redaction_count"]


def test_output_inside_session_is_rejected(tmp_path):
    session = _sealed_session(tmp_path)
    service = AutomaticLogDeliveryService(
        documents_root=session,
        fallback_root=tmp_path / "fallback",
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        service.export(session)


def test_session_inside_output_is_rejected(tmp_path):
    output_parent = tmp_path / "visible"
    sessions_root = output_parent / "掼蛋助手日志" / "nested"
    store = LiveSessionStore(
        sessions_root,
        "profile",
        session_id="game_20260902_141000_nested",
    )
    store.start({})
    store.seal(frame_count=0, dropped_frames=0)

    with pytest.raises(ValueError, match="must be disjoint"):
        AutomaticLogDeliveryService(
            documents_root=output_parent,
            fallback_root=tmp_path / "fallback",
        ).export(store.directory)


def test_output_reparse_point_is_rejected_when_supported(tmp_path):
    session = _sealed_session(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="reparse point"):
        AutomaticLogDeliveryService(
            documents_root=link,
            fallback_root=tmp_path / "fallback",
        ).export(session)


@pytest.mark.parametrize("relative", ["timeline.jsonl", "incidents/INC-9999/escape.json"])
def test_session_file_symlink_is_rejected_before_zip_reads_external_file(
    tmp_path: Path, relative: str
):
    session = _sealed_session(tmp_path)
    target = tmp_path / "outside.txt"
    target.write_text("must never enter archive", encoding="utf-8")
    destination = session / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.symlink(target, destination)
    except OSError as exc:
        pytest.skip(f"file symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="session tree contains|session source"):
        AutomaticLogDeliveryService(
            documents_root=tmp_path / "Documents",
            fallback_root=tmp_path / "fallback",
        ).export(session)


def test_session_directory_junction_or_symlink_is_rejected_before_traversal(tmp_path):
    session = _sealed_session(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    (target / "leak.json").write_text('{"secret":"no"}\n', encoding="utf-8")
    link = session / "incidents" / "external"
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="session tree contains"):
        AutomaticLogDeliveryService(
            documents_root=tmp_path / "Documents",
            fallback_root=tmp_path / "fallback",
        ).export(session)


def test_frozen_portable_mode_writes_beside_bundle_not_inside_it(tmp_path, monkeypatch):
    session = _sealed_session(tmp_path)
    bundle = tmp_path / "DaguandanAssistant"
    bundle.mkdir()
    executable = bundle / "DaguandanAssistant.exe"
    executable.write_bytes(b"exe")
    monkeypatch.setattr(automatic_log_delivery.sys, "frozen", True, raising=False)
    monkeypatch.setattr(automatic_log_delivery.sys, "executable", str(executable))
    monkeypatch.setenv("DAGUANDAN_PORTABLE_MODE", "1")

    result = AutomaticLogDeliveryService(fallback_root=tmp_path / "fallback").export(
        session
    )

    assert result.output_directory.is_relative_to(tmp_path / "DaguandanAssistant_UserData")
    assert not result.output_directory.is_relative_to(bundle)
