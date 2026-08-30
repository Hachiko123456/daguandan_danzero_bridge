from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from daguandan_bridge import support_export
from daguandan_bridge.support_bundle import SupportBundleError, UnsafeSupportSourceError
from daguandan_bridge.support_export import (
    SupportExportRequest,
    export_collected_support_bundle,
)


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _png(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((6, 8, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(bytes(encoded))
    return path


def _archive(path: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        entries = {name: archive.read(name) for name in archive.namelist()}
    return entries, json.loads(entries["support_manifest.json"].decode("utf-8"))


def _diagnostic_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run = tmp_path / "diagnostics" / "runs" / "RUN-1"
    run.mkdir(parents=True)
    (run / "startup.log").write_text(
        r"path=C:\Users\Alice\private\startup.log" + "\n",
        encoding="utf-8",
    )
    (run / "startup.jsonl").write_text(
        json.dumps({"schema": "guandan.startup-event/1", "event": "ready"}) + "\n",
        encoding="utf-8",
    )
    (run / "exceptions.log").write_text("no uncaught exceptions\n", encoding="utf-8")
    (run / "runtime.log").write_text("runtime ready\n", encoding="utf-8")
    _json(run / "runtime_identity.json", {"schema": "guandan.runtime-identity/1", "run_id": "RUN-1"})
    _json(run / "doctor.json", {"schema": "guandan.doctor/1", "status": "PASS"})

    incidents = run / "opening" / "incidents"
    old = incidents / "OPEN-old"
    latest = incidents / "OPEN-latest"
    _json(old / "incident.json", {"code": "OPENING-OLD", "monotonic_ms": 100})
    _json(latest / "incident.json", {"code": "OPENING-LEVEL-MISSING", "monotonic_ms": 200})
    _json(
        latest / "opening_evidence.json",
        {
            "schema": "guandan.opening-evidence/1",
            "frames": [
                {
                    "seq": 7,
                    "monotonic_ms": 1_234,
                    "wall_time": "2026-08-31T00:00:00+08:00",
                    "capture": {"backend": "printwindow", "dpi": 120},
                }
            ],
        },
    )
    _json(
        latest / "repro.json",
        {
            "schema": "guandan.repro-manifest/1",
            "incident_id": "OPEN-latest",
            "expected_truth": None,
        },
    )
    (latest / "recognition_trace.jsonl").write_text(
        json.dumps({"frame_seq": 7, "result": "blocked"}) + "\n",
        encoding="utf-8",
    )
    _png(latest / "frames" / "raw_client_000007.png", 30)
    _png(latest / "frames" / "standardized_000007.png", 60)
    _png(latest / "roi" / "level_rank_000007.png", 90)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _json(
        bundle / "build_manifest.json",
        {"schema": "guandan.build-manifest/1", "build_id": "BUILD-portable-test"},
    )
    session = tmp_path / "session"
    session.mkdir()
    _json(
        session / "health_audit.json",
        {"schema": "guandan.session-health/1", "status": "FAIL"},
    )
    return run, bundle, session, latest


def test_collector_auto_selects_latest_opening_incident_and_keeps_images_opt_in(
    tmp_path: Path,
) -> None:
    run, bundle, session, latest = _diagnostic_tree(tmp_path)
    destination = tmp_path / "out" / "support.zip"

    result = export_collected_support_bundle(
        SupportExportRequest(
            destination=destination,
            diagnostics_run_directory=run,
            bundle_root=bundle,
            session_directory=session,
        )
    )

    entries, manifest = _archive(destination)
    assert result.selected_opening_incident == latest.resolve()
    assert result.selected_session_incident is None
    assert manifest["build_id"] == "BUILD-portable-test"
    assert {
        "startup/startup.log",
        "startup/startup.jsonl",
        "startup/exceptions.log",
        "runtime/runtime.log",
        "runtime/runtime_identity.json",
        "doctor/doctor.json",
        "build/build_manifest.json",
        "incident/incident.json",
        "evidence/opening_evidence.json",
        "evidence/frame_index.jsonl",
        "repro/repro.json",
        "health/health_audit.json",
    }.issubset(entries)
    assert not any(name.startswith(("frames/", "roi/")) for name in entries)
    assert "trace/recognition_trace.jsonl" not in entries
    assert "evidence/image_index.json" not in entries
    assert manifest["privacy"]["contains_sensitive_images"] is False
    assert b"C:\\Users\\Alice" not in entries["startup/startup.log"]
    frame_index = entries["evidence/frame_index.jsonl"].decode("utf-8")
    assert '"frame_seq":7' in frame_index
    assert '"monotonic_ms":1234' in frame_index


def test_collector_opt_in_emits_trace_and_correlated_image_index(tmp_path: Path) -> None:
    run, bundle, session, _latest = _diagnostic_tree(tmp_path)
    destination = tmp_path / "support.zip"

    export_collected_support_bundle(
        SupportExportRequest(
            destination=destination,
            diagnostics_run_directory=run,
            bundle_root=bundle,
            session_directory=session,
            include_frames=True,
            include_roi=True,
            include_recognition_trace=True,
        )
    )

    entries, manifest = _archive(destination)
    assert "trace/recognition_trace.jsonl" in entries
    index = json.loads(entries["evidence/image_index.json"].decode("utf-8"))
    assert index["schema"] == "guandan.support-image-index/1"
    assert {(item["kind"], item["field"]) for item in index["entries"]} == {
        ("raw_client", None),
        ("standardized", None),
        ("roi", "level_rank"),
    }
    assert {item["frame_seq"] for item in index["entries"]} == {7}
    assert {item["monotonic_ms"] for item in index["entries"]} == {1_234}
    records = {record["path"]: record for record in manifest["files"]}
    for item in index["entries"]:
        assert item["archive_path"] in entries
        assert item["source_sha256"] == records[item["archive_path"]]["sha256"]
        assert item["source_sha256"] == hashlib.sha256(
            entries[item["archive_path"]]
        ).hexdigest()
    assert manifest["privacy"]["contains_sensitive_images"] is True


def test_collector_falls_back_to_latest_sealed_session_incident(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _json(session / "incidents" / "INC-0001" / "incident.json", {"trigger_ms": 10})
    latest = session / "incidents" / "INC-0002"
    _json(latest / "incident.json", {"trigger_ms": 20, "reason": "model failed"})
    _json(session / "health_audit.json", {"status": "PASS"})
    (session / "recognition_trace.jsonl").write_text(
        json.dumps({"result": "blocked"}) + "\n",
        encoding="utf-8",
    )

    result = export_collected_support_bundle(
        SupportExportRequest(
            destination=tmp_path / "support.zip",
            session_directory=session,
            include_recognition_trace=True,
        )
    )

    entries, _manifest = _archive(tmp_path / "support.zip")
    assert result.selected_opening_incident is None
    assert result.selected_session_incident == latest.resolve()
    assert json.loads(entries["incident/incident.json"])["trigger_ms"] == 20
    assert "trace/recognition_trace.jsonl" in entries


def test_explicit_opening_incident_cannot_escape_run_collection(tmp_path: Path) -> None:
    run, bundle, _session, _latest = _diagnostic_tree(tmp_path)
    outside = tmp_path / "outside"
    _json(outside / "incident.json", {"monotonic_ms": 999})

    with pytest.raises(UnsafeSupportSourceError, match="outside"):
        export_collected_support_bundle(
            SupportExportRequest(
                destination=tmp_path / "support.zip",
                diagnostics_run_directory=run,
                bundle_root=bundle,
                opening_incident_directory=outside,
            )
        )


def test_staging_rejects_a_reparse_source_before_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run, bundle, _session, _latest = _diagnostic_tree(tmp_path)
    original = support_export._is_link_or_reparse
    monkeypatch.setattr(
        support_export,
        "_is_link_or_reparse",
        lambda path: path.name == "startup.log" or original(path),
    )

    with pytest.raises(UnsafeSupportSourceError, match="reparse"):
        export_collected_support_bundle(
            SupportExportRequest(
                destination=tmp_path / "support.zip",
                diagnostics_run_directory=run,
                bundle_root=bundle,
            )
        )


def test_failed_export_preserves_existing_bundle_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run, bundle, _session, _latest = _diagnostic_tree(tmp_path)
    destination = tmp_path / "out" / "support.zip"
    destination.parent.mkdir()
    destination.write_bytes(b"previous-good-bundle")
    monkeypatch.setattr(
        support_export,
        "export_support_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("archive failed")),
    )

    with pytest.raises(OSError, match="archive failed"):
        export_collected_support_bundle(
            SupportExportRequest(
                destination=destination,
                diagnostics_run_directory=run,
                bundle_root=bundle,
            )
        )

    assert destination.read_bytes() == b"previous-good-bundle"
    assert list(destination.parent.glob("daguandan-support-*")) == []


def test_missing_explicit_root_is_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(SupportBundleError, match="not a directory"):
        export_collected_support_bundle(
            SupportExportRequest(
                destination=tmp_path / "support.zip",
                diagnostics_run_directory=tmp_path / "missing-run",
            )
        )
