from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from daguandan_bridge import problem_bundle as pb
from daguandan_bridge.problem_bundle import ProblemBundleRequest, export_problem_bundle
from daguandan_bridge.support_bundle import SupportBundleError, UnsafeSupportSourceError

pytestmark = pytest.mark.unit


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _pair(directory: Path, index=1, *, session_id="game", case_id=None):
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    pixels[0, 0, 0] = index
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    image = encoded.tobytes()
    metadata = {
        "schema": "guandan.session-diagnostic-frame/v1", "sequence": index,
        "source": "live_listener_frame", "session_id": session_id,
        "width": 16, "height": 12, "channels": 3, "dtype": "uint8",
        "raw_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "png_sha256": hashlib.sha256(image).hexdigest(),
    }
    if case_id is not None:
        metadata["case_id"] = case_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{index:06d}.png").write_bytes(image)
    _write(directory / f"{index:06d}.json", metadata)
    return image, metadata


def _contents(result):
    path = Path(result["archive_path"])
    assert path.is_absolute() and path.is_file()
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        return {name: archive.read(name) for name in archive.namelist()}


def _reasons(result):
    return {item["reason"] for item in result["missing"] + result["omitted"]}


@pytest.fixture
def request_data(tmp_path, monkeypatch):
    monkeypatch.delenv("DAGUANDAN_SESSIONS_ROOT", raising=False)
    root = tmp_path / "中文 项目" / "diagnostics"
    case = root / "cases" / "case_测试"
    run = root / "runs" / "run_01"
    profile = tmp_path / "配置" / "profile_A"
    session = profile / "sessions" / "game"
    bundle = tmp_path / "安装 目录"
    _write(case / "case.json", {
        "schema": "guandan.problem-case/1", "case_id": case.name,
        "session_id": "game", "session_directory": str(session),
        "run_id": run.name, "run_directory": str(run), "profile_name": profile.name,
        "profile_directory": str(profile),
    })
    for name in pb._PROFILE_FILES:
        _write(profile / name, {"generation": "export_time"})
        _write(case / "config" / name, {"generation": "case_creation"})
    _write(run / "startup_report.json", {"run_id": run.name, "marker": "right-run"})
    (run / "startup.jsonl").write_bytes(b'{"event":"ready"}\n')
    (run / "exceptions.log").write_bytes(b"no exception\n")
    _write(session / "manifest.json", {"session_id": "game", "sealed": False, "recording_phase": "live"})
    (session / "recognition_trace.jsonl").write_bytes(b'{"phase":"live"}\n')
    (session / "observations.jsonl.part").write_bytes(b'{"revision":1}\n')
    _write(bundle / "build_manifest.json", {
        "schema": "guandan.build-manifest/1", "build_id": "build-01",
        "source": {"commit": "abcd", "private_path": "must-not-leave"},
        "resources": {"recognition": {"sha256": "0" * 64, "files": ["not-exported"]}},
    })
    _pair(case / "frames", case_id=case.name)
    return ProblemBundleRequest(root, case, run, session, profile, bundle)


def test_full_export_exact_pixels_associations_chinese_paths_and_config_snapshots(request_data):
    before = {p: p.read_bytes() for root in (request_data.case_directory, request_data.run_directory,
              request_data.session_directory, request_data.profile_directory) for p in root.rglob("*") if p.is_file()}
    result = export_problem_bundle(request_data)
    contents = _contents(result)
    assert result["status"] == "SUCCESS", result
    assert result["case_id"] == request_data.case_directory.name
    assert result["has_formal_session"] is True
    image = contents["case/frames/000001.png"]
    assert image == before[request_data.case_directory / "frames/000001.png"]
    metadata = json.loads(contents["case/frames/000001.json"])
    assert hashlib.sha256(image).hexdigest() == metadata["png_sha256"]
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == metadata["raw_sha256"]
    assert json.loads(contents["case/config/profile.json"])["generation"] == "case_creation"
    assert json.loads(contents["profile/profile.json"])["generation"] == "export_time"
    assert json.loads(contents["profile/export_annotation.json"])["export_time"]
    assert b"must-not-leave" not in contents["build/build_manifest_summary.json"]
    assert "离散" in contents["README.txt"].decode("utf-8")
    assert "不自动上传" in contents["README.txt"].decode("utf-8") or "不会自动上传" in contents["README.txt"].decode("utf-8")
    for path, data in before.items():
        assert path.read_bytes() == data


def test_export_without_session_or_screenshots_and_missing_root(tmp_path):
    result = export_problem_bundle(ProblemBundleRequest(tmp_path / "未建立" / "diagnostics"))
    data = _contents(result)
    assert result["status"] == "PARTIAL"
    assert result["has_formal_session"] is False
    assert result["image_count"] == 0
    assert set(data) == {"README.txt", "problem_manifest.json"}
    assert "无正式对局" in data["README.txt"].decode("utf-8")


@pytest.mark.parametrize("use_case_id", [False, True])
def test_manual_preopening_formal_frames_share_one_case(request_data, use_case_id):
    case = request_data.case_directory
    document = json.loads((case / "case.json").read_text("utf-8"))
    document["session_links"] = [
        {"session_id": "episode_1", "kind": "preopening"},
        {"session_id": "game", "kind": "session"},
    ]
    _write(case / "case.json", document)
    expected = {}
    for i, session in enumerate((case.name, "episode_1", "game"), 1):
        expected[i] = _pair(case / "frames", i, session_id=session,
                            case_id=case.name if use_case_id else None)[0]
    _pair(case / "frames", 4, session_id="intruder", case_id="other_case" if use_case_id else None)
    result = export_problem_bundle(request_data)
    contents = _contents(result)
    assert result["image_count"] == 3
    for i, raw in expected.items():
        assert contents[f"case/frames/{i:06d}.png"] == raw
    assert "case/frames/000004.png" not in contents
    assert "frame_association_mismatch" in _reasons(result)


def test_legacy_diagnostic_frames(request_data):
    _pair(request_data.session_directory / "diagnostic_frames", 7)
    result = export_problem_bundle(request_data)
    assert "session/diagnostic_frames/000007.png" in _contents(result)


def test_no_images_does_not_even_read_or_decode_media(request_data, monkeypatch):
    original_open = pb.os.open
    def guarded_open(path, *args, **kwargs):
        assert Path(path).suffix not in {".png", ".jpg", ".mp4", ".avi"}
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(pb.os, "open", guarded_open)
    monkeypatch.setattr(pb, "_validated_image_info", lambda *a, **k: pytest.fail("decoded media"))
    (request_data.session_directory / "video.mp4").write_bytes(b"private video")
    result = export_problem_bundle(replace(request_data, include_images=False))
    contents = _contents(result)
    assert result["include_images"] is False and result["image_count"] == 0
    assert all(Path(name).suffix in {".json", ".jsonl", ".log", ".txt"} for name in contents)
    assert "images_disabled_by_request" in _reasons(result)


@pytest.mark.parametrize("damage,reason", [
    ("bad_png", "invalid_png"), ("missing_json", "incomplete_frame_pair"),
    ("raw_hash", "raw_sha256_mismatch"), ("png_hash", "png_sha256_mismatch"),
    ("missing_iend", "incomplete_png"), ("bad_json", "invalid_frame_pair"),
])
def test_invalid_or_incomplete_pair_is_never_exported(request_data, damage, reason):
    frames = request_data.case_directory / "frames"
    png, meta = frames / "000001.png", frames / "000001.json"
    if damage == "bad_png":
        png.write_bytes(b"not png")
    elif damage == "missing_iend":
        png.write_bytes(png.read_bytes()[:-12])
    elif damage == "missing_json":
        meta.unlink()
    elif damage == "bad_json":
        meta.write_text("{")
    else:
        document = json.loads(meta.read_text("utf-8"))
        document["raw_sha256" if damage == "raw_hash" else "png_sha256"] = "f" * 64
        _write(meta, document)
    result = export_problem_bundle(request_data)
    contents = _contents(result)
    assert "case/frames/000001.png" not in contents and "case/frames/000001.json" not in contents
    assert result["status"] == "PARTIAL" and reason in _reasons(result)


def test_queued_append_preserves_only_snapshot_complete_prefix(request_data, monkeypatch):
    path = request_data.run_directory / "startup.jsonl"
    path.write_bytes(b'{"event":"first"}\n{"event":"second"')
    original = pb._Collector._read_snapshot
    def append(self, snapshot, **kwargs):
        if snapshot.path == path:
            with path.open("ab") as handle:
                handle.write(b'}\n{"event":"after-start"}\n')
        return original(self, snapshot, **kwargs)
    monkeypatch.setattr(pb._Collector, "_read_snapshot", append)
    result = export_problem_bundle(request_data)
    contents = _contents(result)
    assert [json.loads(line)["event"] for line in contents["run/startup.jsonl"].splitlines()] == ["first"]
    assert {"concurrent_append", "incomplete_line_tail"} <= _reasons(result)
    assert b"after-start" in path.read_bytes()


def test_queue_new_frames_are_reported_not_waited_for(request_data, monkeypatch):
    original = pb._Collector._read_snapshot
    def append(self, snapshot, **kwargs):
        if snapshot.name == "run/startup.jsonl":
            _pair(request_data.case_directory / "frames", 2, case_id=request_data.case_directory.name)
        return original(self, snapshot, **kwargs)
    monkeypatch.setattr(pb._Collector, "_read_snapshot", append)
    result = export_problem_bundle(request_data)
    assert result["image_count"] == 1
    assert "directory_changed_after_inventory" in _reasons(result)


def test_concurrent_replacement_not_claimed_complete(request_data, monkeypatch):
    path = request_data.run_directory / "startup_report.json"
    original = pb._Collector._read_snapshot
    def change(self, snapshot, **kwargs):
        if snapshot.path == path:
            _write(path, {"changed": "after-start"})
        return original(self, snapshot, **kwargs)
    monkeypatch.setattr(pb._Collector, "_read_snapshot", change)
    result = export_problem_bundle(request_data)
    assert "run/startup_report.json" not in _contents(result)
    assert "concurrent_change" in _reasons(result)


def test_redacts_secrets_and_rejects_case_paths_without_following(request_data):
    case = request_data.case_directory
    secret_file = request_data.diagnostics_root.parent / "private" / "manifest.json"
    _write(secret_file, {"private": "PRIVATE-FILE-MARKER"})
    document = json.loads((case / "case.json").read_text("utf-8"))
    document.update(session_directory=str(secret_file.parent), apiKey="top-secret-key",
                    environment={"private": "private-env"}, password="plaintext-secret",
                    image_base64="raw-media", nickname_email="person@example.com")
    _write(case / "case.json", document)
    (request_data.run_directory / "exceptions.log").write_text(
        'Authorization: Bearer abcdef123456\npassword=hello-world\nsk-0123456789abcdef\n', encoding="utf-8")
    result = export_problem_bundle(replace(request_data, session_directory=None, profile_directory=None))
    contents = _contents(result)
    text = b"\n".join(data for name, data in contents.items() if not name.endswith(".png"))
    for secret in (b"top-secret-key", b"private-env", b"plaintext-secret", b"raw-media", b"person@example.com",
                   b"hello-world", b"abcdef123456", b"sk-0123456789abcdef", b"PRIVATE-FILE-MARKER"):
        assert secret not in text
    assert "unbound_reference_not_followed" in _reasons(result)


def test_bound_roots_reject_traversal_and_wrong_case_root(request_data):
    with pytest.raises(UnsafeSupportSourceError):
        export_problem_bundle(replace(request_data, case_directory=request_data.case_directory / ".." / request_data.case_directory.name))
    with pytest.raises(UnsafeSupportSourceError):
        export_problem_bundle(replace(request_data, case_directory=request_data.profile_directory))


def test_hardlink_input_omitted(request_data, tmp_path):
    path = request_data.run_directory / "exceptions.log"
    path.unlink()
    private = tmp_path / "private.log"
    private.write_text("PRIVATE-LINK-MARKER\n")
    os.link(private, path)
    result = export_problem_bundle(request_data)
    assert "run/exceptions.log" not in _contents(result)
    assert "unsafe_source" in _reasons(result)


def test_reparse_file_is_omitted(request_data, monkeypatch):
    path = request_data.case_directory / "frames/000001.png"
    original = Path.lstat
    def fake(self, *args, **kwargs):
        info = original(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path, "lstat", fake)
    result = export_problem_bundle(request_data)
    assert "unsafe_source" in _reasons(result)
    assert "case/frames/000001.png" not in _contents(result)


def test_source_permission_failure_returns_partial(request_data, monkeypatch):
    original = pb.os.open
    def denied(path, *args, **kwargs):
        if Path(path).name == "exceptions.log":
            raise PermissionError("secret local path")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pb.os, "open", denied)
    result = export_problem_bundle(request_data)
    assert result["status"] == "PARTIAL" and "unreadable" in _reasons(result)
    assert b"secret local path" not in _contents(result)["problem_manifest.json"]


@pytest.mark.parametrize("failure", ["replace", "fsync", "write", "verify"])
def test_atomic_failure_never_publishes_or_leaves_temp(request_data, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise PermissionError("failed")
    if failure in {"replace", "fsync"}:
        monkeypatch.setattr(pb.os, failure, fail)
    elif failure == "write":
        monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    else:
        monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda self: "broken-entry")
    with pytest.raises((OSError, SupportBundleError)):
        export_problem_bundle(request_data)
    output = request_data.diagnostics_root / "exports"
    assert not list(output.iterdir())


def test_same_second_exports_have_unique_random_names(request_data, monkeypatch):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 26, 12, tzinfo=UTC)
    monkeypatch.setattr(pb, "datetime", Frozen)
    one, two = export_problem_bundle(request_data), export_problem_bundle(request_data)
    assert one["archive_path"] != two["archive_path"]
    assert Path(one["archive_path"]).exists() and Path(two["archive_path"]).exists()


def test_file_and_archive_budgets_preserve_pair_atomicity(request_data, monkeypatch):
    monkeypatch.setattr(pb, "_MAX_TEXT_BYTES", 500)
    _write(request_data.run_directory / "startup_report.json", {"large": "x" * 600})
    result = export_problem_bundle(request_data)
    assert "file_size_budget" in _reasons(result)
    monkeypatch.setattr(pb, "_MAX_TOTAL_BYTES", pb._METADATA_RESERVE + 600)
    result = export_problem_bundle(request_data)
    contents = _contents(result)
    assert "archive_budget" in _reasons(result)
    assert ("case/frames/000001.png" in contents) == ("case/frames/000001.json" in contents)


def test_restarted_case_resolves_exact_legacy_run_and_authorized_session(request_data):
    run = request_data.run_directory
    legacy = request_data.bundle_root / "logs/diagnostics/runs" / run.name
    legacy.mkdir(parents=True)
    for path in run.iterdir():
        path.replace(legacy / path.name)
    _write(request_data.diagnostics_root / "runs/wrong_new/startup_report.json", {"marker": "WRONG-CASE"})
    result = export_problem_bundle(replace(request_data, run_directory=None, session_directory=None))
    contents = _contents(result)
    assert b"right-run" in contents["run/startup_report.json"]
    assert "session/recognition_trace.jsonl" in contents
    assert b"WRONG-CASE" not in b"".join(contents.values())


def test_external_sessions_pointer_authorizes_only_matching_session(request_data, tmp_path):
    external = tmp_path / "external sessions"
    _write(external / "game/manifest.json", {"session_id": "game", "sealed": False})
    (external / "game/recognition_trace.jsonl").write_text('{"marker":"external"}\n', encoding="utf-8")
    _write(request_data.profile_directory / ".sessions_root.json", {
        "schema": "guandan.sessions-root/1", "root": str(external)})
    result = export_problem_bundle(replace(request_data, session_directory=None))
    assert b"external" in _contents(result)["session/recognition_trace.jsonl"]


def test_wrong_explicit_run_and_session_not_mixed_into_case(request_data):
    wrong_run = request_data.diagnostics_root / "runs/wrong"
    _write(wrong_run / "startup_report.json", {"private": "WRONG-CASE"})
    wrong_session = request_data.profile_directory / "sessions/other"
    _write(wrong_session / "manifest.json", {"session_id": "other", "private": "WRONG-CASE"})
    result = export_problem_bundle(replace(request_data, run_directory=wrong_run, session_directory=wrong_session))
    assert "different_case_run_not_collected" in _reasons(result)
    assert "different_case_session_not_collected" in _reasons(result)
    assert b"WRONG-CASE" not in b"".join(_contents(result).values())


def test_real_opening_evidence_monitor_artifacts_are_exported_without_media(request_data, tmp_path, monkeypatch):
    from daguandan_bridge.opening_evidence import OpeningEvidenceMonitor
    from test_opening_evidence import _result, _snapshot

    run = request_data.run_directory
    monkeypatch.setattr("daguandan_bridge.opening_evidence.current_startup_diagnostics",
                        lambda: SimpleNamespace(run_directory=run))
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=run,
        max_bytes=8 * 1024 * 1024,
        field_timeout_seconds=1,
        max_run_image_bytes=0,
        max_total_image_bytes=0,
        clock_ms=lambda: 0,
    )
    monitor.begin(monotonic_ms=0)
    snapshot = _snapshot(77)
    monitor.observe_frame(snapshot, monotonic_ms=1)
    monitor.observe_recognition(
        snapshot, _result(hand=("2S",)), {"schema": "guandan.recognition-trace/1", "score": 0.12,
                                           "failure_reason": "opening score below threshold"},
    )
    monitor.emit_incident("OPENING-LEVEL-MISSING", field="round_level",
                          reason="opening score below threshold", monotonic_ms=2,
                          evidence={"score": 0.12, "failure_reason": "not ready"})
    assert monitor.flush(5)
    monitor.close()

    result = export_problem_bundle(request_data)
    contents = _contents(result)
    assert "run/opening/latest.json" in contents
    incident_dirs = [name.split("/")[3] for name in contents
                     if name.startswith("run/opening/incidents/") and name.endswith("/incident.json")]
    assert incident_dirs and incident_dirs[0].startswith("OPEN-")
    incident_prefix = f"run/opening/incidents/{incident_dirs[0]}/"
    for filename in ("incident.json", "opening_evidence.json", "repro.json", "recognition_trace.jsonl"):
        assert incident_prefix + filename in contents
    assert b"opening score below threshold" in contents[incident_prefix + "incident.json"]
    assert b"opening score below threshold" in contents[incident_prefix + "recognition_trace.jsonl"]


def test_incident_manifest_under_case_frames_is_kept_when_images_disabled(request_data):
    manifest = request_data.case_directory / "frames" / "incident_OPEN-case.json"
    _write(manifest, {
        "schema": "guandan.listener-incident/v1", "incident_id": "OPEN-case",
        "reason": "failure", "frames": [{"role": "prior", "image_path": "000001.png",
                                             "metadata_path": "000001.json"}],
        "context": {"recovery": "linked"},
    })
    result = export_problem_bundle(replace(request_data, include_images=False))
    contents = _contents(result)
    assert "case/frames/incident_OPEN-case.json" in contents
    document = json.loads(contents["case/frames/incident_OPEN-case.json"])
    assert document["problem_bundle_images_requested"] is False
    assert document["frames"][0]["image_exported"] is False

@pytest.mark.parametrize("include_images", [True, False])
def test_real_listener_evidence_prior_failure_context_recovery_manifest(request_data, include_images):
    from daguandan_bridge.application.listener_evidence import ListenerEvidence, ListenerFrame
    from daguandan_bridge.application.session_diagnostic_frames import SessionDiagnosticFrameStore

    case = request_data.case_directory
    store = SessionDiagnosticFrameStore()
    completed = []
    def persist(frame):
        return store.save_snapshot(
            case, frame.snapshot, session_id=frame.session_id,
            capture_generation=frame.capture_generation, capture_seq=frame.capture_seq,
            source_phase=frame.source_phase,
            diagnostic_context={"case_id": case.name, **frame.details},
        ).to_dict()
    evidence = ListenerEvidence(persist, lambda value, error: completed.append((value, error)))
    frames = []
    for i in range(1, 5):
        snapshot = SimpleNamespace(image=np.full((12, 16, 3), i, dtype=np.uint8),
                                   captured_monotonic_ms=i, evidence_frame_id=f"frame_{i}")
        frames.append(ListenerFrame(case, "game", snapshot, 1, i, "live_listener",
                                    details={"listener_phase": "live"}))
    try:
        evidence.remember(frames[0])
        evidence.remember(frames[1])
        assert evidence.incident("failure", frames[1], details={"password": "PRIVATE-PASSWORD"})
        assert evidence.writer.wait_idle(5)
        evidence.recover(frames[2])
        assert evidence.writer.wait_idle(5)
        assert evidence.incident("without_failure_frame", frames[3], failure_role="context")
        assert evidence.writer.wait_idle(5)
        assert all(error is None for _, error in completed), completed
        sources = list((case / "frames").glob("incident_*.json"))
        assert len(sources) == 2
        result = export_problem_bundle(replace(request_data, include_images=include_images))
        contents = _contents(result)
        roles = set()
        for source in sources:
            data = contents[f"case/frames/{source.name}"]
            assert b"PRIVATE-PASSWORD" not in data
            manifest = json.loads(data)
            assert manifest["problem_bundle_images_requested"] is include_images
            for frame in manifest["frames"]:
                roles.add(frame["role"])
                assert frame["image_path"].startswith("case/frames/")
                assert frame["image_exported"] is include_images
                if include_images:
                    assert contents[frame["image_path"]] == (case / "frames" / Path(frame["image_path"]).name).read_bytes()
        assert {"prior", "failure", "context", "recovery"} <= roles
    finally:
        assert evidence.writer.close()

@pytest.mark.parametrize("sealed", [False, True])
def test_real_live_session_observations_export_as_sanitized_jsonl(request_data, sealed):
    from daguandan_bridge.live.session_store import LiveSessionStore

    profile = request_data.profile_directory
    store = LiveSessionStore(profile.parent, profile.name, session_id="state_machine_trace",
                             automatic_log_delivery_enabled=False)
    store.start({})
    store.append_observation({"phase": "reduce", "revision": 1, "password": "OBS-SECRET"})
    if sealed:
        store.seal(frame_count=0, dropped_frames=0)
    case_file = request_data.case_directory / "case.json"
    document = json.loads(case_file.read_text("utf-8"))
    document.update(session_id=store.session_id, session_directory=str(store.directory))
    _write(case_file, document)
    source = store.observations_gzip_path if sealed else store.observations_part_path
    before = source.read_bytes()
    result = export_problem_bundle(replace(request_data, session_directory=store.directory))
    data = _contents(result)
    assert "session/observations.jsonl" in data
    record = json.loads(data["session/observations.jsonl"])
    assert record["phase"] == "reduce" and record["revision"] == 1
    assert record["password"] == "<REDACTED>"
    assert not any(name.endswith((".part", ".gz")) for name in data)
    assert source.read_bytes() == before


def test_observation_queue_snapshot_does_not_include_new_records(request_data, monkeypatch):
    path = request_data.session_directory / "observations.jsonl.part"
    path.write_bytes(b'{"revision":1}\n{"revision":2')
    original = pb._Collector._read_snapshot
    def append(self, snapshot, **kwargs):
        if snapshot.path == path:
            with path.open("ab") as handle:
                handle.write(b'}\n{"revision":3}\n')
        return original(self, snapshot, **kwargs)
    monkeypatch.setattr(pb._Collector, "_read_snapshot", append)
    result = export_problem_bundle(request_data)
    assert json.loads(_contents(result)["session/observations.jsonl"])["revision"] == 1
    assert {"concurrent_append", "incomplete_line_tail"} <= _reasons(result)


@pytest.mark.parametrize("kind,reason", [
    ("bomb", "decompressed_size_budget"), ("truncated", "incomplete_or_concatenated_gzip"),
    ("corrupt", "invalid_gzip"), ("concatenated", "incomplete_or_concatenated_gzip"),
])
def test_gzip_observations_limits_and_corruption(request_data, monkeypatch, kind, reason):
    import gzip
    root = request_data.session_directory
    (root / "observations.jsonl.part").unlink()
    content = gzip.compress(b'{"phase":"a"}\n')
    if kind == "bomb":
        monkeypatch.setattr(pb, "_MAX_TEXT_BYTES", 4096)
        content = gzip.compress(b" " * 65536)
    elif kind == "truncated":
        content = content[:-5]
    elif kind == "corrupt":
        content = b"not a gzip"
    else:
        content += content
    (root / "observations.jsonl.gz").write_bytes(content)
    result = export_problem_bundle(request_data)
    data = _contents(result)
    assert "session/observations.jsonl" not in data
    assert reason in _reasons(result)
    assert not any(name.endswith((".gz", ".part")) for name in data)


def test_rewrite_then_append_is_not_a_snapshot(request_data, monkeypatch):
    path = request_data.run_directory / "startup.jsonl"
    original = pb._Collector._read_snapshot
    def rewrite(self, snapshot, **kwargs):
        if snapshot.path == path:
            path.write_bytes(b'{"event":"fake"}\n{"later":true}\n')
        return original(self, snapshot, **kwargs)
    monkeypatch.setattr(pb._Collector, "_read_snapshot", rewrite)
    result = export_problem_bundle(request_data)
    assert "run/startup.jsonl" not in _contents(result)
    assert "concurrent_change" in _reasons(result)


def test_runtime_context_preserves_request_time_game_state_without_secrets(request_data):
    context = {"session_id": "game", "revision": 7, "game_state": {"turn": 2, "hand": ["3S"]},
               "apiKey": "PRIVATE-RUNTIME-SECRET"}
    result = export_problem_bundle(replace(request_data, runtime_context=context))
    data = json.loads(_contents(result)["runtime_context.json"])
    assert data["game_state"] == context["game_state"] and data["revision"] == 7
    assert data["apiKey"] == "<REDACTED>"
    assert context["apiKey"] == "PRIVATE-RUNTIME-SECRET"
