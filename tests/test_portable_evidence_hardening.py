from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
from threading import Event
import time
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from daguandan_bridge.annotation_service import RegionRecord
from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.diagnostic_root_cause import diagnose_root_cause
from daguandan_bridge.gui.workers import CaptureWorker
from daguandan_bridge.image_io import standardize_to_base
from daguandan_bridge.models import Box, ClientRect
from daguandan_bridge.opening_evidence import (
    NonBlockingOpeningEvidenceSink,
    OpeningEvidenceMonitor,
)
from daguandan_bridge.opening_gate import evaluate_opening_gate
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.resource_fingerprint import recognition_resource_identity
from daguandan_bridge.support_repro import (
    SupportReproError,
    compare_repro_reports,
    reproduce_support_bundle,
    reproduce_support_suite,
    verify_support_archive,
    write_truth_annotation,
)
from daguandan_bridge.window_capture import CapturedStandardizedFrame


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _snapshot(value: int = 127, *, frame_id: str | None = None) -> FrameSnapshot:
    raw = np.full((6, 8, 3), value, dtype=np.uint8)
    snapshot = FrameSnapshot(
        CapturedStandardizedFrame(
            standardization=standardize_to_base(
                raw,
                (8, 6),
                detect_black_bars=False,
            ),
            rect=ClientRect(0, 0, 8, 6),
            backend="test",
            dpi=96,
            window_title="test",
            raw_image=raw,
        )
    )
    return replace(snapshot, evidence_frame_id=frame_id) if frame_id else snapshot


def _result(
    level: str | None,
    hand=HAND,
    *,
    buttons=(),
    lead_player=None,
    current_player=None,
    events=(),
):
    return SimpleNamespace(
        round_level=level,
        wild_rank=level,
        my_hand=tuple(hand),
        buttons=tuple(buttons),
        lead_player=lead_player,
        current_player=current_player,
        events=tuple(events),
        field_confidences={},
        sources={},
        unresolved_fields=(),
        diagnostics=(),
        elapsed_ms=0.0,
    )


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded)


def _pixel_sha(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes(order="C")).hexdigest()


def _manifested_zip(path: Path, payloads: dict[str, bytes], *, build_id="BUILD-remote") -> Path:
    records = [
        {
            "path": name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "classification": "test",
            "capability": "test",
        }
        for name, content in sorted(payloads.items())
    ]
    manifest = {
        "schema": "guandan.support-bundle/1",
        "support_id": "SUP-hardening",
        "build_id": build_id,
        "files": records,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
        archive.writestr("support_manifest.json", json.dumps(manifest).encode())
    return path


def _evidence_support(
    tmp_path: Path,
    images: list[np.ndarray],
    *,
    transform_override: dict[str, object] | None = None,
    roi_box: list[int] | None = None,
    standard_images: list[np.ndarray] | None = None,
    roi_saved_box: list[int] | None = None,
) -> Path:
    payloads: dict[str, bytes] = {}
    frames: list[dict[str, object]] = []
    index: list[dict[str, object]] = []
    for seq, raw in enumerate(images, start=1):
        standard = (
            standard_images[seq - 1].copy()
            if standard_images is not None
            else raw.copy()
        )
        frame_id = f"frame-{seq}"
        standardization: dict[str, object] = {
            "source_size": [raw.shape[1], raw.shape[0]],
            "source_viewport": [0, 0, raw.shape[1], raw.shape[0]],
            "content_box": [0, 0, raw.shape[1], raw.shape[0]],
            "standardized_size": [raw.shape[1], raw.shape[0]],
            "interpolation": "INTER_LINEAR",
            "border_value": 0,
        }
        if transform_override:
            standardization.update(transform_override)
        artifacts: list[dict[str, object]] = []
        for kind, image, archive_path, incident_path, field in (
            (
                "raw_client",
                raw,
                f"frames/raw_{seq}.png",
                f"frames/raw_client_{seq:06d}.png",
                None,
            ),
            (
                "standardized",
                standard,
                f"frames/standard_{seq}.png",
                f"frames/standardized_{seq:06d}.png",
                None,
            ),
        ):
            content = _png(image)
            payloads[archive_path] = content
            artifact = {
                "frame_id": frame_id,
                "frame_seq": seq,
                "kind": kind,
                "field": field,
                "path": incident_path,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "pixel_sha256": _pixel_sha(image),
            }
            artifacts.append(artifact)
            index.append(
                {
                    "archive_path": archive_path,
                    "frame_seq": seq,
                    "monotonic_ms": seq * 100,
                    "kind": kind,
                    "field": field,
                    "source_sha256": artifact["sha256"],
                    "pixel_sha256": artifact["pixel_sha256"],
                    "incident_path": incident_path,
                    "incident_sha256": artifact["sha256"],
                    "frame_id": frame_id,
                }
            )
        if roi_box is not None:
            saved_box = roi_saved_box or roi_box
            x, y, width, height = saved_box
            roi = standard[y : y + height, x : x + width]
            content = _png(roi)
            archive_path = f"roi/level_{seq}.png"
            incident_path = f"roi/level_rank_{seq:06d}.png"
            payloads[archive_path] = content
            artifact = {
                "frame_id": frame_id,
                "frame_seq": seq,
                "kind": "roi",
                "field": "level_rank",
                "path": incident_path,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "pixel_sha256": _pixel_sha(roi),
                "box": list(roi_box),
            }
            artifacts.append(artifact)
            index.append(
                {
                    "archive_path": archive_path,
                    "frame_seq": seq,
                    "monotonic_ms": seq * 100,
                    "kind": "roi",
                    "field": "level_rank",
                    "source_sha256": artifact["sha256"],
                    "pixel_sha256": artifact["pixel_sha256"],
                    "incident_path": incident_path,
                    "incident_sha256": artifact["sha256"],
                    "frame_id": frame_id,
                }
            )
        frames.append(
            {
                "frame_id": frame_id,
                "seq": seq,
                "monotonic_ms": seq * 100,
                "anchor_score": 0.90,
                "capture": {
                    "backend": "test",
                    "pixel_max": int(raw.max()),
                    "standardization": standardization,
                },
                "artifacts": artifacts,
            }
        )
    opening = {
        "schema": "guandan.opening-evidence/1",
        "resource_identity": {"status": "identified", "sha256": "remote"},
        "frames": frames,
    }
    payloads["evidence/opening_evidence.json"] = json.dumps(opening).encode()
    payloads["evidence/image_index.json"] = json.dumps(
        {"schema": "guandan.support-image-index/1", "entries": index}
    ).encode()
    payloads["repro/repro.json"] = json.dumps(
        {"schema": "guandan.repro-manifest/1", "symptom_code": "OPENING-TEST"}
    ).encode()
    return _manifested_zip(tmp_path / f"support-{len(list(tmp_path.glob('*.zip')))}.zip", payloads)


class _FixedRecognizer:
    def __init__(self, level: str = "7") -> None:
        self.level = level

    def recognize(self, image, *, allow_unknown_suit=False):
        assert allow_unknown_suit is True
        return _result(self.level)

    def get_last_diagnostic_trace(self):
        return {
            "candidates": [
                {
                    "field": "level_rank",
                    "label": self.level,
                    "score": 0.9,
                    "threshold": 0.6,
                    "accepted": True,
                    "rejection_reason": None,
                }
            ]
        }


class _PixelRecognizer(_FixedRecognizer):
    def recognize(self, image, *, allow_unknown_suit=False):
        self.level = "7" if int(image[0, 0, 0]) >= 50 else "2"
        return super().recognize(image, allow_unknown_suit=allow_unknown_suit)


def test_e001_canonical_resource_fingerprint_is_root_and_order_independent(tmp_path):
    roots = (tmp_path / "source", tmp_path / "frozen")
    for root, order in zip(roots, (("b.png", "a.png"), ("a.png", "b.png"))):
        profile = root / "p"
        (profile / "templates").mkdir(parents=True)
        for name in ("profile.json", "regions_config.json", "templates_config.json"):
            (profile / name).write_text(name, encoding="utf-8")
        for name in order:
            (profile / "templates" / name).write_bytes(name.encode())
    first = recognition_resource_identity(roots[0], "p")
    second = recognition_resource_identity(roots[1], "p")
    assert first["sha256"] == second["sha256"]
    (roots[1] / "p" / "templates" / "a.png").write_bytes(b"changed")
    assert recognition_resource_identity(roots[1], "p")["sha256"] != first["sha256"]


def test_e002_typed_worker_error_and_correlated_incident_episode(tmp_path):
    class TypedError(RuntimeError):
        code = "RESOURCE-MISMATCH"

    observed = []
    worker = CaptureWorker(lambda: (_ for _ in ()).throw(TypedError("broken")), 0.01)
    worker.error.connect(observed.append)
    worker.run()
    assert len(observed) == 1 and isinstance(observed[0], TypedError)

    monitor = OpeningEvidenceMonitor(diagnostics_root=tmp_path)
    monitor.begin()
    snapshot = _snapshot()
    monitor.observe_frame(snapshot)
    assert monitor.observe_failure(observed[0], stage="recognition", snapshot=snapshot)
    monitor.observe_failure(observed[0], stage="recognition", snapshot=snapshot)
    assert monitor.flush(5)
    incidents = list((tmp_path / "opening" / "incidents").glob("*/incident.json"))
    assert len(incidents) == 1
    incident = json.loads(incidents[0].read_text(encoding="utf-8"))
    assert incident["code"] == "OPENING-RESOURCE-ERROR"
    assert incident["evidence"]["error_type"] == "TypedError"
    assert incident["evidence"]["source_error_code"] == "RESOURCE-MISMATCH"
    assert incident["evidence"]["frame_id"] == snapshot.evidence_frame_id
    assert incident["episode"]["occurrence_count"] == 2
    monitor.close()


def test_e003_pure_opening_gate_covers_anchor_settlement_lead_and_illegal_hand():
    assert evaluate_opening_gate(_result("7"), anchor_score=0.85).ready is True
    assert evaluate_opening_gate(_result("7"), anchor_score=0.849).reason == "table_anchor_unresolved"
    assert evaluate_opening_gate(
        _result("7", buttons=("continue_game",)), anchor_score=1.0
    ).reason == "settlement_screen"
    assert evaluate_opening_gate(
        _result("7", lead_player="right", current_player="self"),
        anchor_score=1.0,
    ).reason == "opening_seed_invalid"
    invalid = HAND[:-1] + ("ZZ",)
    assert evaluate_opening_gate(_result("7", invalid), anchor_score=1.0).reason == "hand_invalid"


def test_e003_single_frame_truth_is_independent_of_multiframe_gate(tmp_path):
    support = _evidence_support(tmp_path, [np.full((6, 8, 3), 70, np.uint8)])
    report = reproduce_support_bundle(
        support,
        expected_level="7",
        repeats=2,
        recognizer_factory=_FixedRecognizer,
    )
    assert report["truth"]["correct_runs"] == 2
    assert report["outcomes"][0]["single_frame_gate"] == "ready"
    assert report["outcomes"][0]["multi_frame_gate"]["status"] == "PENDING"


def test_e004_truth_is_legal_sequence_bound_and_hashed(tmp_path):
    support = _evidence_support(tmp_path, [np.full((6, 8, 3), 70, np.uint8)])
    with pytest.raises(SupportReproError, match="non-empty"):
        write_truth_annotation(tmp_path / "empty.json", support, expected_hand=[])
    with pytest.raises(SupportReproError, match="legal GuanDan rank"):
        write_truth_annotation(tmp_path / "bad.json", support, expected_level="ZZ")
    truth_path = tmp_path / "truth.json"
    truth = write_truth_annotation(truth_path, support, expected_level="7")
    truth["input_sequence_sha256"] = "0" * 64
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(SupportReproError, match="different input sequence"):
        reproduce_support_bundle(support, truth_path=truth_path, recognizer_factory=_FixedRecognizer)

    truth = write_truth_annotation(truth_path, support, expected_level="7")
    report = reproduce_support_bundle(
        support,
        truth_path=truth_path,
        recognizer_factory=_FixedRecognizer,
        repeats=1,
    )
    assert report["truth_identity"]["sha256"] == hashlib.sha256(truth_path.read_bytes()).hexdigest()
    assert report["truth_identity"]["input_sequence_sha256"] == truth["input_sequence_sha256"]


def test_e004_fix_gate_requires_roles_same_truth_and_distinct_builds(tmp_path):
    support = _evidence_support(tmp_path, [np.full((6, 8, 3), 70, np.uint8)] * 2)
    reference = reproduce_support_bundle(
        support,
        expected_level="7",
        repeats=20,
        role="reference",
        recognizer_factory=lambda: _FixedRecognizer("2"),
    )
    candidate = reproduce_support_bundle(
        support,
        expected_level="7",
        repeats=20,
        role="candidate",
        recognizer_factory=_FixedRecognizer,
    )
    reference["runner"]["build_id"] = "BUILD-old"
    candidate["runner"]["build_id"] = "BUILD-new"
    assert compare_repro_reports(reference, candidate)["status"] == "PASS"
    candidate["verification_role"] = "reference"
    candidate["runner"]["build_id"] = "BUILD-old"
    candidate["truth_identity"]["sha256"] = "f" * 64
    failures = compare_repro_reports(reference, candidate)["failures"]
    assert {"candidate_role_invalid", "truth_sha256_mismatch", "reference_candidate_build_not_distinct"}.issubset(failures)


def test_e005_raw_standard_and_roi_pixels_reproduce_exactly_and_tamper_fails(tmp_path):
    raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))
    support = _evidence_support(tmp_path, [raw, raw], roi_box=[1, 1, 3, 2])
    report = reproduce_support_bundle(
        support,
        repeats=1,
        recognizer_factory=_FixedRecognizer,
    )
    layer = {item["layer"]: item for item in report["root_cause"]["layers"]}
    assert layer["standardization_roi"]["status"] == "PASS"

    tampered = _evidence_support(
        tmp_path,
        [raw, raw],
        transform_override={"content_box": [0, 0, 7, 6]},
        roi_box=[1, 1, 3, 2],
    )
    report = reproduce_support_bundle(tampered, repeats=1, recognizer_factory=_FixedRecognizer)
    layer = {item["layer"]: item for item in report["root_cause"]["layers"]}
    assert layer["standardization_roi"]["status"] == "FAIL"

    changed_standard = raw.copy()
    changed_standard[0, 0] = 255
    tampered_standard = _evidence_support(
        tmp_path,
        [raw, raw],
        standard_images=[changed_standard, changed_standard],
        roi_box=[1, 1, 3, 2],
    )
    report = reproduce_support_bundle(
        tampered_standard,
        repeats=1,
        recognizer_factory=_FixedRecognizer,
    )
    layer = {item["layer"]: item for item in report["root_cause"]["layers"]}
    assert layer["standardization_roi"]["status"] == "FAIL"

    tampered_roi = _evidence_support(
        tmp_path,
        [raw, raw],
        roi_box=[1, 1, 3, 2],
        roi_saved_box=[2, 1, 3, 2],
    )
    report = reproduce_support_bundle(
        tampered_roi,
        repeats=1,
        recognizer_factory=_FixedRecognizer,
    )
    layer = {item["layer"]: item for item in report["root_cause"]["layers"]}
    assert layer["standardization_roi"]["status"] == "FAIL"


def test_e006_production_consensus_marks_oscillating_sequence_fail(tmp_path):
    images = [
        np.full((6, 8, 3), value, np.uint8)
        for value in (70, 20, 70)
    ]
    report = reproduce_support_bundle(
        _evidence_support(tmp_path, images),
        repeats=2,
        recognizer_factory=_PixelRecognizer,
    )
    assert report["outcomes"][0]["multi_frame_gate"] == {
        "status": "FAIL",
        "reason": "opening_seed_oscillation",
    }
    assert report["root_cause"]["primary_layer"] == "multi_frame_stability"


def test_e006_suite_runs_ordinary_deterministic_and_child_and_wires_probe(monkeypatch, tmp_path):
    calls = []

    def fake_reproduce(_support, **kwargs):
        calls.append(kwargs)
        child = kwargs.get("child_probe")
        return {
            "schema": "guandan.repro-report/1",
            "support": {"sha256": "s"},
            "inputs": [{"frame_seq": 1, "pixel_sha256": "p"}],
            "outcomes": [
                {
                    "output_fingerprint": "o",
                    "opening_gate": "ready",
                    "single_frame_level": "7",
                    "single_frame_hand": list(HAND),
                }
            ],
            "child_wired": child,
        }

    monkeypatch.setattr(
        "daguandan_bridge.support_repro.reproduce_support_bundle",
        fake_reproduce,
    )
    monkeypatch.setattr(
        "daguandan_bridge.support_repro.run_child_probe",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "report": fake_reproduce("child", deterministic=True),
        },
    )
    report = reproduce_support_suite("support.zip", output_path=tmp_path / "report.json")
    assert any(item.get("deterministic") is False for item in calls)
    assert any(item.get("deterministic") is True for item in calls)
    assert calls[-1]["child_probe"]["matches_same_process"] is True
    assert report["child_wired"]["status"] == "PASS"


def test_e007_frame_id_and_pixel_correlation_never_falls_back_to_latest(tmp_path):
    monitor = OpeningEvidenceMonitor(diagnostics_root=tmp_path)
    monitor.begin(monotonic_ms=0)
    first = _snapshot(10, frame_id="reused")
    second = _snapshot(20, frame_id="reused")
    monitor.observe_frame(first, monotonic_ms=1)
    monitor.observe_frame(second, monotonic_ms=2)
    trace = {"input_sha256": _pixel_sha(second.image)}
    monitor.observe_recognition(second, _result("7"), trace)
    orphan = _snapshot(20, frame_id="reused")
    monitor.observe_recognition(orphan, _result("2"), trace)
    assert monitor.metrics().orphan_recognitions == 1
    assert len({item.frame_id for item in monitor._ring}) == 2
    assert monitor._ring[-1].recognition["round_level"] == "7"
    monitor.close()

    pressured = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path / "pressure",
        max_bytes=6 * 8 * 3 * 2 + 1,
    )
    pressured.begin()
    old, latest = _snapshot(30), _snapshot(40)
    pressured.observe_frame(old)
    pressured.observe_frame(latest)
    pressured.observe_recognition(
        old,
        _result("2"),
        {"input_sha256": _pixel_sha(old.image)},
    )
    assert pressured.metrics().orphan_recognitions == 1
    assert pressured._ring[-1].recognition is None
    pressured.close()


def test_e008_trace_records_every_peak_geometry_and_rejected_is_not_pass():
    service = ScreenshotRecognitionService(diagnostic_tracing=True)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[4:8, 5:9] = 255
    template = np.full((4, 4, 3), 255, dtype=np.uint8)
    region = RegionRecord("level_rank", "generic", Box(0, 0, 20, 20), (0, 0, 1, 1))
    service._diagnostic_local.collector = []
    matches = service._matches_for_region(
        image,
        region,
        (({"label": "7", "kind": "rank", "file": "7.png"}, template),),
        predicate=lambda _raw: True,
        threshold=0.5,
        limit=2,
    )
    trace = service._diagnostic_local.collector
    assert matches
    assert len(trace) == 2
    assert all({"peak_index", "search_box", "roi_box", "peak_location", "match_box", "threshold", "accepted", "rejection_reason"}.issubset(item) for item in trace)

    root = diagnose_root_cause(
        support_verified=True,
        opening_evidence={"frames": []},
        outcomes=[
            {
                "output_fingerprint": "one",
                "candidate_vector": [
                    {
                        "field": "level_rank",
                        "label": "7",
                        "score": 0.9,
                        "threshold": 0.6,
                        "accepted": False,
                        "rejection_reason": "center_outside_roi",
                    }
                ],
            }
        ],
        truth={"expected_level": "7"},
        local_resource_identity=None,
        support_build_id=None,
        runner_build_id=None,
    )
    matcher = {item["layer"]: item for item in root["layers"]}["matcher_threshold_margin"]
    assert matcher["status"] == "FAIL"


def test_e009_conflict_episode_recovers_and_pending_bytes_stay_hard_bounded(tmp_path, monkeypatch):
    now = [0]
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        max_bytes=6 * 8 * 3 * 4,
        clock_ms=lambda: now[0],
    )
    monitor.begin(monotonic_ms=0)
    snapshot = _snapshot()
    monitor.observe_frame(snapshot, monotonic_ms=1)
    for level in ("7", "2", "7", "7", "7", "2"):
        now[0] += 1
        monitor.observe_recognition(snapshot, _result(level), {"input_sha256": _pixel_sha(snapshot.image)})
    assert monitor.flush(5)
    conflicts = []
    for path in (tmp_path / "opening" / "incidents").glob("*/incident.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["code"] == "OPENING-LEVEL-CONFLICT":
            conflicts.append(document)
    assert len(conflicts) == 2
    assert conflicts[0]["episode"]["occurrence_count"] >= 2
    monitor.close()

    blocked = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path / "blocked",
        max_bytes=6 * 8 * 3 * 4,
    )
    blocked.begin(monotonic_ms=0)
    first, second = _snapshot(1), _snapshot(2)
    blocked.observe_frame(first, monotonic_ms=1)
    blocked.observe_frame(second, monotonic_ms=2)
    release = Event()
    original = blocked._write_incident

    def slow(*args):
        release.wait(2)
        return original(*args)

    monkeypatch.setattr(blocked, "_write_incident", slow)
    assert blocked.emit_incident("OPENING-PENDING", field="test", reason="pending")
    blocked.observe_frame(_snapshot(3), monotonic_ms=3)
    metrics = blocked.metrics()
    assert metrics.retained_bytes <= blocked.max_bytes
    assert metrics.pending_snapshot_bytes <= blocked.max_bytes
    release.set()
    assert blocked.flush(5)
    assert blocked.metrics().pending_snapshot_bytes == 0
    blocked.close()


def test_e010_close_has_one_deadline_and_rejects_new_tasks(tmp_path, monkeypatch):
    monitor = OpeningEvidenceMonitor(diagnostics_root=tmp_path)
    monitor.begin(monotonic_ms=0)
    monitor.observe_frame(_snapshot(), monotonic_ms=1)
    release = Event()
    monkeypatch.setattr(monitor, "_write_incident", lambda *_args: release.wait(5))
    assert monitor.emit_incident("OPENING-SLOW", field="test", reason="slow")
    started = time.monotonic()
    monitor.close(timeout=0.05)
    assert time.monotonic() - started < 0.25
    assert monitor.emit_incident("OPENING-AFTER-CLOSE", field="test", reason="closed") is False
    release.set()

    entered = Event()
    unblock = Event()

    class SlowSink:
        def observe_frame(self, *_args, **_kwargs):
            entered.set()
            unblock.wait(5)

    proxy = NonBlockingOpeningEvidenceSink(SlowSink())
    proxy.observe_frame(object())
    assert entered.wait(1)
    started = time.monotonic()
    proxy.close(timeout=0.05)
    assert time.monotonic() - started < 0.25
    before = proxy.dropped_calls
    proxy.observe_frame(object())
    assert proxy.dropped_calls == before + 1
    unblock.set()


def _fake_png_header(width: int, height: int, *, bit_depth=8, color_type=2) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + bytes((bit_depth, color_type, 0, 0, 0))
    )


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ([_fake_png_header(10_000, 10_000)], "pixel limit"),
        ([_fake_png_header(5_000, 5_000)] * 6, "cumulative image pixels"),
        ([_fake_png_header(8_000, 4_000, bit_depth=16, color_type=6)] * 3, "decoded image bytes"),
        ([_fake_png_header(1, 1)] * 257, "image count"),
    ],
)
def test_e011_image_budgets_reject_headers_before_decode(tmp_path, monkeypatch, headers, message):
    payloads = {f"frames/image_{index}.png": value for index, value in enumerate(headers)}
    support = _manifested_zip(tmp_path / "bomb.zip", payloads)
    monkeypatch.setattr(cv2, "imdecode", lambda *_args, **_kwargs: pytest.fail("decode called"))
    with pytest.raises(SupportReproError, match=message):
        verify_support_archive(support)
