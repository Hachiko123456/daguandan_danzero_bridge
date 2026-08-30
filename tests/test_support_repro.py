from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from daguandan_bridge.support_repro import (
    SupportReproError,
    compare_repro_reports,
    reproduce_support_bundle,
    verify_support_archive,
    write_truth_annotation,
)


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + (
    "8S",
    "8H",
    "8C",
)


def _image_bytes(value: int = 100) -> tuple[bytes, str]:
    image = np.full((72, 128, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded), hashlib.sha256(memoryview(image)).hexdigest()


def _support_zip(tmp_path: Path, *, undeclared: bool = False) -> Path:
    image, pixel_hash = _image_bytes()
    payloads: dict[str, bytes] = {
        "evidence/opening_evidence.json": json.dumps(
            {
                "schema": "guandan.opening-evidence/1",
                "resource_identity": {"status": "identified", "sha256": "remote"},
                "frames": [],
            }
        ).encode(),
        "repro/repro.json": json.dumps(
            {"schema": "guandan.repro-manifest/1", "symptom_code": "OPENING-LEVEL-MISSING"}
        ).encode(),
        "frames/frame_1.png": image,
        "frames/frame_2.png": image,
    }
    image_index = {
        "schema": "guandan.support-image-index/1",
        "entries": [
            {
                "archive_path": f"frames/frame_{index}.png",
                "frame_seq": index,
                "monotonic_ms": index * 100,
                "kind": "standardized",
                "field": None,
                "source_sha256": hashlib.sha256(image).hexdigest(),
                "pixel_sha256": pixel_hash,
            }
            for index in (1, 2)
        ],
    }
    payloads["evidence/image_index.json"] = json.dumps(image_index).encode()
    records = [
        {
            "path": name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "classification": "sensitive-image" if name.endswith(".png") else "sanitized-json",
            "capability": "frames" if name.endswith(".png") else "opening_evidence",
        }
        for name, content in sorted(payloads.items())
    ]
    manifest = {
        "schema": "guandan.support-bundle/1",
        "support_id": "SUP-test",
        "build_id": "gb-remote",
        "files": records,
    }
    destination = tmp_path / "support.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
        if undeclared:
            archive.writestr("extra.txt", b"not declared")
        archive.writestr("support_manifest.json", json.dumps(manifest).encode())
    return destination


class _Recognizer:
    def __init__(self, level: str) -> None:
        self.level = level
        self.calls = 0
        self.annotation_service = SimpleNamespace(profiles_root=Path("missing"), profile_name="profile")

    def recognize(self, image, *, allow_unknown_suit=False):
        assert image.shape == (72, 128, 3)
        assert allow_unknown_suit is True
        self.calls += 1
        return SimpleNamespace(round_level=self.level, my_hand=HAND)

    def get_last_diagnostic_trace(self):
        return {
            "input_sha256": None,
            "candidates": [
                {
                    "field": "level_rank",
                    "label": "7",
                    "score": 0.71 if self.level == "7" else 0.51,
                    "threshold": 0.60,
                },
                {
                    "field": "level_rank",
                    "label": "2",
                    "score": 0.40 if self.level == "7" else 0.72,
                    "threshold": 0.60,
                },
            ],
        }


def test_reproducer_runs_same_sequence_twenty_times_with_truth(tmp_path):
    support = _support_zip(tmp_path)
    before = (support.stat().st_size, support.stat().st_mtime_ns, hashlib.sha256(support.read_bytes()).hexdigest())

    report = reproduce_support_bundle(
        support,
        expected_level="7",
        repeats=20,
        recognizer_factory=lambda: _Recognizer("7"),
    )

    assert report["schema"] == "guandan.repro-report/1"
    assert report["repeatability"]["repeatable"] is True
    assert report["truth"]["correct_runs"] == 20
    assert report["comparison"]["candidate_correct_20_of_20"] is True
    assert len(report["outcomes"]) == 20
    assert all(item["opening_gate"] == "ready" for item in report["outcomes"])
    after = (support.stat().st_size, support.stat().st_mtime_ns, hashlib.sha256(support.read_bytes()).hexdigest())
    assert after == before


def test_no_truth_can_only_claim_repeatable_symptom(tmp_path):
    report = reproduce_support_bundle(
        _support_zip(tmp_path),
        repeats=3,
        recognizer_factory=lambda: _Recognizer("2"),
    )

    assert report["repeatability"]["repeatable"] is True
    assert report["truth"]["eligible_for_fix_verification"] is False
    assert report["truth"]["correct_runs"] is None


def test_truth_annotation_is_external_and_bound_to_zip_hash(tmp_path):
    support = _support_zip(tmp_path)
    truth_path = tmp_path / "truth.json"
    before = hashlib.sha256(support.read_bytes()).hexdigest()

    truth = write_truth_annotation(truth_path, support, expected_level="7")
    report = reproduce_support_bundle(
        support,
        truth_path=truth_path,
        repeats=2,
        recognizer_factory=lambda: _Recognizer("7"),
    )

    assert truth["support_sha256"] == before
    assert hashlib.sha256(support.read_bytes()).hexdigest() == before
    assert report["truth"]["correct_runs"] == 2


def test_old_new_gate_requires_same_support_failure_then_correct_20_of_20(tmp_path):
    support = _support_zip(tmp_path)
    reference = reproduce_support_bundle(
        support,
        expected_level="7",
        repeats=20,
        recognizer_factory=lambda: _Recognizer("2"),
    )
    candidate = reproduce_support_bundle(
        support,
        expected_level="7",
        repeats=20,
        recognizer_factory=lambda: _Recognizer("7"),
    )

    gate = compare_repro_reports(reference, candidate)

    assert gate["schema"] == "guandan.repro-gate/1"
    assert gate["status"] == "PASS"
    assert gate["failures"] == []


def test_support_verifier_rejects_undeclared_payload(tmp_path):
    with pytest.raises(SupportReproError, match="declaration mismatch"):
        verify_support_archive(_support_zip(tmp_path, undeclared=True))


def test_reproducer_refuses_correctness_when_truth_targets_other_zip(tmp_path):
    support = _support_zip(tmp_path)
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "schema": "guandan.repro-truth/1",
                "support_sha256": "0" * 64,
                "expected_level": "7",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SupportReproError, match="different support ZIP"):
        reproduce_support_bundle(
            support,
            truth_path=truth,
            recognizer_factory=lambda: _Recognizer("7"),
        )
