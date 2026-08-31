from __future__ import annotations

"""Deterministic offline reproduction of opening failures from support ZIPs."""

from contextlib import contextmanager
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import stat
import struct
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4
import zipfile

import cv2
import numpy as np

from .diagnostic_root_cause import diagnose_root_cause
from .danzero.state import GuanDanState, RANKS
from .opening_gate import evaluate_opening_gate, serialized_result
from .resource_fingerprint import recognition_resource_identity
from .runtime_identity import get_runtime_identity
from .storage import atomic_write_json


REPRO_REPORT_SCHEMA = "guandan.repro-report/1"
REPRO_TRUTH_SCHEMA = "guandan.repro-truth/1"
REPRO_GATE_SCHEMA = "guandan.repro-gate/1"
SUPPORT_SCHEMA = "guandan.support-bundle/1"

_MAX_ENTRIES = 4096
_MAX_ENTRY_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 1000.0
_MAX_IMAGE_PIXELS = 32_000_000
_MAX_IMAGES = 256
_MAX_TOTAL_IMAGE_PIXELS = 128_000_000
_MAX_TOTAL_DECODED_BYTES = 512 * 1024 * 1024


class SupportReproError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedSupport:
    path: Path
    sha256: str
    manifest: dict[str, object]
    payloads: dict[str, bytes]


@dataclass(frozen=True)
class _ImageHeader:
    width: int
    height: int
    decoded_bytes: int


@dataclass
class _ImageBudget:
    count: int = 0
    pixels: int = 0
    decoded_bytes: int = 0

    def add(self, header: _ImageHeader, name: str) -> None:
        self.count += 1
        self.pixels += int(header.width) * int(header.height)
        self.decoded_bytes += int(header.decoded_bytes)
        if self.count > _MAX_IMAGES:
            raise SupportReproError("support image count exceeds hard limit")
        if self.pixels > _MAX_TOTAL_IMAGE_PIXELS:
            raise SupportReproError("support cumulative image pixels exceed hard limit")
        if self.decoded_bytes > _MAX_TOTAL_DECODED_BYTES:
            raise SupportReproError("support cumulative decoded image bytes exceed hard limit")


def verify_support_archive(path: Path | str) -> VerifiedSupport:
    source = Path(path)
    if not source.is_file():
        raise SupportReproError("support ZIP does not exist")
    archive_hash = _sha256_file(source)
    payloads: dict[str, bytes] = {}
    identities: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _MAX_ENTRIES:
                raise SupportReproError("support ZIP entry count is invalid")
            for info in infos:
                name = _safe_archive_name(info.filename)
                identity = name.casefold()
                if identity in identities:
                    raise SupportReproError(f"duplicate case-insensitive ZIP entry: {name}")
                identities.add(identity)
                if info.flag_bits & 0x1:
                    raise SupportReproError(f"encrypted ZIP entry is not allowed: {name}")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise SupportReproError(f"ZIP links are not allowed: {name}")
                if info.is_dir():
                    raise SupportReproError(f"directory entries are not allowed: {name}")
                if info.file_size < 0 or info.file_size > _MAX_ENTRY_BYTES:
                    raise SupportReproError(f"support entry exceeds size limit: {name}")
                if info.compress_size == 0 and info.file_size > 0:
                    raise SupportReproError(f"invalid compressed size: {name}")
                ratio = info.file_size / max(1, info.compress_size)
                if ratio > _MAX_COMPRESSION_RATIO:
                    raise SupportReproError(f"support entry compression ratio is unsafe: {name}")
                total += info.file_size
                if total > _MAX_TOTAL_BYTES:
                    raise SupportReproError("support ZIP exceeds total size limit")
                content = archive.read(info)
                if len(content) != info.file_size:
                    raise SupportReproError(f"support entry changed while reading: {name}")
                payloads[name] = content
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, SupportReproError):
            raise
        raise SupportReproError("support ZIP is unreadable") from exc

    manifest_bytes = payloads.get("support_manifest.json")
    if manifest_bytes is None:
        raise SupportReproError("support manifest is missing")
    manifest = _json_object(manifest_bytes, "support manifest")
    if manifest.get("schema") != SUPPORT_SCHEMA:
        raise SupportReproError("unsupported support manifest schema")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise SupportReproError("support manifest file list is invalid")
    declared: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise SupportReproError("support manifest file record is invalid")
        name = _safe_archive_name(str(raw.get("path", "")))
        if name.casefold() in {item.casefold() for item in declared}:
            raise SupportReproError(f"duplicate declared support entry: {name}")
        declared.add(name)
        content = payloads.get(name)
        if content is None:
            raise SupportReproError(f"declared support entry is missing: {name}")
        if int(raw.get("size", -1)) != len(content):
            raise SupportReproError(f"declared support entry size mismatch: {name}")
        if str(raw.get("sha256", "")) != hashlib.sha256(content).hexdigest():
            raise SupportReproError(f"declared support entry hash mismatch: {name}")
    actual = set(payloads) - {"support_manifest.json"}
    if actual != declared:
        extra = sorted(actual - declared)
        missing = sorted(declared - actual)
        raise SupportReproError(
            f"support payload declaration mismatch: extra={extra}, missing={missing}"
        )
    _preflight_support_images(payloads)
    _validate_incident_artifact_bindings(payloads)
    if _sha256_file(source) != archive_hash:
        raise SupportReproError("support ZIP changed during verification")
    return VerifiedSupport(source, archive_hash, manifest, payloads)


def write_truth_annotation(
    destination: Path | str,
    support_zip: Path | str,
    *,
    input_pixel_sha256: str | None = None,
    expected_level: str | None = None,
    expected_hand: Sequence[str] | None = None,
) -> dict[str, object]:
    normalized_level, normalized_hand = _validated_expected_truth(
        expected_level,
        expected_hand,
    )
    verified = verify_support_archive(support_zip)
    opening = _optional_json(
        verified.payloads,
        ("evidence/opening_evidence.json", "opening/opening_evidence.json"),
    )
    indexed = _image_index(verified.payloads, opening)
    frames = _standardized_frames(verified, indexed)
    if not frames:
        raise SupportReproError("truth annotation needs at least one standardized input")
    if input_pixel_sha256 is None:
        hashes = sorted({str(item["pixel_sha256"]) for item in frames})
        if len(hashes) != 1:
            raise SupportReproError(
                "truth annotation needs input_pixel_sha256 when support contains multiple distinct inputs"
            )
        input_pixel_sha256 = hashes[0]
    elif input_pixel_sha256 not in {
        str(item["pixel_sha256"]) for item in frames
    }:
        raise SupportReproError("truth input_pixel_sha256 is not in the support sequence")
    sequence_sha256 = _input_sequence_sha256(frames)
    document: dict[str, object] = {
        "schema": REPRO_TRUTH_SCHEMA,
        "support_sha256": verified.sha256,
        "input_pixel_sha256": input_pixel_sha256,
        "input_sequence_sha256": sequence_sha256,
        "expected_level": normalized_level,
        "expected_hand": list(normalized_hand) if normalized_hand is not None else None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(Path(destination), document)
    return document


def reproduce_support_bundle(
    support_zip: Path | str,
    *,
    output_path: Path | str | None = None,
    truth_path: Path | str | None = None,
    expected_level: str | None = None,
    expected_hand: Sequence[str] | None = None,
    repeats: int = 20,
    recognizer_factory: Callable[[], object] | None = None,
    deterministic: bool = True,
    child_probe: Mapping[str, object] | None = None,
    role: str = "unspecified",
) -> dict[str, object]:
    if repeats <= 0 or repeats > 1000:
        raise SupportReproError("repeats must be between 1 and 1000")
    before_stat = Path(support_zip).stat()
    verified = verify_support_archive(support_zip)
    opening = _optional_json(
        verified.payloads,
        ("evidence/opening_evidence.json", "opening/opening_evidence.json"),
    )
    repro_manifest = _optional_json(
        verified.payloads,
        ("repro/repro.json", "repro.json", "evidence/repro.json"),
    )
    image_index = _image_index(verified.payloads, opening)
    frames = _standardized_frames(verified, image_index)
    if not frames:
        raise SupportReproError(
            "support bundle has no standardized frames; export again with explicit image opt-in"
        )
    truth = _load_truth(
        truth_path,
        support_sha256=verified.sha256,
        frames=frames,
        expected_level=expected_level,
        expected_hand=expected_hand,
    )
    if truth is not None and truth.get("input_pixel_sha256"):
        allowed_hashes = {str(item["pixel_sha256"]) for item in frames}
        if str(truth["input_pixel_sha256"]) not in allowed_hashes:
            raise SupportReproError("truth annotation is bound to a different input frame")
    factory = recognizer_factory or _default_recognizer_factory
    outcomes: list[dict[str, object]] = []
    local_resources: dict[str, object] | None = None
    with _deterministic_runtime(enabled=deterministic):
        recognizer = factory()
        local_resources = _local_resource_identity(recognizer)
        for repeat_index in range(1, int(repeats) + 1):
            outcome = _run_sequence(recognizer, frames, repeat_index)
            outcomes.append(outcome)
    fingerprints = [str(item["output_fingerprint"]) for item in outcomes]
    repeatable = len(set(fingerprints)) == 1
    truth_evaluation = _evaluate_truth(outcomes, truth)
    runtime = get_runtime_identity()
    runner_build_id = str(runtime.get("build_id") or "") or None
    support_build_id = str(verified.manifest.get("build_id") or "") or None
    standard_equal, roi_equal = _standardization_equality(
        verified,
        image_index,
        opening,
    )
    root_cause = diagnose_root_cause(
        support_verified=True,
        opening_evidence=opening,
        outcomes=outcomes,
        truth=truth,
        local_resource_identity=local_resources,
        support_build_id=support_build_id,
        runner_build_id=runner_build_id,
        standardization_equal=standard_equal,
        roi_equal=roi_equal,
        child_probe=child_probe,
    )
    report: dict[str, object] = {
        "schema": REPRO_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "frozen" if getattr(sys, "frozen", False) else "source",
        "verification_role": str(role),
        "deterministic": bool(deterministic),
        "support": {
            "sha256": verified.sha256,
            "schema": verified.manifest.get("schema"),
            "build_id": support_build_id,
            "support_id": verified.manifest.get("support_id"),
        },
        "runner": {
            "build_id": runner_build_id,
            "runtime_identity": runtime,
            "resource_identity": local_resources,
        },
        "repro_manifest": repro_manifest,
        "inputs": [
            {
                "frame_seq": frame["frame_seq"],
                "archive_path": frame["archive_path"],
                "file_sha256": frame["file_sha256"],
                "pixel_sha256": frame["pixel_sha256"],
                "shape": list(frame["image"].shape),
            }
            for frame in frames
        ],
        "repeat_count": int(repeats),
        "repeatability": {
            "repeatable": repeatable,
            "matching_runs": max(Counter(fingerprints).values()),
            "total_runs": len(outcomes),
            "unique_outcome_count": len(set(fingerprints)),
        },
        "truth": truth_evaluation,
        "truth_identity": (
            {
                "sha256": truth.get("_truth_sha256"),
                "support_sha256": truth.get("support_sha256"),
                "input_sequence_sha256": truth.get("input_sequence_sha256"),
            }
            if truth is not None
            else None
        ),
        "outcomes": outcomes,
        "root_cause": root_cause,
        "comparison": {
            "support_and_runner_build_match": bool(
                support_build_id and runner_build_id and support_build_id == runner_build_id
            ),
            "reference_failure_20_of_20": bool(
                repeats == 20
                and repeatable
                and truth_evaluation.get("eligible_for_fix_verification")
                and truth_evaluation.get("correct_runs") == 0
            ),
            "candidate_correct_20_of_20": bool(
                repeats == 20
                and truth_evaluation.get("eligible_for_fix_verification")
                and truth_evaluation.get("correct_runs") == 20
            ),
        },
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), report)
    after_stat = Path(support_zip).stat()
    if (
        before_stat.st_size != after_stat.st_size
        or before_stat.st_mtime_ns != after_stat.st_mtime_ns
        or _sha256_file(Path(support_zip)) != verified.sha256
    ):
        raise SupportReproError("support ZIP was modified during reproduction")
    return report


def run_child_probe(
    support_zip: Path | str,
    output_path: Path | str,
    *,
    truth_path: Path | str | None = None,
    repeats: int = 20,
    deterministic: bool = True,
    timeout_seconds: float = 180.0,
) -> dict[str, object]:
    output = Path(output_path)
    output.unlink(missing_ok=True)
    command = _entrypoint_command()
    command.extend(
        [
            "--_repro-probe",
            str(Path(support_zip)),
            "--repro-output",
            str(output),
            "--repro-repeats",
            str(int(repeats)),
        ]
    )
    if truth_path is not None:
        command.extend(["--repro-truth", str(Path(truth_path))])
    if deterministic:
        command.append("--repro-deterministic")
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
            text=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
        return {
            "schema": "guandan.repro-probe/1",
            "status": "FAIL",
            "reason": "timeout",
            "exit_code": None,
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        }
    except OSError as exc:
        return {
            "schema": "guandan.repro-probe/1",
            "status": "FAIL",
            "reason": "spawn_failed",
            "error_type": type(exc).__name__,
            "exit_code": None,
        }
    if completed.returncode != 0 or not output.is_file():
        return {
            "schema": "guandan.repro-probe/1",
            "status": "FAIL",
            "exit_code": int(completed.returncode),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        }
    report = _json_object(output.read_bytes(), "child repro report")
    return {
        "schema": "guandan.repro-probe/1",
        "status": "PASS",
        "exit_code": 0,
        "report": report,
    }


def reproduce_support_suite(
    support_zip: Path | str,
    *,
    output_path: Path | str | None = None,
    truth_path: Path | str | None = None,
    expected_level: str | None = None,
    expected_hand: Sequence[str] | None = None,
    repeats: int = 20,
    role: str = "unspecified",
) -> dict[str, object]:
    """Automatically compare ordinary, deterministic, and fresh-child probes."""

    ordinary = reproduce_support_bundle(
        support_zip,
        truth_path=truth_path,
        expected_level=expected_level,
        expected_hand=expected_hand,
        repeats=repeats,
        deterministic=False,
        role=role,
    )
    deterministic_report = reproduce_support_bundle(
        support_zip,
        truth_path=truth_path,
        expected_level=expected_level,
        expected_hand=expected_hand,
        repeats=repeats,
        deterministic=True,
        role=role,
    )
    with tempfile.TemporaryDirectory(prefix="daguandan-repro-child-") as temporary:
        child = run_child_probe(
            support_zip,
            Path(temporary) / "child-report.json",
            truth_path=truth_path,
            repeats=repeats,
            deterministic=True,
        )
    ordinary_matches = _normalized_probe_signature(ordinary) == _normalized_probe_signature(
        deterministic_report
    )
    child_report = child.get("report") if isinstance(child, Mapping) else None
    child_matches = bool(
        isinstance(child_report, Mapping)
        and _normalized_probe_signature(child_report)
        == _normalized_probe_signature(deterministic_report)
    )
    child_summary = {
        **dict(child),
        "matches_same_process": child_matches,
    }
    final_report = reproduce_support_bundle(
        support_zip,
        truth_path=truth_path,
        expected_level=expected_level,
        expected_hand=expected_hand,
        repeats=repeats,
        deterministic=True,
        child_probe=child_summary,
        role=role,
    )
    final_report["probes"] = {
        "same_process_ordinary": {
            "normalized_sha256": _normalized_probe_signature(ordinary),
        },
        "same_process_deterministic": {
            "normalized_sha256": _normalized_probe_signature(deterministic_report),
            "matches_ordinary": ordinary_matches,
        },
        "fresh_child_deterministic": child_summary,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), final_report)
    return final_report


def _normalized_probe_signature(report: Mapping[str, object]) -> str:
    normalized = {
        "support_sha256": _nested(report, "support", "sha256"),
        "inputs": [
            {
                "frame_seq": item.get("frame_seq"),
                "pixel_sha256": item.get("pixel_sha256"),
            }
            for item in report.get("inputs", [])
            if isinstance(item, Mapping)
        ],
        "outcomes": [
            {
                "output_fingerprint": item.get("output_fingerprint"),
                "opening_gate": item.get("opening_gate"),
                "single_frame_level": item.get("single_frame_level"),
                "single_frame_hand": item.get("single_frame_hand"),
            }
            for item in report.get("outcomes", [])
            if isinstance(item, Mapping)
        ],
    }
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def compare_repro_reports(
    reference: Mapping[str, object] | Path | str,
    candidate: Mapping[str, object] | Path | str,
    *,
    output_path: Path | str | None = None,
) -> dict[str, object]:
    old = _report_document(reference)
    new = _report_document(candidate)
    failures: list[str] = []
    if _nested(old, "support", "sha256") != _nested(new, "support", "sha256"):
        failures.append("support_sha256_mismatch")
    old_inputs = [item.get("pixel_sha256") for item in old.get("inputs", []) if isinstance(item, Mapping)]
    new_inputs = [item.get("pixel_sha256") for item in new.get("inputs", []) if isinstance(item, Mapping)]
    if old_inputs != new_inputs:
        failures.append("input_pixel_hash_mismatch")
    if old.get("verification_role") != "reference":
        failures.append("reference_role_invalid")
    if new.get("verification_role") != "candidate":
        failures.append("candidate_role_invalid")
    if _nested(old, "truth_identity", "sha256") != _nested(
        new, "truth_identity", "sha256"
    ):
        failures.append("truth_sha256_mismatch")
    old_build = _nested(old, "runner", "build_id")
    new_build = _nested(new, "runner", "build_id")
    if not old_build or not new_build:
        failures.append("runner_build_id_missing")
    elif old_build == new_build:
        failures.append("reference_candidate_build_not_distinct")
    if old.get("repeat_count") != 20 or not _nested(old, "repeatability", "repeatable"):
        failures.append("reference_not_repeatable_20_of_20")
    if not _nested(old, "truth", "eligible_for_fix_verification"):
        failures.append("reference_truth_missing")
    if _nested(old, "truth", "correct_runs") != 0:
        failures.append("reference_does_not_reproduce_failure_20_of_20")
    if new.get("repeat_count") != 20 or not _nested(new, "repeatability", "repeatable"):
        failures.append("candidate_not_repeatable_20_of_20")
    if not _nested(new, "truth", "eligible_for_fix_verification"):
        failures.append("candidate_truth_missing")
    if _nested(new, "truth", "correct_runs") != 20:
        failures.append("candidate_not_correct_20_of_20")
    report = {
        "schema": REPRO_GATE_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "support_sha256": _nested(old, "support", "sha256"),
        "input_pixel_sha256": old_inputs,
        "truth_sha256": _nested(old, "truth_identity", "sha256"),
        "reference_build_id": old_build,
        "candidate_build_id": new_build,
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), report)
    return report


def _run_sequence(
    recognizer: object,
    frames: Sequence[dict[str, object]],
    repeat_index: int,
) -> dict[str, object]:
    frame_results: list[dict[str, object]] = []
    stable_seed: object | None = None
    previous_seed: object | None = None
    observed_seed_fingerprints: set[str] = set()
    for frame in frames:
        operation = getattr(recognizer, "recognize", None)
        if not callable(operation):
            raise SupportReproError("recognizer does not expose recognize()")
        result = operation(frame["image"], allow_unknown_suit=True)
        trace_reader = getattr(recognizer, "get_last_diagnostic_trace", None)
        trace = trace_reader() if callable(trace_reader) else None
        level = str(getattr(result, "round_level", "") or "")
        hand = tuple(sorted(str(card) for card in getattr(result, "my_hand", ()) or ()))
        opening_frame = frame.get("opening_frame")
        anchor_score = (
            _number(opening_frame.get("anchor_score"), default=-1.0)
            if isinstance(opening_frame, Mapping)
            else None
        )
        gate = evaluate_opening_gate(result, anchor_score=anchor_score)
        seed = gate.seed
        if seed is not None and seed == previous_seed:
            stable_seed = seed
        if seed is not None:
            observed_seed_fingerprints.add(
                hashlib.sha256(_canonical_json(_opening_seed_document(seed))).hexdigest()
            )
        previous_seed = seed
        frame_results.append(
            {
                "frame_seq": frame["frame_seq"],
                "round_level": level or None,
                "hand": list(hand),
                "hand_count": len(hand),
                "production_gate": "ready" if gate.ready else gate.reason,
                "candidate_vector": list(trace.get("candidates", []))
                if isinstance(trace, Mapping)
                else [],
                "input_sha256": frame["pixel_sha256"],
                "trace_input_sha256": (
                    trace.get("input_sha256")
                    if isinstance(trace, Mapping)
                    else None
                ),
            }
        )
    if stable_seed is not None:
        consensus = {"status": "READY", "reason": "two_frame_consensus"}
    elif len(observed_seed_fingerprints) > 1:
        consensus = {"status": "FAIL", "reason": "opening_seed_oscillation"}
    else:
        consensus = {"status": "PENDING", "reason": "opening_stability_pending"}
    normalized = {
        "stable_level": getattr(stable_seed, "round_level", None),
        "stable_hand": list(getattr(stable_seed, "hand", ())) if stable_seed else None,
        "multi_frame_gate": consensus,
        "frames": [
            {
                "frame_seq": item["frame_seq"],
                "round_level": item["round_level"],
                "hand": item["hand"],
                "production_gate": item["production_gate"],
            }
            for item in frame_results
        ],
    }
    fingerprint = hashlib.sha256(_canonical_json(normalized)).hexdigest()
    final = frame_results[-1]
    return {
        "repeat_index": repeat_index,
        "output_fingerprint": fingerprint,
        "stable_level": normalized["stable_level"],
        "stable_hand": normalized["stable_hand"],
        "final_level": final["round_level"],
        "final_hand": final["hand"],
        "single_frame_level": final["round_level"],
        "single_frame_hand": final["hand"],
        "single_frame_gate": final["production_gate"],
        "multi_frame_gate": consensus,
        "opening_gate": (
            "ready" if consensus["status"] == "READY" else consensus["reason"]
        ),
        "candidate_vector": final["candidate_vector"],
        "frame_results": frame_results,
    }


def _opening_seed_document(seed: object) -> dict[str, object]:
    opening_action = getattr(seed, "opening_action", None)
    return {
        "round_level": getattr(seed, "round_level", None),
        "hand": list(getattr(seed, "hand", ())),
        "lead_player": getattr(seed, "lead_player", None),
        "opening_action": (
            {
                "actor": getattr(opening_action, "actor", None),
                "cards": list(getattr(opening_action, "cards", ())),
                "next_player": getattr(opening_action, "next_player", None),
            }
            if opening_action is not None
            else None
        ),
    }


def _evaluate_truth(
    outcomes: Sequence[Mapping[str, object]],
    truth: Mapping[str, object] | None,
) -> dict[str, object]:
    if truth is None:
        return {
            "status": "not_provided",
            "eligible_for_fix_verification": False,
            "correct_runs": None,
            "message": "symptom repeatability only; correctness cannot be claimed without independent truth",
        }
    expected_level = truth.get("expected_level")
    expected_hand = truth.get("expected_hand")
    target_input = str(truth.get("input_pixel_sha256") or "")
    expected_hand_normalized = (
        sorted(str(card) for card in expected_hand)
        if isinstance(expected_hand, list)
        else None
    )
    correct = 0
    for item in outcomes:
        target_frames = [
            frame
            for frame in item.get("frame_results", [])
            if isinstance(frame, Mapping)
            and str(frame.get("input_sha256") or "") == target_input
        ]
        target = target_frames[-1] if target_frames else None
        # Correctness is a single-frame truth assertion.  Multi-frame
        # readiness is reported separately and must never erase a correct
        # one-frame observation.
        level_ok = bool(
            target is not None
            and (expected_level is None or target.get("round_level") == expected_level)
        )
        hand_ok = bool(
            target is not None
            and (
                expected_hand_normalized is None
                or target.get("hand") == expected_hand_normalized
            )
        )
        if level_ok and hand_ok:
            correct += 1
    return {
        "status": "provided",
        "eligible_for_fix_verification": True,
        "expected_level": expected_level,
        "expected_hand": expected_hand_normalized,
        "input_pixel_sha256": target_input,
        "correct_runs": correct,
        "total_runs": len(outcomes),
        "all_correct": correct == len(outcomes),
    }


def _load_truth(
    path: Path | str | None,
    *,
    support_sha256: str,
    frames: Sequence[Mapping[str, object]],
    expected_level: str | None,
    expected_hand: Sequence[str] | None,
) -> dict[str, object] | None:
    sequence_sha256 = _input_sequence_sha256(frames)
    if path is not None:
        raw = Path(path).read_bytes()
        document = _json_object(raw, "truth annotation")
        if document.get("schema") != REPRO_TRUTH_SCHEMA:
            raise SupportReproError("unsupported truth annotation schema")
        if document.get("support_sha256") != support_sha256:
            raise SupportReproError("truth annotation is bound to a different support ZIP")
        if document.get("input_sequence_sha256") != sequence_sha256:
            raise SupportReproError("truth annotation is bound to a different input sequence")
        input_hash = str(document.get("input_pixel_sha256") or "")
        allowed_hashes = {str(item.get("pixel_sha256") or "") for item in frames}
        if input_hash not in allowed_hashes:
            raise SupportReproError("truth annotation is bound to a different input frame")
        normalized_level, normalized_hand = _validated_expected_truth(
            document.get("expected_level"),
            document.get("expected_hand") if isinstance(document.get("expected_hand"), list) else None,
        )
        document["expected_level"] = normalized_level
        document["expected_hand"] = (
            list(normalized_hand) if normalized_hand is not None else None
        )
        document["_truth_sha256"] = hashlib.sha256(raw).hexdigest()
        return document
    if expected_level is None and expected_hand is None:
        return None
    normalized_level, normalized_hand = _validated_expected_truth(
        expected_level,
        expected_hand,
    )
    document = {
        "schema": REPRO_TRUTH_SCHEMA,
        "support_sha256": support_sha256,
        "input_sequence_sha256": sequence_sha256,
        "input_pixel_sha256": str(frames[-1].get("pixel_sha256") or ""),
        "expected_level": normalized_level,
        "expected_hand": list(normalized_hand) if normalized_hand is not None else None,
    }
    document["_truth_sha256"] = hashlib.sha256(_canonical_json(document)).hexdigest()
    return document


def _validated_expected_truth(
    expected_level: object,
    expected_hand: Sequence[object] | None,
) -> tuple[str | None, tuple[str, ...] | None]:
    level = None if expected_level is None or str(expected_level) == "" else str(expected_level)
    hand_values = None if expected_hand is None else tuple(str(card) for card in expected_hand)
    if level is None and not hand_values:
        raise SupportReproError(
            "truth annotation needs at least one non-empty expected value"
        )
    if level is not None and level not in RANKS:
        raise SupportReproError("truth expected_level is not a legal GuanDan rank")
    normalized_hand: tuple[str, ...] | None = None
    if hand_values is not None:
        if len(hand_values) != 27:
            raise SupportReproError("truth expected_hand must contain exactly 27 cards")
        try:
            state = GuanDanState()
            state.confirm_hand(hand_values)
        except Exception as exc:
            raise SupportReproError("truth expected_hand is not a legal GuanDan hand") from exc
        normalized_hand = tuple(state.my_hand)
    return level, normalized_hand


def _input_sequence_sha256(frames: Sequence[Mapping[str, object]]) -> str:
    sequence = [
        {
            "frame_seq": int(item.get("frame_seq", index)),
            "pixel_sha256": str(item.get("pixel_sha256") or ""),
        }
        for index, item in enumerate(frames, start=1)
    ]
    return hashlib.sha256(_canonical_json(sequence)).hexdigest()


def _image_index(
    payloads: Mapping[str, bytes],
    opening: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    document = _optional_json(payloads, ("evidence/image_index.json", "image_index.json"))
    if isinstance(document, Mapping):
        if document.get("schema") != "guandan.support-image-index/1":
            raise SupportReproError("unsupported support image index schema")
        entries = document.get("entries")
        if isinstance(entries, list):
            return [dict(item) for item in entries if isinstance(item, Mapping)]
    return []


def _standardized_frames(
    support: VerifiedSupport,
    image_index: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    opening = _optional_json(
        support.payloads,
        ("evidence/opening_evidence.json", "opening/opening_evidence.json"),
    )
    opening_by_seq = {
        int(item.get("seq", item.get("frame_seq"))): item
        for item in (opening.get("frames", []) if isinstance(opening, Mapping) else [])
        if isinstance(item, Mapping)
        and _safe_int(item.get("seq", item.get("frame_seq"))) is not None
    }
    selected: list[tuple[Mapping[str, object], str, bytes, _ImageHeader]] = []
    budget = _ImageBudget()
    identities: set[tuple[int, str, str | None]] = set()
    for raw in image_index:
        kind = str(raw.get("kind", "")).casefold()
        if kind not in {"raw_client", "standardized", "roi"}:
            raise SupportReproError("support image index kind is invalid")
        frame_seq = _safe_int(raw.get("frame_seq"))
        if frame_seq is None or frame_seq < 0:
            raise SupportReproError("support image index frame_seq is invalid")
        field = None if raw.get("field") in {None, ""} else str(raw.get("field"))
        identity = (frame_seq, kind, field)
        if identity in identities:
            raise SupportReproError("support image index contains duplicate identity")
        identities.add(identity)
        if kind != "standardized":
            continue
        name = _safe_archive_name(str(raw.get("archive_path", "")))
        content = support.payloads.get(name)
        if content is None:
            raise SupportReproError(f"indexed standardized image is missing: {name}")
        header = _image_header(content, name)
        budget.add(header, name)
        selected.append((raw, name, content, header))
    frames: list[dict[str, object]] = []
    for raw, name, content, _header in selected:
        if raw.get("source_sha256") and raw.get("source_sha256") != hashlib.sha256(content).hexdigest():
            raise SupportReproError(f"indexed standardized image hash mismatch: {name}")
        image = _decode_image(content, name)
        pixel_hash = hashlib.sha256(memoryview(np.ascontiguousarray(image))).hexdigest()
        if raw.get("pixel_sha256") and raw.get("pixel_sha256") != pixel_hash:
            raise SupportReproError(f"indexed standardized pixel hash mismatch: {name}")
        frames.append(
            {
                "frame_seq": int(raw.get("frame_seq", len(frames) + 1)),
                "archive_path": name,
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "pixel_sha256": pixel_hash,
                "image": image,
                "opening_frame": opening_by_seq.get(int(raw.get("frame_seq", -1))),
            }
        )
    return sorted(frames, key=lambda item: int(item["frame_seq"]))


def _decode_image(content: bytes, name: str) -> np.ndarray:
    _image_header(content, name)
    data = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.size == 0:
        raise SupportReproError(f"support image is invalid: {name}")
    if image.shape[0] * image.shape[1] > _MAX_IMAGE_PIXELS:
        raise SupportReproError(f"support image exceeds pixel limit: {name}")
    return image


def _preflight_support_images(payloads: Mapping[str, bytes]) -> None:
    budget = _ImageBudget()
    for name, content in sorted(payloads.items()):
        if not name.casefold().endswith((".png", ".jpg", ".jpeg")):
            continue
        budget.add(_image_header(content, name), name)


def _validate_incident_artifact_bindings(payloads: Mapping[str, bytes]) -> None:
    opening = _optional_json(
        payloads,
        ("evidence/opening_evidence.json", "opening/opening_evidence.json"),
    )
    index = _optional_json(payloads, ("evidence/image_index.json", "image_index.json"))
    if not isinstance(opening, Mapping) or not isinstance(index, Mapping):
        return
    declared: dict[str, tuple[int, str, str | None, str, str, str | None]] = {}
    has_artifacts = False
    frames = opening.get("frames")
    for frame in frames if isinstance(frames, list) else []:
        if not isinstance(frame, Mapping):
            continue
        seq = _safe_int(frame.get("seq", frame.get("frame_seq")))
        frame_id = str(frame.get("frame_id") or "") or None
        artifacts = frame.get("artifacts")
        for artifact in artifacts if isinstance(artifacts, list) else []:
            if not isinstance(artifact, Mapping):
                continue
            has_artifacts = True
            path = _safe_archive_name(str(artifact.get("path") or ""))
            artifact_seq = _safe_int(artifact.get("frame_seq"))
            kind = str(artifact.get("kind") or "").casefold()
            field = None if artifact.get("field") in {None, ""} else str(artifact.get("field"))
            file_sha = str(artifact.get("sha256") or "")
            pixel_sha = str(artifact.get("pixel_sha256") or "")
            artifact_frame_id = str(artifact.get("frame_id") or "") or None
            if (
                seq is None
                or artifact_seq != seq
                or kind not in {"raw_client", "standardized", "roi"}
                or (kind == "roi") != (field is not None)
                or re_full_sha256(file_sha) is False
                or re_full_sha256(pixel_sha) is False
                or (frame_id is not None and artifact_frame_id != frame_id)
            ):
                raise SupportReproError("opening incident artifact declaration is invalid")
            if path in declared:
                raise SupportReproError("opening incident artifact path is duplicated")
            declared[path] = (seq, kind, field, file_sha, pixel_sha, frame_id)
    if not has_artifacts:
        return
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise SupportReproError("support image index is missing incident bindings")
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SupportReproError("support image index entry is invalid")
        incident_path = _safe_archive_name(str(entry.get("incident_path") or ""))
        expected = declared.get(incident_path)
        if expected is None:
            raise SupportReproError("support image is not declared by the incident")
        seq, kind, field, file_sha, pixel_sha, frame_id = expected
        if (
            _safe_int(entry.get("frame_seq")) != seq
            or str(entry.get("kind") or "").casefold() != kind
            or (None if entry.get("field") in {None, ""} else str(entry.get("field"))) != field
            or str(entry.get("incident_sha256") or "") != file_sha
            or str(entry.get("source_sha256") or "") != file_sha
            or str(entry.get("pixel_sha256") or "") != pixel_sha
            or (frame_id is not None and str(entry.get("frame_id") or "") != frame_id)
        ):
            raise SupportReproError("support image index disagrees with incident artifact")
        if incident_path in seen_paths:
            raise SupportReproError("support incident artifact is indexed more than once")
        seen_paths.add(incident_path)
    image_payloads = {
        name
        for name in payloads
        if name.casefold().endswith((".png", ".jpg", ".jpeg"))
    }
    indexed_payloads = {
        _safe_archive_name(str(entry.get("archive_path") or ""))
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if image_payloads != indexed_payloads:
        raise SupportReproError("support image payloads are not exactly indexed")


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _image_header(content: bytes, name: str) -> _ImageHeader:
    lowered = name.casefold()
    if lowered.endswith(".png"):
        if (
            len(content) < 29
            or not content.startswith(b"\x89PNG\r\n\x1a\n")
            or content[12:16] != b"IHDR"
        ):
            raise SupportReproError(f"support PNG header is invalid: {name}")
        width, height = struct.unpack(">II", content[16:24])
        bit_depth = int(content[24])
        color_type = int(content[25])
        channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
        channels = channels_by_type.get(color_type)
        if channels is None or bit_depth not in {1, 2, 4, 8, 16}:
            raise SupportReproError(f"support PNG format is unsafe: {name}")
        bytes_per_channel = 2 if bit_depth == 16 else 1
        decoded_bytes = int(width) * int(height) * max(3, channels) * bytes_per_channel
    elif lowered.endswith((".jpg", ".jpeg")):
        width, height, channels = _jpeg_header(content, name)
        decoded_bytes = int(width) * int(height) * max(3, channels)
    else:
        raise SupportReproError(f"support image type is unsupported: {name}")
    if width <= 0 or height <= 0 or int(width) * int(height) > _MAX_IMAGE_PIXELS:
        raise SupportReproError(f"support image exceeds pixel limit: {name}")
    if decoded_bytes <= 0 or decoded_bytes > _MAX_TOTAL_DECODED_BYTES:
        raise SupportReproError(f"support image decoded size exceeds hard limit: {name}")
    return _ImageHeader(int(width), int(height), int(decoded_bytes))


def _jpeg_header(content: bytes, name: str) -> tuple[int, int, int]:
    if len(content) < 4 or not content.startswith(b"\xff\xd8"):
        raise SupportReproError(f"support JPEG header is invalid: {name}")
    position = 2
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while position < len(content):
        while position < len(content) and content[position] != 0xFF:
            position += 1
        while position < len(content) and content[position] == 0xFF:
            position += 1
        if position >= len(content):
            break
        marker = int(content[position])
        position += 1
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        if position + 2 > len(content):
            break
        segment_length = int.from_bytes(content[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(content):
            break
        if marker in sof_markers and segment_length >= 8:
            height = int.from_bytes(content[position + 3 : position + 5], "big")
            width = int.from_bytes(content[position + 5 : position + 7], "big")
            channels = int(content[position + 7])
            if channels not in {1, 3, 4}:
                raise SupportReproError(f"support JPEG channel count is unsafe: {name}")
            return width, height, channels
        position += segment_length
    raise SupportReproError(f"support JPEG header is invalid: {name}")


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _number(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _standardization_equality(
    support: VerifiedSupport,
    image_index: Sequence[Mapping[str, object]],
    opening: Mapping[str, object] | None,
) -> tuple[bool | None, bool | None]:
    frames = {
        int(item.get("seq", item.get("frame_seq"))): item
        for item in (opening.get("frames", []) if isinstance(opening, Mapping) else [])
        if isinstance(item, Mapping)
        and _safe_int(item.get("seq", item.get("frame_seq"))) is not None
    }
    by_identity: dict[tuple[int, str, str | None], Mapping[str, object]] = {}
    for item in image_index:
        seq = _safe_int(item.get("frame_seq"))
        if seq is None:
            continue
        kind = str(item.get("kind") or "").casefold()
        field = None if item.get("field") in {None, ""} else str(item.get("field"))
        by_identity[(seq, kind, field)] = item
    paired = sorted(
        seq
        for seq in frames
        if (seq, "raw_client", None) in by_identity
        and (seq, "standardized", None) in by_identity
    )
    standard_equal: bool | None = None
    standardized_images: dict[int, np.ndarray] = {}
    if paired:
        standard_equal = True
        for seq in paired:
            raw = _decode_indexed_image(support, by_identity[(seq, "raw_client", None)])
            saved = _decode_indexed_image(
                support,
                by_identity[(seq, "standardized", None)],
            )
            regenerated = _regenerate_standardized(raw, frames[seq])
            if regenerated is None or not np.array_equal(regenerated, saved):
                standard_equal = False
            standardized_images[seq] = saved
    roi_entries = [
        (identity, item)
        for identity, item in by_identity.items()
        if identity[1] == "roi"
    ]
    roi_equal: bool | None = None
    if roi_entries:
        roi_equal = True
        for (seq, _kind, field), entry in roi_entries:
            standard = standardized_images.get(seq)
            if standard is None and (seq, "standardized", None) in by_identity:
                standard = _decode_indexed_image(
                    support,
                    by_identity[(seq, "standardized", None)],
                )
                standardized_images[seq] = standard
            artifact = _opening_artifact(frames.get(seq), "roi", field)
            box = artifact.get("box") if isinstance(artifact, Mapping) else None
            if standard is None or not _valid_box(box, standard):
                roi_equal = False
                continue
            x, y, width, height = (int(value) for value in box)
            expected = standard[y : y + height, x : x + width]
            actual = _decode_indexed_image(support, entry, unchanged=True)
            if not np.array_equal(expected, actual):
                roi_equal = False
    return standard_equal, roi_equal


def _decode_indexed_image(
    support: VerifiedSupport,
    entry: Mapping[str, object],
    *,
    unchanged: bool = False,
) -> np.ndarray:
    name = _safe_archive_name(str(entry.get("archive_path") or ""))
    content = support.payloads.get(name)
    if content is None:
        raise SupportReproError(f"indexed image is missing: {name}")
    _image_header(content, name)
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), flag)
    if image is None or image.size == 0:
        raise SupportReproError(f"support image is invalid: {name}")
    return image


def _regenerate_standardized(
    raw: np.ndarray,
    frame: Mapping[str, object],
) -> np.ndarray | None:
    capture = frame.get("capture")
    standardization = (
        capture.get("standardization") if isinstance(capture, Mapping) else None
    )
    if not isinstance(standardization, Mapping):
        return None
    viewport = standardization.get("source_viewport")
    content_box = standardization.get("content_box")
    target_size = standardization.get("standardized_size")
    if not _valid_box(viewport, raw) or not (
        isinstance(content_box, list) and len(content_box) == 4
    ):
        return None
    try:
        vx, vy, vw, vh = (int(value) for value in viewport)
        cx, cy, cw, ch = (int(value) for value in content_box)
        width, height = (int(value) for value in target_size)
    except (TypeError, ValueError):
        return None
    if min(cx, cy) < 0 or min(cw, ch, width, height) <= 0:
        return None
    if cx + cw > width or cy + ch > height:
        return None
    interpolation_name = str(standardization.get("interpolation") or "")
    interpolation = {
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_LINEAR": cv2.INTER_LINEAR,
    }.get(interpolation_name)
    if interpolation is None:
        return None
    crop = raw[vy : vy + vh, vx : vx + vw]
    resized = cv2.resize(crop, (cw, ch), interpolation=interpolation)
    shape = (height, width, *resized.shape[2:])
    canvas = np.zeros(shape, dtype=resized.dtype)
    canvas[cy : cy + ch, cx : cx + cw] = resized
    return canvas


def _valid_box(value: object, image: np.ndarray) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x, y, width, height = (int(item) for item in value)
    except (TypeError, ValueError):
        return False
    return bool(
        x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= image.shape[1]
        and y + height <= image.shape[0]
    )


def _opening_artifact(
    frame: Mapping[str, object] | None,
    kind: str,
    field: str | None,
) -> Mapping[str, object] | None:
    artifacts = frame.get("artifacts") if isinstance(frame, Mapping) else None
    for item in artifacts or []:
        if not isinstance(item, Mapping):
            continue
        item_field = None if item.get("field") in {None, ""} else str(item.get("field"))
        if str(item.get("kind") or "").casefold() == kind and item_field == field:
            return item
    return None


def _local_resource_identity(recognizer: object) -> dict[str, object] | None:
    annotation = getattr(recognizer, "annotation_service", None)
    profiles_root = getattr(annotation, "profiles_root", None)
    profile_name = getattr(annotation, "profile_name", None)
    if profiles_root is None or profile_name is None:
        return None
    return recognition_resource_identity(Path(profiles_root), str(profile_name))


def _default_recognizer_factory() -> object:
    from .recognition_service import ScreenshotRecognitionService

    return ScreenshotRecognitionService()


@contextmanager
def _deterministic_runtime(*, enabled: bool):
    if not enabled:
        yield
        return
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    previous_threads = cv2.getNumThreads()
    previous_opencl = bool(cv2.ocl.useOpenCL())
    try:
        random.seed(0)
        np.random.seed(0)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)


def _entrypoint_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, str(Path(__file__).resolve().parents[2] / "run.py")]


def _optional_json(
    payloads: Mapping[str, bytes],
    names: Iterable[str],
) -> dict[str, object] | None:
    for name in names:
        content = payloads.get(name)
        if content is not None:
            return _json_object(content, name)
    return None


def _json_object(content: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SupportReproError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SupportReproError(f"{label} must be a JSON object")
    return value


def _safe_archive_name(raw: str) -> str:
    if not raw or "\\" in raw or ":" in raw or raw.startswith("/"):
        raise SupportReproError(f"unsafe support ZIP path: {raw!r}")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SupportReproError(f"unsafe support ZIP path: {raw!r}")
    return path.as_posix()


def _report_document(value: Mapping[str, object] | Path | str) -> dict[str, object]:
    if isinstance(value, Mapping):
        document = dict(value)
    else:
        document = _json_object(Path(value).read_bytes(), "repro report")
    if document.get("schema") != REPRO_REPORT_SCHEMA:
        raise SupportReproError("unsupported repro report schema")
    return document


def _nested(value: Mapping[str, object], *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "REPRO_GATE_SCHEMA",
    "REPRO_REPORT_SCHEMA",
    "REPRO_TRUTH_SCHEMA",
    "SupportReproError",
    "VerifiedSupport",
    "compare_repro_reports",
    "reproduce_support_bundle",
    "reproduce_support_suite",
    "run_child_probe",
    "verify_support_archive",
    "write_truth_annotation",
]
