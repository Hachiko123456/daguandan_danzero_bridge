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
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4
import zipfile

import cv2
import numpy as np

from .diagnostic_root_cause import diagnose_root_cause
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


class SupportReproError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedSupport:
    path: Path
    sha256: str
    manifest: dict[str, object]
    payloads: dict[str, bytes]


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
    if expected_level is None and expected_hand is None:
        raise SupportReproError("truth annotation needs expected_level or expected_hand")
    verified = verify_support_archive(support_zip)
    if input_pixel_sha256 is None:
        opening = _optional_json(
            verified.payloads,
            ("evidence/opening_evidence.json", "opening/opening_evidence.json"),
        )
        indexed = _image_index(verified.payloads, opening)
        frames = _standardized_frames(verified, indexed)
        hashes = sorted({str(item["pixel_sha256"]) for item in frames})
        if len(hashes) != 1:
            raise SupportReproError(
                "truth annotation needs input_pixel_sha256 when support contains multiple distinct inputs"
            )
        input_pixel_sha256 = hashes[0]
    document: dict[str, object] = {
        "schema": REPRO_TRUTH_SCHEMA,
        "support_sha256": verified.sha256,
        "input_pixel_sha256": input_pixel_sha256,
        "expected_level": str(expected_level) if expected_level is not None else None,
        "expected_hand": sorted(str(card) for card in expected_hand)
        if expected_hand is not None
        else None,
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
    standard_equal = _standardization_equality(verified, image_index, opening)
    root_cause = diagnose_root_cause(
        support_verified=True,
        opening_evidence=opening,
        outcomes=outcomes,
        truth=truth,
        local_resource_identity=local_resources,
        support_build_id=support_build_id,
        runner_build_id=runner_build_id,
        standardization_equal=standard_equal,
        roi_equal=None,
        child_probe=child_probe,
    )
    report: dict[str, object] = {
        "schema": REPRO_REPORT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "frozen" if getattr(sys, "frozen", False) else "source",
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
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(1.0, float(timeout_seconds)),
        check=False,
        text=False,
    )
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
        "reference_build_id": _nested(old, "runner", "build_id"),
        "candidate_build_id": _nested(new, "runner", "build_id"),
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
    stable_seed: tuple[str, tuple[str, ...]] | None = None
    previous_seed: tuple[str, tuple[str, ...]] | None = None
    for frame in frames:
        operation = getattr(recognizer, "recognize", None)
        if not callable(operation):
            raise SupportReproError("recognizer does not expose recognize()")
        result = operation(frame["image"], allow_unknown_suit=True)
        trace_reader = getattr(recognizer, "get_last_diagnostic_trace", None)
        trace = trace_reader() if callable(trace_reader) else None
        level = str(getattr(result, "round_level", "") or "")
        hand = tuple(sorted(str(card) for card in getattr(result, "my_hand", ()) or ()))
        seed = (level, hand) if level and len(hand) == 27 else None
        if seed is not None and seed == previous_seed:
            stable_seed = seed
        previous_seed = seed
        frame_results.append(
            {
                "frame_seq": frame["frame_seq"],
                "round_level": level or None,
                "hand": list(hand),
                "hand_count": len(hand),
                "gate": "ready" if stable_seed is not None else _gate_reason(level, hand),
                "candidate_vector": list(trace.get("candidates", []))
                if isinstance(trace, Mapping)
                else [],
                "input_sha256": trace.get("input_sha256")
                if isinstance(trace, Mapping)
                else frame["pixel_sha256"],
            }
        )
    normalized = {
        "stable_level": stable_seed[0] if stable_seed else None,
        "stable_hand": list(stable_seed[1]) if stable_seed else None,
        "frames": [
            {
                "frame_seq": item["frame_seq"],
                "round_level": item["round_level"],
                "hand": item["hand"],
                "gate": item["gate"],
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
        "opening_gate": final["gate"] if stable_seed is None else "ready",
        "candidate_vector": final["candidate_vector"],
        "frame_results": frame_results,
    }


def _gate_reason(level: str, hand: Sequence[str]) -> str:
    if not level:
        return "round_level_unresolved"
    if len(hand) != 27:
        return "hand_count_mismatch"
    return "opening_stability_pending"


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
    expected_hand_normalized = (
        sorted(str(card) for card in expected_hand)
        if isinstance(expected_hand, list)
        else None
    )
    correct = 0
    for item in outcomes:
        level_ok = expected_level is None or item.get("stable_level") == expected_level
        hand_ok = expected_hand_normalized is None or item.get("stable_hand") == expected_hand_normalized
        if level_ok and hand_ok:
            correct += 1
    return {
        "status": "provided",
        "eligible_for_fix_verification": True,
        "expected_level": expected_level,
        "expected_hand": expected_hand_normalized,
        "correct_runs": correct,
        "total_runs": len(outcomes),
        "all_correct": correct == len(outcomes),
    }


def _load_truth(
    path: Path | str | None,
    *,
    support_sha256: str,
    expected_level: str | None,
    expected_hand: Sequence[str] | None,
) -> dict[str, object] | None:
    if path is not None:
        document = _json_object(Path(path).read_bytes(), "truth annotation")
        if document.get("schema") != REPRO_TRUTH_SCHEMA:
            raise SupportReproError("unsupported truth annotation schema")
        if document.get("support_sha256") != support_sha256:
            raise SupportReproError("truth annotation is bound to a different support ZIP")
        return document
    if expected_level is None and expected_hand is None:
        return None
    return {
        "schema": REPRO_TRUTH_SCHEMA,
        "support_sha256": support_sha256,
        "expected_level": expected_level,
        "expected_hand": sorted(str(card) for card in expected_hand)
        if expected_hand is not None
        else None,
    }


def _image_index(
    payloads: Mapping[str, bytes],
    opening: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    document = _optional_json(payloads, ("evidence/image_index.json", "image_index.json"))
    if isinstance(document, Mapping):
        entries = document.get("entries")
        if isinstance(entries, list):
            return [dict(item) for item in entries if isinstance(item, Mapping)]
    entries: list[dict[str, object]] = []
    for name in payloads:
        if "standardized" in name.casefold() and name.casefold().endswith((".png", ".jpg", ".jpeg")):
            entries.append(
                {
                    "archive_path": name,
                    "frame_seq": len(entries) + 1,
                    "kind": "standardized",
                    "field": None,
                }
            )
    return entries


def _standardized_frames(
    support: VerifiedSupport,
    image_index: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for raw in image_index:
        if str(raw.get("kind", "")).casefold() != "standardized":
            continue
        name = _safe_archive_name(str(raw.get("archive_path", "")))
        content = support.payloads.get(name)
        if content is None:
            raise SupportReproError(f"indexed standardized image is missing: {name}")
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
            }
        )
    return sorted(frames, key=lambda item: int(item["frame_seq"]))


def _decode_image(content: bytes, name: str) -> np.ndarray:
    data = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.size == 0:
        raise SupportReproError(f"support image is invalid: {name}")
    if image.shape[0] * image.shape[1] > _MAX_IMAGE_PIXELS:
        raise SupportReproError(f"support image exceeds pixel limit: {name}")
    return image


def _standardization_equality(
    support: VerifiedSupport,
    image_index: Sequence[Mapping[str, object]],
    opening: Mapping[str, object] | None,
) -> bool | None:
    del support, opening
    raw_seq = {
        int(item.get("frame_seq", -1))
        for item in image_index
        if item.get("kind") == "raw_client"
    }
    standard_seq = {
        int(item.get("frame_seq", -1))
        for item in image_index
        if item.get("kind") == "standardized"
    }
    if not raw_seq or not standard_seq:
        return None
    # Exact pixel regeneration is performed by a later layer when viewport
    # parameters are available. Presence alone must never be labelled equal.
    return None


def _local_resource_identity(recognizer: object) -> dict[str, object] | None:
    annotation = getattr(recognizer, "annotation_service", None)
    profiles_root = getattr(annotation, "profiles_root", None)
    profile_name = getattr(annotation, "profile_name", None)
    if profiles_root is None or profile_name is None:
        return None
    root = Path(profiles_root) / str(profile_name)
    try:
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and (
                path.name in {"profile.json", "regions_config.json", "templates_config.json"}
                or "templates" in path.relative_to(root).parts
            )
        ]
        records = [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        ]
        return {
            "status": "identified",
            "sha256": hashlib.sha256(_canonical_json(records)).hexdigest(),
            "files": records,
        }
    except OSError:
        return {"status": "unavailable"}


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
    "run_child_probe",
    "verify_support_archive",
    "write_truth_annotation",
]
