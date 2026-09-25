"""Read-only qualification of recorded sessions for replay tiers.

This module deliberately does not treat TruthLog or manifest claims as proof that
an opening was visible in the source video.  Strict replay requires explicit
opening evidence tied to source-frame references and an exact resource identity
match.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..resource_fingerprint import compare_resource_identities, recognition_resource_identity
from .session_locator import SessionDescriptor, inspect_session, inspect_sessions

CLASSIFICATIONS = frozenset({
    "strict_replay", "behavioral_replay", "diagnostic_only",
    "source_not_observable", "resource_mismatch", "invalid",
})

_VIDEO_SAMPLE_LIMIT = 8
_OPENING_EVIDENCE_NAMES = (
    "opening_evidence.json",
    "opening_candidates.json",
    "opening_observability.json",
    "opening_probe.json",
)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _profile_context(profile_root: Path, manifest: dict[str, object]) -> tuple[Path, str]:
    root = Path(profile_root).expanduser().resolve()
    manifest_name = str(manifest.get("profile") or "").strip()
    if (root / "profile.json").is_file():
        return root.parent, root.name
    if manifest_name and (root / manifest_name).is_dir():
        return root, manifest_name
    if manifest_name:
        return root, manifest_name
    return root, root.name


def _legacy_resource_identity(profile_dir: Path, profile_name: str) -> dict[str, object]:
    config = profile_dir / "profile.json"
    templates = profile_dir / "templates_config.json"
    return {
        "status": "identified" if config.is_file() and templates.is_file() else "unavailable",
        "algorithm": "legacy-file-hashes",
        "profile_name": profile_name,
        "configuration_hash": _sha256(config),
        "template_manifest_hash": _sha256(templates),
    }


def _resource_match(manifest: dict[str, object], profile_root: Path) -> dict[str, object]:
    profile_parent, profile_name = _profile_context(profile_root, manifest)
    actual = recognition_resource_identity(profile_parent, profile_name)
    expected = manifest.get("recognition_resource_identity")
    if not isinstance(expected, dict):
        expected = manifest.get("resource_identity")
    if isinstance(expected, dict):
        comparison = compare_resource_identities(expected, actual)
        return {
            **comparison,
            "method": "recognition_resource_identity",
            "expected": expected,
            "actual": actual,
        }

    expected_config = manifest.get("configuration_hash")
    expected_templates = manifest.get("template_manifest_hash")
    if isinstance(expected_config, str) or isinstance(expected_templates, str):
        legacy = _legacy_resource_identity(profile_parent / profile_name, profile_name)
        config_match = isinstance(expected_config, str) and expected_config == legacy.get("configuration_hash")
        template_match = isinstance(expected_templates, str) and expected_templates == legacy.get("template_manifest_hash")
        matched = config_match and template_match
        return {
            "status": "match" if matched else "mismatch",
            "matched": matched,
            "method": "legacy-file-hashes",
            "expected_configuration_hash": expected_config,
            "actual_configuration_hash": legacy.get("configuration_hash"),
            "expected_template_manifest_hash": expected_templates,
            "actual_template_manifest_hash": legacy.get("template_manifest_hash"),
            "expected": {"configuration_hash": expected_config, "template_manifest_hash": expected_templates},
            "actual": legacy,
        }

    return {
        "status": "unavailable",
        "matched": None,
        "method": "missing-source-identity",
        "expected": None,
        "actual": actual,
    }


def _read_index_count(path: Path) -> tuple[int | None, str | None]:
    if not path.is_file():
        return None, "frame_index_missing"
    count = 0
    previous = -1
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None, f"frame_index_record_invalid:{line_number}"
                index = value.get("frame_index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    return None, f"frame_index_value_invalid:{line_number}"
                if index <= previous:
                    return None, "frame_index_not_monotonic"
                previous = index
                count += 1
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "frame_index_unreadable"
    if count == 0:
        return None, "frame_index_empty"
    return count, None


def _video_probe(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "unavailable", "decoded_frames": 0, "sample_limit": _VIDEO_SAMPLE_LIMIT,
        "width": None, "height": None, "fps": None, "frame_count": None,
    }
    if not path.is_file():
        result.update(status="missing", reason="video_missing")
        return result
    try:
        import cv2  # type: ignore
    except Exception as exc:
        result.update(status="unverified", reason=f"opencv_unavailable:{type(exc).__name__}")
        return result
    try:
        capture = cv2.VideoCapture(str(path))
        if not bool(capture.isOpened()):
            result.update(status="invalid", reason="video_not_decodable")
            return result
        result["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        result["frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        result["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        result["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        for _ in range(_VIDEO_SAMPLE_LIMIT):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            result["decoded_frames"] = int(result["decoded_frames"]) + 1
        capture.release()
        if int(result["decoded_frames"]) <= 0:
            result.update(status="invalid", reason="video_has_no_decodable_prefix")
        else:
            result["status"] = "ok"
    except Exception as exc:
        result.update(status="invalid", reason=f"video_probe_failed:{type(exc).__name__}")
        try:
            capture.release()
        except Exception:
            pass
    return result


def _evidence_has_frame_refs(value: dict[str, object]) -> bool:
    for key in ("frame_indices", "frames", "source_frames", "evidence_frames"):
        items = value.get(key)
        if isinstance(items, (list, tuple)) and bool(items):
            return True
    for key in ("frame_index", "source_frame_index", "first_frame_index"):
        item = value.get(key)
        if isinstance(item, int) and item >= 0:
            return True
    return False


def _opening_observability(session: Path, manifest: dict[str, object], evidence_root: Path | None) -> dict[str, object]:
    candidates: list[Path] = []
    for name in _OPENING_EVIDENCE_NAMES:
        candidates.append(session / name)
        if evidence_root is not None:
            candidates.extend((evidence_root / session.name / name, evidence_root / name))
    for path in candidates:
        value = _read_json(path)
        if not isinstance(value, dict):
            continue
        observable = value.get("observable") is True or str(value.get("status", "")).lower() in {"observable", "confirmed", "proven"}
        hand_count = value.get("hand_count")
        if not observable or not _evidence_has_frame_refs(value):
            continue
        if hand_count is not None and hand_count != 27:
            continue
        return {
            "status": "proven", "source": str(path), "frame_references": True,
            "hand_count": hand_count, "reason": "explicit_opening_evidence_with_source_frames",
        }
    if str(manifest.get("recording_phase", "")).lower() == "ended_without_initial_state" or str(manifest.get("initial_state_status", "")).lower() == "unconfirmed":
        return {"status": "not_observable", "source": None, "reason": "manifest_marks_initial_state_unconfirmed"}
    return {
        "status": "unproven", "source": None,
        "reason": "manifest_or_truth_claim_does_not_prove_video_opening_observability",
    }


def _descriptor(session: Path | SessionDescriptor) -> SessionDescriptor:
    return inspect_session(session) if isinstance(session, (str, Path)) else session


def qualify_session(
    session: Path | SessionDescriptor,
    profile_root: Path,
    *,
    evidence_root: Path | None = None,
) -> dict[str, object]:
    """Return a conservative, read-only qualification record for one session."""
    descriptor = _descriptor(session)
    root = Path(descriptor.root).expanduser().resolve()
    manifest = _read_json(Path(descriptor.manifest_path)) or {}
    reasons: list[str] = []
    if not descriptor.manifest_readable:
        reasons.append("manifest_unreadable")
    if not descriptor.has_video:
        reasons.append("video_missing")
    index_count, index_error = _read_index_count(Path(descriptor.frame_index_path))
    if index_error:
        reasons.append(index_error)
    declared_count = manifest.get("frame_count")
    if isinstance(declared_count, int) and declared_count >= 0 and index_count is not None and declared_count != index_count:
        reasons.append("manifest_frame_count_mismatch")
    video_probe = _video_probe(Path(descriptor.video_path))
    if video_probe.get("status") in {"invalid", "missing"}:
        reasons.append(str(video_probe.get("reason", "video_invalid")))
    if isinstance(declared_count, int) and declared_count > 0 and int(video_probe.get("frame_count") or 0) > 0 and abs(declared_count - int(video_probe["frame_count"])) > max(2, declared_count // 100):
        reasons.append("video_frame_count_mismatch")
    if str(manifest.get("status", "sealed")).lower() == "aborted":
        reasons.append("manifest_aborted")
    resource_match = _resource_match(manifest, Path(profile_root))
    observability = _opening_observability(root, manifest, evidence_root)
    truth_verified = descriptor.truth_status == "verified"
    structural_invalid = any(reason in {"manifest_unreadable", "video_missing", "video_not_decodable", "video_has_no_decodable_prefix", "frame_index_unreadable", "frame_index_empty", "frame_index_not_monotonic", "frame_index_missing", "manifest_frame_count_mismatch", "video_frame_count_mismatch", "manifest_aborted"} or reason.startswith(("frame_index_record_invalid", "frame_index_value_invalid", "video_probe_failed")) for reason in reasons)
    resource_status = str(resource_match.get("status"))
    if structural_invalid:
        classification = "invalid"
    elif observability.get("status") == "not_observable":
        classification = "source_not_observable"
        reasons.append("opening_not_observable")
    elif resource_status in {"mismatch", "unavailable"} and truth_verified:
        classification = "resource_mismatch"
        reasons.append("resource_identity_missing" if resource_status == "unavailable" else "resource_identity_mismatch")
    elif truth_verified and observability.get("status") == "proven" and resource_status == "match":
        classification = "strict_replay"
    elif truth_verified and resource_status == "match":
        classification = "behavioral_replay"
        reasons.append("opening_observability_unproven")
    elif truth_verified:
        classification = "resource_mismatch"
        reasons.append("resource_identity_missing" if resource_status == "unavailable" else "resource_identity_mismatch")
    else:
        classification = "diagnostic_only"
        reasons.append("truth_log_not_verified")
    reasons = list(dict.fromkeys(reasons))
    strict_eligible = classification == "strict_replay" and bool(truth_verified and observability.get("status") == "proven" and resource_status == "match" and not structural_invalid)
    return {
        "session_id": descriptor.session_id,
        "classification": classification,
        "reasons": reasons,
        "resource_match": resource_match,
        "opening_observability": {**observability, "video_probe": video_probe},
        "strict_eligible": strict_eligible,
        "truth_status": descriptor.truth_status,
        "integrity": {
            "manifest_readable": descriptor.manifest_readable,
            "video": descriptor.has_video,
            "frame_index": descriptor.has_frame_index,
            "frame_index_count": index_count,
            "declared_frame_count": declared_count,
            "video_probe": video_probe,
        },
        "source": str(root),
    }


def qualify_sessions(
    sessions_root: Path,
    profile_root: Path,
    *,
    evidence_root: Path | None = None,
    session_paths: Iterable[Path | str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Qualify sessions in deterministic session-id order without modifying sources."""
    if session_paths is None:
        descriptors = inspect_sessions(sessions_root)
    else:
        descriptors = tuple(inspect_session(path) for path in session_paths)
    records = [qualify_session(item, profile_root, evidence_root=evidence_root) for item in descriptors]
    return tuple(sorted(records, key=lambda item: str(item.get("session_id", "")).lower()))


__all__ = ["CLASSIFICATIONS", "qualify_session", "qualify_sessions"]
