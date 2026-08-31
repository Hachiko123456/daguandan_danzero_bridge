from __future__ import annotations

"""Evidence-based root-cause classification for offline support replay."""

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


ROOT_CAUSE_LAYERS: tuple[str, ...] = (
    "bundle_integrity",
    "capture_raw",
    "standardization_roi",
    "resource_hash",
    "matcher_threshold_margin",
    "multi_frame_stability",
    "runtime_native_build",
)
CONFIDENCE_LEVELS = ("PROVEN", "HIGH_CONFIDENCE", "SUSPECTED", "UNKNOWN")


def diagnose_root_cause(
    *,
    support_verified: bool,
    opening_evidence: Mapping[str, object] | None,
    outcomes: Sequence[Mapping[str, object]],
    truth: Mapping[str, object] | None,
    local_resource_identity: Mapping[str, object] | None,
    support_build_id: str | None,
    runner_build_id: str | None,
    standardization_equal: bool | None = None,
    roi_equal: bool | None = None,
    child_probe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return one primary layer and a complete seven-layer audit trail."""

    layers = {name: _layer(name) for name in ROOT_CAUSE_LAYERS}
    if not support_verified:
        layers["bundle_integrity"] = _layer(
            "bundle_integrity",
            status="FAIL",
            confidence="PROVEN",
            finding="support manifest hash/shape verification failed",
        )
        return _conclusion(layers, "bundle_integrity")
    layers["bundle_integrity"] = _layer(
        "bundle_integrity",
        status="PASS",
        confidence="PROVEN",
        finding="every declared support payload hash and size verified",
    )

    frames = _frames(opening_evidence)
    black = [
        item
        for item in frames
        if _number(_nested(item, "capture", "pixel_max"), default=255.0) <= 1.0
    ]
    backends = sorted(
        {
            str(_nested(item, "capture", "backend") or "unknown")
            for item in frames
        }
    )
    if black:
        layers["capture_raw"] = _layer(
            "capture_raw",
            status="FAIL",
            confidence="PROVEN",
            finding="one or more captured frames are effectively black",
            evidence={"black_frame_count": len(black), "backends": backends},
        )
    else:
        layers["capture_raw"] = _layer(
            "capture_raw",
            status="PASS" if frames else "UNKNOWN",
            confidence="PROVEN" if frames else "UNKNOWN",
            finding="captured frame metadata is non-black" if frames else "no raw capture evidence",
            evidence={"frame_count": len(frames), "backends": backends},
        )

    if standardization_equal is False or roi_equal is False:
        layers["standardization_roi"] = _layer(
            "standardization_roi",
            status="FAIL",
            confidence="PROVEN",
            finding="saved raw pixels do not reproduce standardized/ROI pixels",
            evidence={
                "standardization_equal": standardization_equal,
                "roi_equal": roi_equal,
            },
        )
    elif standardization_equal is True and roi_equal is not False:
        layers["standardization_roi"] = _layer(
            "standardization_roi",
            status="PASS",
            confidence="PROVEN",
            finding="saved standardization and ROI pixels reproduce exactly",
            evidence={
                "standardization_equal": standardization_equal,
                "roi_equal": roi_equal,
            },
        )

    remote_resources = (
        opening_evidence.get("resource_identity")
        if isinstance(opening_evidence, Mapping)
        else None
    )
    remote_hash = _mapping_text(remote_resources, "sha256")
    local_hash = _mapping_text(local_resource_identity, "sha256")
    if remote_hash and local_hash and remote_hash != local_hash:
        layers["resource_hash"] = _layer(
            "resource_hash",
            status="FAIL",
            confidence="PROVEN",
            finding="remote and local recognition resources have different hashes",
            evidence={"remote_sha256": remote_hash, "local_sha256": local_hash},
        )
    elif remote_hash and local_hash:
        layers["resource_hash"] = _layer(
            "resource_hash",
            status="PASS",
            confidence="PROVEN",
            finding="remote and local recognition resource hashes match",
            evidence={"sha256": remote_hash},
        )

    expected_level = str(truth.get("expected_level") or "") if truth else ""
    all_candidates = [
        candidate
        for outcome in outcomes
        for candidate in _candidate_records(outcome)
        if str(candidate.get("field", "")) == "level_rank"
    ]
    expected_candidates = [
        item for item in all_candidates if str(item.get("label", "")) == expected_level
    ]
    if expected_level and expected_candidates:
        best = max(expected_candidates, key=lambda item: _number(item.get("score")))
        score = _number(best.get("score"))
        threshold = _number(best.get("threshold"))
        other_scores = sorted(
            (
                _number(item.get("score"))
                for item in all_candidates
                if str(item.get("label", "")) != expected_level
            ),
            reverse=True,
        )
        runner_up = other_scores[0] if other_scores else None
        margin = score - runner_up if runner_up is not None else None
        accepted = best.get("accepted") is True
        rejection_reason = best.get("rejection_reason")
        if score < threshold or not accepted:
            layers["matcher_threshold_margin"] = _layer(
                "matcher_threshold_margin",
                status="FAIL",
                confidence="HIGH_CONFIDENCE",
                finding=(
                    "truth candidate was rejected by the production matcher"
                    if score >= threshold
                    else "truth candidate is consistently below the production threshold"
                ),
                evidence={
                    "expected_level": expected_level,
                    "score": score,
                    "threshold": threshold,
                    "runner_up": runner_up,
                    "margin": margin,
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                },
            )
        else:
            layers["matcher_threshold_margin"] = _layer(
                "matcher_threshold_margin",
                status="PASS",
                confidence="PROVEN",
                finding="truth candidate clears the production threshold",
                evidence={
                    "expected_level": expected_level,
                    "score": score,
                    "threshold": threshold,
                    "runner_up": runner_up,
                    "margin": margin,
                },
            )

    consensus_failures = [
        item.get("multi_frame_gate")
        for item in outcomes
        if isinstance(item.get("multi_frame_gate"), Mapping)
        and str(item["multi_frame_gate"].get("status")) == "FAIL"
    ]
    fingerprints = [str(item.get("output_fingerprint", "")) for item in outcomes]
    unique = sorted(set(value for value in fingerprints if value))
    if consensus_failures:
        layers["multi_frame_stability"] = _layer(
            "multi_frame_stability",
            status="FAIL",
            confidence="PROVEN",
            finding="the production opening consensus detected an oscillating frame sequence",
            evidence={
                "failed_runs": len(consensus_failures),
                "reasons": sorted(
                    {
                        str(item.get("reason") or "unknown")
                        for item in consensus_failures
                        if isinstance(item, Mapping)
                    }
                ),
            },
        )
    elif len(unique) > 1:
        layers["multi_frame_stability"] = _layer(
            "multi_frame_stability",
            status="FAIL",
            confidence="HIGH_CONFIDENCE",
            finding="identical replay inputs produced more than one normalized outcome",
            evidence={
                "unique_outcome_count": len(unique),
                "distribution": dict(Counter(fingerprints)),
            },
        )
    elif unique:
        layers["multi_frame_stability"] = _layer(
            "multi_frame_stability",
            status="PASS",
            confidence="PROVEN",
            finding="all repeated runs produced one normalized outcome",
            evidence={"unique_outcome_count": 1},
        )

    child_status = str(child_probe.get("status", "")) if child_probe else ""
    child_matches = child_probe.get("matches_same_process") if child_probe else None
    build_differs = bool(support_build_id and runner_build_id and support_build_id != runner_build_id)
    if child_status == "PASS" and child_matches is False:
        layers["runtime_native_build"] = _layer(
            "runtime_native_build",
            status="FAIL",
            confidence="HIGH_CONFIDENCE",
            finding="fresh deterministic child disagrees with the same-process probe",
            evidence={"support_build_id": support_build_id, "runner_build_id": runner_build_id},
        )
    elif child_status == "FAIL":
        layers["runtime_native_build"] = _layer(
            "runtime_native_build",
            status="UNKNOWN",
            confidence="SUSPECTED",
            finding="fresh deterministic child probe could not complete",
            evidence={
                "reason": child_probe.get("reason") if child_probe else None,
                "exit_code": child_probe.get("exit_code") if child_probe else None,
                "support_build_id": support_build_id,
                "runner_build_id": runner_build_id,
            },
        )
    elif build_differs:
        layers["runtime_native_build"] = _layer(
            "runtime_native_build",
            status="UNKNOWN",
            confidence="SUSPECTED",
            finding="support and runner build identities differ",
            evidence={"support_build_id": support_build_id, "runner_build_id": runner_build_id},
        )
    elif support_build_id and runner_build_id:
        layers["runtime_native_build"] = _layer(
            "runtime_native_build",
            status="PASS",
            confidence="PROVEN",
            finding="support and runner build identities match",
            evidence={"build_id": support_build_id},
        )

    priority = (
        "capture_raw",
        "standardization_roi",
        "resource_hash",
        "matcher_threshold_margin",
        "multi_frame_stability",
        "runtime_native_build",
    )
    primary = next((name for name in priority if layers[name]["status"] == "FAIL"), None)
    if primary is None:
        primary = next(
            (name for name in priority if layers[name]["confidence"] == "SUSPECTED"),
            "runtime_native_build",
        )
    return _conclusion(layers, primary)


def _layer(
    name: str,
    *,
    status: str = "UNKNOWN",
    confidence: str = "UNKNOWN",
    finding: str = "insufficient evidence",
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(f"invalid root-cause confidence: {confidence}")
    return {
        "layer": name,
        "status": status,
        "confidence": confidence,
        "finding": finding,
        "evidence": dict(evidence or {}),
    }


def _conclusion(layers: Mapping[str, Mapping[str, object]], primary: str) -> dict[str, object]:
    item = layers[primary]
    return {
        "primary_layer": primary,
        "confidence": item.get("confidence", "UNKNOWN"),
        "finding": item.get("finding", "insufficient evidence"),
        "layers": [dict(layers[name]) for name in ROOT_CAUSE_LAYERS],
    }


def _frames(value: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    raw = value.get("frames") if isinstance(value, Mapping) else None
    return [item for item in raw or [] if isinstance(item, Mapping)]


def _nested(value: Mapping[str, object], *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping_text(value: object, key: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get(key)
    return str(raw) if raw not in {None, ""} else None


def _candidate_records(outcome: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    raw = outcome.get("candidate_vector")
    return (item for item in raw or [] if isinstance(item, Mapping))


def _number(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "CONFIDENCE_LEVELS",
    "ROOT_CAUSE_LAYERS",
    "diagnose_root_cause",
]
