from __future__ import annotations

"""Safe assembly of one support bundle from several runtime roots.

The low-level exporter intentionally accepts one root-relative allowlist.  A
real diagnostic run stores evidence below the writable diagnostics root while
the immutable build manifest lives beside the executable and a sealed health
report may live below a session root.  This module copies only fixed, bounded
inputs into a private staging directory, then delegates privacy filtering and
atomic ZIP publication to :mod:`daguandan_bridge.support_bundle`.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterable, Mapping

from .support_bundle import (
    RedactionContext,
    SupportBundleError,
    SupportBundleResult,
    SupportBundleSources,
    SupportImageSource,
    UnsafeSupportSourceError,
    export_support_bundle,
)


_MAX_STAGE_FILE_BYTES = 32 * 1024 * 1024
_MAX_STAGE_TOTAL_BYTES = 256 * 1024 * 1024
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_FRAME_IMAGE = re.compile(r"(raw_client|standardized)_(\d+)\.(png|jpe?g)", re.IGNORECASE)
_ROI_IMAGE = re.compile(
    r"([A-Za-z][A-Za-z0-9_.-]{0,63})_(\d+)\.(png|jpe?g)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SupportExportRequest:
    destination: Path
    diagnostics_run_directory: Path | None = None
    # A launcher may run a fresh doctor while exporting a prior failed run.
    # Keeping the roots explicit lets the bundle contain both without writing
    # new files into the original evidence directory.
    evidence_run_directory: Path | None = None
    bundle_root: Path | None = None
    session_directory: Path | None = None
    opening_incident_directory: Path | None = None
    include_frames: bool = False
    include_roi: bool = False
    include_recognition_trace: bool = False
    redaction: RedactionContext | None = None


@dataclass(frozen=True)
class CollectedSupportBundleResult:
    bundle: SupportBundleResult
    selected_opening_incident: Path | None
    selected_session_incident: Path | None


class SupportExportService:
    """Collect an allowlisted support bundle without traversing arbitrary data."""

    def export(self, request: SupportExportRequest) -> CollectedSupportBundleResult:
        destination = Path(request.destination)
        if destination.suffix.lower() != ".zip":
            raise SupportBundleError("support bundle destination must end with .zip")
        run_root = _optional_root(request.diagnostics_run_directory, "diagnostics run")
        evidence_run_root = _optional_root(
            request.evidence_run_directory,
            "diagnostics evidence run",
        )
        bundle_root = _optional_root(request.bundle_root, "bundle")
        session_root = _optional_root(request.session_directory, "session")
        opening_incident = _select_opening_incident(
            evidence_run_root or run_root,
            request.opening_incident_directory,
        )
        session_incident = (
            None
            if opening_incident is not None or session_root is None
            else _select_latest_incident(session_root / "incidents")
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="daguandan-support-",
            dir=destination.parent,
            ignore_cleanup_errors=True,
        ) as temporary:
            staging = Path(temporary)
            collector = _StagingCollector(staging)
            staged: dict[str, Path] = {}

            if run_root is not None:
                _collect_optional(
                    collector,
                    staged,
                    "startup_log",
                    run_root,
                    Path("startup.log"),
                    Path("startup/startup.log"),
                )
                _collect_optional(
                    collector,
                    staged,
                    "startup_events",
                    run_root,
                    Path("startup.jsonl"),
                    Path("startup/startup.jsonl"),
                )
                _collect_optional(
                    collector,
                    staged,
                    "exceptions_log",
                    run_root,
                    Path("exceptions.log"),
                    Path("startup/exceptions.log"),
                )
                _collect_first_optional(
                    collector,
                    staged,
                    "runtime_log",
                    run_root,
                    (Path("runtime.log"), Path("app.log")),
                    Path("runtime/runtime.log"),
                )
                _collect_optional(
                    collector,
                    staged,
                    "runtime_identity",
                    run_root,
                    Path("runtime_identity.json"),
                    Path("runtime/runtime_identity.json"),
                )
                _collect_optional(
                    collector,
                    staged,
                    "doctor",
                    run_root,
                    Path("doctor.json"),
                    Path("doctor/doctor.json"),
                )

            if bundle_root is not None:
                _collect_optional(
                    collector,
                    staged,
                    "build_manifest",
                    bundle_root,
                    Path("build_manifest.json"),
                    Path("build/build_manifest.json"),
                )

            incident_root = opening_incident or session_incident
            if incident_root is not None:
                _collect_optional(
                    collector,
                    staged,
                    "incident",
                    incident_root,
                    Path("incident.json"),
                    Path("incident/incident.json"),
                )

            evidence_document: dict[str, object] | None = None
            if opening_incident is not None:
                _collect_optional(
                    collector,
                    staged,
                    "opening_evidence",
                    opening_incident,
                    Path("opening_evidence.json"),
                    Path("evidence/opening_evidence.json"),
                )
                _collect_optional(
                    collector,
                    staged,
                    "repro_manifest",
                    opening_incident,
                    Path("repro.json"),
                    Path("repro/repro.json"),
                )
                if "opening_evidence" in staged:
                    evidence_document = _read_json_bounded(
                        staging / staged["opening_evidence"],
                        root=staging,
                    )
                _collect_first_optional(
                    collector,
                    staged,
                    "frame_index",
                    opening_incident,
                    (Path("frame_index.jsonl"), Path("frames/frame_index.jsonl")),
                    Path("evidence/frame_index.jsonl"),
                )
                if "frame_index" not in staged and evidence_document is not None:
                    generated = Path("evidence/frame_index.jsonl")
                    _write_generated_frame_index(
                        staging / generated,
                        evidence_document,
                    )
                    if (staging / generated).is_file():
                        collector.account_generated(staging / generated)
                        staged["frame_index"] = generated

            if session_root is not None:
                _collect_first_optional(
                    collector,
                    staged,
                    "health_audit",
                    session_root,
                    (Path("health_audit.json"), Path("health/session_health.json")),
                    Path("health/health_audit.json"),
                )
                if "frame_index" not in staged:
                    _collect_optional(
                        collector,
                        staged,
                        "frame_index",
                        session_root,
                        Path("video/frame_index.jsonl"),
                        Path("evidence/frame_index.jsonl"),
                    )

            if request.include_recognition_trace:
                trace_root = opening_incident or session_root
                if trace_root is not None:
                    _collect_optional(
                        collector,
                        staged,
                        "recognition_trace",
                        trace_root,
                        Path("recognition_trace.jsonl"),
                        Path("trace/recognition_trace.jsonl"),
                    )

            images: tuple[SupportImageSource, ...] = ()
            if opening_incident is not None and (
                request.include_frames or request.include_roi
            ):
                images = _collect_opening_images(
                    collector,
                    opening_incident,
                    evidence_document,
                    include_frames=request.include_frames,
                    include_roi=request.include_roi,
                )

            sources = SupportBundleSources(
                root=staging,
                startup_log=staged.get("startup_log"),
                startup_events=staged.get("startup_events"),
                exceptions_log=staged.get("exceptions_log"),
                runtime_log=staged.get("runtime_log"),
                runtime_identity=staged.get("runtime_identity"),
                doctor=staged.get("doctor"),
                build_manifest=staged.get("build_manifest"),
                incident=staged.get("incident"),
                opening_evidence=staged.get("opening_evidence"),
                repro_manifest=staged.get("repro_manifest"),
                frame_index=staged.get("frame_index"),
                health_audit=staged.get("health_audit"),
                recognition_trace=staged.get("recognition_trace"),
                images=images,
            )
            bundle = export_support_bundle(
                destination,
                sources,
                include_frames=request.include_frames,
                include_roi=request.include_roi,
                include_recognition_trace=request.include_recognition_trace,
                redaction=request.redaction,
            )
        return CollectedSupportBundleResult(
            bundle=bundle,
            selected_opening_incident=opening_incident,
            selected_session_incident=session_incident,
        )


def export_collected_support_bundle(
    request: SupportExportRequest,
) -> CollectedSupportBundleResult:
    return SupportExportService().export(request)


class _StagingCollector:
    def __init__(self, root: Path) -> None:
        self.root = _validated_root(root, "support staging")
        self.used = 0

    def copy(self, source_root: Path, relative: Path, destination: Path) -> Path | None:
        source = _safe_source_file(source_root, relative)
        if source is None:
            return None
        size = int(source.stat().st_size)
        self._ensure_budget(size, relative)
        target = _safe_stage_destination(self.root, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        try:
            with source.open("rb") as reader, target.open("xb") as writer:
                while True:
                    chunk = reader.read(min(1024 * 1024, _MAX_STAGE_FILE_BYTES + 1 - copied))
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > _MAX_STAGE_FILE_BYTES or self.used + copied > _MAX_STAGE_TOTAL_BYTES:
                        raise SupportBundleError(
                            f"support staging budget exceeded while copying: {relative}"
                        )
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        if copied != size:
            target.unlink(missing_ok=True)
            raise SupportBundleError(f"support source changed while staging: {relative}")
        self.used += copied
        return target.relative_to(self.root)

    def account_generated(self, path: Path) -> None:
        path = Path(path)
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise UnsafeSupportSourceError("generated support evidence escaped staging") from exc
        size = int(resolved.stat().st_size)
        self._ensure_budget(size, resolved.name)
        self.used += size

    def _ensure_budget(self, size: int, label: object) -> None:
        if size < 0 or size > _MAX_STAGE_FILE_BYTES:
            raise SupportBundleError(
                f"support staging source exceeds {_MAX_STAGE_FILE_BYTES} bytes: {label}"
            )
        if self.used + size > _MAX_STAGE_TOTAL_BYTES:
            raise SupportBundleError(
                f"support staging payload exceeds {_MAX_STAGE_TOTAL_BYTES} bytes: {label}"
            )


def _collect_optional(
    collector: _StagingCollector,
    staged: dict[str, Path],
    capability: str,
    root: Path,
    source: Path,
    destination: Path,
) -> None:
    relative = collector.copy(root, source, destination)
    if relative is not None:
        staged[capability] = relative


def _collect_first_optional(
    collector: _StagingCollector,
    staged: dict[str, Path],
    capability: str,
    root: Path,
    candidates: Iterable[Path],
    destination: Path,
) -> None:
    for source in candidates:
        relative = collector.copy(root, source, destination)
        if relative is not None:
            staged[capability] = relative
            return


def _collect_opening_images(
    collector: _StagingCollector,
    incident_root: Path,
    evidence: Mapping[str, object] | None,
    *,
    include_frames: bool,
    include_roi: bool,
) -> tuple[SupportImageSource, ...]:
    monotonic_by_seq = _monotonic_by_frame_seq(evidence)
    candidates: list[tuple[Path, int, int, str, str | None]] = []
    if include_frames:
        frames_root = incident_root / "frames"
        for path in _safe_image_files(frames_root):
            match = _FRAME_IMAGE.fullmatch(path.name)
            if match is None:
                continue
            kind = match.group(1).lower()
            seq = int(match.group(2))
            candidates.append(
                (
                    path,
                    seq,
                    monotonic_by_seq.get(seq, 0),
                    kind,
                    None,
                )
            )
    if include_roi:
        roi_root = incident_root / "roi"
        for path in _safe_image_files(roi_root):
            match = _ROI_IMAGE.fullmatch(path.name)
            if match is None:
                continue
            field = match.group(1)
            seq = int(match.group(2))
            candidates.append(
                (
                    path,
                    seq,
                    monotonic_by_seq.get(seq, 0),
                    "roi",
                    field,
                )
            )
    images: list[SupportImageSource] = []
    for index, (source, seq, monotonic_ms, kind, field) in enumerate(
        sorted(candidates, key=lambda item: (item[1], item[3], item[4] or "", item[0].name)),
        start=1,
    ):
        destination = Path("sensitive") / f"image_{index:06d}{source.suffix.lower()}"
        relative_source = source.relative_to(incident_root)
        staged_path = collector.copy(incident_root, relative_source, destination)
        if staged_path is None:
            raise SupportBundleError(f"support image disappeared while staging: {relative_source}")
        images.append(
            SupportImageSource(
                path=staged_path,
                frame_seq=seq,
                monotonic_ms=monotonic_ms,
                kind=kind,
                field=field,
            )
        )
    return tuple(images)


def _monotonic_by_frame_seq(
    evidence: Mapping[str, object] | None,
) -> dict[int, int]:
    if not isinstance(evidence, Mapping):
        return {}
    frames = evidence.get("frames")
    if not isinstance(frames, list):
        return {}
    result: dict[int, int] = {}
    for raw in frames:
        if not isinstance(raw, Mapping):
            continue
        try:
            seq = int(raw.get("seq", raw.get("frame_seq")))
            monotonic_ms = int(raw.get("monotonic_ms"))
        except (TypeError, ValueError):
            continue
        if seq >= 0 and monotonic_ms >= 0:
            result[seq] = monotonic_ms
    return result


def _write_generated_frame_index(path: Path, evidence: Mapping[str, object]) -> None:
    frames = evidence.get("frames")
    if not isinstance(frames, list):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for raw in frames:
            if not isinstance(raw, Mapping):
                continue
            record = {
                "schema": "guandan.support-frame-index-entry/1",
                "frame_seq": raw.get("seq", raw.get("frame_seq")),
                "monotonic_ms": raw.get("monotonic_ms"),
                "wall_time": raw.get("wall_time"),
                "capture": raw.get("capture"),
            }
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    default=str,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_bounded(path: Path, *, root: Path) -> dict[str, object]:
    relative = path.relative_to(root)
    source = _safe_source_file(root, relative)
    if source is None:
        return {}
    size = int(source.stat().st_size)
    if size > _MAX_STAGE_FILE_BYTES:
        raise SupportBundleError(f"support JSON exceeds safe limit: {relative}")
    with source.open("rb") as handle:
        raw = handle.read(size + 1)
    if len(raw) != size:
        raise SupportBundleError(f"support JSON changed while reading: {relative}")
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupportBundleError(f"support JSON is invalid: {relative}") from exc
    return dict(document) if isinstance(document, Mapping) else {}


def _select_opening_incident(
    run_root: Path | None,
    explicit: Path | None,
) -> Path | None:
    if explicit is not None and run_root is None:
        raise SupportBundleError(
            "an explicit opening incident requires a diagnostics run directory"
        )
    if run_root is None:
        return None
    incidents_root = run_root / "opening" / "incidents"
    if explicit is None:
        return _select_latest_incident(incidents_root)
    validated_incidents = _optional_root(incidents_root, "opening incidents")
    if validated_incidents is None:
        raise SupportBundleError("opening incidents directory does not exist")
    candidate = Path(explicit)
    candidate = (
        candidate
        if candidate.is_absolute()
        else validated_incidents / candidate
    )
    selected = _validated_root(candidate, "opening incident")
    try:
        relative = selected.relative_to(validated_incidents)
    except ValueError as exc:
        raise UnsafeSupportSourceError(
            "explicit opening incident is outside the diagnostics run"
        ) from exc
    if len(relative.parts) != 1:
        raise UnsafeSupportSourceError(
            "explicit opening incident must be a direct incident directory"
        )
    return selected


def _select_latest_incident(incidents_root: Path) -> Path | None:
    if not incidents_root.exists():
        return None
    root = _validated_root(incidents_root, "incident collection")
    candidates: list[tuple[int, int, str, Path]] = []
    for path in root.iterdir():
        if path.name.startswith("."):
            continue
        if _is_link_or_reparse(path):
            raise UnsafeSupportSourceError("incident collection contains a reparse point")
        if not path.is_dir():
            continue
        incident_path = _safe_source_file(path, Path("incident.json"))
        if incident_path is None:
            continue
        document = _read_json_bounded(incident_path, root=path)
        try:
            monotonic_ms = int(document.get("monotonic_ms", document.get("trigger_ms", -1)))
        except (TypeError, ValueError):
            monotonic_ms = -1
        candidates.append(
            (monotonic_ms, int(incident_path.stat().st_mtime_ns), path.name, path)
        )
    return max(candidates)[-1] if candidates else None


def _optional_root(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    return _validated_root(Path(path), label)


def _validated_root(path: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_dir():
        raise SupportBundleError(f"{label} root is not a directory: {candidate}")
    _assert_no_reparse_path_chain(candidate, label)
    return candidate.resolve(strict=True)


def _assert_no_reparse_path_chain(path: Path, label: str) -> None:
    current = Path(path).absolute()
    while True:
        if current.exists() and _is_link_or_reparse(current):
            raise UnsafeSupportSourceError(
                f"{label} path cannot traverse a symlink, junction, or reparse point"
            )
        parent = current.parent
        if parent == current:
            return
        current = parent


def _safe_source_file(root: Path, relative: Path) -> Path | None:
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise UnsafeSupportSourceError(f"unsafe support staging source: {relative}")
    if not relative.parts or any(":" in part for part in relative.parts):
        raise UnsafeSupportSourceError(f"invalid support staging source: {relative}")
    current = root
    for part in relative.parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise UnsafeSupportSourceError(
                f"support staging source traverses a reparse point: {relative}"
            )
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafeSupportSourceError(
            f"support staging source escaped its root: {relative}"
        ) from exc
    if not current.exists():
        return None
    if not current.is_file():
        raise UnsafeSupportSourceError(f"support staging source is not a file: {relative}")
    if int(current.stat().st_size) > _MAX_STAGE_FILE_BYTES:
        raise SupportBundleError(
            f"support staging source exceeds {_MAX_STAGE_FILE_BYTES} bytes: {relative}"
        )
    return current


def _safe_image_files(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    validated = _validated_root(root, "support image")
    files: list[Path] = []
    pending = [validated]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if _is_link_or_reparse(path):
                raise UnsafeSupportSourceError(
                    "support image tree contains a symlink, junction, or reparse point"
                )
            if path.is_dir():
                pending.append(path)
            elif path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(validated).as_posix()))


def _safe_stage_destination(root: Path, relative: Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise UnsafeSupportSourceError(f"unsafe support staging destination: {relative}")
    target = root.joinpath(relative)
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafeSupportSourceError(
            f"support staging destination escaped its root: {relative}"
        ) from exc
    return target


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if is_junction is not None and is_junction(path):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attributes & reparse_flag)
    except OSError:
        return False


__all__ = [
    "CollectedSupportBundleResult",
    "SupportExportRequest",
    "SupportExportService",
    "export_collected_support_bundle",
]
