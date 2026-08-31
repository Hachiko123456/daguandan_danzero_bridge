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
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
    _validated_image_info,
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
    if not isinstance(evidence, Mapping):
        raise SupportBundleError(
            "opening image export requires opening_evidence.json artifact declarations"
        )
    frames = evidence.get("frames")
    if not isinstance(frames, list):
        raise SupportBundleError("opening evidence frame artifact list is invalid")
    candidates: list[
        tuple[Path, int, int, str, str | None, str, str, str, str | None]
    ] = []
    declared_selected: set[str] = set()
    for raw_frame in frames:
        if not isinstance(raw_frame, Mapping):
            raise SupportBundleError("opening evidence frame record is invalid")
        try:
            seq = int(raw_frame.get("seq", raw_frame.get("frame_seq")))
            monotonic_ms = int(raw_frame.get("monotonic_ms"))
        except (TypeError, ValueError) as exc:
            raise SupportBundleError("opening evidence frame identity is invalid") from exc
        if seq < 0 or monotonic_ms < 0:
            raise SupportBundleError("opening evidence frame identity is negative")
        frame_id = str(raw_frame.get("frame_id") or "") or None
        artifacts = raw_frame.get("artifacts")
        if not isinstance(artifacts, list):
            raise SupportBundleError("opening evidence frame artifacts are missing")
        for raw_artifact in artifacts:
            if not isinstance(raw_artifact, Mapping):
                raise SupportBundleError("opening evidence artifact record is invalid")
            try:
                artifact_seq = int(raw_artifact.get("frame_seq"))
            except (TypeError, ValueError) as exc:
                raise SupportBundleError("opening artifact frame_seq is invalid") from exc
            if artifact_seq != seq:
                raise SupportBundleError("opening artifact frame_seq does not match its frame")
            artifact_frame_id = str(raw_artifact.get("frame_id") or "") or None
            if frame_id is not None and artifact_frame_id != frame_id:
                raise SupportBundleError("opening artifact frame_id does not match its frame")
            kind = str(raw_artifact.get("kind") or "").strip().lower()
            field = (
                str(raw_artifact.get("field")).strip()
                if raw_artifact.get("field") not in {None, ""}
                else None
            )
            if kind not in {"raw_client", "standardized", "roi"}:
                raise SupportBundleError("opening artifact kind is invalid")
            if (kind == "roi") != (field is not None):
                raise SupportBundleError("opening artifact field/kind declaration is invalid")
            selected = (kind in {"raw_client", "standardized"} and include_frames) or (
                kind == "roi" and include_roi
            )
            if not selected:
                continue
            portable = _portable_incident_artifact_path(raw_artifact.get("path"))
            expected_root = "roi" if kind == "roi" else "frames"
            if not portable.parts or portable.parts[0] != expected_root:
                raise SupportBundleError("opening artifact path does not match its kind")
            source = _safe_source_file(incident_root, Path(*portable.parts))
            if source is None:
                raise SupportBundleError(f"declared opening artifact is missing: {portable}")
            file_sha = str(raw_artifact.get("sha256") or "")
            pixel_sha = str(raw_artifact.get("pixel_sha256") or "")
            _verify_incident_artifact(
                source,
                raw_artifact,
                expected_file_sha=file_sha,
                expected_pixel_sha=pixel_sha,
            )
            identity = portable.as_posix().casefold()
            if identity in declared_selected:
                raise SupportBundleError(f"duplicate opening artifact path: {portable}")
            declared_selected.add(identity)
            candidates.append(
                (
                    source,
                    seq,
                    monotonic_ms,
                    kind,
                    field,
                    portable.as_posix(),
                    file_sha,
                    pixel_sha,
                    frame_id,
                )
            )
    _reject_unmanifested_opening_images(
        incident_root,
        declared_selected,
        include_frames=include_frames,
        include_roi=include_roi,
    )
    images: list[SupportImageSource] = []
    for index, (
        source,
        seq,
        monotonic_ms,
        kind,
        field,
        incident_path,
        file_sha,
        pixel_sha,
        frame_id,
    ) in enumerate(
        sorted(candidates, key=lambda item: (item[1], item[3], item[4] or "", item[5])),
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
                incident_path=incident_path,
                incident_sha256=file_sha,
                incident_pixel_sha256=pixel_sha,
                frame_id=frame_id,
            )
        )
    return tuple(images)


def _portable_incident_artifact_path(value: object) -> PurePosixPath:
    raw = str(value or "")
    if not raw or "\\" in raw or ":" in raw or raw.startswith("/"):
        raise SupportBundleError("opening artifact path is unsafe")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SupportBundleError("opening artifact path is unsafe")
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise SupportBundleError("opening artifact is not a supported image")
    return path


def _verify_incident_artifact(
    path: Path,
    artifact: Mapping[str, object],
    *,
    expected_file_sha: str,
    expected_pixel_sha: str,
) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", expected_file_sha) is None or re.fullmatch(
        r"[0-9a-f]{64}", expected_pixel_sha
    ) is None:
        raise SupportBundleError("opening artifact hashes are invalid")
    size = int(path.stat().st_size)
    try:
        declared_size = int(artifact.get("bytes"))
    except (TypeError, ValueError) as exc:
        raise SupportBundleError("opening artifact byte count is invalid") from exc
    if size != declared_size:
        raise SupportBundleError(f"opening artifact size changed: {path.name}")
    with path.open("rb") as handle:
        content = handle.read(_MAX_STAGE_FILE_BYTES + 1)
    if len(content) != size or len(content) > _MAX_STAGE_FILE_BYTES:
        raise SupportBundleError(f"opening artifact changed while reading: {path.name}")
    if hashlib.sha256(content).hexdigest() != expected_file_sha:
        raise SupportBundleError(f"opening artifact file hash changed: {path.name}")
    try:
        *_metadata, actual_pixel_sha = _validated_image_info(
            content,
            suffix=path.suffix.lower(),
            relative=Path(path.name),
        )
    except Exception as exc:
        raise SupportBundleError(f"opening artifact cannot be decoded: {path.name}") from exc
    if actual_pixel_sha != expected_pixel_sha:
        raise SupportBundleError(f"opening artifact pixel hash changed: {path.name}")


def _reject_unmanifested_opening_images(
    incident_root: Path,
    declared: set[str],
    *,
    include_frames: bool,
    include_roi: bool,
) -> None:
    roots = []
    if include_frames:
        roots.append(incident_root / "frames")
    if include_roi:
        roots.append(incident_root / "roi")
    actual = {
        path.relative_to(incident_root).as_posix().casefold()
        for root in roots
        for path in _safe_image_files(root)
    }
    extras = sorted(actual - declared)
    if extras:
        raise SupportBundleError(
            f"opening incident contains unmanifested image artifacts: {extras}"
        )


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
                "analysis": raw.get("analysis"),
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
