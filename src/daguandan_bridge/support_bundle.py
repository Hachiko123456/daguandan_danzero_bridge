from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import stat
import struct
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from uuid import uuid4


SUPPORT_BUNDLE_SCHEMA = "guandan.support-bundle/1"
SUPPORT_IMAGE_INDEX_SCHEMA = "guandan.support-image-index/1"

__all__ = [
    "RedactionContext",
    "SUPPORT_BUNDLE_SCHEMA",
    "SupportBundleError",
    "SupportBundleResult",
    "SupportBundleSources",
    "SupportImageSource",
    "SUPPORT_IMAGE_INDEX_SCHEMA",
    "UnsafeSupportSourceError",
    "export_support_bundle",
]

_MAX_TEXT_FILE_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_FILE_BYTES = 32 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
_MAX_IMAGE_PIXELS = 32_000_000
_IMAGE_FIELD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
_IMAGE_KINDS = frozenset({"raw_client", "standardized", "roi"})
_BANNED_SUFFIXES = {".ckpt", ".npz"}
_ENVIRONMENT_DUMP_NAMES = {
    ".env",
    "env.json",
    "env.txt",
    "environ.json",
    "environ.txt",
    "environment.json",
    "environment.txt",
}
_ENVIRONMENT_DUMP_KEYS = {
    "environment_variables",
    "process_environment",
    "os_environ",
    "os.environ",
}
_SENSITIVE_JSON_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|key|authorization|bearer|cookie|"
    r"password|passwd|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_KEY_VALUE_SECRET = re.compile(
    r"\b(api[_-]?key|access[_-]?key|authorization|password|passwd|secret|token)"
    r"(\s*[:=]\s*)(\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)",
    re.IGNORECASE,
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_KNOWN_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,})\b",
    re.IGNORECASE,
)
_QUOTED_WINDOWS_PATH = re.compile(
    r"(?P<quote>[\"'])(?:(?:[A-Za-z]:[\\/])|(?:[\\/]{2}[^\\/\r\n]+[\\/]))"
    r".*?(?P=quote)"
)
_UNQUOTED_WINDOWS_PATH_TO_EOL = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:(?:[A-Za-z]:[\\/])|(?:\\\\[^\\\r\n]+\\))"
    r"[^\r\n]*"
)


class SupportBundleError(RuntimeError):
    """The offline support package could not be created safely."""


class UnsafeSupportSourceError(SupportBundleError):
    """A requested input escaped the source allowlist or used a reparse point."""


@dataclass(frozen=True)
class SupportBundleSources:
    """Explicit, source-root-relative evidence selected for one support bundle.

    Paths in this object are *not* archive paths.  Every supplied path must be
    relative to ``root``.  The exporter maps them to fixed archive entry names,
    so a caller cannot smuggle an absolute or ``..`` path into the ZIP.
    """

    root: Path
    startup_log: Path | None = None
    runtime_identity: Path | None = None
    doctor: Path | None = None
    build_manifest: Path | None = None
    incident: Path | None = None
    recognition_trace: Path | None = None
    # Compatibility declarations retained in their original positional order
    # for callers of the phase-1A API.
    frames: tuple[Path, ...] = ()
    roi: tuple[Path, ...] = ()
    # Additive v1 evidence follows the frozen phase-1A fields above.
    startup_events: Path | None = None
    exceptions_log: Path | None = None
    runtime_log: Path | None = None
    opening_evidence: Path | None = None
    repro_manifest: Path | None = None
    frame_index: Path | None = None
    health_audit: Path | None = None
    images: tuple["SupportImageSource", ...] = ()


@dataclass(frozen=True)
class SupportImageSource:
    """One explicitly classified image selected for a support bundle.

    ``path`` remains root-relative and never controls its archive name.
    ``frame_seq`` and ``monotonic_ms`` correlate the generated archive entry
    with opening evidence without exposing a machine-local source filename.
    """

    path: Path
    frame_seq: int
    monotonic_ms: int
    kind: str
    field: str | None = None
    # Immutable identity from the selected incident's artifact manifest.
    # Generic callers may omit it; SupportExportService always supplies it.
    incident_path: str | None = None
    incident_sha256: str | None = None
    incident_pixel_sha256: str | None = None
    frame_id: str | None = None


@dataclass(frozen=True)
class RedactionContext:
    """Additional machine-local values that must not leave the support ZIP."""

    usernames: tuple[str, ...] = ()
    computer_names: tuple[str, ...] = ()
    extra_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupportBundleResult:
    destination: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _Payload:
    capability: str
    archive_path: str
    classification: str
    content: bytes
    redactions: int = 0


@dataclass(frozen=True)
class _NormalizedImageSource:
    path: Path
    frame_seq: int
    monotonic_ms: int
    kind: str
    field: str | None
    archive_stem: str
    incident_path: str | None
    incident_sha256: str | None
    incident_pixel_sha256: str | None
    frame_id: str | None


@dataclass
class _PayloadBudget:
    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def ensure_source_fits(self, size: int, relative: Path) -> None:
        if size > self.remaining:
            raise SupportBundleError(
                f"support bundle payload exceeds {self.limit} bytes before reading: {relative}"
            )

    def add_payload(self, size: int, relative: Path) -> None:
        if size > self.remaining:
            raise SupportBundleError(
                f"support bundle payload exceeds {self.limit} bytes after sanitizing: {relative}"
            )
        self.used += size


def export_support_bundle(
    destination: Path,
    sources: SupportBundleSources,
    *,
    include_frames: bool = False,
    include_roi: bool = False,
    include_recognition_trace: bool = False,
    redaction: RedactionContext | None = None,
) -> SupportBundleResult:
    """Create one sanitized, atomic, offline support ZIP.

    The three potentially large or sensitive capabilities are deliberately
    opt-in.  Merely supplying frame, ROI, or recognition-trace paths does not
    include them.
    """

    destination = Path(destination)
    if destination.suffix.lower() != ".zip":
        raise SupportBundleError("support bundle destination must end with .zip")
    source_root = _validated_source_root(sources.root)
    # Validate every declaration even when a sensitive capability is disabled.
    # Disabled inputs are never read, but accepting an unsafe declaration would
    # make the API's root-relative contract depend on an unrelated toggle.
    for declared in (
        sources.startup_log,
        sources.startup_events,
        sources.exceptions_log,
        sources.runtime_log,
        sources.runtime_identity,
        sources.doctor,
        sources.build_manifest,
        sources.incident,
        sources.opening_evidence,
        sources.repro_manifest,
        sources.frame_index,
        sources.health_audit,
        sources.recognition_trace,
        *(image.path for image in sources.images),
        *sources.frames,
        *sources.roi,
    ):
        if declared is not None:
            _resolve_source(source_root, declared)
    image_sources = _normalized_image_sources(sources)
    context = _effective_redaction_context(redaction)
    payloads: list[_Payload] = []
    budget = _PayloadBudget(_MAX_PAYLOAD_BYTES)
    missing: list[dict[str, str]] = []
    capabilities: dict[str, bool] = {
        "startup_log": False,
        "startup_events": False,
        "exceptions_log": False,
        "runtime_log": False,
        "runtime_identity": False,
        "doctor": False,
        "build_manifest": False,
        "incident": False,
        "opening_evidence": False,
        "repro_manifest": False,
        "frame_index": False,
        "health_audit": False,
        "frames": False,
        "roi": False,
        "image_index": False,
        "recognition_trace": False,
    }

    fixed_text_sources = (
        ("startup_log", sources.startup_log, "startup/startup.log", "sanitized-log"),
        (
            "startup_events",
            sources.startup_events,
            "startup/startup.jsonl",
            "sanitized-jsonl",
        ),
        (
            "exceptions_log",
            sources.exceptions_log,
            "startup/exceptions.log",
            "sanitized-log",
        ),
        ("runtime_log", sources.runtime_log, "runtime/runtime.log", "sanitized-log"),
        (
            "runtime_identity",
            sources.runtime_identity,
            "runtime/runtime_identity.json",
            "sanitized-json",
        ),
        ("doctor", sources.doctor, "doctor/doctor.json", "sanitized-json"),
        (
            "build_manifest",
            sources.build_manifest,
            "build/build_manifest.json",
            "sanitized-json",
        ),
        ("incident", sources.incident, "incident/incident.json", "sanitized-json"),
        (
            "opening_evidence",
            sources.opening_evidence,
            "evidence/opening_evidence.json",
            "sanitized-json",
        ),
        (
            "repro_manifest",
            sources.repro_manifest,
            "repro/repro.json",
            "sanitized-json",
        ),
        (
            "frame_index",
            sources.frame_index,
            "evidence/frame_index.jsonl",
            "sanitized-jsonl",
        ),
        (
            "health_audit",
            sources.health_audit,
            "health/health_audit.json",
            "sanitized-json",
        ),
    )
    for capability, relative, archive_path, classification in fixed_text_sources:
        payload = _text_payload(
            source_root,
            relative,
            capability=capability,
            archive_path=archive_path,
            classification=classification,
            context=context,
            missing=missing,
            budget=budget,
        )
        if payload is not None:
            payloads.append(payload)
            capabilities[capability] = True

    if include_recognition_trace:
        trace = _text_payload(
            source_root,
            sources.recognition_trace,
            capability="recognition_trace",
            archive_path="trace/recognition_trace.jsonl",
            classification="sanitized-jsonl",
            context=context,
            missing=missing,
            budget=budget,
        )
        if trace is not None:
            payloads.append(trace)
            capabilities["recognition_trace"] = True
    else:
        missing.append({"capability": "recognition_trace", "reason": "disabled"})

    image_index_entries = _append_structured_image_payloads(
        payloads,
        missing,
        capabilities,
        source_root=source_root,
        images=image_sources,
        include_frames=include_frames,
        include_roi=include_roi,
        budget=budget,
    )
    if image_index_entries:
        index_content = _json_bytes(
            {
                "schema": SUPPORT_IMAGE_INDEX_SCHEMA,
                "entries": image_index_entries,
            }
        )
        index_relative = Path("generated/image_index.json")
        budget.add_payload(len(index_content), index_relative)
        payloads.append(
            _Payload(
                capability="image_index",
                archive_path="evidence/image_index.json",
                classification="generated-json",
                content=index_content,
            )
        )
        capabilities["image_index"] = True
    else:
        image_reason = (
            "disabled"
            if not include_frames and not include_roi
            else "not_provided"
        )
        missing.append({"capability": "image_index", "reason": image_reason})

    total_payload_bytes = sum(len(payload.content) for payload in payloads)
    if total_payload_bytes != budget.used:
        raise SupportBundleError("support bundle payload budget accounting mismatch")

    file_records = [
        {
            "path": payload.archive_path,
            "size": len(payload.content),
            "sha256": hashlib.sha256(payload.content).hexdigest(),
            "classification": payload.classification,
            "capability": payload.capability,
        }
        for payload in sorted(payloads, key=lambda item: item.archive_path)
    ]
    manifest: dict[str, Any] = {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "support_id": f"SUP-{uuid4().hex}",
        "created_at": datetime.now(UTC).isoformat(),
        "build_id": _build_id(payloads),
        "capabilities": capabilities,
        "missing": sorted(missing, key=lambda item: (item["capability"], item["reason"])),
        "files": file_records,
        "privacy": {
            "text_sanitized": True,
            "redaction_count": sum(payload.redactions for payload in payloads),
            "contains_sensitive_images": bool(
                capabilities["frames"] or capabilities["roi"]
            ),
            "images_require_explicit_opt_in": True,
        },
        "manifest_entry": {
            "path": "support_manifest.json",
            "classification": "manifest",
            "sha256": None,
            "note": "The manifest cannot include a stable hash of itself.",
        },
    }
    manifest_bytes = _json_bytes(manifest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for payload in sorted(payloads, key=lambda item: item.archive_path):
                _write_archive_entry(archive, payload.archive_path, payload.content)
            _write_archive_entry(archive, "support_manifest.json", manifest_bytes)
            archive.comment = SUPPORT_BUNDLE_SCHEMA.encode("ascii")
        # Windows rejects fsync on a descriptor opened read-only.  Open the
        # completed archive read/write without modifying it so publication is
        # durable on both Windows and POSIX.
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        with zipfile.ZipFile(temporary) as verification:
            corrupt_entry = verification.testzip()
            if corrupt_entry is not None:
                raise SupportBundleError(
                    f"support archive verification failed: {corrupt_entry}"
                )
        for attempt in range(20):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
    return SupportBundleResult(destination=destination, manifest=manifest)


def _validated_source_root(root: Path) -> Path:
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise SupportBundleError(f"support source root is not a directory: {root}")
    if _is_link_or_reparse(root):
        raise UnsafeSupportSourceError("support source root cannot be a symlink or junction")
    return root.resolve(strict=True)


def _resolve_source(root: Path, relative: Path | None) -> Path | None:
    if relative is None:
        return None
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise UnsafeSupportSourceError(
            f"support source must be root-relative without '..': {relative}"
        )
    if not relative.parts or any(":" in part for part in relative.parts):
        raise UnsafeSupportSourceError(f"invalid support source path: {relative}")
    lowered_name = relative.name.lower()
    if relative.suffix.lower() in _BANNED_SUFFIXES:
        raise UnsafeSupportSourceError(f"model artifacts cannot enter a support bundle: {relative}")
    stem = relative.stem.lower()
    looks_like_environment_dump = (
        lowered_name in _ENVIRONMENT_DUMP_NAMES
        or lowered_name.startswith(".env.")
        or re.fullmatch(
            r"(?:process[_-])?(?:env|environ|environment)"
            r"(?:[_-](?:dump|variables|vars))?",
            stem,
        )
        is not None
    )
    if looks_like_environment_dump:
        raise UnsafeSupportSourceError(f"environment dumps cannot enter a support bundle: {relative}")

    candidate = root.joinpath(relative)
    current = root
    for part in relative.parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise UnsafeSupportSourceError(
                f"support source cannot traverse a symlink, junction, or reparse point: {relative}"
            )
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UnsafeSupportSourceError(f"support source escapes its root: {relative}") from exc
    if not candidate.exists():
        return None
    if not candidate.is_file():
        raise UnsafeSupportSourceError(f"support source is not a regular file: {relative}")
    return candidate


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


def _read_limited(
    path: Path,
    *,
    per_file_limit: int,
    budget: _PayloadBudget,
    relative: Path,
    kind: str,
) -> bytes:
    """Stat first, then read at most the active limit plus one sentinel byte."""

    size = path.stat().st_size
    if size > per_file_limit:
        raise SupportBundleError(
            f"{kind} support source exceeds {per_file_limit} bytes: {relative}"
        )
    budget.ensure_source_fits(size, relative)
    read_limit = min(per_file_limit, budget.remaining)
    with path.open("rb") as handle:
        content = handle.read(read_limit + 1)
    if len(content) > per_file_limit:
        raise SupportBundleError(
            f"{kind} support source exceeds {per_file_limit} bytes while reading: {relative}"
        )
    if len(content) > budget.remaining:
        raise SupportBundleError(
            f"support bundle payload exceeds {budget.limit} bytes while reading: {relative}"
        )
    return content


def _text_payload(
    root: Path,
    relative: Path | None,
    *,
    capability: str,
    archive_path: str,
    classification: str,
    context: RedactionContext,
    missing: list[dict[str, str]],
    budget: _PayloadBudget,
) -> _Payload | None:
    if relative is None:
        missing.append({"capability": capability, "reason": "not_provided"})
        return None
    path = _resolve_source(root, relative)
    if path is None:
        missing.append({"capability": capability, "reason": "not_found"})
        return None
    raw = _read_limited(
        path,
        per_file_limit=_MAX_TEXT_FILE_BYTES,
        budget=budget,
        relative=relative,
        kind="text",
    )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SupportBundleError(f"text support source is not UTF-8: {relative}") from exc
    sanitized, redactions = _sanitize_document(text, classification, context)
    content = sanitized.encode("utf-8")
    budget.add_payload(len(content), relative)
    return _Payload(
        capability=capability,
        archive_path=_validated_archive_path(archive_path),
        classification=classification,
        content=content,
        redactions=redactions,
    )


def _normalized_image_sources(
    sources: SupportBundleSources,
) -> tuple[_NormalizedImageSource, ...]:
    images: list[tuple[SupportImageSource, str | None]] = [
        (image, None) for image in sources.images
    ]
    images.extend(
        (
            SupportImageSource(
                path=relative,
                frame_seq=index,
                monotonic_ms=0,
                kind="standardized",
            ),
            "frames",
        )
        for index, relative in enumerate(sources.frames, start=1)
    )
    images.extend(
        (
            SupportImageSource(
                path=relative,
                frame_seq=index,
                monotonic_ms=0,
                kind="roi",
            ),
            "roi",
        )
        for index, relative in enumerate(sources.roi, start=1)
    )
    seen: set[tuple[str, int, str | None]] = set()
    normalized: list[_NormalizedImageSource] = []
    for image, compatibility_stem in images:
        if not isinstance(image, SupportImageSource):
            raise SupportBundleError("structured support images must use SupportImageSource")
        kind = str(image.kind).strip().lower()
        if kind not in _IMAGE_KINDS:
            raise SupportBundleError(
                "support image kind must be raw_client, standardized, or roi"
            )
        if isinstance(image.frame_seq, bool) or int(image.frame_seq) < 0:
            raise SupportBundleError("support image frame_seq must be a non-negative integer")
        if isinstance(image.monotonic_ms, bool) or int(image.monotonic_ms) < 0:
            raise SupportBundleError("support image monotonic_ms must be a non-negative integer")
        field = None if image.field in {None, ""} else str(image.field).strip()
        if field is not None and _IMAGE_FIELD_PATTERN.fullmatch(field) is None:
            raise SupportBundleError("support image field is not a safe diagnostic identifier")
        if kind != "roi" and field is not None:
            raise SupportBundleError("only ROI support images may declare a field")
        incident_path = None
        if image.incident_path not in {None, ""}:
            incident_path = _validated_archive_path(
                str(image.incident_path).replace("\\", "/")
            )
        for label, digest in (
            ("incident_sha256", image.incident_sha256),
            ("incident_pixel_sha256", image.incident_pixel_sha256),
        ):
            if digest not in {None, ""} and re.fullmatch(
                r"[0-9a-f]{64}", str(digest)
            ) is None:
                raise SupportBundleError(f"support image {label} is not SHA256")
        identity = (kind, int(image.frame_seq), field)
        if identity in seen:
            raise SupportBundleError(
                "duplicate structured support image identity: "
                f"kind={kind}, frame_seq={int(image.frame_seq)}, field={field!r}"
            )
        seen.add(identity)
        normalized.append(
            _NormalizedImageSource(
                path=Path(image.path),
                frame_seq=int(image.frame_seq),
                monotonic_ms=int(image.monotonic_ms),
                kind=kind,
                field=field,
                archive_stem=compatibility_stem or kind,
                incident_path=incident_path,
                incident_sha256=(
                    str(image.incident_sha256)
                    if image.incident_sha256 not in {None, ""}
                    else None
                ),
                incident_pixel_sha256=(
                    str(image.incident_pixel_sha256)
                    if image.incident_pixel_sha256 not in {None, ""}
                    else None
                ),
                frame_id=(
                    str(image.frame_id) if image.frame_id not in {None, ""} else None
                ),
            )
        )
    return tuple(normalized)


def _append_structured_image_payloads(
    payloads: list[_Payload],
    missing: list[dict[str, str]],
    capabilities: dict[str, bool],
    *,
    source_root: Path,
    images: tuple[_NormalizedImageSource, ...],
    include_frames: bool,
    include_roi: bool,
    budget: _PayloadBudget,
) -> list[dict[str, object]]:
    frame_sources = tuple(image for image in images if image.kind != "roi")
    roi_sources = tuple(image for image in images if image.kind == "roi")
    for capability, enabled, declared in (
        ("frames", include_frames, frame_sources),
        ("roi", include_roi, roi_sources),
    ):
        if not enabled:
            missing.append({"capability": capability, "reason": "disabled"})
        elif not declared:
            missing.append({"capability": capability, "reason": "not_provided"})

    selected = tuple(
        image
        for image in images
        if (image.kind == "roi" and include_roi)
        or (image.kind != "roi" and include_frames)
    )
    counters: dict[str, int] = {}
    index_entries: list[dict[str, object]] = []
    for image in selected:
        relative = image.path
        path = _resolve_source(source_root, relative)
        if path is None:
            raise SupportBundleError(
                f"requested image support source was not found: {relative}"
            )
        content = _read_limited(
            path,
            per_file_limit=_MAX_IMAGE_FILE_BYTES,
            budget=budget,
            relative=relative,
            kind="image",
        )
        suffix = path.suffix.lower()
        normalized_suffix = ".jpg" if suffix == ".jpeg" else suffix
        width, height, channels, dtype, pixel_sha256 = _validated_image_info(
            content,
            suffix=suffix,
            relative=relative,
        )
        counters[image.archive_stem] = counters.get(image.archive_stem, 0) + 1
        if image.kind == "roi":
            archive_path = (
                f"roi/{image.archive_stem}_{counters[image.archive_stem]:04d}"
                f"{normalized_suffix}"
            )
            capability = "roi"
        else:
            archive_path = (
                f"frames/{image.archive_stem}_{counters[image.archive_stem]:04d}"
                f"{normalized_suffix}"
            )
            capability = "frames"
        archive_path = _validated_archive_path(archive_path)
        source_sha256 = hashlib.sha256(content).hexdigest()
        if image.incident_sha256 is not None and image.incident_sha256 != source_sha256:
            raise SupportBundleError(
                f"incident artifact file hash changed before export: {relative}"
            )
        if (
            image.incident_pixel_sha256 is not None
            and image.incident_pixel_sha256 != pixel_sha256
        ):
            raise SupportBundleError(
                f"incident artifact pixel hash changed before export: {relative}"
            )
        budget.add_payload(len(content), relative)
        payloads.append(
            _Payload(
                capability=capability,
                archive_path=archive_path,
                classification="sensitive-image",
                content=content,
            )
        )
        capabilities[capability] = True
        index_entries.append(
            {
                "archive_path": archive_path,
                "frame_seq": image.frame_seq,
                "monotonic_ms": image.monotonic_ms,
                "kind": image.kind,
                "field": image.field,
                "source_sha256": source_sha256,
                "pixel_sha256": pixel_sha256,
                "width": width,
                "height": height,
                "channels": channels,
                "dtype": dtype,
                "incident_path": image.incident_path,
                "incident_sha256": image.incident_sha256,
                "frame_id": image.frame_id,
            }
        )
    return index_entries


def _validated_image_info(
    content: bytes,
    *,
    suffix: str,
    relative: Path,
) -> tuple[int, int, int, str, str]:
    if suffix == ".png":
        width, height = _png_dimensions(content, relative)
    elif suffix in {".jpg", ".jpeg"}:
        width, height = _jpeg_dimensions(content, relative)
    else:
        raise UnsafeSupportSourceError(f"unsupported support image type: {relative}")
    if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
        raise UnsafeSupportSourceError(
            f"support image dimensions exceed the safe pixel budget: {relative}"
        )
    try:
        import cv2
        import numpy as np

        decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise UnsafeSupportSourceError(
            f"support image could not be decoded safely: {relative}"
        ) from exc
    if decoded is None or decoded.size == 0:
        raise UnsafeSupportSourceError(f"invalid support image content: {relative}")
    decoded_height, decoded_width = decoded.shape[:2]
    if (int(decoded_width), int(decoded_height)) != (width, height):
        raise UnsafeSupportSourceError(
            f"support image header and decoded dimensions disagree: {relative}"
        )
    channels = 1 if decoded.ndim == 2 else int(decoded.shape[2])
    pixel_sha256 = hashlib.sha256(decoded.tobytes(order="C")).hexdigest()
    return width, height, channels, str(decoded.dtype), pixel_sha256


def _png_dimensions(content: bytes, relative: Path) -> tuple[int, int]:
    if (
        len(content) < 24
        or not content.startswith(b"\x89PNG\r\n\x1a\n")
        or content[12:16] != b"IHDR"
    ):
        raise UnsafeSupportSourceError(f"invalid PNG support source: {relative}")
    return tuple(int(value) for value in struct.unpack(">II", content[16:24]))  # type: ignore[return-value]


def _jpeg_dimensions(content: bytes, relative: Path) -> tuple[int, int]:
    if len(content) < 4 or not content.startswith(b"\xff\xd8"):
        raise UnsafeSupportSourceError(f"invalid JPEG support source: {relative}")
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
        marker = content[position]
        position += 1
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        if position + 2 > len(content):
            break
        segment_length = int.from_bytes(content[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(content):
            break
        if marker in sof_markers and segment_length >= 7:
            height = int.from_bytes(content[position + 3 : position + 5], "big")
            width = int.from_bytes(content[position + 5 : position + 7], "big")
            return width, height
        position += segment_length
    raise UnsafeSupportSourceError(f"invalid JPEG support source: {relative}")


def _sanitize_document(
    text: str,
    classification: str,
    context: RedactionContext,
) -> tuple[str, int]:
    if _looks_like_text_environment_dump(text):
        return "<OMITTED_ENVIRONMENT_DUMP>\n", 1
    if classification == "sanitized-json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            sanitized, count = _redact_text(text, context)
            return _ensure_newline(sanitized), count
        sanitized_value, count = _sanitize_json_value(parsed, context)
        return _json_bytes(sanitized_value).decode("utf-8"), count
    if classification == "sanitized-jsonl":
        lines: list[str] = []
        count = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                clean, line_count = _redact_text(line, context)
                lines.append(clean)
                count += line_count
            else:
                clean_value, line_count = _sanitize_json_value(parsed, context)
                lines.append(
                    json.dumps(
                        clean_value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                count += line_count
        return ("\n".join(lines) + ("\n" if lines else "")), count
    sanitized, count = _redact_text(text, context)
    return _ensure_newline(sanitized), count


def _sanitize_json_value(value: Any, context: RedactionContext) -> tuple[Any, int]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        total = 0
        for raw_key, raw_value in value.items():
            key, key_count = _redact_text(str(raw_key), context)
            total += key_count
            normalized = str(raw_key).strip().lower()
            if _SENSITIVE_JSON_KEY.search(normalized):
                result[key] = "<REDACTED>"
                total += 1
                continue
            if normalized in _ENVIRONMENT_DUMP_KEYS or _looks_like_environment_dump(raw_value):
                result[key] = "<OMITTED_ENVIRONMENT_DUMP>"
                total += 1
                continue
            clean_value, nested_count = _sanitize_json_value(raw_value, context)
            result[key] = clean_value
            total += nested_count
        return result, total
    if isinstance(value, list):
        result_list: list[Any] = []
        total = 0
        for item in value:
            clean, count = _sanitize_json_value(item, context)
            result_list.append(clean)
            total += count
        return result_list, total
    if isinstance(value, tuple):
        clean, count = _sanitize_json_value(list(value), context)
        return clean, count
    if isinstance(value, str):
        return _redact_text(value, context)
    return value, 0


def _looks_like_environment_dump(value: Any) -> bool:
    if not isinstance(value, Mapping) or len(value) < 4:
        return False
    keys = {str(key).upper() for key in value}
    common = {"PATH", "HOME", "USERPROFILE", "USERNAME", "COMPUTERNAME", "TEMP", "TMP"}
    uppercase_like = sum(bool(re.fullmatch(r"[A-Z][A-Z0-9_]{1,}", key)) for key in keys)
    return bool(keys & common) and uppercase_like >= max(3, len(keys) // 2)


def _looks_like_text_environment_dump(value: str) -> bool:
    keys = [
        match.group(1).upper()
        for line in value.splitlines()
        if (match := re.match(r"^([A-Za-z][A-Za-z0-9_]{1,})=", line.strip()))
    ]
    if len(keys) < 4:
        return False
    common = {"PATH", "HOME", "USERPROFILE", "USERNAME", "COMPUTERNAME", "TEMP", "TMP"}
    uppercase_like = sum(bool(re.fullmatch(r"[A-Z][A-Z0-9_]{1,}", key)) for key in keys)
    return bool(set(keys) & common) and uppercase_like >= max(3, len(keys) // 2)


def _redact_text(text: str, context: RedactionContext) -> tuple[str, int]:
    stripped = text.strip()
    if (
        "\n" not in stripped
        and "\r" not in stripped
        and re.match(r"^(?:[A-Za-z]:[\\/]|[\\/]{2}[^\\/]+[\\/])", stripped)
    ):
        return "<PATH>", 1

    count = 0

    def replace(pattern: re.Pattern[str], value: str, replacement: str) -> str:
        nonlocal count
        value, matched = pattern.subn(replacement, value)
        count += matched
        return value

    text = replace(_QUOTED_WINDOWS_PATH, text, '"<PATH>"')
    # An unquoted Windows path has no reliable delimiter when a segment
    # contains spaces.  Redact through end-of-line; retaining less diagnostic
    # prose is safer than leaking a tail such as "Project/private/frame.png".
    text = replace(_UNQUOTED_WINDOWS_PATH_TO_EOL, text, "<PATH>")
    text = replace(_EMAIL, text, "<EMAIL>")
    text = replace(_BEARER, text, "Bearer <REDACTED>")
    text = replace(_JWT, text, "<REDACTED_TOKEN>")
    text = replace(_KNOWN_TOKEN, text, "<REDACTED_TOKEN>")

    def secret_replacement(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}<REDACTED>"

    text, secret_count = _KEY_VALUE_SECRET.subn(secret_replacement, text)
    count += secret_count

    values = (
        *context.usernames,
        *context.computer_names,
        *context.extra_values,
    )
    for raw in sorted(
        {item.strip() for item in values if item.strip()},
        key=len,
        reverse=True,
    ):
        # Unicode-aware word boundaries keep a two-character/Chinese identity
        # private without replacing it as a substring inside a larger word.
        pattern = re.compile(rf"(?<!\w){re.escape(raw)}(?!\w)", re.IGNORECASE)
        text, value_count = pattern.subn("<REDACTED_IDENTITY>", text)
        count += value_count
    return text, count


def _effective_redaction_context(context: RedactionContext | None) -> RedactionContext:
    context = context or RedactionContext()
    usernames = set(context.usernames)
    computer_names = set(context.computer_names)
    extra_values = set(context.extra_values)
    try:
        detected_user = getpass.getuser()
    except (KeyError, OSError, RuntimeError):
        detected_user = ""
    try:
        home = Path.home()
    except (OSError, RuntimeError):
        home = None
    for value in (
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        detected_user,
        home.name if home is not None else "",
    ):
        if value:
            usernames.add(value)
    for value in (os.environ.get("COMPUTERNAME", ""), platform.node()):
        if value:
            computer_names.add(value)
    if home is not None:
        try:
            extra_values.add(str(home.resolve()))
        except OSError:
            pass
    return RedactionContext(
        usernames=tuple(sorted(usernames)),
        computer_names=tuple(sorted(computer_names)),
        extra_values=tuple(sorted(extra_values)),
    )


def _build_id(payloads: Iterable[_Payload]) -> str | None:
    payload = next(
        (item for item in payloads if item.capability == "build_manifest"),
        None,
    )
    if payload is None:
        return None
    try:
        document = json.loads(payload.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    build_id = document.get("build_id") if isinstance(document, Mapping) else None
    return str(build_id) if build_id not in {None, ""} else None


def _validated_archive_path(raw: str) -> str:
    if (
        "\\" in raw
        or ":" in raw
        or raw.startswith("/")
        or any(ord(character) < 32 for character in raw)
    ):
        raise SupportBundleError(f"unsafe support archive path: {raw}")
    path = PurePosixPath(raw)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SupportBundleError(f"unsafe support archive path: {raw}")
    return path.as_posix()


def _write_archive_entry(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    safe_name = _validated_archive_path(name)
    info = zipfile.ZipInfo(safe_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    info.external_attr = 0o600 << 16
    archive.writestr(info, content)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _ensure_newline(value: str) -> str:
    return value if not value or value.endswith("\n") else value + "\n"
