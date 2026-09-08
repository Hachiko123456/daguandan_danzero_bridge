from __future__ import annotations

import json
import hashlib
import gzip
import os
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .storage import atomic_write_json
from .support_bundle import sanitize_support_text


AUTO_LOG_SCHEMA = "guandan.auto-log-delivery/1"
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]+")
_DEFAULT_FILES = (
    "manifest.json",
    "timeline.jsonl",
    "timeline.md",
    "advice.jsonl",
    "decisions.jsonl",
    "health_audit.json",
    "recognition_trace.jsonl",
    "observations.jsonl.gz",
    "video/frame_index.jsonl",
)
_MEDIA_SUFFIXES = frozenset({".avi", ".mp4", ".png", ".jpg", ".jpeg", ".bmp"})


@dataclass(frozen=True)
class AutomaticLogDeliveryResult:
    status: str
    session_id: str
    output_directory: Path
    summary_path: Path | None
    machine_summary_path: Path | None
    diagnostic_zip_path: Path | None
    include_media: bool
    used_fallback: bool
    diagnostic_zip_sha256: str = ""
    redaction_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": AUTO_LOG_SCHEMA,
            "status": self.status,
            "session_id": self.session_id,
            "output_directory": str(self.output_directory),
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "machine_summary_path": (
                str(self.machine_summary_path) if self.machine_summary_path else None
            ),
            "diagnostic_zip_path": (
                str(self.diagnostic_zip_path) if self.diagnostic_zip_path else None
            ),
            "include_media": self.include_media,
            "used_fallback": self.used_fallback,
            "diagnostic_zip_sha256": self.diagnostic_zip_sha256,
            "text_sanitized": True,
            "redaction_count": self.redaction_count,
            "error": self.error,
        }


class AutomaticLogDeliveryService:
    """Publish one sealed session to a stable, user-visible directory."""

    def __init__(
        self,
        *,
        documents_root: Path | None = None,
        fallback_root: Path | None = None,
    ) -> None:
        self._documents_root = Path(documents_root) if documents_root else None
        self._fallback_root = Path(fallback_root) if fallback_root else None

    def export(
        self,
        session_directory: Path,
        *,
        include_media: bool = False,
        destination_name: str | None = None,
    ) -> AutomaticLogDeliveryResult:
        session = Path(session_directory).resolve(strict=True)
        session_id = session.name
        if _SESSION_ID.fullmatch(session_id) is None:
            raise ValueError("unsafe session id")
        manifest = _read_json(session / "manifest.json")
        if manifest.get("status") != "sealed":
            raise RuntimeError("only sealed sessions can be delivered")
        output_root, used_fallback = self._select_output_root(session)
        output = output_root / _session_date(manifest) / session_id
        output.mkdir(parents=True, exist_ok=True)
        # Re-check the fully materialized destination immediately before any
        # evidence is written.  This closes the gap where a path component is
        # replaced by a junction/symlink between root selection and mkdir.
        _validate_output_root(session, output)

        health = _read_json(session / "health_audit.json", required=False)
        summary = _build_summary(
            session,
            manifest=manifest,
            health=health,
            output=output,
            include_media=include_media,
            used_fallback=used_fallback,
        )
        machine_path = output / "summary.json"
        atomic_write_json(machine_path, summary)
        text_path = output / "摘要.txt"
        _atomic_write_text(text_path, _human_summary(summary))
        zip_name = destination_name or (
            "完整诊断.zip" if include_media else "可直接发送的诊断.zip"
        )
        if Path(zip_name).name != zip_name or not zip_name.lower().endswith(".zip"):
            raise ValueError("unsafe diagnostic ZIP name")
        zip_path = output / zip_name
        zip_path, zip_sha256, redaction_count = _write_zip_atomic(
            session,
            zip_path,
            include_media=include_media,
        )
        summary["diagnostic_zip_path"] = str(zip_path)
        summary["diagnostic_zip_sha256"] = zip_sha256
        summary["text_sanitized"] = True
        summary["redaction_count"] = redaction_count
        atomic_write_json(machine_path, summary)
        _atomic_write_text(text_path, _human_summary(summary))
        return AutomaticLogDeliveryResult(
            status="PASS",
            session_id=session_id,
            output_directory=output,
            summary_path=text_path,
            machine_summary_path=machine_path,
            diagnostic_zip_path=zip_path,
            include_media=include_media,
            used_fallback=used_fallback,
            diagnostic_zip_sha256=zip_sha256,
            redaction_count=redaction_count,
        )

    def _select_output_root(self, session: Path) -> tuple[Path, bool]:
        configured = str(os.environ.get("DAGUANDAN_LOG_ROOT") or "").strip()
        portable = str(os.environ.get("DAGUANDAN_PORTABLE_MODE") or "").strip().lower()
        if self._documents_root is not None:
            preferred = self._documents_root / "掼蛋助手日志"
        elif configured:
            configured_path = Path(configured)
            if not configured_path.is_absolute():
                raise ValueError("DAGUANDAN_LOG_ROOT must be an absolute path")
            preferred = configured_path / "掼蛋助手日志"
        elif portable in {"1", "true", "yes", "on"}:
            preferred = _portable_output_directory() / "掼蛋助手日志"
        else:
            preferred = _documents_directory() / "掼蛋助手日志"
        _validate_output_root(session, preferred)
        if _ensure_writable_directory(preferred):
            return preferred, False
        fallback = (self._fallback_root or _fallback_directory(session)) / "掼蛋助手日志"
        _validate_output_root(session, fallback)
        if not _ensure_writable_directory(fallback):
            raise OSError(f"no writable automatic log directory: {fallback}")
        return fallback, True


def export_automatic_session_log(
    session_directory: Path,
    *,
    include_media: bool = False,
    documents_root: Path | None = None,
    fallback_root: Path | None = None,
    destination_name: str | None = None,
) -> AutomaticLogDeliveryResult:
    return AutomaticLogDeliveryService(
        documents_root=documents_root,
        fallback_root=fallback_root,
    ).export(
        session_directory,
        include_media=include_media,
        destination_name=destination_name,
    )


def _documents_directory() -> Path:
    profile = str(os.environ.get("USERPROFILE") or "").strip()
    return Path(profile) / "Documents" if profile else Path.home() / "Documents"


def _fallback_directory(session: Path) -> Path:
    local = str(os.environ.get("LOCALAPPDATA") or "").strip()
    if local:
        return Path(local) / "DaguandanAssistant" / "exports"
    return (session.parents[3] if len(session.parents) > 3 else session.parent) / "exports"


def _portable_output_directory() -> Path:
    """Return a portable data root that is outside a frozen application bundle.

    A portable ZIP is commonly extracted into a directory that is treated as
    immutable by the build manifest.  Writing ``UserData`` below that
    directory therefore both dirties the bundle and makes a later Doctor run
    fail with unexpected files.  Keep source-mode behavior convenient, but in
    frozen mode place the data directory beside the bundle instead.
    """

    executable = Path(sys.executable).resolve()
    bundle_root = executable.parent
    if bool(getattr(sys, "frozen", False)):
        bundle_name = bundle_root.name or "DaguandanAssistant"
        return bundle_root.parent / f"{bundle_name}_UserData"
    return bundle_root / "UserData"


def _ensure_writable_directory(path: Path) -> bool:
    probe = path / f".write-probe-{uuid4().hex}.tmp"
    try:
        path.mkdir(parents=True, exist_ok=True)
        with probe.open("xb") as handle:
            handle.write(b"ok")
            handle.flush()
            os.fsync(handle.fileno())
        probe.unlink()
        return True
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _validate_output_root(session: Path, output_root: Path) -> None:
    source = Path(session).resolve(strict=True)
    requested = Path(output_root).absolute()
    _assert_no_reparse_chain(requested)
    output = requested.resolve(strict=False)
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("automatic log output and canonical session must be disjoint")


def _assert_no_reparse_chain(path: Path) -> None:
    current = Path(path)
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    while True:
        try:
            info = current.lstat()
        except FileNotFoundError:
            info = None
        if info is not None:
            attributes = int(getattr(info, "st_file_attributes", 0) or 0)
            if current.is_symlink() or attributes & reparse_flag:
                raise ValueError(f"automatic log output traverses a reparse point: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent


def _session_date(manifest: dict[str, object]) -> str:
    raw = str(manifest.get("finished_at") or manifest.get("started_at") or "")
    try:
        return datetime.fromisoformat(raw).date().isoformat()
    except ValueError:
        return datetime.now().astimezone().date().isoformat()


def _read_json(path: Path, *, required: bool = True) -> dict[str, object]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path.name}")
    return value


def _build_summary(
    session: Path,
    *,
    manifest: dict[str, object],
    health: dict[str, object],
    output: Path,
    include_media: bool,
    used_fallback: bool,
) -> dict[str, object]:
    issues = [
        str(item.get("code", ""))
        for item in health.get("issues", ())
        if isinstance(item, dict)
    ]
    return {
        "schema": AUTO_LOG_SCHEMA,
        "session_id": session.name,
        "session_status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "frame_count": manifest.get("frame_count"),
        "dropped_frames": manifest.get("dropped_frames"),
        "health_status": health.get("status", "UNKNOWN"),
        "health_issue_codes": issues,
        "recording_integrity": manifest.get("recording_integrity", {}),
        "source_session": str(session),
        "output_directory": str(output),
        "used_fallback_directory": used_fallback,
        "includes_sensitive_media": include_media,
        "privacy": (
            "explicit full diagnostic includes recorded images/video"
            if include_media
            else "automatic diagnostic excludes screenshots and video"
        ),
        "retention_policy": "user-managed; generated copies are never silently deleted",
    }


def _human_summary(summary: dict[str, object]) -> str:
    issues = ", ".join(summary["health_issue_codes"]) or "none"
    integrity = summary.get("recording_integrity")
    integrity_status = (
        integrity.get("status", "UNKNOWN") if isinstance(integrity, dict) else "UNKNOWN"
    )
    return "\n".join(
        (
            f"Session: {summary['session_id']}",
            f"Finished: {summary.get('finished_at') or '-'}",
            f"Frames: {summary.get('frame_count')}",
            f"Health: {summary.get('health_status')} ({issues})",
            f"Recording integrity: {integrity_status}",
            f"Media included: {'yes' if summary['includes_sensitive_media'] else 'no'}",
            f"Log directory: {summary['output_directory']}",
            f"Diagnostic ZIP: {summary.get('diagnostic_zip_path') or '-'}",
            f"ZIP SHA256: {summary.get('diagnostic_zip_sha256') or '-'}",
            "Retention: user-managed; generated copies are never silently deleted.",
            "",
        )
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _zip_sources(session: Path, *, include_media: bool) -> tuple[Path, ...]:
    session = Path(session).resolve(strict=True)
    _assert_safe_session_tree(session)
    selected: list[Path] = []
    for name in _DEFAULT_FILES:
        path = session / name
        if path.is_file():
            selected.append(_validated_session_file(session, path))
    incidents = session / "incidents"
    if incidents.is_dir():
        for path in sorted(incidents.rglob("*")):
            if path.is_file() and (
                include_media or path.suffix.lower() not in _MEDIA_SUFFIXES
            ):
                selected.append(_validated_session_file(session, path))
    if include_media:
        video = session / "video"
        if video.is_dir():
            selected.extend(
                _validated_session_file(session, path)
                for path in sorted(video.rglob("*"))
                if path.is_file()
            )
    return tuple(dict.fromkeys(selected))


def _is_link_or_reparse(path: Path) -> bool:
    """Fail closed when inspecting a session member.

    ``Path.is_file`` follows links, which would allow a selected source to
    read arbitrary content outside the sealed session.  ``lstat`` also sees a
    broken link, and Windows junctions/reparse points are identified from the
    file attributes where the platform exposes them.
    """

    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"cannot inspect session source: {path}") from exc
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        stat.S_ISLNK(info.st_mode)
        or Path(path).is_symlink()
        or attributes & reparse_flag
    ):
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction(path)) if is_junction is not None else False


def _assert_safe_session_tree(session: Path) -> None:
    """Reject links/reparse points anywhere in the sealed session tree."""

    if _is_link_or_reparse(session):
        raise ValueError(f"session root is a symlink, junction, or reparse point: {session}")
    pending = [session]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"cannot inspect sealed session tree: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            if _is_link_or_reparse(path):
                raise ValueError(
                    "session tree contains a symlink, junction, or reparse point: "
                    f"{path}"
                )
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
            except OSError as exc:
                raise ValueError(f"cannot inspect sealed session entry: {path}") from exc


def _validated_session_file(session: Path, path: Path) -> Path:
    """Return a canonical regular file proven to remain inside ``session``."""

    root = Path(session).resolve(strict=True)
    candidate = Path(path)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"session source is outside canonical session: {candidate}") from exc
    if _is_link_or_reparse(candidate):
        raise ValueError(
            f"session source is a symlink, junction, or reparse point: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cannot resolve session source: {candidate}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"session source escapes canonical session: {candidate}") from exc
    if _is_link_or_reparse(resolved) or not resolved.is_file():
        raise ValueError(f"session source is not a regular file: {candidate}")
    return resolved


@dataclass(frozen=True)
class _ArchivePayload:
    path: str
    classification: str
    content: bytes | None = None
    source: Path | None = None
    redactions: int = 0


def _archive_payload(session: Path, source: Path) -> _ArchivePayload:
    # Validate again at the read boundary.  The source list is built from a
    # directory walk, but a caller or a concurrent filesystem mutation must
    # not turn a previously safe-looking entry into a path escape.
    source = _validated_session_file(session, source)
    relative = source.relative_to(session)
    archive_path = (Path("session") / relative).as_posix()
    suffix = source.suffix.lower()
    if suffix in _MEDIA_SUFFIXES:
        return _ArchivePayload(
            archive_path,
            "sensitive-media",
            source=source,
        )
    if source.name == "observations.jsonl.gz":
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            text = handle.read()
        clean, count = sanitize_support_text(
            text,
            classification="sanitized-jsonl",
        )
        return _ArchivePayload(
            archive_path,
            "sanitized-jsonl-gzip",
            content=gzip.compress(clean.encode("utf-8"), compresslevel=9, mtime=0),
            redactions=count,
        )
    classification = (
        "sanitized-json"
        if suffix == ".json"
        else "sanitized-jsonl"
        if suffix in {".jsonl", ".part"}
        else "sanitized-log"
    )
    text = source.read_text(encoding="utf-8", errors="replace")
    if relative.as_posix() == "manifest.json":
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(document, dict):
                # Delivery paths and ZIP hashes are recursive output metadata,
                # not game evidence. Excluding them makes the archive view
                # stable after record_automatic_log_delivery updates manifest.
                document.pop("automatic_log_delivery", None)
                text = json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
    clean, count = sanitize_support_text(text, classification=classification)
    return _ArchivePayload(
        archive_path,
        classification,
        content=clean.encode("utf-8"),
        redactions=count,
    )


def _write_zip_atomic(
    session: Path,
    destination: Path,
    *,
    include_media: bool,
) -> tuple[Path, str, int]:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        sources = _zip_sources(session, include_media=include_media)
        payloads = tuple(_archive_payload(session, source) for source in sources)
        manifest_files = []
        for payload in payloads:
            size = (
                len(payload.content)
                if payload.content is not None
                else payload.source.stat().st_size
                if payload.source is not None
                else 0
            )
            digest = (
                hashlib.sha256(payload.content).hexdigest()
                if payload.content is not None
                else _sha256_file(payload.source)
                if payload.source is not None
                else hashlib.sha256(b"").hexdigest()
            )
            manifest_files.append(
                {
                    "path": payload.path,
                    "size": size,
                    "sha256": digest,
                    "classification": payload.classification,
                    "redactions": payload.redactions,
                }
            )
        redaction_count = sum(payload.redactions for payload in payloads)
        delivery_manifest = {
            "schema": AUTO_LOG_SCHEMA,
            "session_id": session.name,
            "includes_sensitive_media": include_media,
            "privacy": (
                "explicit full diagnostic includes session media"
                if include_media
                else "automatic diagnostic excludes screenshots and video"
            ),
            "text_sanitized": True,
            "redaction_count": redaction_count,
            "files": manifest_files,
        }
        with zipfile.ZipFile(
            temporary,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            manifest_info = zipfile.ZipInfo(
                "automatic_delivery_manifest.json",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                manifest_info,
                json.dumps(
                    delivery_manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            for payload in payloads:
                info = zipfile.ZipInfo(
                    payload.path,
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                if payload.content is not None:
                    archive.writestr(info, payload.content)
                elif payload.source is not None:
                    with payload.source.open("rb") as reader, archive.open(
                        info,
                        "w",
                    ) as writer:
                        while True:
                            chunk = reader.read(1024 * 1024)
                            if not chunk:
                                break
                            writer.write(chunk)
        new_sha256 = _sha256_file(temporary)
        published = destination
        if destination.exists():
            existing_sha256 = _sha256_file(destination)
            if existing_sha256 == new_sha256:
                temporary.unlink()
                return destination, existing_sha256, redaction_count
            retry = 2
            while True:
                candidate = destination.with_name(
                    f"{destination.stem}-retry-{retry:02d}{destination.suffix}"
                )
                if not candidate.exists():
                    published = candidate
                    break
                if _sha256_file(candidate) == new_sha256:
                    temporary.unlink()
                    return candidate, new_sha256, redaction_count
                retry += 1
        temporary.replace(published)
        return published, new_sha256, redaction_count
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AUTO_LOG_SCHEMA",
    "AutomaticLogDeliveryResult",
    "AutomaticLogDeliveryService",
    "export_automatic_session_log",
]
