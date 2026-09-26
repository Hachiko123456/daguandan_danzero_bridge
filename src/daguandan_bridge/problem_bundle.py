"""Offline snapshots; case.json is evidence, never authority to read other roots."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping
from uuid import uuid4
import zipfile
import zlib

from .support_bundle import (
    RedactionContext, SupportBundleError, UnsafeSupportSourceError,
    _validated_image_info, sanitize_support_text,
)

PROBLEM_BUNDLE_SCHEMA = "guandan.problem-bundle/1"
_MAX_TEXT_BYTES = 8 * 1024 * 1024
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_FILES = 1024
_MAX_DIRECTORY_ENTRIES = 4096
_METADATA_RESERVE = 2 * 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_FRAME_NAME = re.compile(r"[0-9]{6,18}\.(?:png|json)\Z")
_INCIDENT_NAME = re.compile(r"incident_[\w-]{1,100}\.json\Z")
_RUN_FILES = (
    "startup_report.json", "startup.jsonl", "exceptions.log", "startup.log",
    "faulthandler.log", "runtime_identity.json", "doctor.json",
)
_SESSION_FILES = (
    "manifest.json", "session.json", "state.json", "state_before.json",
    "state_after.json", "initial_state.json", "final_state.json",
    "recognition_trace.jsonl", "timeline.jsonl", "advice.jsonl",
    "decisions.jsonl", "health_audit.json", "session_health.json",
)
_PROFILE_FILES = ("profile.json", "regions_config.json", "templates_config.json")


@dataclass(frozen=True)
class ProblemBundleRequest:
    diagnostics_root: Path
    case_directory: Path | None = None
    run_directory: Path | None = None
    session_directory: Path | None = None
    profile_directory: Path | None = None
    bundle_root: Path | None = None
    include_images: bool = True
    runtime_context: Mapping | None = None


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    name: str
    info: os.stat_result
    prefix_sha256: str | None = None


def _absolute(path: Path) -> Path:
    value = Path(path)
    if ".." in value.parts or any(":" in p for p in value.parts[1:]):
        raise UnsafeSupportSourceError("problem source contains traversal or alternate stream")
    return Path(os.path.abspath(value))


def _guard(path: Path) -> None:
    """Resolving first would silently erase junctions: check every ancestor."""
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise UnsafeSupportSourceError("problem source contains link/junction/reparse point")


def _root(path: Path) -> Path:
    path = _absolute(path)
    _guard(path)
    if path.exists() and not path.is_dir():
        raise UnsafeSupportSourceError("problem source root is not a directory")
    return path


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _version(info: os.stat_result) -> tuple:
    # CPython/Windows fstat and path.lstat can report different ctime values
    # (change time versus creation time) for unchanged NTFS files. mtime/size
    # plus file identity are comparable; use ctime additionally on POSIX only.
    return (_identity(info), info.st_size, info.st_mtime_ns,
            info.st_ctime_ns if os.name != "nt" else None)


def _append_only(path: Path) -> bool:
    return path.suffix in {".jsonl", ".log"} or path.name == "observations.jsonl.part"


def _regular(info: os.stat_result) -> bool:
    # Hardlinks can smuggle outside files too.
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1


def _privacy(value: Any) -> Any:
    """Never embed environment dumps, credentials, or base64 media in text."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key = str(key)
            compact = re.sub(r"[^a-z0-9]", "", key.lower())
            if any(word in compact for word in (
                "password", "passwd", "secret", "apikey", "accesskey", "authorization",
                "cookie", "credential", "accesstoken", "refreshtoken",
            )) or compact in {"token", "key", "bearer"}:
                result[key] = "<REDACTED>"
            elif compact in {"env", "environ", "environment", "environmentvariables",
                              "processenvironment", "osenviron"}:
                result[key] = "<OMITTED_ENVIRONMENT_DUMP>"
            elif any(word in compact for word in ("base64", "imagedata", "pixeldata")):
                result[key] = "<OMITTED_EMBEDDED_MEDIA>"
            else:
                result[key] = _privacy(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_privacy(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and re.search(r"data:(?:image|audio|video)/", value, re.I):
        return "<OMITTED_EMBEDDED_MEDIA>"
    return value


class _Collector:
    def __init__(self, roots: Mapping[str, Path | None]):
        self.roots = roots
        self.missing: list[dict] = []
        self.omitted: list[dict] = []
        self.snapshots: dict[str, _Snapshot] = {}
        self.payloads: dict[str, bytes] = {}
        self.records: list[dict] = []
        self.used = 0
        self.redaction = RedactionContext(extra_values=tuple(str(p) for p in roots.values() if p))
        self.directories: dict[Path, tuple[str, os.stat_result]] = {}

    def issue(self, name: str, reason: str, *, missing: bool = False, **details: Any) -> None:
        target = self.missing if missing else self.omitted
        item = {"path": name, "reason": reason, **details}
        if item not in target:
            target.append(item)

    def inventory(self, path: Path, name: str, *, required: bool = False) -> None:
        if name in self.snapshots:
            return
        if len(self.snapshots) >= _MAX_FILES:
            self.issue("inventory", "file_count_budget", limit=_MAX_FILES)
            return
        try:
            _guard(path)
            info = path.lstat()
            if not _regular(info):
                raise UnsafeSupportSourceError("not a regular unlinked file")
        except FileNotFoundError:
            if required:
                self.issue(name, "not_found", missing=True)
            return
        except (OSError, SupportBundleError) as exc:
            self.issue(name, "unsafe_source" if isinstance(exc, SupportBundleError)
                       else "unreadable", error_type=type(exc).__name__)
            return
        digest = None
        if _append_only(path):
            try:
                flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                with os.fdopen(os.open(path, flags), "rb") as handle:
                    opened = os.fstat(handle.fileno())
                    prefix = handle.read(min(info.st_size, _MAX_TEXT_BYTES))
                    after = os.fstat(handle.fileno())
                _guard(path)
                current = path.lstat()
                if (not _regular(opened) or _version(opened) != _version(info)
                        or _version(after) != _version(info) or _version(current) != _version(info)
                        or len(prefix) != min(info.st_size, _MAX_TEXT_BYTES)):
                    self.issue(name, "concurrent_change_during_inventory")
                    return
                digest = hashlib.sha256(prefix).hexdigest()
            except (OSError, SupportBundleError) as exc:
                self.issue(name, "unsafe_source" if isinstance(exc, SupportBundleError)
                           else "unreadable", error_type=type(exc).__name__)
                return
        self.snapshots[name] = _Snapshot(path, name, info, digest)

    def directory(self, path: Path, name: str) -> list[Path]:
        try:
            _guard(path)
            info = path.stat()
            if not stat.S_ISDIR(info.st_mode):
                raise UnsafeSupportSourceError("not a directory")
            self.directories[path] = (name, info)
            entries = []
            with os.scandir(path) as iterator:
                for entry in iterator:
                    if len(entries) >= _MAX_DIRECTORY_ENTRIES:
                        self.issue(name, "directory_entry_budget", limit=_MAX_DIRECTORY_ENTRIES)
                        break
                    entries.append(Path(entry.path))
            return sorted(entries)
        except FileNotFoundError:
            return []
        except (OSError, SupportBundleError) as exc:
            self.issue(name, "unsafe_source" if isinstance(exc, SupportBundleError)
                       else "unreadable", error_type=type(exc).__name__)
            return []

    def read(self, name: str, *, media: bool = False) -> bytes | None:
        snapshot = self.snapshots.get(name)
        if snapshot is None:
            return None
        try:
            return self._read_snapshot(snapshot, media=media)
        except (OSError, SupportBundleError) as exc:
            self.issue(name, "unsafe_source" if isinstance(exc, SupportBundleError)
                       else "unreadable", error_type=type(exc).__name__)
            return None

    def _read_snapshot(self, snapshot: _Snapshot, *, media: bool = False) -> bytes | None:
        path, name, start = snapshot.path, snapshot.name, snapshot.info
        limit = _MAX_IMAGE_BYTES if media else _MAX_TEXT_BYTES
        append_only = _append_only(path)
        if start.st_size > limit and not append_only:
            self.issue(name, "file_size_budget", source_bytes=start.st_size, limit=limit)
            return None
        _guard(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not _regular(before) or _identity(before) != _identity(start):
                self.issue(name, "concurrent_replacement")
                return None
            if before.st_size < start.st_size or (not append_only and _version(before) != _version(start)):
                self.issue(name, "concurrent_change")
                return None
            if append_only and before.st_size == start.st_size and _version(before) != _version(start):
                self.issue(name, "concurrent_rewrite")
                return None
            data = handle.read(min(start.st_size, limit))
            handle.seek(0)
            repeated = handle.read(len(data))
            after = os.fstat(handle.fileno())
        _guard(path)
        current = path.lstat()
        if (len(data) != min(start.st_size, limit) or data != repeated
                or (snapshot.prefix_sha256 is not None
                    and hashlib.sha256(data).hexdigest() != snapshot.prefix_sha256)
                or _identity(current) != _identity(start) or not _regular(current)
                or after.st_size < start.st_size or current.st_size < after.st_size
                or (not append_only and (_version(after) != _version(start)
                                         or _version(current) != _version(start)))
                or (append_only and current.st_size == start.st_size
                    and _version(current) != _version(start))):
            self.issue(name, "concurrent_change")
            return None
        if append_only:
            if current.st_size > start.st_size:
                self.issue(name, "concurrent_append", captured_bytes=len(data),
                           snapshot_bytes=start.st_size, observed_bytes=current.st_size)
            if start.st_size > limit:
                self.issue(name, "file_size_budget", captured_bytes=len(data), source_bytes=start.st_size)
            complete = data.rfind(b"\n") + 1
            if complete != len(data):
                self.issue(name, "incomplete_line_tail", omitted_bytes=len(data) - complete)
                data = data[:complete]
        return data

    def clean(self, data: bytes, name: str) -> bytes | None:
        try:
            text = data.decode("utf-8-sig")
            if name.endswith(".json"):
                value = _privacy(json.loads(text))
                text = _json(value).decode("utf-8")
                classification = "sanitized-json"
            elif name.endswith(".jsonl"):
                lines = []
                for index, line in enumerate(text.splitlines()):
                    if not line.strip():
                        continue
                    try:
                        value = _privacy(json.loads(line))
                        lines.append(json.dumps(value, ensure_ascii=False, allow_nan=False))
                    except (ValueError, TypeError, RecursionError):
                        self.issue(name, "invalid_jsonl_prefix", first_invalid_line=index + 1)
                        break
                text = "\n".join(lines) + ("\n" if lines else "")
                classification = "sanitized-jsonl"
            else:
                classification = "sanitized-log"
            sanitized, _ = sanitize_support_text(text, classification=classification, redaction=self.redaction)
            return sanitized.encode("utf-8")
        except (ValueError, TypeError, UnicodeError, RecursionError):
            self.issue(name, "invalid_text_or_json")
            return None

    def add(self, parts: list[tuple[str, bytes]], *, pair: bool = False) -> bool:
        cost = sum(len(data) + 512 + 2 * len(name.encode("utf-8")) for name, data in parts)
        if len(self.payloads) + len(parts) > _MAX_FILES or self.used + cost > _MAX_TOTAL_BYTES - _METADATA_RESERVE:
            self.issue(parts[0][0], "archive_budget", paired_paths=[p[0] for p in parts])
            return False
        self.used += cost
        for name, data in parts:
            self.payloads[name] = data
            source = self.snapshots.get(name)
            self.records.append({
                "path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "source_bytes_at_snapshot": source.info.st_size if source else None,
                "snapshot": "validated_pair" if pair else (
                    "complete_line_prefix" if name.endswith((".jsonl", ".log")) else "stable_file"),
            })
        return True

    def text(self, name: str) -> dict | None:
        raw = self.read(name)
        if raw is None:
            return None
        clean = self.clean(raw, name)
        if clean is None:
            return None
        self.add([(name, clean)])
        if name.endswith(".json"):
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        return None

    def check_directories(self) -> None:
        for snapshot in self.snapshots.values():
            try:
                _guard(snapshot.path)
                if _version(snapshot.path.lstat()) != _version(snapshot.info):
                    self.issue(snapshot.name, "source_changed_after_inventory")
            except (OSError, SupportBundleError):
                self.issue(snapshot.name, "source_changed_after_inventory")
        for path, (name, before) in self.directories.items():
            try:
                _guard(path)
                after = path.stat()
                if _version(after) != _version(before):
                    self.issue(name, "directory_changed_after_inventory")
            except (OSError, SupportBundleError):
                self.issue(name, "directory_changed_after_inventory")


def _validate_png(data: bytes, metadata: dict, sequence: int) -> None:
    # Decoders may tolerate missing IEND or trailing data. Check complete chunks
    # and CRCs before decoding/hashing BGR/BGRA pixels; never re-encode the image.
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid_png")
    position, ended = 8, False
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError("incomplete_png")
        length = struct.unpack_from(">I", data, position)[0]
        end = position + 12 + length
        if end > len(data):
            raise ValueError("incomplete_png")
        kind = data[position + 4:position + 8]
        if zlib.crc32(data[position + 4:end - 4]) & 0xFFFFFFFF != struct.unpack_from(">I", data, end - 4)[0]:
            raise ValueError("invalid_png_crc")
        position = end
        if kind == b"IEND":
            ended = length == 0 and position == len(data)
            break
    if not ended:
        raise ValueError("incomplete_png")
    width, height, channels, dtype, raw_hash = _validated_image_info(
        data, suffix=".png", relative=Path("frame.png"))
    if metadata.get("png_sha256") != hashlib.sha256(data).hexdigest():
        raise ValueError("png_sha256_mismatch")
    if metadata.get("raw_sha256") != raw_hash:
        raise ValueError("raw_sha256_mismatch")
    for field, expected in (("width", width), ("height", height), ("channels", channels),
                            ("dtype", dtype), ("sequence", sequence)):
        if field in metadata and metadata[field] != expected:
            raise ValueError("frame_metadata_mismatch")


def _frames(collector: _Collector, case: dict, session: dict) -> int:
    count = 0
    names = set(collector.snapshots)
    prefixes = sorted({name.rsplit(".", 1)[0] for name in names
                       if _FRAME_NAME.fullmatch(Path(name).name)})
    for prefix in prefixes:
        png, sidecar = prefix + ".png", prefix + ".json"
        if png not in names or sidecar not in names:
            collector.issue(prefix, "incomplete_frame_pair", missing=True)
            continue
        image = collector.read(png, media=True)
        metadata_raw = collector.read(sidecar)
        if image is None or metadata_raw is None:
            collector.issue(prefix, "frame_pair_unavailable")
            continue
        try:
            metadata = json.loads(metadata_raw)
            if not isinstance(metadata, dict):
                raise ValueError("invalid_frame_metadata")
            _validate_png(image, metadata, int(Path(prefix).name))
            expected_case = case.get("case_id")
            allowed_sessions = {str(value) for value in (
                expected_case, case.get("session_id"),
            ) if value}
            links = case.get("session_links", [])
            if isinstance(links, list):
                allowed_sessions.update(str(link["session_id"]) for link in links
                                        if isinstance(link, dict) and link.get("session_id"))
            if metadata.get("case_id") is not None:
                if not expected_case or metadata["case_id"] != expected_case:
                    raise ValueError("frame_association_mismatch")
            elif not allowed_sessions or str(metadata.get("session_id") or "") not in allowed_sessions:
                raise ValueError("frame_association_mismatch")
            for name in (png, sidecar):
                snapshot = collector.snapshots[name]
                _guard(snapshot.path)
                if _version(snapshot.path.lstat()) != _version(snapshot.info):
                    raise ValueError("concurrent_frame_pair_change")
        except (ValueError, OSError, SupportBundleError, RecursionError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) and str(exc) in {
                "invalid_png", "incomplete_png", "invalid_png_crc", "png_sha256_mismatch",
                "raw_sha256_mismatch", "frame_metadata_mismatch", "frame_association_mismatch",
                "concurrent_frame_pair_change", "invalid_frame_metadata",
            } else "invalid_frame_pair"
            collector.issue(prefix, reason)
            continue
        clean = collector.clean(metadata_raw, sidecar)
        if clean is not None and collector.add([(png, image), (sidecar, clean)], pair=True):
            count += 1
    return count


_OBSERVATION_SOURCES = ("session/observations.jsonl.part", "session/observations.jsonl.gz")


def _observations(collector: _Collector) -> None:
    output = "session/observations.jsonl"
    candidates = [name for name in _OBSERVATION_SOURCES if name in collector.snapshots]
    if not candidates:
        if collector.roots["session"] is not None:
            collector.issue(output, "not_found", missing=True)
        return
    for name in candidates:
        raw = collector.read(name)
        if raw is None:
            continue
        if name.endswith(".gz"):
            try:
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                # A single fixed-size output allocation stops compression bombs;
                # never gzip.decompress untrusted bytes without an output bound.
                raw = decoder.decompress(raw, _MAX_TEXT_BYTES + 1)
                if len(raw) > _MAX_TEXT_BYTES or decoder.unconsumed_tail:
                    collector.issue(name, "decompressed_size_budget", limit=_MAX_TEXT_BYTES)
                    continue
                if not decoder.eof or decoder.unused_data:
                    collector.issue(name, "incomplete_or_concatenated_gzip")
                    continue
            except zlib.error:
                collector.issue(name, "invalid_gzip")
                continue
            complete = raw.rfind(b"\n") + 1
            if complete != len(raw):
                collector.issue(name, "incomplete_line_tail", omitted_bytes=len(raw) - complete)
                raw = raw[:complete]
        clean = collector.clean(raw, output)
        if clean is not None and collector.add([(output, clean)]):
            collector.records[-1].update({
                "source_path": name,
                "source_bytes_at_snapshot": collector.snapshots[name].info.st_size,
                "snapshot": "bounded_decompressed_jsonl" if name.endswith(".gz") else "complete_line_prefix",
            })
            for other in candidates:
                if other != name:
                    collector.issue(other, "alternate_observations_representation_not_used")
            return
    collector.issue(output, "no_consistent_observations_snapshot", missing=True)


def _incident(collector: _Collector, name: str, *, include_images: bool) -> None:
    raw = collector.read(name)
    if raw is None:
        return
    try:
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError
        records = document.get("frames", [])
        if isinstance(records, list):
            for record in records:
                if not isinstance(record, dict):
                    continue
                for field in ("image_path", "metadata_path"):
                    pointer = record.get(field)
                    if not isinstance(pointer, str):
                        continue
                    path = Path(pointer)
                    expected_parent = collector.snapshots[name].path.parent
                    if not path.is_absolute():
                        path = expected_parent / path
                    if (".." not in path.parts and path.parent == expected_parent
                            and _FRAME_NAME.fullmatch(path.name)):
                        relative = str(Path(name).parent / path.name).replace("\\", "/")
                        record[field] = relative
                        record[field.replace("_path", "_exported")] = relative in collector.payloads
                    else:
                        record[field] = "<UNBOUND_REFERENCE>"
                        record[field.replace("_path", "_exported")] = False
        document["problem_bundle_images_requested"] = bool(include_images)
        clean = collector.clean(_json(document), name)
        if clean is not None:
            collector.add([(name, clean)])
    except (ValueError, TypeError, RecursionError):
        collector.issue(name, "invalid_incident_manifest")


def _binding(collector: _Collector, case: dict) -> None:
    for field, root_key in (("session_directory", "session"), ("run_directory", "run"),
                            ("profile_directory", "profile")):
        pointer = case.get(field)
        if not pointer:
            continue
        bound = collector.roots.get(root_key)
        if bound is None:
            collector.issue("case/" + field, "unbound_reference_not_followed")
            continue
        try:
            candidate = Path(str(pointer))
            if not candidate.is_absolute():
                candidate = collector.roots["diagnostics"] / candidate
            candidate = _absolute(candidate)
            _guard(candidate)
            if candidate != bound:
                raise ValueError
        except (ValueError, OSError, SupportBundleError):
            collector.issue("case/" + field, "reference_mismatch_not_followed")
    for field, root_key in (("run_id", "run"), ("profile_name", "profile")):
        if case.get(field) and collector.roots.get(root_key) and case[field] != collector.roots[root_key].name:
            collector.issue("case/" + field, "reference_mismatch_not_followed")


def _build_summary(collector: _Collector) -> None:
    raw = collector.read("build/build_manifest.json")
    if raw is None:
        return
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        source = value.get("source", {})
        resources = value.get("resources", {})
        summary = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "verification": "metadata_only_not_integrity_verification",
            **{key: value.get(key) for key in ("schema", "build_id", "created_at", "version")},
            "source": {key: source.get(key) for key in ("commit", "tree", "dirty", "status_sha256")}
            if isinstance(source, dict) else {},
            "resources": {key: {field: entry.get(field) for field in ("sha256", "file_count", "bytes")}
                          for key, entry in resources.items() if isinstance(entry, dict)}
            if isinstance(resources, dict) else {},
        }
        name = "build/build_manifest_summary.json"
        clean = collector.clean(_json(summary), name)
        if clean is not None:
            collector.add([(name, clean)])
    except (ValueError, TypeError, RecursionError):
        collector.issue("build/build_manifest.json", "invalid_text_or_json")

_OPENING_INCIDENT_NAME = re.compile(r"OPEN-[A-Za-z0-9_.-]{1,100}\Z")
_OPENING_FILES = ("incident.json", "opening_evidence.json", "repro.json", "recognition_trace.jsonl")
_MAX_OPENING_INCIDENTS = 3


def _collect_opening_evidence(collector: _Collector, run_root: Path | None) -> None:
    """Collect the opening monitor's fixed text allowlist for one exact run."""
    if run_root is None:
        return
    opening = run_root / "opening"
    latest_name = "run/opening/latest.json"
    collector.inventory(opening / "latest.json", latest_name)
    latest_raw = collector.read(latest_name)
    latest: dict[str, Any] = {}
    if latest_raw is not None:
        try:
            value = json.loads(latest_raw)
            if isinstance(value, dict):
                latest = value
        except (ValueError, UnicodeError, RecursionError):
            collector.issue(latest_name, "invalid_text_or_json")
    incident_root = opening / "incidents"
    candidates: list[tuple[int, str, Path]] = []
    for path in collector.directory(incident_root, "run/opening/incidents"):
        if not _OPENING_INCIDENT_NAME.fullmatch(path.name):
            continue
        try:
            candidates.append((path.stat().st_mtime_ns, path.name, path))
        except OSError:
            continue
    selected: list[Path] = []
    latest_id = latest.get("incident_id")
    if isinstance(latest_id, str) and _OPENING_INCIDENT_NAME.fullmatch(latest_id):
        expected = f"incidents/{latest_id}"
        if latest.get("relative_directory", expected) != expected:
            collector.issue(latest_name, "invalid_reference_not_followed")
        else:
            selected.extend(path for _, name, path in candidates if name == latest_id)
            if not selected:
                collector.issue(expected, "associated_opening_incident_not_found", missing=True)
    elif latest_id is not None:
        collector.issue(latest_name, "invalid_reference_not_followed")
    for _, _, path in sorted(candidates, reverse=True):
        if path not in selected:
            selected.append(path)
        if len(selected) >= _MAX_OPENING_INCIDENTS:
            break
    if len(candidates) > len(selected):
        collector.issue("run/opening/incidents", "opening_incident_count_budget",
                        omitted_count=len(candidates) - len(selected), limit=_MAX_OPENING_INCIDENTS)
    for path in selected:
        incident_id = path.name
        for filename in _OPENING_FILES:
            collector.inventory(path / filename,
                                f"run/opening/incidents/{incident_id}/{filename}", required=True)


def _collect_files(collector: _Collector, root: Path | None, key: str, names: tuple[str, ...], required: tuple[str, ...] = ()) -> None:
    if root is None:
        return
    for filename in names:
        collector.inventory(root / filename, f"{key}/{filename}", required=filename in required)


def _segment(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[\w-]{1,128}", value):
        return value
    return None


def _selection_document(path: Path) -> dict | None:
    # Shared bounded, identity-checked read; never Path.read_bytes on discovery.
    collector = _Collector({})
    collector.inventory(path, "selection.json")
    raw = collector.read("selection.json")
    if raw is None:
        return None
    try:
        document = json.loads(raw)
        return document if isinstance(document, dict) else None
    except (ValueError, UnicodeError, RecursionError):
        return None


def _run_roots(diagnostics: Path, bundle: Path | None) -> tuple[Path, ...]:
    # The only legacy namespace we authorize comes from the caller's app root,
    # not from case.json's absolute run_directory.
    paths = [diagnostics / "runs"]
    if bundle is not None:
        paths.append(bundle / "logs" / "diagnostics" / "runs")
    return tuple(dict.fromkeys(paths))


def _valid_run(path: Path) -> bool:
    if _selection_document(path / "startup_report.json") is not None:
        return True
    collector = _Collector({})
    collector.inventory(path / "startup.jsonl", "startup.jsonl")
    data = collector.read("startup.jsonl")
    if data:
        try:
            return isinstance(json.loads(data.splitlines()[0]), dict)
        except (ValueError, UnicodeError):
            pass
    return False


def select_problem_bundle_sources(
    diagnostics_root: Path, *, exclude_run: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Select latest case; prefer its EXACT run ID, not somebody else's logs."""
    root = _root(diagnostics_root)
    collector = _Collector({})
    excluded = _absolute(exclude_run) if exclude_run is not None else None
    cases = []
    for path in collector.directory(root / "cases", "cases"):
        if not path.name.startswith("case_"):
            continue
        doc = _selection_document(path / "case.json")
        if doc is not None:
            try:
                cases.append(((path / "case.json").stat().st_mtime_ns, path.name, path, doc))
            except OSError:
                continue
    if cases:
        _, _, case, document = max(cases)
        run_id = _segment(document.get("run_id"))
        if run_id:
            path = root / "runs" / run_id
            return case, path if path != excluded and _valid_run(path) else None
        # No association is proof of missing run, not authority to pick another.
        return case, None
    runs = []
    for path in collector.directory(root / "runs", "runs"):
        if path == excluded or not _valid_run(path):
            continue
        try:
            runs.append((path.stat().st_mtime_ns, path.name, path))
        except OSError:
            continue
    return None, max(runs)[2] if runs else None


def select_problem_profile_directory(case_directory: Path | None, profiles_root: Path) -> Path | None:
    """CLI binds the runtime-selected profiles root, never a case absolute path."""
    if case_directory is None:
        return None
    doc = _selection_document(_root(case_directory) / "case.json") or {}
    name = _segment(doc.get("profile_name"))
    if name is None:
        return None
    path = _root(profiles_root) / name
    return _root(path)


def _resolve_associations(collector: _Collector, case: dict) -> None:
    roots = collector.roots
    run_id = _segment(case.get("run_id"))
    if case.get("run_id") and not run_id:
        collector.issue("case/run_id", "invalid_reference_not_followed")
        roots["run"] = None
    elif run_id:
        explicit = roots["run"]
        if explicit is not None and explicit.name != run_id:
            collector.issue("run", "different_case_run_not_collected")
            roots["run"] = None
        if roots["run"] is None:
            for base in _run_roots(roots["diagnostics"], roots["bundle"]):
                candidate = base / run_id
                if _valid_run(candidate):
                    roots["run"] = _root(candidate)
                    break
        if roots["run"] is None:
            collector.issue("run", "associated_run_not_found", missing=True)

    profile = roots["profile"]
    if profile is not None and case.get("profile_name") and case["profile_name"] != profile.name:
        collector.issue("profile", "different_case_profile_not_collected")
        profile = roots["profile"] = None
    session_id = _segment(case.get("session_id"))
    links = case.get("session_links", [])
    if not session_id and isinstance(links, list):
        for link in reversed(links):
            if isinstance(link, dict) and link.get("kind") in {"session", "preopening"}:
                session_id = _segment(link.get("session_id"))
                if session_id:
                    break
    explicit = roots["session"]
    if session_id and explicit is not None:
        state = _selection_document(explicit / "manifest.json") or {}
        if state.get("session_id") != session_id:
            collector.issue("session", "different_case_session_not_collected")
            roots["session"] = None
    if session_id and roots["session"] is None and profile is not None:
        try:
            from .session_paths import resolve_sessions_root

            # Bound pointer before the existing resolver reads it; it cannot be a
            # symlink/huge arbitrary file. This config grants the external root.
            pointer = profile / ".sessions_root.json"
            _guard(pointer)
            if pointer.exists() and _selection_document(pointer) is None:
                raise ValueError("invalid sessions root pointer")
            _guard(profile / "sessions")
            base = _root(resolve_sessions_root(profile.parent, profile.name))
            for group in ("", ".preopening", ".episodes"):
                candidate = base / group / session_id
                state = _selection_document(candidate / "manifest.json")
                if state and state.get("session_id") == session_id:
                    roots["session"] = _root(candidate)
                    break
        except (OSError, ValueError, SupportBundleError):
            collector.issue("session", "invalid_authorized_sessions_root")
    if session_id and roots["session"] is None:
        collector.issue("session", "associated_session_not_found", missing=True)
    collector.redaction = RedactionContext(extra_values=tuple(str(p) for p in roots.values() if p))


def export_problem_bundle(request: ProblemBundleRequest) -> dict:
    """Create an atomic, offline ZIP; partial evidence is explicit, not failure."""
    started = datetime.now(UTC)
    roots: dict[str, Path | None] = {"diagnostics": _root(request.diagnostics_root)}
    for key, value in (("case", request.case_directory), ("run", request.run_directory),
                       ("session", request.session_directory), ("profile", request.profile_directory),
                       ("bundle", request.bundle_root)):
        roots[key] = _root(value) if value is not None else None
    diagnostics = roots["diagnostics"]
    assert diagnostics is not None
    for key, parent in (("case", "cases"), ("run", "runs")):
        value = roots[key]
        allowed = _run_roots(diagnostics, roots["bundle"]) if key == "run" else (diagnostics / parent,)
        if value is not None and value.parent not in allowed:
            raise UnsafeSupportSourceError(f"{key} is outside caller-bound diagnostics roots")
    case_root = roots["case"]
    if case_root is not None and not case_root.name.startswith("case_"):
        raise UnsafeSupportSourceError("case directory must start with case_")

    collector = _Collector(roots)
    if case_root is not None:
        collector.inventory(case_root / "case.json", "case/case.json", required=True)
    case = collector.text("case/case.json") or {}
    _resolve_associations(collector, case)
    for key in ("case", "run", "session", "profile", "bundle"):
        if roots[key] is None:
            collector.issue(key, "not_bound", missing=True)
    if case_root is not None:
        _collect_files(collector, case_root / "config", "case/config", _PROFILE_FILES, _PROFILE_FILES)
    _collect_files(collector, roots["run"], "run", _RUN_FILES, ("startup_report.json", "startup.jsonl"))
    _collect_opening_evidence(collector, roots["run"])
    _collect_files(collector, roots["session"], "session", _SESSION_FILES, ("manifest.json", "recognition_trace.jsonl"))
    if roots["session"] is not None:
        for name in _OBSERVATION_SOURCES:
            collector.inventory(roots["session"] / Path(name).name, name)
    _collect_files(collector, roots["profile"], "profile", _PROFILE_FILES, _PROFILE_FILES)
    if roots["bundle"] is not None:
        collector.inventory(roots["bundle"] / "build_manifest.json", "build/build_manifest.json", required=True)

    for key, root in (("case", case_root), ("session", roots["session"])):
        if root is None:
            continue
        for subdir in ("", "incidents", "frames", "diagnostic_frames"):
            directory = root / subdir
            archive_prefix = f"{key}/{subdir + '/' if subdir else ''}"
            for path in collector.directory(directory, f"{key}/{subdir}".rstrip("/")):
                if _INCIDENT_NAME.fullmatch(path.name):
                    # Incident manifests are evidence even with --problem-no-images;
                    # their prior/failure/context/recovery linkage must remain.
                    collector.inventory(path, archive_prefix + path.name)
                elif request.include_images and _FRAME_NAME.fullmatch(path.name):
                    collector.inventory(path, archive_prefix + path.name)

    session = collector.text("session/manifest.json") or {}
    _binding(collector, case)
    for name in list(collector.snapshots):
        if name in {"case/case.json", "session/manifest.json", "build/build_manifest.json", *_OBSERVATION_SOURCES}:
            continue
        if not _FRAME_NAME.fullmatch(Path(name).name) and not _INCIDENT_NAME.fullmatch(Path(name).name):
            collector.text(name)
    _observations(collector)
    _build_summary(collector)
    if request.runtime_context is not None:
        clean = collector.clean(_json(_privacy(dict(request.runtime_context))), "runtime_context.json")
        if clean is not None:
            collector.add([("runtime_context.json", clean)])
    profile_hashes = {name.rsplit("/", 1)[-1]: hashlib.sha256(data).hexdigest()
                      for name, data in collector.payloads.items() if name.startswith("profile/")}
    if profile_hashes:
        collector.add([("profile/resource_identity.json", _json({
            "scope": "sanitized_profile_configs_only_no_template_or_model_scan",
            "config_sha256": profile_hashes,
        }))])
        # Keep source profile bytes faithful.  The export-time marker is a
        # separate generated annotation, never written back to profile.json.
        annotation = collector.clean(_json({
            "export_time": started.isoformat(),
            "profile_name": case.get("profile_name"),
            "source": "current_profile_at_export",
            "source_config_files": sorted(profile_hashes),
        }), "profile/export_annotation.json")
        if annotation is not None:
            collector.add([("profile/export_annotation.json", annotation)])

    image_count = _frames(collector, case, session) if request.include_images else 0
    if not request.include_images:
        collector.issue("frames", "images_disabled_by_request")
    elif image_count == 0:
        collector.issue("frames", "no_valid_complete_frame_pairs", missing=True)
    for name in collector.snapshots:
        if _INCIDENT_NAME.fullmatch(Path(name).name):
            _incident(collector, name, include_images=request.include_images)
    collector.check_directories()

    case_id = case.get("case_id") or (case_root.name if case_root else None)
    formal = (bool(session.get("session_id")) and roots["session"] is not None
              and roots["session"].parent.name not in {".preopening", ".episodes"}
              and session.get("recording_phase") not in {"waiting_for_initial_state", "preopening", "aborted_before_initial_state"})
    formal_text = "有正式对局证据" if formal else "无正式对局证据"
    readme = (
        "大掼蛋助手 · 离线问题包\n\n"
        f"导出时刻（UTC）：{started.isoformat()}\n{formal_text}；导出不要求 sealed，不停止监听，也不等待写入队列清空。\n"
        "这是有界快照，不是跨文件事务快照；并发变化、缺失、损坏和预算遗漏见 problem_manifest.json。\n"
        "正写入的 JSONL/日志只包含导出开始时已完整写入的行前缀，不能视为完整文件。\n"
        "截图是离散诊断事实，不是连续录像，不能据此推断每一帧都被记录。\n"
        f"包含图片：{'是' if request.include_images else '否'}；有效完整 PNG+JSON 对：{image_count}。\n"
        "PNG 原字节未修改；图片可能含昵称等个人信息，不会自动上传。文本已脱敏，分享前仍请自行检查。\n"
        "未扫描全部历史对局、模型、模板原图或环境凭据。case.json 路径只用于核对传入绑定，不会授权任意文件读取。\n"
    ).encode("utf-8")
    status = "PARTIAL" if collector.missing or collector.omitted else "SUCCESS"
    manifest = {
        "schema": PROBLEM_BUNDLE_SCHEMA,
        "status": status,
        "created_at": started.isoformat(),
        "case_id": case_id,
        "run_id": roots["run"].name if roots["run"] else None,
        "session_id": session.get("session_id") or case.get("session_id"),
        "has_formal_session": formal,
        "configuration": {
            "evidence_priority": "case/config",
            "case/config": "case_creation_snapshot",
            "profile": "export_time",
            "profile_export_time": started.isoformat() if profile_hashes else None,
        },
        "include_images": bool(request.include_images),
        "image_count": image_count,
        "missing": collector.missing,
        "omitted": collector.omitted,
        "files": sorted(collector.records, key=lambda item: item["path"]),
        "limits": {"max_files": _MAX_FILES, "max_text_file_bytes": _MAX_TEXT_BYTES,
                   "max_image_file_bytes": _MAX_IMAGE_BYTES, "max_archive_bytes": _MAX_TOTAL_BYTES},
        "snapshot_policy": "bounded_inventory_then_stable_files_or_complete_append_only_prefix",
        "privacy": {"text_sanitized": True, "images_unmodified": True, "automatic_upload": False},
    }
    manifest_bytes = collector.clean(_json(manifest), "problem_manifest.json")
    if manifest_bytes is None:
        raise SupportBundleError("problem manifest serialization failed")
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES or len(manifest_bytes) + len(readme) + 2048 > _METADATA_RESERVE:
        raise SupportBundleError("problem manifest exceeds metadata budget")

    output = diagnostics / "exports"
    _guard(output)
    output.mkdir(parents=True, exist_ok=True)
    _guard(output)
    for _ in range(10):
        destination = output / f"DaguandanAssistant_problem_{started:%Y%m%d_%H%M%S}_{uuid4().hex}.zip"
        if not destination.exists():
            break
    else:
        raise SupportBundleError("could not allocate unique problem archive name")
    temporary = output / f".{destination.stem}.{uuid4().hex}.tmp"
    temporary_owned = False
    try:
        _guard(temporary)
        with temporary.open("xb") as stream:
            temporary_owned = True
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, data in sorted(collector.payloads.items()):
                    archive.writestr(name, data)
                archive.writestr("README.txt", readme)
                archive.writestr("problem_manifest.json", manifest_bytes)
        if temporary.stat().st_size > _MAX_TOTAL_BYTES:
            raise SupportBundleError("problem ZIP exceeds archive budget")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise SupportBundleError("problem ZIP verification failed")
        if destination.exists():
            raise FileExistsError("problem ZIP collision; existing archive not replaced")
        _guard(destination)
        _guard(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary_owned:
            _guard(temporary)
            temporary.unlink(missing_ok=True)

    result = json.loads(manifest_bytes)
    result["archive_path"] = str(destination.absolute())
    result["message"] = "问题包已导出" if status == "SUCCESS" else "问题包已导出（部分证据缺失或遗漏，请查看清单）"
    return result
