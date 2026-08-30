from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from uuid import uuid4


SCHEMA = "guandan.build-manifest/1"
RELEASE_RECORD_SCHEMA = "guandan.release-record/1"
BUILD_MANIFEST_FILENAME = "build_manifest.json"
DEFAULT_EXECUTABLE_NAME = "DaguandanAssistant.exe"
DEFAULT_PROFILE_NAME = "tencent_daguandan"

_KEY_DEPENDENCIES: tuple[str, ...] = (
    "PyInstaller",
    "pyinstaller-hooks-contrib",
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
    "PySide6-Fluent-Widgets",
    "PySideSix-Frameless-Window",
    "opencv-python",
    "numpy",
    "torch",
    "mss",
    "pywin32",
    "rlcard",
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BuildManifestError(RuntimeError):
    """A portable release manifest cannot be created or trusted."""


@dataclass(frozen=True)
class IntegrityVerification:
    ok: bool
    build_id: str | None
    checked_files: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    mutable_differences: tuple[str, ...] = ()
    unexpected_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "build_id": self.build_id,
            "checked_files": self.checked_files,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "mutable_differences": list(self.mutable_differences),
            "unexpected_files": list(self.unexpected_files),
        }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_identity(project_root: Path | str) -> dict[str, object]:
    root = _safe_existing_directory(project_root, field="project root")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    branch = _git(root, "branch", "--show-current") or None
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    return {
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest() if status else None,
    }


def collect_python_identity() -> dict[str, str]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "architecture": platform.machine() or "unknown",
    }


def collect_dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _KEY_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def create_build_manifest(
    project_root: Path | str,
    bundle_root: Path | str,
    *,
    executable_name: str = DEFAULT_EXECUTABLE_NAME,
    profile_name: str = DEFAULT_PROFILE_NAME,
    source_identity: Mapping[str, object] | None = None,
    python_identity: Mapping[str, str] | None = None,
    dependency_versions: Mapping[str, str | None] | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, object]:
    root = _safe_existing_directory(bundle_root, field="bundle root")

    relative_executable = _portable_relative_path(executable_name)
    executable_path = root.joinpath(*PurePosixPath(relative_executable).parts)
    if not executable_path.is_file():
        raise BuildManifestError(f"bundle executable is missing: {relative_executable}")

    excluded = {root / BUILD_MANIFEST_FILENAME}
    if manifest_path is not None:
        candidate = _safe_absolute_path(manifest_path, field="manifest path")
        try:
            candidate.relative_to(root)
        except ValueError:
            pass
        else:
            excluded.add(candidate)

    files = _bundle_entries(root, excluded=excluded, executable=relative_executable)
    executable = next(
        (entry for entry in files if entry["path"] == relative_executable),
        None,
    )
    if executable is None:
        raise BuildManifestError("bundle executable was not included in the file tree")

    source = _normalize_source_identity(
        source_identity if source_identity is not None else collect_source_identity(project_root)
    )
    python = _normalize_string_mapping(
        python_identity if python_identity is not None else collect_python_identity(),
        field="python",
    )
    dependencies = _normalize_dependencies(
        dependency_versions
        if dependency_versions is not None
        else collect_dependency_versions()
    )
    normalized_profile_name = _portable_segment(profile_name, field="profile name")
    profile_root = f"data/profiles/{normalized_profile_name}"
    document: dict[str, object] = {
        "schema": SCHEMA,
        "source": source,
        "python": python,
        "pyinstaller": {"version": dependencies.get("PyInstaller")},
        "dependencies": dependencies,
        "executable": _without_kind(executable),
        "bundle_tree": _tree_summary(files, include_files=True),
        "profile": {
            "name": normalized_profile_name,
            "root": profile_root,
        },
        "resources": {
            "profile": _resource_summary(
                entry for entry in files if entry["kind"] == "profile"
            ),
            "templates": _resource_summary(
                entry for entry in files if entry["kind"] == "template"
            ),
            "models": _resource_summary(
                entry for entry in files if entry["kind"] == "model"
            ),
        },
    }
    document["build_id"] = compute_build_id(document)
    return document


def compute_build_id(manifest: Mapping[str, object]) -> str:
    payload = {
        key: manifest.get(key)
        for key in (
            "schema",
            "source",
            "python",
            "pyinstaller",
            "dependencies",
            "executable",
            "bundle_tree",
            "profile",
            "resources",
        )
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"gb-{digest[:24]}"


def write_build_manifest(
    project_root: Path | str,
    bundle_root: Path | str,
    output_path: Path | str | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    root = _safe_existing_directory(bundle_root, field="bundle root")
    output = (
        _safe_absolute_path(output_path, field="manifest output path")
        if output_path
        else root / BUILD_MANIFEST_FILENAME
    )
    document = create_build_manifest(
        project_root,
        root,
        manifest_path=output,
        **kwargs,
    )
    _atomic_write_json(output, document)
    return document


def load_build_manifest(path: Path | str) -> dict[str, object]:
    safe_path = _safe_existing_file(path, field="build manifest")
    try:
        value = json.loads(safe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildManifestError("build manifest is missing or invalid JSON") from exc
    if not isinstance(value, dict):
        raise BuildManifestError("build manifest must be a JSON object")
    return value


def verify_build_manifest(
    bundle_root: Path | str,
    manifest: Mapping[str, object] | Path | str,
    *,
    strict: bool = False,
) -> IntegrityVerification:
    try:
        root = _safe_existing_directory(bundle_root, field="bundle root")
    except BuildManifestError as exc:
        return IntegrityVerification(False, None, 0, (str(exc),))
    errors: list[str] = []
    warnings: list[str] = []
    mutable_differences: list[str] = []
    unexpected: list[str] = []
    manifest_path: Path | None = None
    if isinstance(manifest, (str, Path)):
        try:
            manifest_path = _safe_existing_file(manifest, field="build manifest")
            document = load_build_manifest(manifest_path)
        except BuildManifestError as exc:
            return IntegrityVerification(False, None, 0, (str(exc),))
    elif isinstance(manifest, Mapping):
        document = dict(manifest)
    else:
        return IntegrityVerification(False, None, 0, ("manifest has an unsupported type",))

    schema = document.get("schema")
    if schema != SCHEMA:
        errors.append(f"unsupported manifest schema: {schema!r}")
    build_id = document.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        errors.append("manifest build_id is missing")
        normalized_build_id = None
    else:
        normalized_build_id = build_id
        try:
            expected_build_id = compute_build_id(document)
        except (TypeError, ValueError):
            errors.append("manifest signed fields are not canonical JSON values")
        else:
            if build_id != expected_build_id:
                errors.append("manifest build_id does not match its signed fields")

    bundle_tree = document.get("bundle_tree")
    raw_files = bundle_tree.get("files") if isinstance(bundle_tree, dict) else None
    if not isinstance(raw_files, list):
        errors.append("manifest bundle_tree.files must be a list")
        raw_files = []

    executable = document.get("executable")
    executable_path: str | None = None
    if not isinstance(executable, dict):
        errors.append("manifest executable entry is missing")
    else:
        try:
            executable_path = _portable_relative_path(executable.get("path"))
        except BuildManifestError as exc:
            errors.append(str(exc))

    normalized_entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        entry = _validated_entry(
            raw,
            index=index,
            errors=errors,
            executable=executable_path,
        )
        if entry is None:
            continue
        relative = str(entry["path"])
        identity = relative.casefold()
        if identity in seen:
            errors.append(f"duplicate bundle path: {relative}")
            continue
        seen.add(identity)
        normalized_entries.append(entry)

        try:
            candidate = _resolve_bundle_member(root, relative)
        except BuildManifestError as exc:
            errors.append(str(exc))
            continue
        if _is_link_or_reparse(candidate):
            errors.append(
                f"bundle entry must not be a symlink, junction, or reparse point: {relative}"
            )
            continue
        if not candidate.is_file():
            _record_file_difference(
                entry,
                f"missing bundle file: {relative}",
                errors=errors,
                warnings=warnings,
                mutable_differences=mutable_differences,
            )
            continue
        actual_size = candidate.stat().st_size
        if actual_size != entry["bytes"]:
            _record_file_difference(
                entry,
                f"bundle file size mismatch: {relative}",
                errors=errors,
                warnings=warnings,
                mutable_differences=mutable_differences,
            )
            continue
        actual_hash = sha256_file(candidate)
        if _is_link_or_reparse(candidate):
            errors.append(
                f"bundle entry changed into a symlink, junction, or reparse point: {relative}"
            )
            continue
        if actual_hash != entry["sha256"]:
            _record_file_difference(
                entry,
                f"bundle file SHA256 mismatch: {relative}",
                errors=errors,
                warnings=warnings,
                mutable_differences=mutable_differences,
            )

    expected_tree = _tree_summary(normalized_entries, include_files=True)
    if isinstance(bundle_tree, dict):
        for key in ("algorithm", "file_count", "bytes", "sha256"):
            if bundle_tree.get(key) != expected_tree[key]:
                errors.append(f"bundle_tree.{key} does not match file entries")

    if isinstance(executable, dict) and not any(
        _without_kind(entry) == executable for entry in normalized_entries
    ):
        errors.append("manifest executable entry is not present in bundle_tree.files")

    resources = document.get("resources")
    if not isinstance(resources, dict):
        errors.append("manifest resources entry is missing")
    else:
        for label, kind in (
            ("profile", "profile"),
            ("templates", "template"),
            ("models", "model"),
        ):
            expected = _resource_summary(
                entry for entry in normalized_entries if entry.get("kind") == kind
            )
            if resources.get(label) != expected:
                errors.append(f"resources.{label} does not match bundle_tree.files")

    if strict and root.is_dir():
        exclusions = {root / BUILD_MANIFEST_FILENAME}
        if manifest_path is not None:
            try:
                manifest_path.relative_to(root)
            except ValueError:
                pass
            else:
                exclusions.add(manifest_path)
        try:
            actual = _bundle_file_paths(root, excluded=exclusions)
        except BuildManifestError as exc:
            errors.append(str(exc))
            actual = set()
        expected = {str(entry["path"]) for entry in normalized_entries}
        unexpected = sorted(actual - expected, key=str.casefold)
        for path in unexpected:
            disposition = _unexpected_file_disposition(path)
            if disposition == "allow":
                continue
            message = f"unexpected bundle file: {path}"
            if disposition == "warning":
                warnings.append(message)
            else:
                errors.append(message)

    return IntegrityVerification(
        ok=not errors,
        build_id=normalized_build_id,
        checked_files=len(normalized_entries),
        errors=tuple(errors),
        warnings=tuple(warnings),
        mutable_differences=tuple(mutable_differences),
        unexpected_files=tuple(unexpected),
    )


def write_release_record(
    manifest_path: Path | str,
    archive_path: Path | str,
    record_path: Path | str,
    checksum_path: Path | str,
) -> dict[str, object]:
    manifest_file = _safe_existing_file(manifest_path, field="build manifest")
    archive = _safe_existing_file(archive_path, field="release archive")
    document = load_build_manifest(manifest_file)
    if (
        document.get("schema") != SCHEMA
        or not isinstance(document.get("build_id"), str)
        or document.get("build_id") != compute_build_id(document)
    ):
        raise BuildManifestError("release record requires a valid build manifest")
    archive_hash = sha256_file(archive)
    record: dict[str, object] = {
        "schema": RELEASE_RECORD_SCHEMA,
        "release_id": f"gr-{archive_hash[:24]}",
        "build_id": document["build_id"],
        "source": document.get("source"),
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": archive_hash,
        },
        "build_manifest": {
            "path": BUILD_MANIFEST_FILENAME,
            "sha256": sha256_file(manifest_file),
        },
    }
    _atomic_write_bytes(
        Path(checksum_path),
        f"{archive_hash}  {archive.name}\n".encode("ascii"),
    )
    # Treat the release record as the completion marker: publish it only after
    # the conventional checksum sidecar has been durably replaced.
    _atomic_write_json(Path(record_path), record)
    return record


def _bundle_entries(
    root: Path,
    *,
    excluded: set[Path],
    executable: str | None,
) -> list[dict[str, object]]:
    excluded_absolute = {_path_identity(path) for path in excluded}
    entries: list[dict[str, object]] = []
    identities: set[str] = set()
    for path in _walk_bundle_files(root):
        if _path_identity(path) in excluded_absolute:
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise BuildManifestError("bundle file is outside bundle root") from exc
        relative = _portable_relative_path(relative)
        identity = relative.casefold()
        if identity in identities:
            raise BuildManifestError(f"bundle contains duplicate case-insensitive path: {relative}")
        identities.add(identity)
        if _is_link_or_reparse(path):
            raise BuildManifestError(
                f"bundle entry must not be a symlink, junction, or reparse point: {relative}"
            )
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if _is_link_or_reparse(path):
            raise BuildManifestError(
                f"bundle entry changed into a symlink, junction, or reparse point: {relative}"
            )
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise BuildManifestError(f"bundle file changed while hashing: {relative}")
        policy = _entry_policy(relative, executable=executable)
        entries.append(
            {
                "path": relative,
                "bytes": after.st_size,
                "sha256": digest,
                **policy,
            }
        )
    return entries


def _bundle_file_paths(root: Path, *, excluded: set[Path]) -> set[str]:
    excluded_absolute = {_path_identity(path) for path in excluded}
    paths: set[str] = set()
    identities: set[str] = set()
    for path in _walk_bundle_files(root):
        if _path_identity(path) in excluded_absolute:
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise BuildManifestError("bundle file is outside bundle root") from exc
        relative = _portable_relative_path(relative)
        identity = relative.casefold()
        if identity in identities:
            raise BuildManifestError(f"bundle contains duplicate case-insensitive path: {relative}")
        identities.add(identity)
        paths.add(relative)
    return paths


def _walk_bundle_files(root: Path) -> list[Path]:
    """Enumerate without ever traversing a symlink, junction, or reparse point."""

    pending = [root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        if _is_link_or_reparse(directory):
            relative = "." if directory == root else directory.relative_to(root).as_posix()
            raise BuildManifestError(
                f"bundle directory must not be a symlink, junction, or reparse point: {relative}"
            )
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name.casefold())
        except OSError as exc:
            raise BuildManifestError("bundle directory could not be enumerated safely") from exc
        for entry in entries:
            path = Path(entry.path)
            if _directory_entry_is_link_or_reparse(entry, path):
                relative = path.relative_to(root).as_posix()
                raise BuildManifestError(
                    f"bundle entry must not be a symlink, junction, or reparse point: {relative}"
                )
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
                else:
                    relative = path.relative_to(root).as_posix()
                    raise BuildManifestError(
                        f"bundle entry must be a regular file or directory: {relative}"
                    )
            except OSError as exc:
                raise BuildManifestError("bundle entry changed while being enumerated") from exc
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().casefold())


def _tree_summary(
    entries: list[dict[str, object]],
    *,
    include_files: bool,
) -> dict[str, object]:
    normalized = sorted(
        (dict(entry) for entry in entries),
        key=lambda entry: str(entry["path"]).casefold(),
    )
    summary: dict[str, object] = {
        "algorithm": "sha256",
        "file_count": len(normalized),
        "bytes": sum(int(entry["bytes"]) for entry in normalized),
        "sha256": hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest(),
    }
    if include_files:
        summary["files"] = normalized
    return summary


def _resource_summary(entries: Any) -> dict[str, object]:
    normalized = [_without_kind(entry) for entry in entries]
    return _tree_summary(normalized, include_files=True)


def _without_kind(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": entry["path"],
        "bytes": entry["bytes"],
        "sha256": entry["sha256"],
        "classification": entry["classification"],
        "mutability": entry["mutability"],
    }


def _entry_policy(relative: str, *, executable: str | None) -> dict[str, str]:
    if executable is not None and relative.casefold() == executable.casefold():
        return {
            "kind": "executable",
            "classification": "executable",
            "mutability": "immutable",
        }
    parts = PurePosixPath(relative).parts
    folded = tuple(part.casefold() for part in parts)
    if folded and folded[0] == "_internal":
        return {
            "kind": "bundle",
            "classification": "internal",
            "mutability": "immutable",
        }
    if len(folded) >= 4 and folded[:2] == ("data", "profiles"):
        if "templates" in folded[3:]:
            return {
                "kind": "template",
                "classification": "template",
                "mutability": "mutable",
            }
        if "models" in folded[3:]:
            return {
                "kind": "model",
                "classification": "model",
                "mutability": "mutable",
            }
        classification = (
            "profile_config"
            if len(folded) == 4
            and folded[-1]
            in {"profile.json", "regions_config.json", "templates_config.json"}
            else "profile_asset"
        )
        return {
            "kind": "profile",
            "classification": classification,
            "mutability": "mutable",
        }
    return {
        "kind": "bundle",
        "classification": "bundle",
        "mutability": "immutable",
    }


def _record_file_difference(
    entry: Mapping[str, object],
    message: str,
    *,
    errors: list[str],
    warnings: list[str],
    mutable_differences: list[str],
) -> None:
    if entry.get("mutability") == "mutable":
        warnings.append(message)
        mutable_differences.append(message)
    else:
        errors.append(message)


def _unexpected_file_disposition(relative: str) -> str:
    parts = PurePosixPath(_portable_relative_path(relative)).parts
    folded = tuple(part.casefold() for part in parts)
    if folded and folded[0] in {"logs", "diagnostics"}:
        return "allow"
    if len(folded) >= 3 and folded[:2] == ("data", "profiles"):
        remainder = folded[3:]
        if remainder:
            first = remainder[0]
            if first in {
                "screenshots",
                "sessions",
                "diagnostics",
                "truth_log_batch_reports",
            } or first.startswith("_quarantine"):
                return "allow"
            if remainder == ("hand_template_calibration.json",):
                return "allow"
            if len(remainder) >= 2 and remainder[:2] == ("models", "benchmarks"):
                return "allow"
        return "warning"
    return "error"


def _validated_entry(
    raw: object,
    *,
    index: int,
    errors: list[str],
    executable: str | None,
) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        errors.append(f"bundle file entry {index} must be an object")
        return None
    relative = raw.get("path")
    size = raw.get("bytes")
    digest = raw.get("sha256")
    kind = raw.get("kind")
    classification = raw.get("classification")
    mutability = raw.get("mutability")
    try:
        normalized_path = _portable_relative_path(relative)
    except BuildManifestError as exc:
        errors.append(str(exc))
        return None
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        errors.append(f"invalid bundle file size: {normalized_path}")
        return None
    if not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None:
        errors.append(f"invalid bundle file SHA256: {normalized_path}")
        return None
    if kind not in {"bundle", "executable", "profile", "template", "model"}:
        errors.append(f"invalid bundle file kind: {normalized_path}")
        return None
    if not isinstance(classification, str) or not classification:
        errors.append(f"invalid bundle file classification: {normalized_path}")
        return None
    if mutability not in {"immutable", "mutable"}:
        errors.append(f"invalid bundle file mutability: {normalized_path}")
        return None
    normalized = {
        "path": normalized_path,
        "bytes": size,
        "sha256": digest,
        "kind": kind,
        "classification": classification,
        "mutability": mutability,
    }
    expected = _entry_policy(normalized_path, executable=executable)
    for key, value in expected.items():
        if normalized[key] != value:
            errors.append(f"bundle file {key} violates path policy: {normalized_path}")
            return None
    return normalized


def _resolve_bundle_member(root: Path, relative: str) -> Path:
    pure = PurePosixPath(_portable_relative_path(relative))
    candidate = _safe_absolute_path(
        root.joinpath(*pure.parts),
        field="bundle member",
    )
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BuildManifestError(f"bundle path escapes root: {relative}") from exc
    return candidate


def _safe_existing_directory(path: Path | str, *, field: str) -> Path:
    absolute = _safe_absolute_path(path, field=field)
    if not absolute.is_dir():
        raise BuildManifestError(f"{field} does not exist or is not a directory")
    resolved = absolute.resolve(strict=True)
    _assert_no_reparse_chain(absolute, field=field)
    return resolved


def _safe_existing_file(path: Path | str, *, field: str) -> Path:
    absolute = _safe_absolute_path(path, field=field)
    if not absolute.is_file():
        raise BuildManifestError(f"{field} does not exist or is not a regular file")
    resolved = absolute.resolve(strict=True)
    _assert_no_reparse_chain(absolute, field=field)
    return resolved


def _safe_absolute_path(path: Path | str, *, field: str) -> Path:
    try:
        expanded = Path(path).expanduser()
        absolute = Path(os.path.abspath(os.fspath(expanded)))
    except (OSError, TypeError, ValueError) as exc:
        raise BuildManifestError(f"{field} is invalid") from exc
    _assert_no_reparse_chain(absolute, field=field)
    return absolute


def _assert_no_reparse_chain(path: Path, *, field: str) -> None:
    parts = path.parts
    if not parts:
        raise BuildManifestError(f"{field} is invalid")
    current = Path(parts[0])
    if os.path.lexists(current) and _is_link_or_reparse(current):
        raise BuildManifestError(f"{field} traverses a symlink, junction, or reparse point")
    for part in parts[1:]:
        current /= part
        if not os.path.lexists(current):
            break
        if _is_link_or_reparse(current):
            raise BuildManifestError(
                f"{field} traverses a symlink, junction, or reparse point: {current.name}"
            )


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None:
        try:
            if is_junction(path):
                return True
        except OSError:
            return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _directory_entry_is_link_or_reparse(entry: os.DirEntry[str], path: Path) -> bool:
    try:
        if entry.is_symlink():
            return True
        metadata = entry.stat(follow_symlinks=False)
    except OSError:
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None:
        try:
            if is_junction(path):
                return True
        except OSError:
            return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _path_identity(path: Path | str) -> str:
    absolute = _safe_absolute_path(path, field="bundle exclusion path")
    return os.path.normcase(os.fspath(absolute))


def _portable_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BuildManifestError("bundle path must be a non-empty string")
    if "\\" in value:
        raise BuildManifestError(f"bundle path is not portable: {value}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise BuildManifestError(f"bundle path is not portable: {value}")
    return path.as_posix()


def _portable_segment(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BuildManifestError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if normalized in {".", ".."} or any(character in normalized for character in "/\\:"):
        raise BuildManifestError(f"{field} is not portable")
    return normalized


def _normalize_source_identity(source: Mapping[str, object]) -> dict[str, object]:
    commit = source.get("commit")
    tree = source.get("tree")
    branch = source.get("branch")
    dirty = source.get("dirty")
    status_sha256 = source.get("status_sha256")
    if not isinstance(commit, str) or not commit:
        raise BuildManifestError("source commit is missing")
    if not isinstance(tree, str) or not tree:
        raise BuildManifestError("source tree is missing")
    if branch is not None and not isinstance(branch, str):
        raise BuildManifestError("source branch must be a string or null")
    if not isinstance(dirty, bool):
        raise BuildManifestError("source dirty flag must be boolean")
    if status_sha256 is not None and (
        not isinstance(status_sha256, str)
        or _HEX_SHA256.fullmatch(status_sha256) is None
    ):
        raise BuildManifestError("source status_sha256 is invalid")
    return {
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "dirty": dirty,
        "status_sha256": status_sha256,
    }


def _normalize_string_mapping(
    values: Mapping[str, str],
    *,
    field: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in sorted(values.items(), key=lambda item: str(item[0]).casefold()):
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise BuildManifestError(f"{field} entries must be non-empty strings")
        normalized[key] = value
    return normalized


def _normalize_dependencies(
    values: Mapping[str, str | None],
) -> dict[str, str | None]:
    normalized: dict[str, str | None] = {}
    for key, value in sorted(values.items(), key=lambda item: str(item[0]).casefold()):
        if not isinstance(key, str) or not key:
            raise BuildManifestError("dependency names must be non-empty strings")
        if value is not None and (not isinstance(value, str) or not value):
            raise BuildManifestError(f"dependency version is invalid: {key}")
        normalized[key] = value
    return normalized


def _git(root: Path, *arguments: str) -> str:
    return _git_bytes(root, *arguments).decode("utf-8", errors="strict").strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise BuildManifestError("git is required to create a build manifest") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BuildManifestError(f"git source identity failed: {detail or completed.returncode}")
    return completed.stdout


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = _safe_absolute_path(path, field="atomic output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(path, field="atomic output path")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                _assert_no_reparse_chain(path, field="atomic output path")
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "BUILD_MANIFEST_FILENAME",
    "BuildManifestError",
    "IntegrityVerification",
    "RELEASE_RECORD_SCHEMA",
    "SCHEMA",
    "collect_dependency_versions",
    "collect_python_identity",
    "collect_source_identity",
    "compute_build_id",
    "create_build_manifest",
    "load_build_manifest",
    "sha256_file",
    "verify_build_manifest",
    "write_build_manifest",
    "write_release_record",
]
