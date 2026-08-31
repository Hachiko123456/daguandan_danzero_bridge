"""Resolve and initialize immutable bundle resources and writable user data.

The source checkout intentionally keeps using ``<repo>/data``.  A frozen
application never writes below the directory containing the executable:
packaged data is treated as an immutable seed and copied into a versioned user
generation below ``%LOCALAPPDATA%`` (or ``DAGUANDAN_DATA_ROOT``).

This module is standard-library-only so it can run before Qt, OpenCV, or model
libraries are imported.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import sys
import threading
import time
from typing import BinaryIO, Iterable, Iterator, Mapping
from uuid import uuid4


APP_DIRECTORY_NAME = "DaguandanAssistant"
DATA_ROOT_ENV = "DAGUANDAN_DATA_ROOT"
DIAGNOSTICS_ROOT_ENV = "DAGUANDAN_DIAGNOSTICS_ROOT"
DIAGNOSTICS_ROOT_ENV_ALIAS = "DAGUANDAN_DIAGNOSTICS_DIR"
BUILD_MANIFEST_FILENAME = "build_manifest.json"
BUILD_MANIFEST_SCHEMA = "guandan.build-manifest/1"
RUNTIME_ROOT_SCHEMA = "guandan.user-data-root/1"
GENERATION_SCHEMA = "guandan.user-data-generation/1"
ACTIVE_GENERATION_SCHEMA = "guandan.active-data-generation/1"
DATA_SCHEMA_VERSION = 1
DATA_SCHEMA_DIRECTORY = f"v{DATA_SCHEMA_VERSION}"
RUNTIME_ROOT_MARKER = ".daguandan-user-data-root.json"
GENERATION_MARKER = "runtime_layout.json"
ACTIVE_GENERATION_FILE = "active.json"

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class RuntimeLayoutError(RuntimeError):
    """The runtime storage layout cannot be trusted or initialized safely."""


@dataclass
class _LocalStorageLock:
    mutex: threading.RLock
    depth: int = 0
    owner_thread: int | None = None
    handle: BinaryIO | None = None
    owner: dict[str, object] | None = None


_STORAGE_LOCKS_GUARD = threading.Lock()
_STORAGE_LOCKS: dict[str, _LocalStorageLock] = {}


def runtime_storage_lock_path(runtime_root: Path | str) -> Path:
    """Return the sibling OS-lock file for one absolute runtime root.

    Keeping the lock beside (not inside) the runtime root lets first-time
    ownership checks remain strict while still serializing root creation.
    """

    root = _absolute_path_without_resolving(Path(runtime_root).expanduser())
    if not root.is_absolute():
        raise RuntimeLayoutError("runtime lock root must be absolute")
    identity = hashlib.sha256(_path_identity(root).encode("utf-8")).hexdigest()[:24]
    return root.parent / f".daguandan-{identity}.storage.lock"


def runtime_storage_lock_owner_path(runtime_root: Path | str) -> Path:
    lock = runtime_storage_lock_path(runtime_root)
    return lock.with_suffix(lock.suffix + ".owner.json")


@contextmanager
def runtime_storage_lock(
    runtime_root: Path | str,
    *,
    operation: str,
    timeout_seconds: float = 30.0,
) -> Iterator[dict[str, object]]:
    """Serialize storage/release transactions across threads and processes."""

    if not isinstance(operation, str) or not operation.strip():
        raise RuntimeLayoutError("runtime lock operation is required")
    if timeout_seconds <= 0:
        raise RuntimeLayoutError("runtime lock timeout must be positive")
    lock_path = runtime_storage_lock_path(runtime_root)
    owner_path = runtime_storage_lock_owner_path(runtime_root)
    key = _path_identity(lock_path)
    with _STORAGE_LOCKS_GUARD:
        state = _STORAGE_LOCKS.setdefault(
            key, _LocalStorageLock(mutex=threading.RLock())
        )
    deadline = time.monotonic() + float(timeout_seconds)
    if not state.mutex.acquire(timeout=max(0.0, deadline - time.monotonic())):
        owner = _read_storage_lock_owner(owner_path)
        raise RuntimeLayoutError(_lock_timeout_message(operation, owner))
    current_thread = threading.get_ident()
    outermost = state.depth == 0
    try:
        if not outermost:
            if state.owner_thread != current_thread or state.owner is None:
                raise RuntimeLayoutError("runtime lock reentrancy state is invalid")
            state.depth += 1
            try:
                yield dict(state.owner)
            finally:
                state.depth -= 1
            return

        handle = _open_storage_lock(lock_path)
        while True:
            try:
                _try_lock_file(handle)
                break
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    owner = _read_storage_lock_owner(owner_path)
                    handle.close()
                    raise RuntimeLayoutError(_lock_timeout_message(operation, owner))
                time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))
        owner = {
            "schema": "guandan.runtime-storage-lock/1",
            "state": "held",
            "operation": operation.strip(),
            "pid": os.getpid(),
            "thread_id": current_thread,
            "host": socket.gethostname(),
            "token": uuid4().hex,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
        }
        _write_storage_lock_owner(owner_path, owner)
        state.depth = 1
        state.owner_thread = current_thread
        state.handle = handle
        state.owner = owner
        try:
            yield dict(owner)
        finally:
            released = {
                **owner,
                "state": "released",
                "released_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
            }
            try:
                _write_storage_lock_owner(owner_path, released)
            finally:
                _unlock_file(handle)
                handle.close()
                state.depth = 0
                state.owner_thread = None
                state.handle = None
                state.owner = None
    finally:
        state.mutex.release()


def _open_storage_lock(path: Path) -> BinaryIO:
    _assert_no_reparse_chain(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(path.parent)
    try:
        with path.open("xb") as created:
            created.write(b"0")
            created.flush()
            os.fsync(created.fileno())
    except FileExistsError:
        pass
    if _path_is_reparse(path) or not path.is_file():
        raise RuntimeLayoutError("runtime lock path is unsafe")
    return path.open("r+b", buffering=0)


def _try_lock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_storage_lock_owner(path: Path, owner: Mapping[str, object]) -> None:
    payload = (
        json.dumps(dict(owner), ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_storage_lock_owner(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as handle:
            value = json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _lock_timeout_message(
    operation: str, owner: Mapping[str, object] | None
) -> str:
    if not owner:
        return f"runtime storage lock timed out for {operation}; owner unavailable"
    return (
        f"runtime storage lock timed out for {operation}; "
        f"owner operation={owner.get('operation')!s} pid={owner.get('pid')!s} "
        f"host={owner.get('host')!s} acquired={owner.get('acquired_at_utc')!s}"
    )


@dataclass(frozen=True)
class RuntimeLayout:
    """All immutable and writable roots used by one process."""

    frozen: bool
    bundle_root: Path
    resource_data_dir: Path
    runtime_root: Path
    runtime_root_source: str
    data_schema: int
    build_id: str
    manifest_status: str
    manifest_error: str | None
    generation_id: str
    generation_root: Path
    data_dir: Path
    profiles_root: Path
    logs_root: Path
    diagnostics_root: Path
    diagnostics_root_source: str
    preferences_root: Path
    cache_root: Path
    active_generation_path: Path | None

    def sanitized_identity(self) -> dict[str, object]:
        """Return support-safe identity fields without absolute local paths."""

        return {
            "frozen": self.frozen,
            "data_schema": self.data_schema,
            "build_id": self.build_id,
            "manifest_status": self.manifest_status,
            "generation_id": self.generation_id,
            "runtime_root_source": self.runtime_root_source,
            "diagnostics_root_source": self.diagnostics_root_source,
        }


def resolve_runtime_layout(
    *,
    environ: Mapping[str, str] | None = None,
    frozen: bool | None = None,
    bundle_root: Path | str | None = None,
    executable_path: Path | str | None = None,
) -> RuntimeLayout:
    """Resolve paths without creating or modifying any filesystem entry."""

    values = os.environ if environ is None else environ
    is_frozen = getattr(sys, "frozen", False) if frozen is None else bool(frozen)
    if bundle_root is not None:
        application_root = _absolute_path_without_resolving(Path(bundle_root).expanduser())
    elif is_frozen:
        application_root = _absolute_path_without_resolving(
            Path(executable_path or sys.executable)
        ).parent
    else:
        application_root = Path(__file__).resolve().parents[2]
    resource_data = application_root / "data"

    diagnostics_override = str(
        values.get(DIAGNOSTICS_ROOT_ENV)
        or values.get(DIAGNOSTICS_ROOT_ENV_ALIAS)
        or ""
    ).strip()

    if not is_frozen:
        diagnostics_root, diagnostics_source = _resolve_diagnostics_root(
            values,
            data_override=None,
            diagnostics_override=diagnostics_override,
        )
        return RuntimeLayout(
            frozen=False,
            bundle_root=application_root,
            resource_data_dir=resource_data,
            runtime_root=application_root,
            runtime_root_source="source_checkout",
            data_schema=DATA_SCHEMA_VERSION,
            build_id="source",
            manifest_status="source_checkout",
            manifest_error=None,
            generation_id="source",
            generation_root=application_root,
            data_dir=resource_data,
            profiles_root=resource_data / "profiles",
            logs_root=application_root / "logs",
            diagnostics_root=diagnostics_root,
            diagnostics_root_source=diagnostics_source,
            preferences_root=application_root / "config",
            cache_root=application_root / ".cache",
            active_generation_path=None,
        )

    override = str(values.get(DATA_ROOT_ENV) or "").strip()
    if override:
        runtime_root = _absolute_user_path(override, field=DATA_ROOT_ENV)
        runtime_source = "environment"
    else:
        local_app_data = str(values.get("LOCALAPPDATA") or "").strip()
        if not local_app_data:
            raise RuntimeLayoutError(
                "LOCALAPPDATA is unavailable and DAGUANDAN_DATA_ROOT was not set"
            )
        runtime_root = _absolute_user_path(
            str(Path(local_app_data).expanduser() / APP_DIRECTORY_NAME),
            field="LOCALAPPDATA",
        )
        runtime_source = "local_app_data"
    _assert_external_runtime_root(runtime_root, application_root)

    build_id, manifest_status, manifest_error = _read_build_identity(application_root)
    generations_root = runtime_root / "data" / DATA_SCHEMA_DIRECTORY / "generations"
    active_path = runtime_root / "data" / DATA_SCHEMA_DIRECTORY / ACTIVE_GENERATION_FILE
    generation_id = build_id
    active = _read_active_generation_if_present(active_path, build_id=build_id)
    if active is not None:
        generation_id = active
    generation_root = generations_root / _safe_segment(
        generation_id,
        field="generation id",
    )
    diagnostics_root, diagnostics_source = _resolve_diagnostics_root(
        values,
        data_override=runtime_root,
        diagnostics_override=diagnostics_override,
    )
    _assert_external_runtime_root(diagnostics_root, application_root)
    return RuntimeLayout(
        frozen=True,
        bundle_root=application_root,
        resource_data_dir=resource_data,
        runtime_root=runtime_root,
        runtime_root_source=runtime_source,
        data_schema=DATA_SCHEMA_VERSION,
        build_id=build_id,
        manifest_status=manifest_status,
        manifest_error=manifest_error,
        generation_id=generation_id,
        generation_root=generation_root,
        data_dir=generation_root / "data",
        profiles_root=generation_root / "data" / "profiles",
        logs_root=runtime_root / "logs",
        diagnostics_root=diagnostics_root,
        diagnostics_root_source=diagnostics_source,
        preferences_root=(
            runtime_root
            / "preferences"
            / DATA_SCHEMA_DIRECTORY
            / _safe_segment(build_id, field="build id")
        ),
        cache_root=(
            runtime_root
            / "cache"
            / DATA_SCHEMA_DIRECTORY
            / _safe_segment(build_id, field="build id")
        ),
        active_generation_path=active_path,
    )


def ensure_runtime_layout(layout: RuntimeLayout | None = None) -> RuntimeLayout:
    """Atomically seed and validate the writable generation for a frozen app."""

    selected = layout or resolve_runtime_layout()
    if not selected.frozen:
        return selected
    with runtime_storage_lock(
        selected.runtime_root,
        operation="ensure-runtime-layout",
        timeout_seconds=120.0,
    ):
        if layout is None:
            selected = resolve_runtime_layout()
        selected = _prepare_runtime_layout_locked(selected)
        _write_active_generation(selected, selected.generation_id)
        return selected


def prepare_runtime_layout(layout: RuntimeLayout | None = None) -> RuntimeLayout:
    """Seed/validate one generation without changing the active pointer."""

    selected = layout or resolve_runtime_layout()
    if not selected.frozen:
        return selected
    with runtime_storage_lock(
        selected.runtime_root,
        operation="prepare-runtime-layout",
        timeout_seconds=120.0,
    ):
        if layout is None:
            selected = resolve_runtime_layout()
        return _prepare_runtime_layout_locked(selected)


def _prepare_runtime_layout_locked(selected: RuntimeLayout) -> RuntimeLayout:
    if selected.manifest_status != "identified":
        detail = selected.manifest_error or "build manifest is unavailable"
        raise RuntimeLayoutError(f"cannot seed user data: {detail}")

    _ensure_owned_runtime_root(selected)
    _assert_no_reparse_chain(selected.runtime_root)
    for path in (
        selected.logs_root,
        selected.diagnostics_root,
        selected.preferences_root,
        selected.cache_root,
        selected.generation_root.parent,
    ):
        _ensure_safe_directory(path)

    if selected.generation_root.exists():
        _validate_generation(selected.generation_root, selected)
    elif selected.generation_id != selected.build_id:
        raise RuntimeLayoutError(
            "the selected migrated data generation is missing; refusing to replace it"
        )
    else:
        _seed_generation(selected)
    return selected


def layout_for_generation(layout: RuntimeLayout, generation_id: str) -> RuntimeLayout:
    """Return the same frozen layout targeted at another safe generation."""

    if not layout.frozen:
        raise RuntimeLayoutError("source checkouts do not use data generations")
    safe = _safe_segment(generation_id, field="generation id")
    root = layout.generation_root.parent / safe
    return replace(
        layout,
        generation_id=safe,
        generation_root=root,
        data_dir=root / "data",
        profiles_root=root / "data" / "profiles",
    )


def activate_generation(layout: RuntimeLayout, generation_id: str) -> RuntimeLayout:
    """Validate and atomically select an already-created generation."""

    with runtime_storage_lock(
        layout.runtime_root,
        operation="activate-data-generation",
        timeout_seconds=120.0,
    ):
        selected = layout_for_generation(layout, generation_id)
        _validate_generation(selected.generation_root, selected)
        _write_active_generation(selected, selected.generation_id)
        return selected


def seed_entries(layout: RuntimeLayout) -> tuple[dict[str, object], ...]:
    """Return verified immutable ``data/`` entries from the build manifest."""

    if not layout.frozen:
        raise RuntimeLayoutError("source mode has no frozen seed manifest")
    manifest_path = layout.bundle_root / BUILD_MANIFEST_FILENAME
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeLayoutError("build manifest is unreadable") from exc
    if not isinstance(document, dict) or document.get("schema") != BUILD_MANIFEST_SCHEMA:
        raise RuntimeLayoutError("build manifest schema is unsupported")
    if document.get("build_id") != layout.build_id:
        raise RuntimeLayoutError("build manifest identity changed during startup")
    tree = document.get("bundle_tree")
    raw_files = tree.get("files") if isinstance(tree, dict) else None
    if not isinstance(raw_files, list):
        raise RuntimeLayoutError("build manifest has no file inventory")

    verified: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            continue
        raw_path = raw.get("path")
        if not isinstance(raw_path, str):
            continue
        relative = _portable_relative(raw_path)
        if not relative.startswith("data/"):
            continue
        parts = PurePosixPath(relative).parts
        if len(parts) >= 4 and parts[:2] == ("data", "profiles"):
            runtime_item = parts[3].casefold()
            if (
                runtime_item
                in {
                    "sessions",
                    "screenshots",
                    "pics",
                    "diagnostics",
                    "truth_log_batch_reports",
                }
                or runtime_item.startswith("_quarantine")
                or runtime_item == "hand_template_calibration.json"
                or (
                    runtime_item == "models"
                    and len(parts) >= 5
                    and parts[4].casefold() == "benchmarks"
                )
            ):
                raise RuntimeLayoutError(
                    f"runtime artifact was included in immutable seed: {relative}"
                )
        identity = relative.casefold()
        if identity in seen:
            raise RuntimeLayoutError(f"duplicate seed path: {relative}")
        seen.add(identity)
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or _HEX_SHA256.fullmatch(digest) is None
        ):
            raise RuntimeLayoutError(f"invalid seed metadata: {relative}")
        source = _bundle_member(layout.bundle_root, relative)
        if _path_is_reparse(source) or not source.is_file():
            raise RuntimeLayoutError(f"seed resource is missing or unsafe: {relative}")
        before = source.stat()
        actual_digest = _sha256_file(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeLayoutError(f"seed resource changed while hashing: {relative}")
        if after.st_size != size or actual_digest != digest:
            raise RuntimeLayoutError(f"seed resource failed integrity check: {relative}")
        verified.append(
            {"path": relative, "bytes": size, "sha256": digest}
        )
    if not verified:
        raise RuntimeLayoutError("build manifest contains no packaged data resources")
    required_suffixes = {
        "profile.json",
        "regions_config.json",
        "templates_config.json",
    }
    present = {PurePosixPath(str(item["path"])).name for item in verified}
    missing = required_suffixes - present
    if missing:
        raise RuntimeLayoutError(
            "build manifest seed is incomplete: " + ", ".join(sorted(missing))
        )
    return tuple(sorted(verified, key=lambda item: str(item["path"]).casefold()))


def copy_seed_resources(
    layout: RuntimeLayout,
    destination_generation: Path | str,
) -> dict[str, object]:
    """Copy verified seed files into a new, empty staging generation."""

    destination = Path(destination_generation)
    if destination.exists():
        raise RuntimeLayoutError("seed staging generation already exists")
    destination.mkdir(parents=True, exist_ok=False)
    _assert_no_reparse_chain(destination)
    entries = seed_entries(layout)
    copied: list[dict[str, object]] = []
    for entry in entries:
        relative = str(entry["path"])
        source = _bundle_member(layout.bundle_root, relative)
        target = destination.joinpath(*PurePosixPath(relative).parts)
        _assert_below(target, destination, field="seed destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_chain(target.parent)
        shutil.copyfile(source, target)
        if target.stat().st_size != entry["bytes"] or _sha256_file(target) != entry["sha256"]:
            raise RuntimeLayoutError(f"copied seed failed verification: {relative}")
        copied.append(dict(entry))
    return _tree_summary(copied)


def write_generation_marker(
    generation_root: Path | str,
    layout: RuntimeLayout,
    *,
    seed_summary: Mapping[str, object],
    migration: Mapping[str, object] | None = None,
) -> Path:
    """Write the completion marker last inside a staging generation."""

    root = Path(generation_root)
    document: dict[str, object] = {
        "schema": GENERATION_SCHEMA,
        "data_schema": layout.data_schema,
        "build_id": layout.build_id,
        # The marker is written inside a uniquely named staging directory and
        # then atomically renamed to the public generation id.
        "generation_id": layout.generation_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "seed": dict(seed_summary),
    }
    if migration is not None:
        document["migration"] = dict(migration)
    marker = root / GENERATION_MARKER
    _atomic_write_json(marker, document)
    return marker


def generation_marker(path: Path | str) -> dict[str, object]:
    """Load one generation marker with a stable error on malformed data."""

    marker = Path(path) / GENERATION_MARKER
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeLayoutError("data generation marker is missing or invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeLayoutError("data generation marker must be a JSON object")
    return value


def assert_safe_tree(path: Path | str) -> None:
    """Reject any symlink, junction, reparse point, or special file in a tree."""

    original = _absolute_path_without_resolving(Path(path))
    _assert_no_reparse_chain(original)
    root = original.resolve(strict=True)
    pending = [root]
    while pending:
        current = pending.pop()
        if _path_is_reparse(current):
            raise RuntimeLayoutError(f"unsafe reparse entry: {current.name}")
        try:
            with os.scandir(current) as iterator:
                entries = list(iterator)
        except OSError as exc:
            raise RuntimeLayoutError("data tree cannot be enumerated safely") from exc
        for entry in entries:
            candidate = Path(entry.path)
            if _directory_entry_is_reparse(entry, candidate):
                raise RuntimeLayoutError(f"unsafe reparse entry: {candidate.name}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(candidate)
            elif not entry.is_file(follow_symlinks=False):
                raise RuntimeLayoutError(f"unsupported filesystem entry: {candidate.name}")


def safe_tree_files(path: Path | str) -> tuple[Path, ...]:
    """Return regular files below a tree after a no-reparse validation."""

    original = _absolute_path_without_resolving(Path(path))
    _assert_no_reparse_chain(original)
    root = original.resolve(strict=True)
    assert_safe_tree(root)
    return tuple(
        sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix().casefold(),
        )
    )


def atomic_write_json(path: Path | str, value: Mapping[str, object]) -> None:
    """Public standard-library atomic JSON helper for migration/install layers."""

    _atomic_write_json(Path(path), value)


def sha256_file(path: Path | str) -> str:
    return _sha256_file(Path(path))


def _seed_generation(layout: RuntimeLayout) -> None:
    parent = layout.generation_root.parent
    _ensure_safe_directory(parent)
    temporary: Path | None = (
        parent / f".{layout.generation_id}.seed-{uuid4().hex}.tmp"
    )
    _assert_below(temporary, parent, field="seed staging directory")
    try:
        summary = copy_seed_resources(layout, temporary)
        write_generation_marker(temporary, layout, seed_summary=summary)
        _assert_no_reparse_chain(temporary)
        try:
            temporary.rename(layout.generation_root)
        except (FileExistsError, PermissionError, OSError):
            if not layout.generation_root.exists():
                raise
            _validate_generation(layout.generation_root, layout)
        else:
            temporary = None
    except RuntimeLayoutError:
        raise
    except OSError as exc:
        raise RuntimeLayoutError("user data seed could not be published") from exc
    finally:
        if temporary is not None and temporary.exists():
            _remove_owned_staging_tree(temporary, parent)
    _validate_generation(layout.generation_root, layout)


def _validate_generation(root: Path, layout: RuntimeLayout) -> None:
    _assert_below(root, root.parent, field="data generation")
    _assert_no_reparse_chain(root)
    if not root.is_dir():
        raise RuntimeLayoutError("data generation directory is missing")
    marker = generation_marker(root)
    if marker.get("schema") != GENERATION_SCHEMA:
        raise RuntimeLayoutError("data generation schema is unsupported")
    if marker.get("data_schema") != layout.data_schema:
        raise RuntimeLayoutError("data generation version does not match this application")
    if marker.get("build_id") != layout.build_id:
        raise RuntimeLayoutError("data generation belongs to another build")
    if marker.get("generation_id") != root.name:
        raise RuntimeLayoutError("data generation identity does not match its directory")
    if not (root / "data" / "profiles").is_dir():
        raise RuntimeLayoutError("data generation has no profiles directory")


def _write_active_generation(layout: RuntimeLayout, generation_id: str) -> None:
    path = layout.active_generation_path
    if path is None:
        return
    _ensure_safe_directory(path.parent)
    _atomic_write_json(
        path,
        {
            "schema": ACTIVE_GENERATION_SCHEMA,
            "data_schema": layout.data_schema,
            "build_id": layout.build_id,
            "generation_id": _safe_segment(generation_id, field="generation id"),
        },
    )


def _read_active_generation_if_present(path: Path, *, build_id: str) -> str | None:
    if not path.exists():
        return None
    if _path_is_reparse(path) or not path.is_file():
        raise RuntimeLayoutError("active data pointer is not a regular file")
    payload: str | None = None
    try:
        for attempt in range(20):
            try:
                payload = path.read_text(encoding="utf-8")
                break
            except (FileNotFoundError, PermissionError):
                if attempt == 19:
                    raise
                # A concurrent atomic ReplaceFile may make the destination
                # briefly unavailable on Windows; retry without accepting a
                # partial or malformed document.
                time.sleep(0.01)
        value = json.loads(payload or "")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeLayoutError("active data pointer is unreadable or invalid") from exc
    if not isinstance(value, dict) or value.get("schema") != ACTIVE_GENERATION_SCHEMA:
        raise RuntimeLayoutError("active data pointer schema is unsupported")
    if value.get("data_schema") != DATA_SCHEMA_VERSION:
        raise RuntimeLayoutError("active data pointer version is unsupported")
    if value.get("build_id") != build_id:
        return None
    return _safe_segment(value.get("generation_id"), field="generation id")


def _ensure_owned_runtime_root(layout: RuntimeLayout) -> None:
    root = layout.runtime_root
    _assert_no_reparse_chain(root)
    root.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(root)
    marker = root / RUNTIME_ROOT_MARKER
    if marker.exists():
        if _path_is_reparse(marker) or not marker.is_file():
            raise RuntimeLayoutError("runtime ownership marker is unsafe")
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeLayoutError("runtime ownership marker is invalid") from exc
        if not isinstance(value, dict) or value.get("schema") != RUNTIME_ROOT_SCHEMA:
            raise RuntimeLayoutError("runtime ownership marker schema is invalid")
        if value.get("application") != APP_DIRECTORY_NAME:
            raise RuntimeLayoutError("runtime root belongs to another application")
        return

    # Startup diagnostics initialize before the data layout.  Adopt an
    # otherwise-empty root containing only that known directory; never adopt
    # an arbitrary non-empty directory selected through the environment.
    existing = list(root.iterdir())

    def bootstrap_entry_allowed(entry: Path) -> bool:
        if entry.name.casefold() == "diagnostics" and entry.is_dir():
            return True
        if entry.name == RUNTIME_ROOT_MARKER and entry.is_file():
            return True
        return (
            entry.is_file()
            and entry.name.startswith(f".{RUNTIME_ROOT_MARKER}.")
            and entry.name.endswith(".tmp")
        )

    if any(not bootstrap_entry_allowed(entry) for entry in existing):
        raise RuntimeLayoutError(
            "refusing to adopt a non-empty runtime root without an ownership marker"
        )
    for entry in existing:
        if entry.is_dir():
            assert_safe_tree(entry)
    _atomic_write_json(
        marker,
        {
            "schema": RUNTIME_ROOT_SCHEMA,
            "application": APP_DIRECTORY_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        },
    )


def _read_build_identity(bundle_root: Path) -> tuple[str, str, str | None]:
    path = bundle_root / BUILD_MANIFEST_FILENAME
    if not path.is_file() or _path_is_reparse(path):
        return "unidentified", "unidentified", "build manifest is missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid", "invalid", "build manifest is unreadable or invalid"
    if not isinstance(value, dict) or value.get("schema") != BUILD_MANIFEST_SCHEMA:
        return "invalid", "invalid", "build manifest schema is unsupported"
    raw_build_id = value.get("build_id")
    try:
        build_id = _safe_segment(raw_build_id, field="build id")
    except RuntimeLayoutError:
        return "invalid", "invalid", "build manifest build_id is invalid"
    return build_id, "identified", None


def _resolve_diagnostics_root(
    environ: Mapping[str, str],
    *,
    data_override: Path | None,
    diagnostics_override: str,
) -> tuple[Path, str]:
    if diagnostics_override:
        return (
            _absolute_user_path(diagnostics_override, field=DIAGNOSTICS_ROOT_ENV),
            "diagnostics_environment",
        )
    if data_override is not None:
        return data_override / "diagnostics", "data_root"
    local_app_data = str(environ.get("LOCALAPPDATA") or "").strip()
    if local_app_data:
        return (
            Path(local_app_data).expanduser().resolve(strict=False)
            / APP_DIRECTORY_NAME
            / "diagnostics",
            "local_app_data",
        )
    temporary = str(environ.get("TEMP") or environ.get("TMP") or "").strip()
    base = Path(temporary).expanduser() if temporary else Path(os.getcwd())
    return base.resolve(strict=False) / APP_DIRECTORY_NAME / "diagnostics", "temporary"


def _absolute_user_path(value: str, *, field: str) -> Path:
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        raise RuntimeLayoutError(f"{field} must be an absolute path")
    resolved = _absolute_path_without_resolving(expanded)
    anchor = Path(resolved.anchor)
    if resolved == anchor:
        raise RuntimeLayoutError(f"{field} must not be a filesystem root")
    return resolved


def _assert_external_runtime_root(runtime_root: Path, bundle_root: Path) -> None:
    runtime = _path_identity(runtime_root)
    bundle = _path_identity(bundle_root)
    bundle_prefix = bundle.rstrip("\\/") + os.sep
    # A future versioned installer may place immutable versions below
    # ``runtime_root/install``.  The writable data root may therefore be an
    # ancestor of the bundle, but it must never equal or live inside the
    # specific immutable bundle directory.
    if runtime == bundle or runtime.startswith(bundle_prefix):
        raise RuntimeLayoutError("frozen user data root must be outside the bundle tree")


def _safe_segment(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_SEGMENT.fullmatch(value):
        raise RuntimeLayoutError(f"{field} is not a safe directory name")
    if value in {".", ".."} or value.rstrip(". ").upper() in _WINDOWS_RESERVED_NAMES:
        raise RuntimeLayoutError(f"{field} is not a safe directory name")
    return value


def _portable_relative(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or (pure.parts and ":" in pure.parts[0])
    ):
        raise RuntimeLayoutError(f"unsafe bundle path: {value!r}")
    return pure.as_posix()


def _bundle_member(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(_portable_relative(relative)).parts)
    _assert_below(candidate, root, field="bundle resource")
    _assert_no_reparse_chain(candidate)
    return candidate


def _assert_below(path: Path, root: Path, *, field: str) -> None:
    resolved = path.resolve(strict=False)
    base = root.resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise RuntimeLayoutError(f"{field} escaped its managed root") from exc
    if resolved == base:
        raise RuntimeLayoutError(f"{field} must be below its managed root")


def _assert_no_reparse_chain(path: Path) -> None:
    current = _absolute_path_without_resolving(path)
    while True:
        if current.exists() and _path_is_reparse(current):
            raise RuntimeLayoutError(f"path traverses a reparse point: {current.name}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _path_is_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeLayoutError("filesystem entry could not be inspected safely") from exc


def _directory_entry_is_reparse(entry: os.DirEntry[str], path: Path) -> bool:
    try:
        if entry.is_symlink():
            return True
        attributes = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError as exc:
        raise RuntimeLayoutError(f"filesystem entry changed during inspection: {path.name}") from exc


def _ensure_safe_directory(path: Path) -> None:
    _assert_no_reparse_chain(path)
    path.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(path)
    if not path.is_dir():
        raise RuntimeLayoutError(f"managed path is not a directory: {path.name}")


def _tree_summary(entries: Iterable[Mapping[str, object]]) -> dict[str, object]:
    normalized = sorted(
        (
            {
                "path": str(entry["path"]),
                "bytes": int(entry["bytes"]),
                "sha256": str(entry["sha256"]),
            }
            for entry in entries
        ),
        key=lambda item: item["path"].casefold(),
    )
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "file_count": len(normalized),
        "bytes": sum(int(item["bytes"]) for item in normalized),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    _ensure_safe_directory(path.parent)
    if path.exists() and (_path_is_reparse(path) or not path.is_file()):
        raise RuntimeLayoutError(f"atomic JSON destination is unsafe: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                dict(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if _path_is_reparse(temporary):
            raise RuntimeLayoutError("atomic JSON temporary file became a reparse point")
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                # Windows may briefly deny two simultaneous ReplaceFile-style
                # publishes to the same destination.  The sibling temp file
                # remains private, so a bounded retry preserves atomicity.
                time.sleep(0.01)
    except RuntimeLayoutError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeLayoutError(f"could not atomically write {path.name}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _remove_owned_staging_tree(path: Path, parent: Path) -> None:
    _assert_below(path, parent, field="staging cleanup")
    if not path.name.startswith(".") or not path.name.endswith(".tmp"):
        raise RuntimeLayoutError("refusing to clean an unrecognized staging directory")
    if path.exists():
        assert_safe_tree(path)
        shutil.rmtree(path)


def _path_identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _absolute_path_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


__all__ = [
    "ACTIVE_GENERATION_FILE",
    "ACTIVE_GENERATION_SCHEMA",
    "APP_DIRECTORY_NAME",
    "BUILD_MANIFEST_FILENAME",
    "DATA_ROOT_ENV",
    "DATA_SCHEMA_VERSION",
    "GENERATION_MARKER",
    "GENERATION_SCHEMA",
    "RUNTIME_ROOT_MARKER",
    "RuntimeLayout",
    "RuntimeLayoutError",
    "activate_generation",
    "assert_safe_tree",
    "atomic_write_json",
    "copy_seed_resources",
    "ensure_runtime_layout",
    "generation_marker",
    "layout_for_generation",
    "prepare_runtime_layout",
    "resolve_runtime_layout",
    "safe_tree_files",
    "seed_entries",
    "sha256_file",
    "write_generation_marker",
]
