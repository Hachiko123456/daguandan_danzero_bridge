"""Explicit copy-on-write migration from a legacy portable release.

Migration is deliberately opt-in.  The legacy directory is only read and is
rehashes after the copy; a new versioned generation is published atomically and
selected for the next process.  Unknown process/cache files are left in the
legacy directory and listed in the local migration receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Iterable, Mapping
from uuid import uuid4

from .runtime_layout import (
    RuntimeLayout,
    RuntimeLayoutError,
    active_generation_pointer,
    assert_safe_tree,
    atomic_write_json,
    compare_and_swap_active_generation,
    copy_seed_resources,
    ensure_runtime_layout,
    generation_marker,
    layout_for_generation,
    make_active_generation_pointer,
    prepare_runtime_layout,
    resolve_runtime_layout,
    runtime_storage_lock,
    safe_tree_files,
    sha256_file,
    write_generation_marker,
)


MIGRATION_SCHEMA = "guandan.portable-data-migration/1"
MIGRATION_RECEIPT_SCHEMA = "guandan.portable-data-migration-receipt/1"

_CONFIG_FILES = {
    "profile.json",
    "regions_config.json",
    "templates_config.json",
}
_DATA_DIRECTORIES = {
    "templates",
    "models",
    "sessions",
    "screenshots",
    "pics",
    "truth_log_batch_reports",
}
_SAFE_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class PortableMigrationResult:
    migration_id: str
    generation_id: str
    copied_files: int
    copied_bytes: int
    excluded_files: int
    receipt_path: Path
    source_preserved: bool
    activated: bool
    previous_generation_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MIGRATION_SCHEMA,
            "migration_id": self.migration_id,
            "generation_id": self.generation_id,
            "copied_files": self.copied_files,
            "copied_bytes": self.copied_bytes,
            "excluded_files": self.excluded_files,
            "receipt_file": self.receipt_path.name,
            "receipt_path": str(self.receipt_path.resolve()),
            "source_preserved": self.source_preserved,
            "activated": self.activated,
            "activation": "immediate_next_start",
            "previous_generation_id": self.previous_generation_id,
        }


@dataclass(frozen=True)
class PortableMigrationRestoreResult:
    migration_id: str
    receipt_path: Path
    restored_generation_id: str | None
    restored: bool
    already_restored: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "guandan.portable-data-migration-restore/1",
            "migration_id": self.migration_id,
            "receipt_path": str(self.receipt_path.resolve()),
            "restored_generation_id": self.restored_generation_id,
            "restored": self.restored,
            "already_restored": self.already_restored,
        }


def migrate_portable_data(
    portable_root: Path | str,
    *,
    layout: RuntimeLayout | None = None,
) -> PortableMigrationResult:
    """Copy allowlisted legacy data into a new generation and activate it."""

    selected = layout or resolve_runtime_layout()
    with runtime_storage_lock(
        selected.runtime_root,
        operation="migrate-portable-data",
        timeout_seconds=300.0,
    ):
        return _migrate_portable_data_locked(portable_root, layout=selected)


def _migrate_portable_data_locked(
    portable_root: Path | str,
    *,
    layout: RuntimeLayout,
) -> PortableMigrationResult:
    selected = ensure_runtime_layout(layout)
    if not selected.frozen:
        raise RuntimeLayoutError(
            "portable migration is available only from the frozen application"
        )
    source = _absolute_without_resolving(Path(portable_root).expanduser())
    if not source.is_dir():
        raise RuntimeLayoutError("portable migration source is not a directory")
    assert_safe_tree(source)
    _assert_disjoint(source, selected.runtime_root, field="runtime root")
    _assert_disjoint(source, selected.bundle_root, field="current bundle")
    profiles_root = _locate_profiles_root(source)

    copied_candidates, excluded = _migration_inventory(profiles_root)
    if not copied_candidates:
        raise RuntimeLayoutError("portable migration source has no supported profile data")
    before = _inventory_summary(copied_candidates, root=profiles_root)
    migration_id = f"pm-{before['sha256'][:24]}"
    generation_id = f"{selected.build_id}-m-{before['sha256'][:12]}"
    target_layout = layout_for_generation(selected, generation_id)
    receipt_directory = selected.runtime_root / "migration_backups"
    receipt_path = receipt_directory / f"{migration_id}.json"
    previous_pointer = active_generation_pointer(selected)

    if target_layout.generation_root.exists():
        marker = generation_marker(target_layout.generation_root)
        migration = marker.get("migration")
        if (
            not isinstance(migration, dict)
            or migration.get("schema") != MIGRATION_SCHEMA
            or migration.get("source_tree_sha256") != before["sha256"]
        ):
            raise RuntimeLayoutError(
                "existing migration generation does not match the requested source"
            )
    else:
        _create_migrated_generation(
            selected,
            target_layout,
            profiles_root=profiles_root,
            candidates=copied_candidates,
            excluded=excluded,
            migration_id=migration_id,
            source_summary=before,
        )

    # The old portable directory is the durable backup.  Prove it did not
    # change before publishing the active pointer.
    after_candidates, after_excluded = _migration_inventory(profiles_root)
    after = _inventory_summary(after_candidates, root=profiles_root)
    if before != after or excluded != after_excluded:
        raise RuntimeLayoutError(
            "portable source changed during migration; new generation was not activated"
        )

    receipt = {
        "schema": MIGRATION_RECEIPT_SCHEMA,
        "migration_id": migration_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "build_id": selected.build_id,
        "data_schema": selected.data_schema,
        "generation_id": generation_id,
        "source": {
            "directory_name": source.name,
            "profiles_directory_name": profiles_root.name,
            "tree": before,
            "preserved_in_place": True,
        },
        "copy": {
            "file_count": before["file_count"],
            "bytes": before["bytes"],
            "excluded_count": len(excluded),
            "excluded_paths": list(excluded),
        },
        "activation": {
            "mode": "immediate_next_start",
            "state": "PREPARED",
            "previous_pointer": previous_pointer,
            "previous_pointer_sha256": _pointer_sha256(previous_pointer),
            "migrated_pointer": make_active_generation_pointer(
                target_layout,
                generation_id,
                transaction_id=f"migration-{migration_id}",
            ),
        },
    }
    receipt["activation"]["migrated_pointer_sha256"] = _pointer_sha256(
        receipt["activation"]["migrated_pointer"]
    )
    if receipt_path.exists():
        existing_receipt = _read_receipt(receipt_path)
        existing_activation = existing_receipt.get("activation")
        if not isinstance(existing_activation, Mapping):
            raise RuntimeLayoutError("existing migration receipt is not recoverable")
        if existing_receipt.get("migration_id") != migration_id:
            raise RuntimeLayoutError("existing migration receipt identity differs")
        if isinstance(existing_receipt.get("restoration"), Mapping):
            raise RuntimeLayoutError(
                "this migration was restored; refusing to silently reactivate it"
            )
        receipt = existing_receipt
    else:
        atomic_write_json(receipt_path, receipt)
    activation = receipt.get("activation")
    if not isinstance(activation, Mapping):
        raise RuntimeLayoutError("migration receipt activation is invalid")
    expected_pointer = _optional_pointer(activation.get("previous_pointer"))
    migrated_pointer = _optional_pointer(activation.get("migrated_pointer"))
    if migrated_pointer is None:
        raise RuntimeLayoutError("migration receipt target pointer is missing")
    current_pointer = active_generation_pointer(selected)
    if current_pointer != migrated_pointer:
        compare_and_swap_active_generation(
            selected,
            expected=expected_pointer,
            desired=migrated_pointer,
            operation="activate-portable-migration",
        )
    receipt = dict(receipt)
    receipt["activation"] = {
        **dict(activation),
        "state": "ACTIVE",
        "activated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
    }
    atomic_write_json(receipt_path, receipt)
    return PortableMigrationResult(
        migration_id=migration_id,
        generation_id=generation_id,
        copied_files=int(before["file_count"]),
        copied_bytes=int(before["bytes"]),
        excluded_files=len(excluded),
        receipt_path=receipt_path,
        source_preserved=True,
        activated=True,
        previous_generation_id=(
            str(expected_pointer.get("generation_id"))
            if expected_pointer is not None
            else None
        ),
    )


def restore_portable_migration(
    receipt_path: Path | str,
    *,
    layout: RuntimeLayout | None = None,
) -> PortableMigrationRestoreResult:
    """Restore the exact pre-migration pointer without clobbering newer data."""

    selected = layout or resolve_runtime_layout()
    with runtime_storage_lock(
        selected.runtime_root,
        operation="restore-portable-migration",
        timeout_seconds=120.0,
    ):
        selected = prepare_runtime_layout(selected)
        path = _absolute_without_resolving(Path(receipt_path).expanduser())
        receipts_root = selected.runtime_root / "migration_backups"
        _assert_below(path, receipts_root)
        if path.parent != receipts_root or path.suffix.casefold() != ".json":
            raise RuntimeLayoutError("migration receipt must be a direct JSON receipt")
        receipt = _read_receipt(path)
        if (
            receipt.get("schema") != MIGRATION_RECEIPT_SCHEMA
            or receipt.get("build_id") != selected.build_id
            or receipt.get("data_schema") != selected.data_schema
        ):
            raise RuntimeLayoutError("migration receipt does not belong to this build")
        activation = receipt.get("activation")
        if not isinstance(activation, Mapping):
            raise RuntimeLayoutError("migration receipt has no reversible activation")
        previous = _optional_pointer(activation.get("previous_pointer"))
        migrated = _optional_pointer(activation.get("migrated_pointer"))
        if migrated is None:
            raise RuntimeLayoutError("migration receipt target pointer is missing")
        if _pointer_sha256(previous) != activation.get("previous_pointer_sha256"):
            raise RuntimeLayoutError("migration previous pointer hash is invalid")
        if _pointer_sha256(migrated) != activation.get("migrated_pointer_sha256"):
            raise RuntimeLayoutError("migration target pointer hash is invalid")
        current = active_generation_pointer(selected)
        restoration = receipt.get("restoration")
        if isinstance(restoration, Mapping):
            if current != previous:
                raise RuntimeLayoutError(
                    "migration was restored but active data has since changed"
                )
            return PortableMigrationRestoreResult(
                migration_id=str(receipt.get("migration_id")),
                receipt_path=path,
                restored_generation_id=(
                    str(previous.get("generation_id")) if previous else None
                ),
                restored=True,
                already_restored=True,
            )
        compare_and_swap_active_generation(
            selected,
            expected=migrated,
            desired=previous,
            operation="restore-portable-migration-pointer",
        )
        receipt["restoration"] = {
            "schema": "guandan.portable-data-migration-restoration/1",
            "status": "RESTORED",
            "restored_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "restored_pointer": previous,
            "restored_pointer_sha256": _pointer_sha256(previous),
        }
        atomic_write_json(path, receipt)
        return PortableMigrationRestoreResult(
            migration_id=str(receipt.get("migration_id")),
            receipt_path=path,
            restored_generation_id=(
                str(previous.get("generation_id")) if previous else None
            ),
            restored=True,
            already_restored=False,
        )


def _create_migrated_generation(
    base_layout: RuntimeLayout,
    target_layout: RuntimeLayout,
    *,
    profiles_root: Path,
    candidates: tuple[Path, ...],
    excluded: tuple[str, ...],
    migration_id: str,
    source_summary: Mapping[str, object],
) -> None:
    parent = target_layout.generation_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{target_layout.generation_id}.migrate-{uuid4().hex}.tmp"
    if temporary.exists():
        raise RuntimeLayoutError("migration staging directory already exists")
    try:
        seed_summary = copy_seed_resources(base_layout, temporary)
        copied_entries: list[dict[str, object]] = []
        for source in candidates:
            relative_profile = source.relative_to(profiles_root)
            portable = _portable_relative(relative_profile)
            target = temporary / "data" / "profiles" / Path(portable)
            _assert_below(target, temporary)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = sha256_file(target)
            size = target.stat().st_size
            if digest != sha256_file(source) or size != source.stat().st_size:
                raise RuntimeLayoutError(
                    f"migrated file failed verification: {source.name}"
                )
            copied_entries.append(
                {"path": portable, "bytes": size, "sha256": digest}
            )
        migration = {
            "schema": MIGRATION_SCHEMA,
            "migration_id": migration_id,
            "source_tree_sha256": source_summary["sha256"],
            "copied": _entry_summary(copied_entries),
            "excluded_count": len(excluded),
            "source_preserved_in_place": True,
        }
        write_generation_marker(
            temporary,
            target_layout,
            seed_summary=seed_summary,
            migration=migration,
        )
        assert_safe_tree(temporary)
        try:
            temporary.rename(target_layout.generation_root)
        except (FileExistsError, PermissionError, OSError):
            if not target_layout.generation_root.exists():
                raise
            marker = generation_marker(target_layout.generation_root)
            existing = marker.get("migration")
            if not isinstance(existing, dict) or existing.get("migration_id") != migration_id:
                raise RuntimeLayoutError(
                    "a different data generation won the migration publish race"
                )
        else:
            temporary = None
    except RuntimeLayoutError:
        raise
    except OSError as exc:
        raise RuntimeLayoutError("migrated generation could not be published") from exc
    finally:
        if temporary is not None and temporary.exists():
            _remove_staging_tree(temporary, parent)


def _locate_profiles_root(source: Path) -> Path:
    candidates = (source / "data" / "profiles", source / "profiles")
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if len(existing) != 1:
        raise RuntimeLayoutError(
            "migration source must contain exactly one data/profiles or profiles directory"
        )
    profiles = existing[0]
    assert_safe_tree(profiles)
    return profiles


def _migration_inventory(
    profiles_root: Path,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    files = safe_tree_files(profiles_root)
    selected: list[Path] = []
    excluded: list[str] = []
    for path in files:
        relative = path.relative_to(profiles_root)
        portable = _portable_relative(relative)
        parts = PurePosixPath(portable).parts
        allowed = False
        if len(parts) >= 2 and _SAFE_PROFILE.fullmatch(parts[0]):
            item = parts[1]
            allowed = (
                (len(parts) == 2 and item in _CONFIG_FILES)
                or item in _DATA_DIRECTORIES
                or item.startswith("_quarantine")
            )
        # Deliberately do not migrate the removed automatic-calibration cache,
        # diagnostics, bytecode, or arbitrary files from a legacy package.
        if allowed and parts[-1].casefold() != "hand_template_calibration.json":
            selected.append(path)
        else:
            excluded.append(portable)
    return (
        tuple(sorted(selected, key=lambda path: path.relative_to(profiles_root).as_posix().casefold())),
        tuple(sorted(excluded, key=str.casefold)),
    )


def _inventory_summary(files: Iterable[Path], *, root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for path in files:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeLayoutError(f"migration source changed while hashing: {path.name}")
        entries.append(
            {
                "path": _portable_relative(path.relative_to(root)),
                "bytes": after.st_size,
                "sha256": digest,
            }
        )
    return _entry_summary(entries)


def _entry_summary(entries: Iterable[Mapping[str, object]]) -> dict[str, object]:
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
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "file_count": len(normalized),
        "bytes": sum(int(item["bytes"]) for item in normalized),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _assert_disjoint(path: Path, protected: Path, *, field: str) -> None:
    left = os.path.normcase(str(path.resolve(strict=False)))
    right = os.path.normcase(str(protected.resolve(strict=False)))
    left_prefix = left.rstrip("\\/") + os.sep
    right_prefix = right.rstrip("\\/") + os.sep
    if left == right or left.startswith(right_prefix) or right.startswith(left_prefix):
        raise RuntimeLayoutError(f"migration source overlaps the {field}")


def _portable_relative(path: Path) -> str:
    value = path.as_posix()
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or ":" in pure.parts[0]
    ):
        raise RuntimeLayoutError("legacy profile contains an unsafe relative path")
    return pure.as_posix()


def _read_receipt(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeLayoutError("migration receipt is unavailable or unsafe")
    assert_safe_tree(path.parent)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeLayoutError("migration receipt is unreadable or invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeLayoutError("migration receipt must be a JSON object")
    return value


def _optional_pointer(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeLayoutError("migration generation pointer is invalid")
    pointer = dict(value)
    if (
        pointer.get("schema") != "guandan.active-data-generation/1"
        or not isinstance(pointer.get("build_id"), str)
        or not isinstance(pointer.get("generation_id"), str)
    ):
        raise RuntimeLayoutError("migration generation pointer is invalid")
    return pointer


def _pointer_sha256(value: Mapping[str, object] | None) -> str:
    payload = json.dumps(
        dict(value) if value is not None else None,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_below(path: Path, root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise RuntimeLayoutError("migration destination escaped its staging root") from exc


def _remove_staging_tree(path: Path, parent: Path) -> None:
    _assert_below(path, parent)
    if not path.name.startswith(".") or not path.name.endswith(".tmp"):
        raise RuntimeLayoutError("refusing to clean an unrecognized migration staging tree")
    assert_safe_tree(path)
    shutil.rmtree(path)


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


__all__ = [
    "MIGRATION_RECEIPT_SCHEMA",
    "MIGRATION_SCHEMA",
    "PortableMigrationResult",
    "PortableMigrationRestoreResult",
    "migrate_portable_data",
    "restore_portable_migration",
]
