from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from .runtime_layout import assert_safe_tree, sha256_file
from .session_paths import (
    SESSIONS_ROOT_SCHEMA,
    clear_sessions_root_override,
    resolve_sessions_root,
    sessions_root_override_path,
    write_sessions_root_override,
)

MIGRATION_SCHEMA = "guandan.sessions-migration/1"
RECEIPT_FILENAME = ".sessions-migration.json"


@dataclass(frozen=True)
class SessionsMigrationResult:
    status: str
    source_root: Path
    target_root: Path
    receipt_path: Path | None
    file_count: int
    bytes_copied: int
    tree_sha256: str
    pointer_path: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MIGRATION_SCHEMA,
            "status": self.status,
            "source_root": str(self.source_root),
            "target_root": str(self.target_root),
            "receipt_path": str(self.receipt_path) if self.receipt_path else None,
            "file_count": self.file_count,
            "bytes_copied": self.bytes_copied,
            "tree_sha256": self.tree_sha256,
            "pointer_path": str(self.pointer_path) if self.pointer_path else None,
        }


def _full(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise ValueError(f"路径必须是绝对路径：{value}")
    return value.resolve(strict=False)


def _is_reparse(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return path.is_symlink() or (bool(checker()) if callable(checker) else False)


def _assert_no_reparse_tree(root: Path) -> None:
    assert_safe_tree(root)


def _inventory(root: Path) -> tuple[tuple[Path, int, str], ...]:
    files: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().casefold()):
        if _is_reparse(path):
            raise ValueError(f"sessions 数据不能包含符号链接或 junction：{path}")
        if path.is_file():
            relative = path.relative_to(root)
            size = path.stat().st_size
            files.append((relative, size, sha256_file(path)))
    return tuple(files)


def _summary(entries: tuple[tuple[Path, int, str], ...]) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    total = 0
    for relative, size, sha in entries:
        total += int(size)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return len(entries), total, digest.hexdigest()


def _has_partial_files(root: Path) -> bool:
    return any(path.is_file() and path.name.endswith(".part") for path in root.rglob("*"))


def migrate_sessions(
    *,
    profiles_root: Path | str,
    profile_name: str,
    target_root: Path | str,
    source_root: Path | str | None = None,
) -> SessionsMigrationResult:
    """Copy, verify, and activate a sessions root without deleting the source."""
    profile_root = _full(Path(profiles_root) / str(profile_name))
    source = _full(source_root) if source_root is not None else resolve_sessions_root(
        profiles_root, profile_name
    )
    target = _full(target_root)
    if source == target:
        raise ValueError("source_root 与 target_root 不能相同")
    if not source.is_dir():
        raise ValueError(f"源 sessions 目录不存在：{source}")
    _assert_no_reparse_tree(source)
    if _has_partial_files(source):
        raise RuntimeError("源 sessions 仍有 .part 文件，请先停止程序并完成封存")
    if target.exists():
        raise FileExistsError(f"目标目录已存在，为避免覆盖数据而拒绝：{target}")
    if str(target).casefold().startswith(str(source).casefold() + os.sep):
        raise ValueError("目标目录不能位于源 sessions 目录内部")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.migration-{uuid4().hex}.tmp"
    source_entries = _inventory(source)
    file_count, total_bytes, tree_sha = _summary(source_entries)
    pointer = sessions_root_override_path(profiles_root, profile_name)
    receipt_path = target / RECEIPT_FILENAME
    try:
        stage.mkdir(parents=True, exist_ok=False)
        for relative, _size, _sha in source_entries:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
        _assert_no_reparse_tree(stage)
        copied_entries = _inventory(stage)
        copied_count, copied_bytes, copied_sha = _summary(copied_entries)
        if (copied_count, copied_bytes, copied_sha) != (file_count, total_bytes, tree_sha):
            raise RuntimeError("目标 staging 校验失败，未切换 sessions 根目录")
        receipt = {
            "schema": MIGRATION_SCHEMA,
            "status": "active",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "profile": str(profile_name),
            "source_root": str(source),
            "target_root": str(target),
            "file_count": file_count,
            "bytes_copied": total_bytes,
            "tree_sha256": tree_sha,
            "source_preserved": True,
            "files": [
                {"path": relative.as_posix(), "bytes": size, "sha256": sha}
                for relative, size, sha in copied_entries
            ],
        }
        (stage / RECEIPT_FILENAME).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stage.replace(target)
        pointer = write_sessions_root_override(
            profiles_root, profile_name, target,
            source_root=source, receipt_path=target / RECEIPT_FILENAME,
        )
        return SessionsMigrationResult(
            "active", source, target, target / RECEIPT_FILENAME,
            file_count, total_bytes, tree_sha, pointer,
        )
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def rollback_sessions(*, profiles_root: Path | str, profile_name: str) -> SessionsMigrationResult:
    pointer = sessions_root_override_path(profiles_root, profile_name)
    if not pointer.is_file():
        raise ValueError("当前没有 sessions 迁移指针")
    target = resolve_sessions_root(profiles_root, profile_name)
    clear_sessions_root_override(profiles_root, profile_name)
    return SessionsMigrationResult(
        "rolled_back", target, Path(profiles_root).expanduser().resolve() / str(profile_name) / "sessions",
        target / RECEIPT_FILENAME if (target / RECEIPT_FILENAME).is_file() else None,
        0, 0, "", pointer,
    )


def sessions_migration_status(*, profiles_root: Path | str, profile_name: str) -> dict[str, object]:
    root = resolve_sessions_root(profiles_root, profile_name)
    pointer = sessions_root_override_path(profiles_root, profile_name)
    receipt = root / RECEIPT_FILENAME
    return {
        "schema": MIGRATION_SCHEMA,
        "profile": str(profile_name),
        "active_root": str(root),
        "source": "pointer" if pointer.is_file() else "profile_default",
        "pointer_path": str(pointer),
        "receipt_path": str(receipt) if receipt.is_file() else None,
    }


__all__ = [
    "RECEIPT_FILENAME", "SessionsMigrationResult", "migrate_sessions",
    "rollback_sessions", "sessions_migration_status",
]
