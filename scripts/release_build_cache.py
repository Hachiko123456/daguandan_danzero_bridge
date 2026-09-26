"""Content-addressed release cache receipts; standard library only (-I -S safe).

Invoke with a trusted bootstrap, never with an interpreter from an unverified
cache. Receipts detect corruption, not an attacker who can replace both cache
and receipt. The caller must serialize builders/readers and keep the cache
unchanged between inspection and execution. No cached code is imported/executed.

All commands write one JSON object to stdout and, unless --output is '-', an
atomic copy to the requested existing parent directory. Help is also JSON.
Receipts live beside, never inside, the cache. No cache contents are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import struct
import sys
import tempfile
from typing import Any

HELPER_VERSION = "1"
RECEIPT_SCHEMA = "guandan.release-build-cache-receipt/1"
KEY_SCHEMA = "guandan.release-build-cache-key/1"
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_ENTRIES = 100_000
MAX_PATH_LENGTH = 4096
MAX_DEPTH = 128
ENVIRONMENT_FILES = (
    "requirements-release.in", "requirements-release.lock",
    "python_runtime.lock.json", "release_toolchain.lock.json", "wheelhouse.lock.json",
)
WORK_FILES = (
    "scripts/package_release.ps1", "scripts/pyinstaller_live_v2_collection.json",
    "app.ico", "run.py",
)
RESOURCE_FILES = (
    "data/profiles/tencent_daguandan/profile.json",
    "data/profiles/tencent_daguandan/regions_config.json",
    "data/profiles/tencent_daguandan/templates_config.json",
    "data/profiles/tencent_daguandan/models/best.npz",
    "src/daguandan_bridge/danzero/_vendor/guandan_rlcard/baselines/danzero/q_network.ckpt",
    "src/daguandan_bridge/fabledan/_vendor/FABLEDAN_LICENSE",
    "src/daguandan_bridge/fabledan/_vendor/FABLEDAN_REVISION",
)
RESOURCE_TREES = ("data/profiles/tencent_daguandan/templates", "release_assets")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789¹²³"
}


class CacheError(Exception):
    """A fail-closed error, rather than an ordinary cache miss."""


class UnsafePath(CacheError):
    pass


class InvalidReceipt(ValueError):
    pass


def _component(name: str) -> None:
    # Apply Windows-safe spelling even on POSIX, so receipts remain unambiguous.
    if (not name or name in (".", "..") or name.endswith((".", " "))
            or any(char in name for char in '\\/:<>"|?*')
            or any(ord(char) < 32 for char in name)
            or name.split(".", 1)[0].upper() in _RESERVED):
        raise UnsafePath(f"unsafe path component: {name!r}")


def _relative(raw: Any) -> str:
    if not isinstance(raw, str):
        raise InvalidReceipt("entry path is not a string")
    if len(raw) > MAX_PATH_LENGTH or len(raw.split("/")) > MAX_DEPTH:
        raise UnsafePath("relative path exceeds safety limits")
    for name in raw.split("/"):
        _component(name)
    return raw


def _absolute(raw: str | Path, *, nonroot: bool = True) -> Path:
    text = os.fspath(raw)
    if not text or "\x00" in text:
        raise UnsafePath("empty path or NUL in path")
    spelling = text.replace("\\", "/")
    if spelling.startswith(("//?/", "//./")):
        raise UnsafePath("device/extended namespace paths are not accepted")
    drive, tail = os.path.splitdrive(text)
    if drive and not tail.startswith(("/", "\\")):
        raise UnsafePath("drive-relative paths are not accepted")
    # Do this BEFORE abspath/Path can erase '..' or a linked path component.
    for name in tail.replace("\\", "/").split("/"):
        if name and name != ".":
            _component(name)
    path = Path(os.path.abspath(text))  # Deliberately never resolve().
    for name in path.parts[1:]:
        _component(name)
    if nonroot and path == path.parent:
        raise UnsafePath("filesystem roots are not accepted")
    return path


def _normal(path: Path) -> str:
    return os.path.normcase(str(path))


def _reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _chain(path: Path) -> os.stat_result | None:
    """lstat every ancestor, including the supplied root, without resolving it."""
    result = None
    for current in (*reversed(path.parents), path):
        try:
            result = current.lstat()
        except FileNotFoundError:
            result = None
            continue
        if _reparse(result):
            raise UnsafePath(f"symlink/junction/reparse point refused: {current}")
        if current != path and not stat.S_ISDIR(result.st_mode):
            raise UnsafePath(f"non-directory ancestor refused: {current}")
    return result


def _require_directory(path: Path) -> None:
    info = _chain(path)
    if info is None or not stat.S_ISDIR(info.st_mode):
        raise CacheError(f"required directory is missing or not a directory: {path}")


def _signature(info: os.stat_result) -> tuple[int, ...]:
    # Windows lstat synthesizes executable bits from the filename; fstat does not.
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _regular(path: Path, info: os.stat_result | None) -> os.stat_result:
    if info is None:
        raise FileNotFoundError(f"required file is missing: {path}")
    if _reparse(info) or not stat.S_ISREG(info.st_mode):
        raise UnsafePath(f"non-regular/reparse file refused: {path}")
    return info


def _read_file(path: Path, *, limit: int | None = None,
               expected: os.stat_result | None = None) -> tuple[int, str | bytes]:
    before = _regular(path, _chain(path))
    if expected is not None and _signature(before) != _signature(expected):
        raise CacheError(f"file changed during inventory: {path}")
    if limit is not None and before.st_size > limit:
        raise InvalidReceipt("receipt exceeds byte limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    with os.fdopen(descriptor, "rb") as stream:
        opened = _regular(path, os.fstat(stream.fileno()))
        # On Windows 3.12 lstat/fstat can expose different ctime semantics.
        # Compare identity/size/mtime across APIs, and ctime within each API.
        path_identity = _signature(before)[:-1] if os.name == 'nt' else _signature(before)
        open_identity = _signature(opened)[:-1] if os.name == 'nt' else _signature(opened)
        if open_identity != path_identity:
            raise CacheError(f"file replaced while opening: {path}")
        while True:
            chunk = stream.read(1024 * 1024 if limit is None else min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if limit is not None:
                if size > limit:
                    raise InvalidReceipt("receipt exceeds byte limit")
                chunks.append(chunk)
            else:
                digest.update(chunk)
        after = os.fstat(stream.fileno())
    final = _regular(path, _chain(path))
    if (size != before.st_size or _signature(opened) != _signature(after)
            or _signature(before) != _signature(final)):
        raise CacheError(f"file changed while reading: {path}")
    return size, b"".join(chunks) if limit is not None else digest.hexdigest()


def _record(path: Path, relative: str, info: os.stat_result | None = None) -> dict[str, Any]:
    size, digest = _read_file(path, expected=info)
    return {"path": relative, "size": size, "sha256": digest}


def _source_generated(relative: str) -> bool:
    parts = relative.split("/")
    return (any(part == "__pycache__" or part.endswith(".egg-info") for part in parts)
            or parts[-1].endswith((".pyc", ".pyo")))


def _tree(path: Path, *, source: bool = False) -> tuple[dict[str, os.stat_result], dict[str, os.stat_result]]:
    _require_directory(path)
    files: dict[str, os.stat_result] = {}
    directories: dict[str, os.stat_result] = {}
    pending = [(path, "")]
    seen: set[str] = set()
    while pending:
        directory, prefix = pending.pop()
        _require_directory(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = _relative(f"{prefix}/{entry.name}" if prefix else entry.name)
                # Windows DirEntry.stat caches zero st_ino/st_dev; lstat is needed
                # for a comparable file identity during the subsequent hash.
                info = os.lstat(entry.path)
                if _reparse(info):
                    raise UnsafePath(f"symlink/junction/reparse point refused: {entry.path}")
                if (source and not stat.S_ISDIR(info.st_mode)
                        and _source_generated(relative) and not relative.lower().endswith('.py')):
                    continue
                if relative.casefold() in seen:
                    raise UnsafePath(f"case-aliased cache paths refused: {relative}")
                seen.add(relative.casefold())
                if len(seen) > MAX_ENTRIES:
                    raise CacheError("tree exceeds entry limit")
                if stat.S_ISDIR(info.st_mode):
                    directories[relative] = info
                    pending.append((Path(entry.path), relative))
                else:
                    _regular(Path(entry.path), info)
                    files[relative] = info
    if source:
        # All .py files count, even in a generated-looking directory; ordinary
        # bytecode/egg metadata and their otherwise-empty directories do not.
        parents = {name.rsplit('/', depth)[0] for name in files
                   for depth in range(1, name.count('/') + 1)}
        directories = {name: info for name, info in directories.items()
                       if not _source_generated(name) or name in parents}
    return files, directories


def _same_tree(first: tuple[dict, dict], second: tuple[dict, dict]) -> bool:
    return all(
        before.keys() == after.keys()
        and all(_signature(info) == _signature(after[name]) for name, info in before.items())
        for before, after in zip(first, second)
    )


def _inventory(directory: Path, *, source: bool = False) -> tuple[list[dict], list[str]]:
    snapshot = _tree(directory, source=source)
    records = [_record(directory / name, name, info) for name, info in sorted(snapshot[0].items())]
    if not _same_tree(snapshot, _tree(directory, source=source)):
        raise CacheError(f"tree changed during inventory: {directory}")
    return records, sorted(snapshot[1])


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def build_keys(project_root: str | Path, python_root: str | Path) -> dict[str, Any]:
    root, python = _absolute(project_root), _absolute(python_root)
    _require_directory(root)
    _require_directory(python)
    environment = {
        "helper_version": HELPER_VERSION,
        "helper_sha256": _record(_absolute(__file__), "helper")["sha256"],
        "project_root": _normal(root),
        "python_root": _normal(python),
        "python": {"version": sys.version, "implementation": sys.implementation.name,
                   "cache_tag": sys.implementation.cache_tag, "platform": sys.platform,
                   "machine": platform.machine(), "pointer_bits": struct.calcsize("P") * 8},
        "files": [_record(root / name, name) for name in ENVIRONMENT_FILES],
    }
    work_files = {name: _record(root / name, name) for name in (*WORK_FILES, *RESOURCE_FILES)}
    work_directories = []
    for subtree in ("src", *RESOURCE_TREES):
        records, directories = _inventory(root / subtree, source=subtree == "src")
        if subtree == "src" and not any(record["path"].endswith(".py") for record in records):
            raise CacheError("required source tree has no Python files")
        work_directories.extend([subtree, *(f"{subtree}/{name}" for name in directories)])
        for record in records:
            relative = f"{subtree}/{record['path']}"
            work_files[relative] = {**record, "path": relative}
    environment_key = _digest({"schema": KEY_SCHEMA, "environment": environment})
    work = {"files": [work_files[name] for name in sorted(work_files)],
            "directories": sorted(work_directories)}
    return {"schema": KEY_SCHEMA, "environment_key": environment_key,
            "work_key": _digest({"schema": KEY_SCHEMA, "environment_key": environment_key, "work": work}),
            "inputs": {"environment": environment, "work": work}}


def _key(key: str) -> str:
    if not isinstance(key, str) or not _HEX.fullmatch(key.lower()):
        raise CacheError("key must contain exactly 64 hexadecimal characters")
    return key.lower()


def _cache_paths(directory: str | Path) -> tuple[Path, Path]:
    path = _absolute(directory)
    receipt = path.with_name(path.name + ".receipt.json")
    _chain(path)
    _chain(receipt)
    return path, receipt


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidReceipt(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load_receipt(path: Path) -> dict[str, Any]:
    _, raw = _read_file(path, limit=MAX_RECEIPT_BYTES)
    value = json.loads(raw, object_pairs_hook=_duplicate_keys)
    if not isinstance(value, dict) or value.get("schema") != RECEIPT_SCHEMA:
        raise InvalidReceipt("unsupported receipt schema")
    if value.get("helper_version") != HELPER_VERSION:
        raise InvalidReceipt("unsupported helper version")
    if value.get("kind") not in ("environment", "work"):
        raise InvalidReceipt("invalid receipt kind")
    if not isinstance(value.get("key"), str) or not _HEX.fullmatch(value["key"]):
        raise InvalidReceipt("invalid receipt key")
    if not isinstance(value.get("directory"), str):
        raise InvalidReceipt("invalid receipt directory")
    # Check spelling only: never follow or open a root supplied by the receipt.
    _absolute(value["directory"])
    files, directories = value.get("files"), value.get("directories")
    if not isinstance(files, list) or not isinstance(directories, list):
        raise InvalidReceipt("missing file/directory inventory")
    if not files or len(files) + len(directories) > MAX_ENTRIES:
        raise InvalidReceipt("empty or oversized inventory")
    if type(value.get("file_count")) is not int or value["file_count"] != len(files):
        raise InvalidReceipt("file count mismatch")
    seen: set[str] = set()
    for item in [*directories, *files]:
        if not isinstance(item, (str, dict)):
            raise InvalidReceipt("invalid inventory entry")
        relative = _relative(item.get("path") if isinstance(item, dict) else item)
        if relative.casefold() in seen:
            raise InvalidReceipt("duplicate/aliased inventory path")
        seen.add(relative.casefold())
    if any(not isinstance(name, str) for name in directories):
        raise InvalidReceipt("invalid directory entry")
    for item in files:
        if (not isinstance(item, dict) or type(item.get("size")) is not int or item["size"] < 0
                or not isinstance(item.get("sha256"), str) or not _HEX.fullmatch(item["sha256"])):
            raise InvalidReceipt("invalid file entry")
    return value


def inspect_cache(directory: str | Path, kind: str, key: str) -> dict[str, Any]:
    key = _key(key)
    path, receipt = _cache_paths(directory)
    count = 0

    def miss(reason: str, **extra: Any) -> dict[str, Any]:
        return {"hit": False, "reason": reason, "file_count": count,
                "receipt_path": str(receipt), **extra}

    if _chain(path) is None:
        return miss("directory_missing")
    if not path.is_dir():
        return miss("not_a_directory")
    try:
        # Even on a key/receipt miss, reject an unsafe tree before the caller can
        # decide to rebuild it. Receipt paths are never used to open cache files.
        snapshot = _tree(path)
        count = len(snapshot[0])
        if _chain(receipt) is None:
            return miss("receipt_missing")
        try:
            value = _load_receipt(receipt)
        except (ValueError, UnicodeError, RecursionError) as exc:
            return miss("receipt_invalid", detail=str(exc))
        if value["directory"] != _normal(path):
            return miss("directory_mismatch")
        if value["kind"] != kind:
            return miss("kind_mismatch")
        if value["key"] != key:
            return miss("key_mismatch")
        expected = {item["path"]: item for item in value["files"]}
        actual = snapshot[0]
        if expected.keys() != actual.keys():
            return miss("file_set_mismatch", missing_count=len(expected.keys() - actual.keys()),
                        extra_count=len(actual.keys() - expected.keys()))
        if set(value["directories"]) != snapshot[1].keys():
            return miss("directory_set_mismatch")
        for relative, info in sorted(actual.items()):
            record = _record(path / relative, relative, info)
            if record["size"] != expected[relative]["size"]:
                return miss("file_size_mismatch", file=relative)
            if record["sha256"] != expected[relative]["sha256"]:
                return miss("file_hash_mismatch", file=relative)
        if not _same_tree(snapshot, _tree(path)):
            return miss("tree_changed")
        return {"hit": True, "reason": "verified", "file_count": count,
                "receipt_path": str(receipt)}
    except UnsafePath:
        raise
    except (OSError, CacheError) as exc:
        return miss("unreadable_or_changed", detail=str(exc))


def _atomic_json(path: Path, value: Any, *, receipt: bool = False) -> None:
    _require_directory(path.parent)
    existing = _chain(path)
    if existing is not None:
        _regular(path, existing)
    data = _json_bytes(value)
    if receipt and len(data) > MAX_RECEIPT_BYTES:
        raise CacheError("generated receipt exceeds byte limit")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _chain(path)
        _regular(temp_path, _chain(temp_path))
        os.replace(temp_path, path)
    finally:
        # Only our random temporary FILE; never remove a cache or directory.
        _chain(temp_path.parent)
        temp_path.unlink(missing_ok=True)


def seal_cache(directory: str | Path, kind: str, key: str) -> dict[str, Any]:
    key = _key(key)
    if kind not in ("environment", "work"):
        raise CacheError("kind must be environment or work")
    path, receipt = _cache_paths(directory)
    files, directories = _inventory(path)
    if not files:
        raise CacheError("refusing to seal an empty cache")
    value = {"schema": RECEIPT_SCHEMA, "helper_version": HELPER_VERSION,
             "directory": _normal(path), "kind": kind, "key": key,
             "file_count": len(files), "files": files, "directories": directories}
    _atomic_json(receipt, value, receipt=True)
    return {"sealed": True, "kind": kind, "key": key, "receipt_path": str(receipt),
            "file_count": len(files), "directory_count": len(directories)}


def _below(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([_normal(path), _normal(root)]) == _normal(root)
    except ValueError:
        return False


def _output_path(args: argparse.Namespace) -> Path | None:
    if args.output == "-":
        return None
    output = _absolute(args.output)
    _require_directory(output.parent)
    existing = _chain(output)
    if existing is not None:
        _regular(output, existing)
    if args.command in ("inspect", "seal"):
        directory, receipt = _cache_paths(args.directory)
        if _below(output, directory) or _normal(output) == _normal(receipt):
            raise UnsafePath("output must be outside the cache and distinct from its receipt")
    else:
        root = _absolute(args.project_root)
        if (any(_normal(output) == _normal(root / name)
                for name in (*ENVIRONMENT_FILES, *WORK_FILES, *RESOURCE_FILES))
                or any(_below(output, root / name) for name in ("src", *RESOURCE_TREES))
                or _normal(output) == _normal(_absolute(__file__))):
            raise UnsafePath("output must not overwrite a build input")
    return output


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CacheError(message)

    def print_help(self, file: Any = None) -> None:
        print(json.dumps({"help": self.format_help()}, ensure_ascii=True), file=file)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    key = commands.add_parser("key", help="Hash environment and source/resource inputs.")
    key.add_argument("--project-root", required=True)
    key.add_argument("--python-root", required=True)
    key.add_argument("--output", required=True, help="JSON output file; '-' means stdout only.")
    for command in ("inspect", "seal"):
        sub = commands.add_parser(command, help=f"{command.capitalize()} a complete cache directory.")
        sub.add_argument("--directory", required=True)
        sub.add_argument("--kind", required=True, choices=("environment", "work"))
        sub.add_argument("--key", required=True)
        sub.add_argument("--output", required=True, help="JSON output file; '-' means stdout only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        output = _output_path(args)
        if args.command == "key":
            value = build_keys(args.project_root, args.python_root)
        elif args.command == "inspect":
            value = inspect_cache(args.directory, args.kind, args.key)
        else:
            value = seal_cache(args.directory, args.kind, args.key)
        if output is not None:
            _atomic_json(output, value)
        print(_json_bytes(value).decode("utf-8"), end="")
        return 0
    except (CacheError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "reason": "unsafe_path" if isinstance(exc, UnsafePath)
                          else "error", "hit": False}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
