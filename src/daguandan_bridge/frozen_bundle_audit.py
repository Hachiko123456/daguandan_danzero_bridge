from __future__ import annotations

"""Fail-closed native/PE provenance audit for a frozen Windows bundle."""

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping, Sequence


NATIVE_AUDIT_SCHEMA = "guandan.native-dependency-audit/1"
_PE_SUFFIXES = {".exe", ".dll", ".pyd"}
_BANNED_SOURCE_TOKENS = ("java", "jdk", "anaconda", "miniconda", "poppler")
_SYSTEM_DLLS = {
    "advapi32.dll",
    "authz.dll",
    "avicap32.dll",
    "bcrypt.dll",
    "bcryptprimitives.dll",
    "cfgmgr32.dll",
    "combase.dll",
    "comctl32.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "d2d1.dll",
    "d3d11.dll",
    "d3d12.dll",
    "d3d9.dll",
    "dbghelp.dll",
    "dcomp.dll",
    "dwmapi.dll",
    "dwrite.dll",
    "dnsapi.dll",
    "dxgi.dll",
    "gdi32.dll",
    "gdi32full.dll",
    "imm32.dll",
    "icuuc.dll",
    "imagehlp.dll",
    "iphlpapi.dll",
    "kernel32.dll",
    "kernelbase.dll",
    "mpr.dll",
    "mf.dll",
    "mfplat.dll",
    "mfreadwrite.dll",
    "msvcp_win.dll",
    "msvcrt.dll",
    "ncrypt.dll",
    "netapi32.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "powrprof.dll",
    "propsys.dll",
    "psapi.dll",
    "rpcrt4.dll",
    "secur32.dll",
    "sechost.dll",
    "setupapi.dll",
    "shell32.dll",
    "shlwapi.dll",
    "user32.dll",
    "userenv.dll",
    "uiautomationcore.dll",
    "usp10.dll",
    "uxtheme.dll",
    "version.dll",
    "winhttp.dll",
    "winmm.dll",
    "wintrust.dll",
    "winspool.drv",
    "wsock32.dll",
    "wtsapi32.dll",
    "wldap32.dll",
    "ws2_32.dll",
}


class FrozenBundleAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenBundleAuditResult:
    document: dict[str, object]

    @property
    def ok(self) -> bool:
        return self.document.get("status") == "PASS"


def collect_pyinstaller_provenance(work_root: Path | str) -> dict[str, str]:
    """Read PyInstaller TOC literals and map bundle destination to source."""

    root = Path(work_root).resolve()
    if not root.is_dir():
        raise FrozenBundleAuditError("PyInstaller work root does not exist")
    result: dict[str, str] = {}
    for toc_path in sorted(root.rglob("*.toc")):
        if _is_link_or_reparse(toc_path):
            raise FrozenBundleAuditError("PyInstaller TOC path is a reparse point")
        try:
            payload = ast.literal_eval(toc_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError, ValueError):
            continue
        for destination, source in _toc_pairs(payload):
            if Path(source).suffix.casefold() not in _PE_SUFFIXES:
                continue
            try:
                normalized = _portable_relative(destination)
            except FrozenBundleAuditError:
                continue
            aliases = [normalized]
            if not normalized.casefold().startswith("_internal/"):
                # PyInstaller 6's onedir COLLECT TOC stores paths relative to
                # the contents directory, while the final bundle places that
                # directory at ``_internal``.
                aliases.append(f"_internal/{normalized}")
            for alias in aliases:
                existing = result.get(alias.casefold())
                if (
                    existing is not None
                    and _path_identity(existing) != _path_identity(source)
                ):
                    raise FrozenBundleAuditError(
                        f"conflicting PyInstaller provenance for {alias}"
                    )
                result[alias.casefold()] = str(Path(source).resolve())
    return result


def audit_frozen_bundle(
    bundle_root: Path | str,
    *,
    provenance: Mapping[str, str] | None = None,
    allowed_venv_root: Path | str | None = None,
    allowed_python_root: Path | str | None = None,
    project_root: Path | str | None = None,
    windows_root: Path | str | None = None,
    executable_name: str = "DaguandanAssistant.exe",
) -> FrozenBundleAuditResult:
    root = Path(bundle_root).resolve()
    if not root.is_dir() or _is_link_or_reparse(root):
        raise FrozenBundleAuditError("bundle root is unavailable or a reparse point")
    allowed = {
        "venv": _optional_root(allowed_venv_root),
        "python": _optional_root(allowed_python_root),
        "project": _optional_root(project_root),
        "windows_system": _optional_root(windows_root or os.environ.get("WINDIR")),
    }
    provenance_map = {str(key).replace("\\", "/").casefold(): str(value) for key, value in (provenance or {}).items()}
    errors: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    pe_paths = [
        path for path in _walk_files(root) if path.suffix.casefold() in _PE_SUFFIXES
    ]
    bundled_names = {path.name.casefold() for path in pe_paths}
    executable_identity = executable_name.casefold()
    for path in pe_paths:
        relative = path.relative_to(root).as_posix()
        content_hash = _sha256_file(path)
        source = provenance_map.get(relative.casefold())
        if source is None:
            # PyInstaller's generated executable has no source TOC entry.
            source_class = "pyinstaller-output" if relative.casefold() == executable_identity else "unknown"
            source_relative = path.name if source_class == "pyinstaller-output" else None
        else:
            source_class, source_relative = _classify_source(Path(source), allowed)
        record: dict[str, object] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": content_hash,
            "source_class": source_class,
            "source_relative": source_relative,
            "imports": [],
            "sections": [],
        }
        source_folded = str(source or "").casefold().replace("/", "\\")
        banned_tokens = [token for token in _BANNED_SOURCE_TOKENS if token in source_folded]
        if banned_tokens:
            errors.append(_error("NATIVE-BANNED-SOURCE", relative, {"tokens": banned_tokens}))
        if source_class == "unknown":
            errors.append(_error("NATIVE-UNKNOWN-PROVENANCE", relative, {}))
        lower_name = path.name.casefold()
        original_name = ""
        try:
            original_name = str(path.stat())  # overwritten by PE metadata when available
        except OSError:
            pass
        if lower_name == "icuuc.dll" or re.fullmatch(r"icu.*78\.dll", lower_name):
            errors.append(_error("NATIVE-CONFLICTING-ICU", relative, {"name": lower_name}))
        if _contains_upx_signature(path):
            errors.append(_error("NATIVE-UPX-DETECTED", relative, {}))
        imports, sections, pe_error = _pe_metadata(path)
        record["imports"] = imports
        record["sections"] = sections
        if pe_error is not None:
            errors.append(_error("NATIVE-INVALID-PE", relative, {"error": pe_error}))
        for imported in imports:
            normalized = imported.casefold()
            if _is_windows_system_import(normalized):
                continue
            if normalized not in bundled_names:
                errors.append(
                    _error(
                        "NATIVE-MISSING-IMPORT",
                        relative,
                        {"import": imported},
                    )
                )
        if lower_name.startswith("qt6") and not (
            source_class == "venv"
            and source_relative is not None
            and "pyside6" in source_relative.casefold()
        ):
            errors.append(_error("NATIVE-QT-PROVENANCE", relative, {"source_class": source_class}))
        if lower_name.startswith(("vcruntime", "msvcp")) and source_class not in {
            "venv",
            "python",
            "windows_system",
        }:
            errors.append(_error("NATIVE-CRT-PROVENANCE", relative, {"source_class": source_class}))
        records.append(record)

    error_codes = [str(item["code"]) for item in errors]
    document: dict[str, object] = {
        "schema": NATIVE_AUDIT_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "policy": {
            "upx_allowed": False,
            "unknown_provenance_allowed": False,
            "banned_source_tokens": list(_BANNED_SOURCE_TOKENS),
            "allowed_source_classes": [key for key, value in allowed.items() if value is not None]
            + ["pyinstaller-output"],
        },
        "summary": {
            "pe_file_count": len(records),
            "error_count": len(errors),
            "error_codes": sorted(set(error_codes)),
        },
        "files": sorted(records, key=lambda item: str(item["path"]).casefold()),
        "errors": errors,
    }
    return FrozenBundleAuditResult(document)


def write_native_audit(
    output_path: Path | str,
    result: FrozenBundleAuditResult,
) -> None:
    from .storage import atomic_write_json

    atomic_write_json(Path(output_path), result.document)


def _toc_pairs(value: object) -> Iterable[tuple[str, str]]:
    if isinstance(value, (list, tuple)):
        if (
            len(value) >= 2
            and isinstance(value[0], str)
            and isinstance(value[1], str)
            and _looks_like_file(value[1])
        ):
            yield value[0], value[1]
        for item in value:
            yield from _toc_pairs(item)
    elif isinstance(value, dict):
        for item in value.items():
            yield from _toc_pairs(item)


def _looks_like_file(value: str) -> bool:
    suffix = Path(value).suffix.casefold()
    return bool(suffix) and (Path(value).is_absolute() or "\\" in value or "/" in value)


def _classify_source(
    source: Path,
    roots: Mapping[str, Path | None],
) -> tuple[str, str | None]:
    resolved = source.resolve()
    for label in ("venv", "python", "project", "windows_system"):
        root = roots.get(label)
        if root is None:
            continue
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return label, relative.as_posix()
    return "unknown", None


def _pe_metadata(path: Path) -> tuple[list[str], list[str], str | None]:
    try:
        import pefile

        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        )
        imports = sorted(
            {
                entry.dll.decode("ascii", errors="replace")
                for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", ())
            },
            key=str.casefold,
        )
        sections = [
            section.Name.rstrip(b"\0").decode("ascii", errors="replace")
            for section in pe.sections
        ]
        pe.close()
        return imports, sections, None
    except Exception as exc:
        return [], [], type(exc).__name__


def _contains_upx_signature(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(min(path.stat().st_size, 2 * 1024 * 1024))
    except OSError:
        return False
    return b"UPX!" in head or b"UPX0" in head or b"UPX1" in head


def _is_windows_system_import(name: str) -> bool:
    return (
        name in _SYSTEM_DLLS
        or name.startswith("api-ms-win-")
        or name.startswith("ext-ms-win-")
    )


def _walk_files(root: Path) -> list[Path]:
    pending = [root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        if _is_link_or_reparse(directory):
            raise FrozenBundleAuditError("bundle tree contains a reparse point")
        with os.scandir(directory) as iterator:
            for entry in iterator:
                path = Path(entry.path)
                if _is_link_or_reparse(path):
                    raise FrozenBundleAuditError("bundle tree contains a reparse point")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().casefold())


def _portable_relative(value: object) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if (
        not path.parts
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise FrozenBundleAuditError("unsafe bundle-relative provenance path")
    return path.as_posix()


def _optional_root(value: Path | str | None) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(value).resolve()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        if callable(is_junction) and is_junction(path):
            return True
        return bool(
            getattr(path.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError:
        return False


def _error(code: str, path: str, evidence: Mapping[str, object]) -> dict[str, object]:
    return {"code": code, "path": path, "evidence": dict(evidence)}


def _path_identity(value: Path | str) -> str:
    # Windows packaged-app LocalAppData may expose the same physical file via
    # both the logical user path and Packages/.../LocalCache.  Provenance is
    # stored resolved, so comparisons must use the same canonicalization on
    # both sides; distinct physical sources still retain distinct identities.
    return os.path.normcase(str(Path(value).resolve(strict=False)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FrozenBundleAuditError",
    "FrozenBundleAuditResult",
    "NATIVE_AUDIT_SCHEMA",
    "audit_frozen_bundle",
    "collect_pyinstaller_provenance",
    "write_native_audit",
]
