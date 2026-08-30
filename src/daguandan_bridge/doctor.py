"""Pre-GUI installation and resource diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Callable, Sequence
from uuid import uuid4

from .build_manifest import BUILD_MANIFEST_FILENAME, verify_build_manifest
from .runtime_identity import application_root, get_runtime_identity
from .runtime_layout import (
    DATA_ROOT_ENV,
    RuntimeLayout,
    RuntimeLayoutError,
    ensure_runtime_layout,
    resolve_runtime_layout,
)
from .startup_diagnostics import current_startup_diagnostics, record_startup_event


DOCTOR_SCHEMA = "guandan.doctor/1"
IMPORT_PROBE_SCHEMA = "guandan.doctor-import-probe/1"
MINIMUM_FREE_BYTES = 512 * 1024 * 1024
MAX_DIAGNOSTIC_MESSAGE_CHARS = 500
_BEARER_SECRET = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_KEY_VALUE_SECRET = re.compile(
    r"\b([A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?key|authorization|password|"
    r"passwd|secret|token))"
    r"(\s*[:=]\s*)(\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)",
    re.IGNORECASE,
)
_JWT_SECRET = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_KNOWN_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DependencySpec:
    check_id: str
    module: str
    distribution: str


DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("DEPENDENCY-NUMPY", "numpy", "numpy"),
    DependencySpec("DEPENDENCY-OPENCV", "cv2", "opencv-python"),
    DependencySpec("DEPENDENCY-PYSIDE6", "PySide6", "PySide6"),
    DependencySpec("DEPENDENCY-PYSIDE6-QTGUI", "PySide6.QtGui", "PySide6"),
    DependencySpec(
        "DEPENDENCY-QFLUENTWIDGETS",
        "qfluentwidgets",
        "PySide6-Fluent-Widgets",
    ),
    DependencySpec(
        "DEPENDENCY-QFRAMELESSWINDOW",
        "qframelesswindow",
        "PySideSix-Frameless-Window",
    ),
    DependencySpec("DEPENDENCY-MSS", "mss", "mss"),
    DependencySpec("DEPENDENCY-PYWIN32", "win32gui", "pywin32"),
    DependencySpec("DEPENDENCY-TORCH", "torch", "torch"),
    DependencySpec("DEPENDENCY-RLCARD", "rlcard.envs", "rlcard"),
    DependencySpec(
        "DEPENDENCY-RLCARD-BLACKJACK",
        "rlcard.envs.blackjack",
        "rlcard",
    ),
)


def collect_doctor_report(
    *,
    root: Path | str | None = None,
    profile_name: str = "tencent_daguandan",
    dependencies: Sequence[DependencySpec] = DEPENDENCIES,
    dependency_probe: Callable[[DependencySpec], dict[str, object]] | None = None,
    startup_state: object | None = None,
    frozen: bool | None = None,
    data_root: Path | str | None = None,
    environ: dict[str, str] | None = None,
    layout: RuntimeLayout | None = None,
) -> dict[str, object]:
    """Collect deterministic checks without constructing GUI/game services."""

    started = perf_counter()
    bundle_root = Path(root) if root is not None else application_root()
    startup = startup_state or current_startup_diagnostics()
    checks: list[dict[str, object]] = []

    identity = get_runtime_identity()
    is_frozen = bool(identity.get("frozen")) if frozen is None else bool(frozen)
    build_status = str(identity.get("build_status") or "invalid")
    checks.append(
        _timed_result(
            "IDENTITY-RUNTIME",
            "WARN" if build_status in {"unidentified", "invalid"} else "PASS",
            "Runtime identity is available",
            {
                "run_id": identity.get("run_id"),
                "build_status": build_status,
                "build_id": identity.get("build_id"),
                "frozen": bool(identity.get("frozen")),
            },
            0.0,
        )
    )
    checks.append(_platform_check())
    checks.append(_python_check())
    checks.append(_architecture_check())
    checks.append(_diagnostics_storage_check(startup))
    checks.append(_build_integrity_check(bundle_root, frozen=is_frozen))

    bundle_data_root = bundle_root / "data"
    checks.append(
        _readable_directory_check(
            "STORAGE-BUNDLE-DATA",
            bundle_data_root,
        )
    )
    selected_layout: RuntimeLayout | None = None
    layout_error: BaseException | None = None
    layout_started = perf_counter()
    try:
        if layout is not None:
            selected_layout = layout
        else:
            layout_environment = dict(os.environ if environ is None else environ)
            if data_root is not None:
                layout_environment[DATA_ROOT_ENV] = str(Path(data_root))
            selected_layout = resolve_runtime_layout(
                environ=layout_environment,
                frozen=is_frozen,
                bundle_root=bundle_root,
            )
        selected_layout = ensure_runtime_layout(selected_layout)
    except BaseException as exc:
        layout_error = exc

    if selected_layout is None or layout_error is not None:
        checks.append(
            _timed_result(
                "STORAGE-RUNTIME-LAYOUT",
                "FAIL",
                "Writable runtime data layout could not be initialized",
                {
                    "frozen": is_frozen,
                    "error_type": type(layout_error).__name__ if layout_error else None,
                    "reason": _sanitize_diagnostic_text(layout_error or "unavailable"),
                },
                (perf_counter() - layout_started) * 1000,
            )
        )
        profile_root = bundle_data_root / "profiles" / profile_name
    else:
        checks.append(
            _timed_result(
                "STORAGE-RUNTIME-LAYOUT",
                "PASS",
                "Writable runtime data layout is initialized",
                selected_layout.sanitized_identity(),
                (perf_counter() - layout_started) * 1000,
            )
        )
        checks.append(
            _storage_check(
                "STORAGE-DATA",
                selected_layout.data_dir,
                must_exist=True,
            )
        )
        profile_root = selected_layout.profiles_root / profile_name

    json_documents: dict[str, object] = {}
    for check_id, filename, expected_key in (
        ("RESOURCE-PROFILE-JSON", "profile.json", None),
        ("RESOURCE-REGIONS-JSON", "regions_config.json", "regions"),
        ("RESOURCE-TEMPLATES-JSON", "templates_config.json", "templates"),
    ):
        result, document = _json_resource_check(
            check_id,
            profile_root / filename,
            expected_key=expected_key,
        )
        checks.append(result)
        if document is not None:
            json_documents[filename] = document

    checks.append(
        _template_files_check(
            profile_root,
            json_documents.get("templates_config.json"),
        )
    )
    checks.append(
        _file_resource_check(
            "RESOURCE-MODEL-FABLEDAN",
            profile_root / "models" / "best.npz",
            logical_name="models/best.npz",
        )
    )
    danzero_model = _danzero_model_path(bundle_root, profile_root)
    checks.append(
        _file_resource_check(
            "RESOURCE-MODEL-DANZERO",
            danzero_model,
            logical_name="models/danzero/q_network.ckpt",
        )
    )

    probe = dependency_probe or _subprocess_dependency_probe
    for dependency in dependencies:
        probe_started = perf_counter()
        try:
            raw = probe(dependency)
            status = str(raw.get("status") or "FAIL")
            summary = str(raw.get("summary") or "Dependency probe completed")
            evidence = raw.get("evidence")
            if not isinstance(evidence, dict):
                evidence = {}
        except BaseException as exc:
            status = "FAIL"
            summary = "Dependency probe raised an internal exception"
            evidence = {"error_type": type(exc).__name__}
        checks.append(
            _timed_result(
                dependency.check_id,
                status,
                summary,
                evidence,
                (perf_counter() - probe_started) * 1000,
            )
        )

    overall_status = "FAIL" if any(item["status"] == "FAIL" for item in checks) else "PASS"
    report = {
        "schema": DOCTOR_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": identity.get("run_id"),
        "overall_status": overall_status,
        "identity": identity,
        "capabilities": {
            "startup_diagnostics": True,
            "resource_checks": True,
            "isolated_dependency_imports": True,
            "window_probe": False,
            "capture_probe": False,
            "recognition_probe": False,
            "support_zip": False,
            "frames": False,
            "roi": False,
            "recognition_trace": False,
        },
        "checks": checks,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
    }
    return report


def run_doctor(output_path: Path | str | None = None) -> int:
    """Write/print one report and return the documented doctor exit code."""

    try:
        report = collect_doctor_report()
        exit_code = 2 if report["overall_status"] == "FAIL" else 0
    except BaseException as exc:
        identity = _safe_identity()
        report = {
            "schema": DOCTOR_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": identity.get("run_id"),
            "overall_status": "ERROR",
            "identity": identity,
            "capabilities": {
                "window_probe": False,
                "capture_probe": False,
                "recognition_probe": False,
                "support_zip": False,
                "frames": False,
                "roi": False,
                "recognition_trace": False,
            },
            "checks": [
                _timed_result(
                    "DOCTOR-INTERNAL",
                    "FAIL",
                    "Doctor encountered an internal exception",
                    {"error_type": type(exc).__name__},
                    0.0,
                )
            ],
            "duration_ms": 0.0,
        }
        exit_code = 3

    try:
        destination = _doctor_output_path(output_path)
        _atomic_write_json(destination, report)
        record_startup_event(
            "doctor_completed",
            {
                "overall_status": report["overall_status"],
                "exit_code": exit_code,
                "output_file": destination.name,
            },
        )
    except BaseException as exc:
        report["overall_status"] = "ERROR"
        report["checks"] = list(report.get("checks", [])) + [
            _timed_result(
                "DOCTOR-OUTPUT",
                "FAIL",
                "Doctor report could not be written",
                {"error_type": type(exc).__name__},
                0.0,
            )
        ]
        exit_code = 3

    try:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException:
        pass
    return exit_code


def run_import_probe(probe_id: str, output_path: Path | str | None) -> int:
    """Internal child-process entry point used to isolate native imports."""

    by_id = {item.check_id: item for item in DEPENDENCIES}
    dependency = by_id.get(str(probe_id))
    started = perf_counter()
    if dependency is None:
        payload = {
            "schema": IMPORT_PROBE_SCHEMA,
            "status": "FAIL",
            "summary": "Unknown dependency probe",
            "evidence": {"probe_id": str(probe_id)},
            "duration_ms": 0.0,
        }
        code = 2
    else:
        version = _distribution_version(dependency.distribution)
        try:
            importlib.import_module(dependency.module)
        except BaseException as exc:
            evidence: dict[str, object] = {
                "module": dependency.module,
                "distribution": dependency.distribution,
                "version": version,
                "error_type": type(exc).__name__,
                "message": _sanitize_diagnostic_text(exc),
            }
            for attribute in ("winerror", "errno"):
                value = getattr(exc, attribute, None)
                if isinstance(value, int) and not isinstance(value, bool):
                    evidence[attribute] = value
            payload = {
                "schema": IMPORT_PROBE_SCHEMA,
                "status": "FAIL",
                "summary": "Dependency could not be imported",
                "evidence": evidence,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
            code = 2
        else:
            payload = {
                "schema": IMPORT_PROBE_SCHEMA,
                "status": "PASS",
                "summary": "Dependency imported successfully",
                "evidence": {
                    "module": dependency.module,
                    "distribution": dependency.distribution,
                    "version": version,
                },
                "duration_ms": round((perf_counter() - started) * 1000, 3),
            }
            code = 0
    if output_path is not None:
        try:
            _atomic_write_json(Path(output_path), payload)
        except BaseException:
            return 3
    try:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except BaseException:
        pass
    return code


def _safe_identity() -> dict[str, object]:
    try:
        return get_runtime_identity()
    except BaseException:
        return {
            "schema": "guandan.runtime-identity/1",
            "run_id": "unavailable",
            "build_status": "invalid",
            "build_id": "invalid",
            "implementation_fingerprint": "unavailable",
            "executable_path": Path(sys.executable).name,
        }


def _timed_result(
    check_id: str,
    status: str,
    summary: str,
    evidence: dict[str, object],
    duration_ms: float,
) -> dict[str, object]:
    return {
        "id": str(check_id),
        "status": status if status in {"PASS", "WARN", "FAIL"} else "FAIL",
        "summary": str(summary),
        "evidence": evidence,
        "duration_ms": round(max(float(duration_ms), 0.0), 3),
    }


def _platform_check() -> dict[str, object]:
    started = perf_counter()
    is_windows = platform.system() == "Windows"
    return _timed_result(
        "ENV-OS",
        "PASS" if is_windows else "WARN",
        "Operating system detected",
        {
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
            "windows_supported": is_windows,
        },
        (perf_counter() - started) * 1000,
    )


def _python_check() -> dict[str, object]:
    started = perf_counter()
    supported = sys.version_info[:2] == (3, 12)
    return _timed_result(
        "ENV-PYTHON",
        "PASS" if supported else "WARN",
        "Python runtime detected",
        {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "supported_version": supported,
        },
        (perf_counter() - started) * 1000,
    )


def _architecture_check() -> dict[str, object]:
    started = perf_counter()
    bits = platform.architecture()[0]
    supported = bits == "64bit"
    return _timed_result(
        "ENV-ARCHITECTURE",
        "PASS" if supported else "FAIL",
        "Process architecture detected",
        {"bits": bits, "machine": platform.machine() or "unknown"},
        (perf_counter() - started) * 1000,
    )


def _diagnostics_storage_check(startup: object) -> dict[str, object]:
    started = perf_counter()
    enabled = bool(getattr(startup, "enabled", False))
    run_directory = getattr(startup, "run_directory", None)
    if not enabled or run_directory is None:
        return _timed_result(
            "STORAGE-DIAGNOSTICS",
            "FAIL",
            "Diagnostics directory is not writable",
            {
                "root_source": getattr(startup, "root_source", "unknown"),
                "error_type": _error_type(getattr(startup, "error", None)),
            },
            (perf_counter() - started) * 1000,
        )
    result = _storage_check(
        "STORAGE-DIAGNOSTICS",
        Path(run_directory),
        must_exist=True,
    )
    result["evidence"]["root_source"] = getattr(startup, "root_source", "unknown")
    result["duration_ms"] = round((perf_counter() - started) * 1000, 3)
    return result


def _build_integrity_check(bundle_root: Path, *, frozen: bool) -> dict[str, object]:
    started = perf_counter()
    manifest_path = bundle_root / BUILD_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return _timed_result(
            "BUILD-INTEGRITY",
            "FAIL" if frozen else "WARN",
            "Frozen build manifest is missing"
            if frozen
            else "Source checkout has no build manifest",
            {
                "manifest_file": BUILD_MANIFEST_FILENAME,
                "manifest_present": False,
                "frozen": bool(frozen),
                "checked_files": 0,
                "build_id": None,
                "errors": ["build_manifest_missing"],
                "warnings": [],
                "mutable_differences": [],
                "unexpected_files": [],
            },
            (perf_counter() - started) * 1000,
        )
    try:
        verification = verify_build_manifest(
            bundle_root,
            manifest_path,
            strict=True,
        )
        errors = [
            _sanitize_integrity_error(error, bundle_root)
            for error in verification.errors
        ]
        warnings = [
            _sanitize_integrity_error(warning, bundle_root)
            for warning in verification.warnings
        ]
        mutable_differences = [
            _sanitize_integrity_error(difference, bundle_root)
            for difference in verification.mutable_differences
        ]
        unexpected_files = [
            _sanitize_integrity_error(path, bundle_root)
            for path in verification.unexpected_files
        ]
        # Runtime resources are copied to a writable generation.  Therefore a
        # profile/template/model difference inside a frozen package is package
        # corruption, even while the phase-one manifest schema still labels
        # those entries as historically mutable.
        if frozen and mutable_differences:
            errors.extend(
                f"immutable bundle resource changed: {difference}"
                for difference in mutable_differences
            )
        if errors or not verification.ok:
            status = "FAIL"
            summary = "Build manifest or an immutable file failed integrity verification"
        elif warnings or mutable_differences:
            status = "WARN"
            summary = "Build is intact but mutable runtime files differ from the release"
        else:
            status = "PASS"
            summary = "Build manifest file integrity verified"
        return _timed_result(
            "BUILD-INTEGRITY",
            status,
            summary,
            {
                "manifest_file": BUILD_MANIFEST_FILENAME,
                "manifest_present": True,
                "frozen": bool(frozen),
                "checked_files": int(verification.checked_files),
                "build_id": (
                    _sanitize_integrity_error(verification.build_id, bundle_root)
                    if verification.build_id is not None
                    else None
                ),
                "errors": errors,
                "warnings": warnings,
                "mutable_differences": mutable_differences,
                "unexpected_files": unexpected_files,
            },
            (perf_counter() - started) * 1000,
        )
    except BaseException as exc:
        return _timed_result(
            "BUILD-INTEGRITY",
            "FAIL",
            "Build integrity verification could not complete",
            {
                "manifest_file": BUILD_MANIFEST_FILENAME,
                "manifest_present": True,
                "frozen": bool(frozen),
                "checked_files": 0,
                "build_id": None,
                "errors": [f"internal:{type(exc).__name__}"],
                "warnings": [],
                "mutable_differences": [],
                "unexpected_files": [],
            },
            (perf_counter() - started) * 1000,
        )


def _sanitize_integrity_error(error: object, bundle_root: Path) -> str:
    return _sanitize_diagnostic_text(error, roots=(bundle_root,))


def _sanitize_diagnostic_text(
    value: object,
    *,
    roots: Sequence[Path] = (),
) -> str:
    """Return a bounded support-safe error detail while preserving root-cause text."""

    text = str(value).replace("\r", " ").replace("\n", " ")
    candidates = {
        str(path)
        for root in roots
        for path in (root, root.resolve())
    }
    candidates.add(str(Path.home()))
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate:
            text = text.replace(candidate, "<PATH>")
            text = text.replace(candidate.replace("\\", "/"), "<PATH>")
    text = re.sub(
        r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/])[^\s\"'<>|?*,;]+",
        "<PATH>",
        text,
    )
    text = re.sub(
        r"(?<![\\/])(?:[\\/]{2})[^\\/\s\"'<>|?*]+[\\/]"
        r"[^\s\"'<>|?*,;]+",
        "<UNC_PATH>",
        text,
    )
    text = _BEARER_SECRET.sub("Bearer <REDACTED>", text)
    text = _JWT_SECRET.sub("<REDACTED_TOKEN>", text)
    text = _KNOWN_SECRET.sub("<REDACTED_TOKEN>", text)
    text = _KEY_VALUE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<REDACTED>",
        text,
    )
    for identity in (
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("COMPUTERNAME", ""),
        Path.home().name,
        platform.node(),
    ):
        if len(identity.strip()) >= 3:
            text = re.sub(
                re.escape(identity.strip()),
                "<REDACTED>",
                text,
                flags=re.IGNORECASE,
            )
    return text[:MAX_DIAGNOSTIC_MESSAGE_CHARS]


def _sanitize_probe_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = _sanitize_diagnostic_text(raw_key)[:80]
        if isinstance(raw_value, str):
            result[key] = _sanitize_diagnostic_text(raw_value)
        elif isinstance(raw_value, (int, float, bool)) or raw_value is None:
            result[key] = raw_value
        else:
            result[key] = _sanitize_diagnostic_text(raw_value)
    return result


def _completed_stderr_summary(completed: object) -> str:
    raw = getattr(completed, "stderr", b"")
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw or "")
    return _sanitize_diagnostic_text(text)


def _storage_check(check_id: str, path: Path, *, must_exist: bool) -> dict[str, object]:
    started = perf_counter()
    if must_exist and not path.is_dir():
        return _timed_result(
            check_id,
            "FAIL",
            "Required directory is missing",
            {"directory": path.name, "exists": False},
            (perf_counter() - started) * 1000,
        )
    probe_path: Path | None = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / f".doctor-write-{uuid4().hex}.tmp"
        probe_path.write_bytes(b"doctor")
        probe_path.unlink()
        probe_path = None
        free_bytes = int(shutil.disk_usage(path).free)
    except BaseException as exc:
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
        return _timed_result(
            check_id,
            "FAIL",
            "Directory is not writable",
            {
                "directory": path.name,
                "exists": path.is_dir(),
                "error_type": type(exc).__name__,
            },
            (perf_counter() - started) * 1000,
        )
    enough_space = free_bytes >= MINIMUM_FREE_BYTES
    return _timed_result(
        check_id,
        "PASS" if enough_space else "FAIL",
        "Directory is writable and disk space was measured",
        {
            "directory": path.name,
            "exists": True,
            "writable": True,
            "free_bytes": free_bytes,
            "required_free_bytes": MINIMUM_FREE_BYTES,
        },
        (perf_counter() - started) * 1000,
    )


def _readable_directory_check(check_id: str, path: Path) -> dict[str, object]:
    """Check immutable bundle data without ever creating or writing a probe."""

    started = perf_counter()
    try:
        exists = path.is_dir()
        if not exists:
            raise FileNotFoundError(path.name)
        # Enumerating one entry proves the directory can be read while keeping
        # the package byte-for-byte unchanged.
        next(iter(path.iterdir()), None)
    except BaseException as exc:
        return _timed_result(
            check_id,
            "FAIL",
            "Immutable bundle data directory is missing or unreadable",
            {
                "directory": path.name,
                "exists": path.is_dir(),
                "error_type": type(exc).__name__,
            },
            (perf_counter() - started) * 1000,
        )
    return _timed_result(
        check_id,
        "PASS",
        "Immutable bundle data directory is readable",
        {"directory": path.name, "exists": True, "write_probe": False},
        (perf_counter() - started) * 1000,
    )


def _json_resource_check(
    check_id: str,
    path: Path,
    *,
    expected_key: str | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    started = perf_counter()
    if not path.is_file():
        return (
            _timed_result(
                check_id,
                "FAIL",
                "Required JSON resource is missing",
                {"file": path.name, "exists": False},
                (perf_counter() - started) * 1000,
            ),
            None,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except BaseException as exc:
        return (
            _timed_result(
                check_id,
                "FAIL",
                "JSON resource is unreadable or invalid",
                {
                    "file": path.name,
                    "exists": True,
                    "error_type": type(exc).__name__,
                },
                (perf_counter() - started) * 1000,
            ),
            None,
        )
    valid = isinstance(payload, dict) and (
        expected_key is None or isinstance(payload.get(expected_key), list)
    )
    count = len(payload.get(expected_key, [])) if valid and expected_key else None
    return (
        _timed_result(
            check_id,
            "PASS" if valid else "FAIL",
            "JSON resource is readable" if valid else "JSON resource has an invalid shape",
            {
                "file": path.name,
                "exists": True,
                "expected_key": expected_key,
                "item_count": count,
            },
            (perf_counter() - started) * 1000,
        ),
        payload if valid else None,
    )


def _template_files_check(
    profile_root: Path,
    templates_document: object,
) -> dict[str, object]:
    started = perf_counter()
    if not isinstance(templates_document, dict):
        return _timed_result(
            "RESOURCE-TEMPLATE-FILES",
            "FAIL",
            "Template manifest is unavailable",
            {"referenced_count": 0, "missing_count": 0},
            (perf_counter() - started) * 1000,
        )
    records = templates_document.get("templates")
    if not isinstance(records, list):
        records = []
    referenced: list[str] = []
    unsafe: list[str] = []
    for record in records:
        if isinstance(record, dict):
            relative = record.get("file")
            if isinstance(relative, str) and relative.strip():
                normalized = relative.strip().replace("\\", "/")
                pure = PurePosixPath(normalized)
                if pure.is_absolute() or any(
                    part in {"", ".", ".."} for part in pure.parts
                ) or (pure.parts and ":" in pure.parts[0]):
                    unsafe.append(PurePosixPath(normalized).name or "invalid")
                else:
                    referenced.append(pure.as_posix())
    unique = tuple(dict.fromkeys(referenced))
    missing: list[str] = []
    unreadable: list[str] = []
    for relative in unique:
        candidate = profile_root / Path(relative)
        if not candidate.is_file():
            missing.append(Path(relative).name)
            continue
        try:
            with candidate.open("rb") as handle:
                handle.read(1)
        except OSError:
            unreadable.append(Path(relative).name)
    valid = bool(unique) and not missing and not unreadable and not unsafe
    return _timed_result(
        "RESOURCE-TEMPLATE-FILES",
        "PASS" if valid else "FAIL",
        "Referenced template files are readable"
        if valid
        else "One or more referenced template files are unavailable",
        {
            "referenced_count": len(unique),
            "missing_count": len(missing),
            "unreadable_count": len(unreadable),
            "unsafe_count": len(unsafe),
            "missing_samples": missing[:5],
            "unreadable_samples": unreadable[:5],
            "unsafe_samples": unsafe[:5],
        },
        (perf_counter() - started) * 1000,
    )


def _file_resource_check(
    check_id: str,
    path: Path,
    *,
    logical_name: str,
) -> dict[str, object]:
    started = perf_counter()
    try:
        with path.open("rb") as handle:
            handle.read(1)
        size = path.stat().st_size
        valid = size > 0
        error_type = None
    except BaseException as exc:
        size = 0
        valid = False
        error_type = type(exc).__name__
    evidence: dict[str, object] = {
        "file": logical_name,
        "exists": path.is_file(),
        "size_bytes": size,
    }
    if error_type:
        evidence["error_type"] = error_type
    return _timed_result(
        check_id,
        "PASS" if valid else "FAIL",
        "Required binary resource is readable"
        if valid
        else "Required binary resource is missing or unreadable",
        evidence,
        (perf_counter() - started) * 1000,
    )


def _danzero_model_path(bundle_root: Path, profile_root: Path) -> Path:
    bundled = profile_root / "models" / "danzero" / "q_network.ckpt"
    if bundled.is_file():
        return bundled
    return (
        bundle_root
        / "src"
        / "daguandan_bridge"
        / "danzero"
        / "_vendor"
        / "guandan_rlcard"
        / "baselines"
        / "danzero"
        / "q_network.ckpt"
    )


def _subprocess_dependency_probe(dependency: DependencySpec) -> dict[str, object]:
    startup = current_startup_diagnostics()
    scratch_root = getattr(startup, "run_directory", None)
    if scratch_root is None:
        scratch_root = Path(tempfile.gettempdir()) / "DaguandanAssistant" / "doctor-probes"
    scratch = Path(scratch_root) / "dependency-probes"
    scratch.mkdir(parents=True, exist_ok=True)
    output = scratch / f"{dependency.check_id.lower()}-{uuid4().hex}.json"
    scratch_file = f"dependency-probes/{output.name}"
    base_evidence: dict[str, object] = {
        "module": dependency.module,
        "distribution": dependency.distribution,
        "version": _distribution_version(dependency.distribution),
        "probe_id": dependency.check_id,
        "scratch_file": scratch_file,
    }
    if getattr(sys, "frozen", False):
        command = [
            sys.executable,
            "--_doctor-import-probe",
            dependency.check_id,
            "--doctor-output",
            str(output),
        ]
    else:
        entrypoint = application_root() / "run.py"
        command = [
            sys.executable,
            str(entrypoint),
            "--_doctor-import-probe",
            dependency.check_id,
            "--doctor-output",
            str(output),
        ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
            text=False,
        )
    except subprocess.TimeoutExpired:
        _remove_probe_output(output)
        return {
            "status": "FAIL",
            "summary": "Dependency import probe timed out",
            "evidence": {
                **base_evidence,
                "probe_exit_code": "timeout",
            },
        }
    except BaseException as exc:
        _remove_probe_output(output)
        evidence = {
            **base_evidence,
            "probe_exit_code": "not_started",
            "error_type": type(exc).__name__,
            "message": _sanitize_diagnostic_text(exc),
        }
        for attribute in ("winerror", "errno"):
            value = getattr(exc, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                evidence[attribute] = value
        return {
            "status": "FAIL",
            "summary": "Dependency import probe could not start",
            "evidence": evidence,
        }

    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except BaseException:
        payload = None
    finally:
        _remove_probe_output(output)
    valid_payload = (
        isinstance(payload, dict)
        and payload.get("schema") == IMPORT_PROBE_SCHEMA
    )
    reported_status = str(payload.get("status") or "") if valid_payload else ""
    child_evidence = (
        _sanitize_probe_evidence(payload.get("evidence"))
        if valid_payload
        else {}
    )
    evidence = {
        **child_evidence,
        **base_evidence,
        "probe_exit_code": int(completed.returncode),
    }
    if valid_payload:
        evidence["reported_status"] = _sanitize_diagnostic_text(reported_status)
        evidence["reported_summary"] = _sanitize_diagnostic_text(
            payload.get("summary") or ""
        )
    stderr_summary = _completed_stderr_summary(completed)

    if int(completed.returncode) != 0:
        evidence["stderr_summary"] = stderr_summary or "unavailable"
        return {
            "status": "FAIL",
            "summary": "Dependency import probe process exited with a non-zero status",
            "evidence": evidence,
        }

    if not valid_payload:
        evidence["stderr_summary"] = stderr_summary or "unavailable"
        return {
            "status": "FAIL",
            "summary": "Dependency import probe exited without a valid result",
            "evidence": evidence,
        }

    if reported_status not in {"PASS", "WARN"}:
        return {
            "status": "FAIL",
            "summary": (
                "Dependency import probe reported a failure"
                if reported_status == "FAIL"
                else "Dependency import probe returned an invalid status"
            ),
            "evidence": evidence,
        }
    return {
        "status": reported_status,
        "summary": (
            "Dependency import probe passed"
            if reported_status == "PASS"
            else "Dependency import probe completed with a warning"
        ),
        "evidence": evidence,
    }


def _distribution_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"
    except BaseException:
        return "unavailable"


def _remove_probe_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _doctor_output_path(output_path: Path | str | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    startup = current_startup_diagnostics()
    directory = getattr(startup, "run_directory", None)
    if directory is None:
        raise OSError("startup diagnostics directory is unavailable")
    return Path(directory) / "doctor.json"


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _error_type(raw_error: object) -> str | None:
    if not raw_error:
        return None
    text = str(raw_error)
    return text.split(":", 1)[0] or "Error"
