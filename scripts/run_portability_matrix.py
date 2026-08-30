from __future__ import annotations

"""Run deterministic path/DPI qualification cases against one frozen bundle."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.build_manifest import sha256_file  # noqa: E402
from daguandan_bridge.storage import atomic_write_json  # noqa: E402


MATRIX_SCHEMA = "guandan.portability-matrix/1"
_DPI_CASES = (96, 120, 144)


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    description: str
    data_root: Path
    simulated_dpi: int
    read_only_bundle: bool


def run_matrix(
    *,
    executable: Path,
    bundle_root: Path,
    output_root: Path,
    data_root: Path | None = None,
    doctor_runner: Callable[[Path, Path, Path, int], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    executable = executable.resolve()
    bundle_root = bundle_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not executable.is_file():
        raise ValueError(f"frozen executable is missing: {executable}")
    if not bundle_root.is_dir():
        raise ValueError(f"frozen bundle is missing: {bundle_root}")
    before = _tree_hash(bundle_root)
    base = (data_root or output_root / "matrix-data").resolve()
    base.mkdir(parents=True, exist_ok=True)
    cases = (
        MatrixCase("unicode-space-path", "中文和空格数据路径", base / "中文 数据", 96, False),
        MatrixCase("readonly-bundle", "资源包只读（模拟）", base / "readonly data", 96, True),
        *(MatrixCase(f"dpi-{dpi}", f"模拟 {dpi}/96*100% DPI", base / f"dpi-{dpi}", dpi, False) for dpi in _DPI_CASES),
    )
    results: list[dict[str, object]] = []
    for case in cases:
        case.data_root.mkdir(parents=True, exist_ok=True)
        report_path = output_root / f"{case.case_id}.doctor.json"
        if doctor_runner is None:
            environment = os.environ.copy()
            environment["DAGUANDAN_DATA_ROOT"] = str(case.data_root)
            environment["DAGUANDAN_PORTABILITY_SIMULATED_DPI"] = str(case.simulated_dpi)
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            completed = subprocess.run(
                [str(executable), "--doctor", "--doctor-output", str(report_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=180,
                check=False,
            )
            report = _read_json(report_path)
            checks = report.get("checks") if isinstance(report.get("checks"), list) else []
            failed_checks = [
                item.get("id")
                for item in checks
                if isinstance(item, Mapping) and item.get("status") == "FAIL"
            ]
            passed = completed.returncode == 0 and not failed_checks and report.get("schema") == "guandan.doctor/1"
            result = {
                "case_id": case.case_id,
                "description": case.description,
                "simulated": True,
                "simulated_dpi": case.simulated_dpi,
                "read_only_bundle": case.read_only_bundle,
                "data_root_name": case.data_root.name,
                "doctor_exit_code": int(completed.returncode),
                "failed_checks": failed_checks,
                "passed": bool(passed),
                "physical_dpi_validated": False,
            }
        else:
            result = dict(doctor_runner(executable, bundle_root, case.data_root, case.simulated_dpi))
            result.setdefault("case_id", case.case_id)
            result.setdefault("simulated", True)
            result.setdefault("physical_dpi_validated", False)
        results.append(result)
    after = _tree_hash(bundle_root)
    bundle_unchanged = before == after
    if not bundle_unchanged:
        for result in results:
            result["passed"] = False
            result.setdefault("errors", []).append("bundle tree changed during matrix")
    return {
        "schema": MATRIX_SCHEMA,
        "status": "PASS" if bundle_unchanged and all(bool(item.get("passed")) for item in results) else "FAIL",
        "bundle_root_name": bundle_root.name,
        "bundle_before_sha256": before,
        "bundle_after_sha256": after,
        "bundle_unchanged": bundle_unchanged,
        "physical_dpi_validation": "NOT_RUN",
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    executable = args.executable or args.bundle_root / "DaguandanAssistant.exe"
    try:
        report = run_matrix(
            executable=executable,
            bundle_root=args.bundle_root,
            output_root=args.output.parent,
            data_root=args.data_root,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        report = {
            "schema": MATRIX_SCHEMA,
            "status": "FAIL",
            "errors": [{"type": type(exc).__name__, "message": str(exc)}],
        }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("status") == "PASS" else 2


def _tree_hash(root: Path) -> str:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().casefold()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
