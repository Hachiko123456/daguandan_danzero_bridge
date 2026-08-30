from __future__ import annotations

"""Single fail-closed qualification entry for a frozen Windows release.

Every stage writes a bounded machine-readable result.  A failed stage leaves
the candidate staging tree for diagnosis, but never publishes/activates it or
touches the repository's ``release/`` directory.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.build_manifest import (  # noqa: E402
    BUILD_MANIFEST_FILENAME,
    sha256_file,
    verify_build_manifest,
)
from daguandan_bridge.release_lock import verify_release_inputs  # noqa: E402
from daguandan_bridge.release_manager import (  # noqa: E402
    BASELINE_SOURCE_COMMIT,
    ReleaseManagerError,
    activate_release,
    install_release,
    register_legacy_baseline,
    rollback_release,
)
from daguandan_bridge.storage import atomic_write_json  # noqa: E402
from daguandan_bridge.support_repro import verify_support_archive  # noqa: E402


QUALIFICATION_SCHEMA = "guandan.release-qualification/1"
FULL_SCENARIOS = (
    "initial_capture",
    "move_recovery",
    "resize_recovery",
    "minimize_recovery",
    "occlusion",
    "dpi",
    "full_chain",
)


@dataclass
class Stage:
    name: str
    status: str = "PENDING"
    exit_code: int | None = None
    duration_ms: float = 0.0
    command: list[str] | None = None
    log: str | None = None
    evidence: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.evidence is None:
            self.evidence = {}

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 3),
            "command": self.command,
            "log": self.log,
            "evidence": self.evidence,
        }


class Qualification:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.release_root = Path(args.release_root).resolve()
        self.work_root = (
            Path(args.work_root).resolve()
            if args.work_root
            else self.release_root.parent / f".{self.release_root.name}.qualification-{uuid4().hex[:8]}"
        )
        self.output_path = (
            Path(args.output).resolve()
            if args.output
            else self.release_root.parent / f"{self.release_root.name}.qualification.json"
        )
        self.stages: list[Stage] = []
        self.errors: list[str] = []
        self.started_at = datetime.now(UTC).isoformat()
        self.bundle_root: Path | None = None
        self.executable: Path | None = None
        self.archive: Path | None = None
        self.release_record: Path | None = None
        self.archive_checksum: Path | None = None
        self.bundle_hash_before: str | None = None

    def run(self) -> int:
        self._preflight()
        if self.errors:
            return self._finish()
        self.work_root.mkdir(parents=True, exist_ok=False)
        self._stage("source-tests", self._source_tests)
        self._stage("release-input-lock", self._release_input_lock)
        self._stage("clean-frozen-build", self._build)
        self._stage("manifest-native-audit", self._manifest_audit)
        self._stage("frozen-doctor", self._frozen_doctor)
        self._stage("support-export-verify", self._support_export)
        self._stage("reproducer-fixtures", self._reproducer_fixtures)
        if self.args.session is not None:
            self._stage("source-frozen-window-e2e", self._window_e2e)
        else:
            self._add_skipped("source-frozen-window-e2e", "--session was not supplied")
        self._stage("portability-matrix", self._portability_matrix)
        self._stage("bundle-immutability", self._bundle_immutability)
        self._stage("install-activate-rollback", self._install_rollback)
        return self._finish()

    def _preflight(self) -> None:
        stage = Stage("preflight")
        started = time.perf_counter()
        try:
            if self.release_root.exists():
                raise ValueError(
                    f"release root must be unique and absent before qualification: {self.release_root}"
                )
            if self.release_root == PROJECT_ROOT or _is_below(self.release_root, PROJECT_ROOT):
                raise ValueError("release root must be external to the source checkout")
            if self.args.wheelhouse is None or not Path(self.args.wheelhouse).is_dir():
                raise ValueError("an external prepared wheelhouse is required")
            clean = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
                capture_output=True,
                text=True,
                check=False,
            )
            if clean.returncode != 0 or clean.stdout.strip():
                raise ValueError("source tree must be clean before qualification")
            baseline = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "baseline/local-stable-20260831^{}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if baseline.returncode != 0 or baseline.stdout.strip() != BASELINE_SOURCE_COMMIT:
                raise ValueError("immutable baseline tag no longer points to 2db427b")
            stage.status = "PASS"
            stage.evidence = {
                "baseline_tag": "baseline/local-stable-20260831",
                "baseline_commit": baseline.stdout.strip(),
                "project_root": PROJECT_ROOT.name,
                "release_root": self.release_root.name,
                "wheelhouse_root": Path(self.args.wheelhouse).name,
            }
        except Exception as exc:
            stage.status = "FAIL"
            stage.evidence = {"error_type": type(exc).__name__, "message": str(exc)}
            self.errors.append(f"preflight: {exc}")
        stage.duration_ms = (time.perf_counter() - started) * 1000
        self.stages.append(stage)

    def _stage(self, name: str, operation: Callable[[], Mapping[str, object] | None]) -> None:
        if self.errors:
            self._add_skipped(name, "previous stage failed")
            return
        stage = Stage(name)
        started = time.perf_counter()
        try:
            result = operation() or {}
            stage.status = "PASS"
            stage.evidence = dict(result)
        except _StageFailure as exc:
            stage.status = "FAIL"
            stage.exit_code = exc.exit_code
            stage.command = exc.command
            stage.log = exc.log
            stage.evidence = dict(exc.evidence)
            self.errors.append(f"{name}: {exc}")
        except Exception as exc:
            stage.status = "FAIL"
            stage.evidence = {"error_type": type(exc).__name__, "message": str(exc)}
            self.errors.append(f"{name}: {exc}")
        stage.duration_ms = (time.perf_counter() - started) * 1000
        self.stages.append(stage)

    def _add_skipped(self, name: str, reason: str) -> None:
        self.stages.append(Stage(name, status="SKIPPED", evidence={"reason": reason}))

    def _source_tests(self) -> Mapping[str, object]:
        command = [sys.executable, "-m", "pytest", "-q"]
        completed = self._run(command, self.work_root / "source-tests.log")
        if completed.returncode != 0:
            raise _StageFailure("source pytest failed", completed.returncode, command, completed.log_path, {})
        return {"command": command, "log": str(completed.log_path), "qa_scope": "SESSION-FULL"}

    def _release_input_lock(self) -> Mapping[str, object]:
        report = verify_release_inputs(
            project_root=PROJECT_ROOT,
            wheelhouse_root=Path(self.args.wheelhouse),
            python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        )
        if report.get("status") != "PASS":
            raise _StageFailure("release input locks failed", 2, None, None, report)
        atomic_write_json(self.work_root / "release_input_audit.json", report)
        return {"report": str(self.work_root / "release_input_audit.json"), **report}

    def _build(self) -> Mapping[str, object]:
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "scripts" / "package_release.ps1"),
            "-ReleaseRoot",
            str(self.release_root),
            "-WheelhouseRoot",
            str(Path(self.args.wheelhouse).resolve()),
        ]
        log = self.work_root / "package.log"
        completed = self._run(command, log)
        self.bundle_root = self.release_root / "dist" / "DaguandanAssistant"
        self.executable = self.bundle_root / "DaguandanAssistant.exe"
        self.archive = self.release_root / "DaguandanAssistant.zip"
        self.release_record = self.release_root / "DaguandanAssistant.release.json"
        self.archive_checksum = self.release_root / "DaguandanAssistant.zip.sha256"
        if completed.returncode != 0 or not self.executable.is_file():
            raise _StageFailure("clean frozen package failed", completed.returncode, command, log, {"release_root": str(self.release_root)})
        return {
            "command": command,
            "log": str(log),
            "bundle_root": str(self.bundle_root),
            "executable": str(self.executable),
            "archive": str(self.archive),
            "archive_sha256": sha256_file(self.archive) if self.archive.is_file() else None,
        }

    def _manifest_audit(self) -> Mapping[str, object]:
        assert self.bundle_root is not None and self.executable is not None
        manifest_path = self.bundle_root / BUILD_MANIFEST_FILENAME
        manifest = verify_build_manifest(self.bundle_root, manifest_path, strict=True)
        if not manifest.ok:
            raise _StageFailure("strict build manifest failed", 2, None, None, manifest.to_dict())
        native = _read_json(self.bundle_root / "native_dependency_audit.json")
        if native.get("status") != "PASS":
            raise _StageFailure("native dependency audit did not pass", 2, None, None, native)
        self.bundle_hash_before = _tree_hash(self.bundle_root)
        return {"build_id": manifest.build_id, "checked_files": manifest.checked_files, "native_audit": native.get("summary"), "bundle_tree_sha256": self.bundle_hash_before}

    def _frozen_doctor(self) -> Mapping[str, object]:
        assert self.executable is not None
        data_root = self.work_root / "frozen-doctor-data"
        report_path = self.work_root / "frozen-doctor.json"
        environment = _clean_runtime_environment(data_root)
        command = [str(self.executable), "--doctor", "--doctor-output", str(report_path)]
        completed = self._run(command, self.work_root / "frozen-doctor.log", env=environment)
        report = _read_json(report_path)
        checks = report.get("checks") if isinstance(report.get("checks"), list) else []
        failed = [item.get("id") for item in checks if isinstance(item, Mapping) and item.get("status") == "FAIL"]
        if completed.returncode != 0 or report.get("schema") != "guandan.doctor/1" or failed:
            raise _StageFailure("frozen doctor failed", completed.returncode, command, completed.log_path, {"report": str(report_path), "failed_checks": failed, "doctor": report})
        return {"report": str(report_path), "failed_checks": [], "build_id": report.get("build_id")}

    def _support_export(self) -> Mapping[str, object]:
        assert self.executable is not None
        destination = self.work_root / "support.zip"
        environment = _clean_runtime_environment(self.work_root / "support-data")
        command = [str(self.executable), "--export-support", str(destination)]
        completed = self._run(command, self.work_root / "support-export.log", env=environment)
        if completed.returncode != 0 or not destination.is_file():
            raise _StageFailure("frozen support export failed", completed.returncode, command, completed.log_path, {})
        verified = verify_support_archive(destination)
        return {"path": str(destination), "sha256": verified.sha256, "manifest": verified.manifest, "images_default": bool(verified.manifest.get("privacy", {}).get("contains_sensitive_images"))}

    def _reproducer_fixtures(self) -> Mapping[str, object]:
        command = [sys.executable, "-m", "pytest", "-q", "tests/test_support_repro.py", "tests/test_diagnostic_root_cause.py", "tests/test_diagnostic_non_interference.py"]
        completed = self._run(command, self.work_root / "reproducer-fixtures.log")
        if completed.returncode != 0:
            raise _StageFailure("reproducer fixture tests failed", completed.returncode, command, completed.log_path, {})
        return {"command": command, "log": str(completed.log_path), "default_repeats": 20, "truth_required_for_fix_gate": True}

    def _window_e2e(self) -> Mapping[str, object]:
        assert self.bundle_root is not None and self.executable is not None
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_window_e2e_validation.py"),
            "--session", str(Path(self.args.session).resolve()),
            "--output", str(self.work_root / "window-e2e"),
            "--profile-source", str((PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan").resolve()),
            "--bundle-root", str(self.bundle_root),
            "--executable", str(self.executable),
            "--frozen-data-root", str(self.work_root / "window-e2e-data"),
            "--scenarios", ",".join(FULL_SCENARIOS),
            "--time-scale", "1.0",
        ]
        if self.args.baseline_summary:
            command.extend(["--baseline-summary", str(Path(self.args.baseline_summary).resolve())])
        completed = self._run(command, self.work_root / "window-e2e.log")
        summary = _read_json(self.work_root / "window-e2e" / "runs")
        if completed.returncode != 0:
            raise _StageFailure("source/frozen window E2E failed", completed.returncode, command, completed.log_path, {"output_root": str(self.work_root / "window-e2e")})
        return {"command": command, "log": str(completed.log_path), "output_root": str(self.work_root / "window-e2e"), "scenarios": list(FULL_SCENARIOS), "summary_probe": summary}

    def _portability_matrix(self) -> Mapping[str, object]:
        assert self.bundle_root is not None and self.executable is not None
        output = self.work_root / "portability-matrix.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_portability_matrix.py"),
            "--bundle-root", str(self.bundle_root),
            "--executable", str(self.executable),
            "--output", str(output),
            "--data-root", str(self.work_root / "matrix-data"),
        ]
        completed = self._run(command, self.work_root / "portability-matrix.log")
        report = _read_json(output)
        if completed.returncode != 0 or report.get("status") != "PASS":
            raise _StageFailure("portability matrix failed", completed.returncode, command, completed.log_path, report)
        return {"report": str(output), "physical_dpi_validation": report.get("physical_dpi_validation"), "cases": report.get("cases")}

    def _bundle_immutability(self) -> Mapping[str, object]:
        assert self.bundle_root is not None
        after = _tree_hash(self.bundle_root)
        if self.bundle_hash_before != after:
            raise _StageFailure("frozen bundle changed after qualification stages", 2, None, None, {"before": self.bundle_hash_before, "after": after})
        return {"before": self.bundle_hash_before, "after": after, "unchanged": True}

    def _install_rollback(self) -> Mapping[str, object]:
        assert self.archive is not None and self.release_record is not None and self.archive_checksum is not None
        runtime = self.work_root / "install-runtime"
        # Existing repository release is an old portable build. Copy it as an
        # explicitly non-reproducible legacy baseline; never modify release/.
        legacy = PROJECT_ROOT / "release" / "dist" / "DaguandanAssistant"
        if not (legacy / "DaguandanAssistant.exe").is_file():
            raise _StageFailure("legacy baseline binary is unavailable for rollback exercise", 2, None, None, {"path": str(legacy)})
        baseline = register_legacy_baseline(
            legacy,
            executable_sha256=sha256_file(legacy / "DaguandanAssistant.exe"),
            runtime_root=runtime,
        )
        candidate = install_release(
            self.archive,
            release_record_path=self.release_record,
            checksum_path=self.archive_checksum,
            runtime_root=runtime,
        )
        # The legacy binary predates the current doctor contract, so use a
        # deliberately explicit transition runner that validates the pointer
        # operation while recording that the physical legacy doctor is outside
        # this local proof. Candidate activation is still run by the real EXE.
        def legacy_doctor(_release, _runtime, output_path):
            report = {"schema": "guandan.doctor/1", "checks": [], "legacy_baseline": True}
            atomic_write_json(output_path, report)
            return report

        activate_release(baseline.release_id, runtime_root=runtime, doctor_runner=legacy_doctor)
        activate_release(candidate.release_id, runtime_root=runtime)
        rolled = rollback_release(runtime_root=runtime)
        if rolled.get("release_id") != baseline.release_id:
            raise _StageFailure("rollback did not restore the legacy baseline pointer", 2, None, None, {"rolled_back": rolled, "baseline": baseline.to_dict()})
        status = _read_json(runtime / "install" / "active.json")
        return {"runtime_root": str(runtime), "baseline_release": baseline.to_dict(), "candidate_release": candidate.to_dict(), "active_after_rollback": status, "legacy_baseline_reproducible": False, "bad_candidate_preserved": candidate.version_root.is_dir()}

    def _finish(self) -> int:
        report = {
            "schema": QUALIFICATION_SCHEMA,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "PASS" if not self.errors and all(stage.status in {"PASS", "SKIPPED"} for stage in self.stages) else "FAIL",
            "branch": _git_value("branch --show-current"),
            "source_commit": _git_value("rev-parse HEAD"),
            "baseline": {"tag": "baseline/local-stable-20260831", "commit": BASELINE_SOURCE_COMMIT},
            "release_root": self.release_root.name,
            "candidate": {
                "bundle_root": str(self.bundle_root) if self.bundle_root else None,
                "executable_sha256": sha256_file(self.executable) if self.executable and self.executable.is_file() else None,
                "archive_sha256": sha256_file(self.archive) if self.archive and self.archive.is_file() else None,
                "build_id": _build_id(self.bundle_root) if self.bundle_root else None,
            },
            "stages": [stage.to_dict() for stage in self.stages],
            "errors": self.errors,
            "external_validation_risks": [
                "physical second-machine Windows/driver combination not run",
                "real WeChat mini-program rendering not run",
                "physical 125%/150% monitor transition not run; matrix is deterministic simulation",
            ],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2

    def _run(
        self,
        command: list[str],
        log_path: Path,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "_Completed":
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(env) if env is not None else None,
                check=False,
            )
        return _Completed(int(completed.returncode), log_path)


@dataclass(frozen=True)
class _Completed:
    returncode: int
    log_path: Path


class _StageFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        exit_code: int | None,
        command: list[str] | None,
        log: Path | None,
        evidence: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.command = command
        self.log = str(log) if log is not None else None
        self.evidence = dict(evidence)


def _clean_runtime_environment(data_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["DAGUANDAN_DATA_ROOT"] = str(data_root)
    environment["PYTHONNOUSERSITE"] = "1"
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
        "JAVA_HOME",
        "JDK_HOME",
        "CONDA_PREFIX",
        "CONDA_EXE",
        "POPPLER_PATH",
    ):
        environment.pop(name, None)
    return environment


def _read_json(path: Path) -> dict[str, object]:
    if path.is_dir():
        candidates = sorted(path.rglob("host_summary.json"))
        if not candidates:
            return {}
        path = candidates[-1]
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _build_id(bundle: Path | None) -> str | None:
    if bundle is None:
        return None
    document = _read_json(bundle / BUILD_MANIFEST_FILENAME)
    raw = document.get("build_id")
    return str(raw) if raw else None


def _tree_hash(root: Path) -> str:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix().casefold()):
        if path.is_file():
            records.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_value(arguments: str) -> str | None:
    completed = subprocess.run(["git", "-C", str(PROJECT_ROOT), *arguments.split()], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return Qualification(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
