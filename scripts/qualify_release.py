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
    collect_source_identity,
    load_build_manifest,
    sha256_file,
    verify_build_manifest,
    verify_source_identity,
)
from daguandan_bridge.release_lock import verify_release_inputs  # noqa: E402
from daguandan_bridge.doctor import validate_frozen_doctor_report  # noqa: E402
from daguandan_bridge.release_manager import (  # noqa: E402
    BASELINE_SOURCE_COMMIT,
    ReleaseManagerError,
    activate_release,
    install_release,
    register_legacy_baseline,
    release_status,
    rollback_release,
    verify_baseline_auth,
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
        self.archive_hash_before: str | None = None
        self._output_owned = False
        self.source_identity: dict[str, object] | None = None

    def run(self) -> int:
        self._preflight()
        if self.errors:
            return self._finish()
        self.work_root.mkdir(parents=True, exist_ok=False)
        self._stage("source-tests", self._source_tests)
        self._stage("release-input-lock", self._release_input_lock)
        self._stage("clean-frozen-build", self._build)
        self._stage("source-identity-immutability", self._source_immutability)
        self._stage("manifest-native-audit", self._manifest_audit)
        self._stage("frozen-doctor", self._frozen_doctor)
        self._stage("support-export-verify", self._support_export)
        self._stage("frozen-repro-gate", self._frozen_repro_gate)
        self._stage("source-frozen-window-e2e", self._window_e2e)
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
            if self.work_root.exists():
                raise ValueError(f"work root must be unique and absent: {self.work_root}")
            if self.output_path.exists():
                raise ValueError(f"qualification output must be new: {self.output_path}")
            if self.args.wheelhouse is None or not Path(self.args.wheelhouse).is_dir():
                raise ValueError("an external prepared wheelhouse is required")
            required_files = {
                "baseline summary": Path(self.args.baseline_summary),
                "repro support": Path(self.args.repro_support),
                "repro truth": Path(self.args.repro_truth),
                "reference repro report": Path(self.args.reference_repro_report),
                "baseline auth": Path(self.args.baseline_auth),
            }
            missing = [label for label, path in required_files.items() if not path.is_file()]
            if missing:
                raise ValueError("required formal inputs are missing: " + ", ".join(missing))
            if not Path(self.args.session).is_dir():
                raise ValueError("formal window E2E session is unavailable")
            if not Path(self.args.baseline_bundle).is_dir():
                raise ValueError("external baseline bundle is unavailable")
            overlaps = _root_overlap_failures(
                project_root=PROJECT_ROOT,
                release_root=self.release_root,
                work_root=self.work_root,
                wheelhouse_root=Path(self.args.wheelhouse),
                output_path=self.output_path,
            )
            if overlaps:
                raise ValueError("; ".join(overlaps))
            baseline_bundle = Path(self.args.baseline_bundle).resolve()
            if _paths_overlap(baseline_bundle, PROJECT_ROOT):
                raise ValueError("baseline bundle must be a pre-stored external artifact")
            baseline_auth_path = Path(self.args.baseline_auth).resolve()
            if _paths_overlap(baseline_auth_path, PROJECT_ROOT):
                raise ValueError("baseline auth must be external to the source checkout")
            for label, managed in (
                ("release root", self.release_root),
                ("work root", self.work_root),
                ("wheelhouse", Path(self.args.wheelhouse).resolve()),
                ("qualification output", self.output_path),
            ):
                if _paths_overlap(baseline_bundle, managed) or _paths_overlap(baseline_auth_path, managed):
                    raise ValueError(f"baseline artifact/auth must be disjoint from {label}")
            baseline_auth = verify_baseline_auth(
                baseline_bundle,
                Path(self.args.baseline_auth),
            )
            repro_inputs = _validate_repro_inputs(
                Path(self.args.repro_support),
                Path(self.args.repro_truth),
                Path(self.args.reference_repro_report),
            )
            if repro_inputs.get("status") != "PASS":
                raise ValueError(
                    "formal repro inputs are invalid: "
                    + ", ".join(str(item) for item in repro_inputs.get("failures", []))
                )
            # From this point the destination is a new, disjoint file owned by
            # this qualification attempt, so later failures may be published
            # there without overwriting source/build inputs.
            self._output_owned = True
            source_identity = collect_source_identity(PROJECT_ROOT)
            if source_identity.get("dirty") is not False:
                raise ValueError("source tree must be clean before qualification")
            self.source_identity = source_identity
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
                "source_identity": source_identity,
                "project_root": PROJECT_ROOT.name,
                "release_root": self.release_root.name,
                "wheelhouse_root": Path(self.args.wheelhouse).name,
                "formal_inputs": {
                    label.replace(" ", "_"): sha256_file(path)
                    for label, path in required_files.items()
                },
                "baseline_artifact": {
                    "tree_sha256": baseline_auth["artifact"]["tree_sha256"],
                    "executable_sha256": baseline_auth["executable"]["sha256"],
                    "build_identity_sha256": baseline_auth["build_identity_sha256"],
                },
                "repro_inputs": repro_inputs,
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
        self.archive_hash_before = sha256_file(self.archive) if self.archive.is_file() else None
        return {
            "command": command,
            "log": str(log),
            "bundle_root": str(self.bundle_root),
            "executable": str(self.executable),
            "archive": str(self.archive),
            "archive_sha256": self.archive_hash_before,
        }

    def _source_immutability(self) -> Mapping[str, object]:
        if self.source_identity is None:
            raise ValueError("preflight source identity is unavailable")
        current = verify_source_identity(
            PROJECT_ROOT,
            self.source_identity,
            require_clean=True,
        )
        return {
            "status": "PASS",
            "commit": current["commit"],
            "tree": current["tree"],
            "dirty": current["dirty"],
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
        source_validation = _validate_formal_manifest_source(
            load_build_manifest(manifest_path),
            self.source_identity,
        )
        if source_validation.get("status") != "PASS":
            raise _StageFailure(
                "build manifest source identity is not the expected clean HEAD",
                2,
                None,
                None,
                source_validation,
            )
        self.bundle_hash_before = _tree_hash(self.bundle_root)
        return {"build_id": manifest.build_id, "checked_files": manifest.checked_files, "native_audit": native.get("summary"), "bundle_tree_sha256": self.bundle_hash_before, "source_identity": source_validation}

    def _frozen_doctor(self) -> Mapping[str, object]:
        assert self.executable is not None
        data_root = self.work_root / "frozen-doctor-data"
        report_path = self.work_root / "frozen-doctor.json"
        environment = _clean_runtime_environment(data_root)
        command = [str(self.executable), "--doctor", "--doctor-output", str(report_path)]
        completed = self._run(command, self.work_root / "frozen-doctor.log", env=environment)
        report = _read_json(report_path)
        validation = validate_frozen_doctor_report(
            report,
            expected_build_id=str(_build_id(self.bundle_root) or ""),
        )
        if completed.returncode != 0 or validation.get("status") != "PASS":
            raise _StageFailure("frozen doctor failed", completed.returncode, command, completed.log_path, {"report": str(report_path), "validation": validation, "doctor": report})
        return {"report": str(report_path), **validation}

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

    def _frozen_repro_gate(self) -> Mapping[str, object]:
        assert self.executable is not None and self.bundle_root is not None
        support = Path(self.args.repro_support).resolve()
        truth = Path(self.args.repro_truth).resolve()
        reference_path = Path(self.args.reference_repro_report).resolve()
        candidate_path = self.work_root / "candidate-repro.json"
        environment = _clean_runtime_environment(self.work_root / "repro-data")
        candidate_command = [
            str(self.executable),
            "--repro-support",
            str(support),
            "--repro-truth",
            str(truth),
            "--repro-repeats",
            "20",
            "--repro-deterministic",
            "--repro-role",
            "candidate",
            "--repro-output",
            str(candidate_path),
        ]
        candidate_run = self._run(
            candidate_command,
            self.work_root / "candidate-repro.log",
            env=environment,
        )
        if candidate_run.returncode != 0 or not candidate_path.is_file():
            raise _StageFailure(
                "candidate frozen reproducer failed",
                candidate_run.returncode,
                candidate_command,
                candidate_run.log_path,
                {},
            )
        reference = _read_json(reference_path)
        candidate = _read_json(candidate_path)
        validation = _validate_release_repro_equivalence(
            reference,
            candidate,
            expected_support_sha256=sha256_file(support),
            expected_candidate_build_id=_build_id(self.bundle_root),
        )
        if validation.get("status") != "PASS":
            raise _StageFailure(
                "source/candidate release reproduction equivalence failed",
                2,
                candidate_command,
                candidate_run.log_path,
                validation,
            )
        return {
            **validation,
            "reference_report": str(reference_path),
            "candidate_report": str(candidate_path),
            "candidate_command": candidate_command,
        }

    def _window_e2e(self) -> Mapping[str, object]:
        assert self.bundle_root is not None and self.executable is not None
        run_id = f"qualification-{uuid4().hex[:12]}"
        output_root = self.work_root / "window-e2e"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_window_e2e_validation.py"),
            "--session", str(Path(self.args.session).resolve()),
            "--output", str(output_root),
            "--run-id", run_id,
            "--profile-source", str((PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan").resolve()),
            "--bundle-root", str(self.bundle_root),
            "--executable", str(self.executable),
            "--frozen-data-root", str(self.work_root / "window-e2e-data"),
            "--scenarios", ",".join(FULL_SCENARIOS),
            "--time-scale", "1.0",
            "--baseline-summary", str(Path(self.args.baseline_summary).resolve()),
        ]
        completed = self._run(command, self.work_root / "window-e2e.log")
        host_summary_path = output_root / "runs" / run_id / "host_summary.json"
        validation = _validate_formal_window_e2e(host_summary_path)
        if completed.returncode != 0 or validation.get("status") != "PASS":
            raise _StageFailure("source/frozen window E2E failed", completed.returncode, command, completed.log_path, validation)
        return {"command": command, "log": str(completed.log_path), "output_root": str(output_root), "host_summary": str(host_summary_path), **validation}

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
        if (
            completed.returncode != 0
            or report.get("status") != "PASS"
            or report.get("physical_dpi_validation") != "NOT_RUN"
            or report.get("formal_acceptance_contribution") is not False
        ):
            raise _StageFailure("portability matrix failed", completed.returncode, command, completed.log_path, report)
        return {"report": str(output), "physical_dpi_validation": report.get("physical_dpi_validation"), "formal_acceptance_contribution": False, "dpi_covered_by": "source-frozen-window-e2e/dpi", "cases": report.get("cases")}

    def _bundle_immutability(self) -> Mapping[str, object]:
        assert self.bundle_root is not None
        after = _tree_hash(self.bundle_root)
        if self.bundle_hash_before != after:
            raise _StageFailure("frozen bundle changed after qualification stages", 2, None, None, {"before": self.bundle_hash_before, "after": after})
        return {"before": self.bundle_hash_before, "after": after, "unchanged": True}

    def _install_rollback(self) -> Mapping[str, object]:
        assert self.archive is not None and self.release_record is not None and self.archive_checksum is not None
        runtime = self.work_root / "install-runtime"
        baseline = register_legacy_baseline(
            Path(self.args.baseline_bundle),
            baseline_auth_path=Path(self.args.baseline_auth),
            runtime_root=runtime,
        )
        activate_release(baseline.release_id, runtime_root=runtime)
        verify_before_write = self._verify_active_launcher(
            runtime,
            label="baseline-before-write",
        )
        initial_legacy_probe = self._legacy_baseline_probe(
            baseline,
            require_runtime_write=True,
            label="before-candidate",
        )
        # VerifyOnly: the immutable externally authorized source tree must
        # still match after the separate mutable run copy wrote runtime data.
        verify_baseline_auth(
            Path(self.args.baseline_bundle),
            Path(self.args.baseline_auth),
        )
        release_status(runtime)
        verify_after_write = self._verify_active_launcher(
            runtime,
            label="baseline-after-write",
        )
        candidate = install_release(
            self.archive,
            release_record_path=self.release_record,
            checksum_path=self.archive_checksum,
            runtime_root=runtime,
        )
        activate_release(candidate.release_id, runtime_root=runtime)
        rolled = rollback_release(runtime_root=runtime)
        if rolled.get("release_id") != baseline.release_id:
            raise _StageFailure("rollback did not restore the legacy baseline pointer", 2, None, None, {"rolled_back": rolled, "baseline": baseline.to_dict()})
        status = _read_json(runtime / "install" / "active.json")
        verify_after_rollback = self._verify_active_launcher(
            runtime,
            label="baseline-after-rollback",
        )
        rollback_legacy_probe = self._legacy_baseline_probe(
            baseline,
            require_runtime_write=False,
            label="after-rollback",
        )
        verify_baseline_auth(
            Path(self.args.baseline_bundle),
            Path(self.args.baseline_auth),
        )
        release_status(runtime)
        verify_after_relaunch = self._verify_active_launcher(
            runtime,
            label="baseline-after-relaunch",
        )
        return {
            "runtime_root": str(runtime),
            "baseline_release": baseline.to_dict(),
            "candidate_release": candidate.to_dict(),
            "active_after_rollback": status,
            "legacy_baseline_reproducible": False,
            "legacy_mutable_probe_before_candidate": initial_legacy_probe,
            "legacy_probe_after_rollback": rollback_legacy_probe,
            "launcher_verify_before_write": verify_before_write,
            "launcher_verify_after_write": verify_after_write,
            "launcher_verify_after_rollback": verify_after_rollback,
            "launcher_verify_after_relaunch": verify_after_relaunch,
            "immutable_baseline_reverified_after_each_probe": True,
            "bad_candidate_preserved": candidate.version_root.is_dir(),
        }

    def _legacy_baseline_probe(
        self,
        release: object,
        *,
        require_runtime_write: bool,
        label: str,
    ) -> dict[str, object]:
        executable = Path(getattr(release, "executable"))
        run_root = executable.parent
        before = _tree_hash(run_root)
        command = [str(executable), "--fabledan-fixed-benchmark"]
        completed = self._run(
            command,
            self.work_root / f"legacy-baseline-{label}.log",
            timeout_seconds=300.0,
        )
        after = _tree_hash(run_root)
        if completed.returncode != 0:
            raise _StageFailure(
                f"legacy baseline {label} launch failed",
                completed.returncode,
                command,
                completed.log_path,
                {"tree_before": before, "tree_after": after},
            )
        if require_runtime_write and before == after:
            raise _StageFailure(
                "legacy baseline mutable run copy produced no runtime write",
                2,
                command,
                completed.log_path,
                {"tree_before": before, "tree_after": after},
            )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "tree_before": before,
            "tree_after": after,
            "runtime_write_observed": before != after,
        }

    def _verify_active_launcher(
        self,
        runtime_root: Path,
        *,
        label: str,
    ) -> dict[str, object]:
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "release_assets" / "Launch_DaguandanAssistant.ps1"),
            "-RuntimeRoot",
            str(runtime_root),
            "-VerifyOnly",
        ]
        completed = self._run(
            command,
            self.work_root / f"launcher-{label}.log",
            timeout_seconds=180.0,
        )
        if completed.returncode != 0:
            raise _StageFailure(
                f"launcher VerifyOnly failed: {label}",
                completed.returncode,
                command,
                completed.log_path,
                {},
            )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "log": str(completed.log_path),
        }

    def _finish(self) -> int:
        if self.source_identity is not None:
            try:
                final_source = verify_source_identity(
                    PROJECT_ROOT,
                    self.source_identity,
                    require_clean=True,
                )
            except Exception as exc:
                self.errors.append(f"final-source-immutability: {exc}")
                self.stages.append(
                    Stage(
                        "final-source-immutability",
                        status="FAIL",
                        evidence={"error_type": type(exc).__name__, "message": str(exc)},
                    )
                )
            else:
                self.stages.append(
                    Stage(
                        "final-source-immutability",
                        status="PASS",
                        evidence=final_source,
                    )
                )
        report = {
            "schema": QUALIFICATION_SCHEMA,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "PASS" if not self.errors and all(stage.status == "PASS" for stage in self.stages) else "FAIL",
            "branch": _git_value("branch --show-current"),
            "source_commit": _git_value("rev-parse HEAD"),
            "source_identity": self.source_identity,
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
        if self._output_owned:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_new_json(self.output_path, report)
            post_report = _post_report_artifact_hashes(
                bundle_root=self.bundle_root,
                archive=self.archive,
                expected_bundle=self.bundle_hash_before,
                expected_archive=self.archive_hash_before,
            )
            report["post_report_immutability"] = post_report
            if post_report.get("status") == "FAIL":
                report["status"] = "FAIL"
                report["errors"] = [*self.errors, "artifacts changed while publishing qualification report"]
            atomic_write_json(self.output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2

    def _run(
        self,
        command: list[str],
        log_path: Path,
        *,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> "_Completed":
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as handle:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(env) if env is not None else None,
                timeout=timeout_seconds,
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


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_below(first, second) or _is_below(second, first)


def _root_overlap_failures(
    *,
    project_root: Path,
    release_root: Path,
    work_root: Path,
    wheelhouse_root: Path,
    output_path: Path,
) -> list[str]:
    roots = {
        "project_root": project_root.resolve(),
        "release_root": release_root.resolve(),
        "work_root": work_root.resolve(),
        "wheelhouse_root": wheelhouse_root.resolve(),
        "output_path": output_path.resolve(),
    }
    failures: list[str] = []
    labels = list(roots)
    for index, first in enumerate(labels):
        for second in labels[index + 1 :]:
            if _paths_overlap(roots[first], roots[second]):
                failures.append(f"{first} overlaps {second}")
    return failures


def _validate_formal_window_e2e(host_summary_path: Path | str) -> dict[str, object]:
    host_path = Path(host_summary_path).resolve()
    host = _read_json(host_path)
    failures: list[str] = []
    if host.get("schema") != "guandan.window-e2e-host-summary/1":
        failures.append("host_schema_invalid")
    for field in ("execution_ok", "acceptance_eligible", "acceptance_passed"):
        if host.get(field) is not True:
            failures.append(f"host_{field}_not_true")
    for field in ("complete_source_and_bundle_matrix", "same_hwnd", "integrity_unchanged"):
        if host.get(field) is not True:
            failures.append(f"host_{field}_not_true")
    if host.get("source_only_debug_run") is not False:
        failures.append("host_source_only_debug_run_not_false")
    if host.get("forced_simulator_termination") is not False:
        failures.append("host_forced_simulator_termination_not_false")
    for field in ("source_exit_code", "package_exit_code", "bundle_exit_code"):
        if host.get(field) != 0:
            failures.append(f"host_{field}_not_zero")

    scenario_evidence: dict[str, list[str]] = {}
    for label, expected_kind in (("source", "source"), ("frozen", "frozen_exe")):
        raw_path = host.get(f"{label if label == 'source' else 'bundle'}_summary")
        summary_path = Path(str(raw_path)).resolve() if raw_path else Path()
        expected_path = host_path.parent / ("source" if label == "source" else "bundle") / "summary.json"
        if summary_path != expected_path.resolve():
            failures.append(f"{label}_summary_path_not_exact")
        summary = _read_json(summary_path)
        if summary.get("schema") != "guandan.window-e2e-summary/1":
            failures.append(f"{label}_summary_schema_invalid")
        if summary.get("run_kind") != expected_kind:
            failures.append(f"{label}_run_kind_invalid")
        for field in ("execution_ok", "acceptance_eligible", "acceptance_passed"):
            if summary.get(field) is not True:
                failures.append(f"{label}_{field}_not_true")
        raw_scenarios = summary.get("scenarios")
        scenarios = raw_scenarios if isinstance(raw_scenarios, Mapping) else {}
        if set(scenarios) != set(FULL_SCENARIOS):
            failures.append(f"{label}_scenario_set_invalid")
        for name in FULL_SCENARIOS:
            value = scenarios.get(name)
            if not isinstance(value, Mapping) or value.get("passed") is not True:
                failures.append(f"{label}_scenario_{name}_not_passed")
        scenario_evidence[f"{label}_scenarios"] = [
            name for name in FULL_SCENARIOS if isinstance(scenarios.get(name), Mapping)
        ]
    return {
        "status": "PASS" if not failures else "FAIL",
        "host_summary": str(host_path),
        **scenario_evidence,
        "failures": failures,
    }


def _validate_release_repro_equivalence(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    expected_support_sha256: str,
    expected_candidate_build_id: str | None,
) -> dict[str, object]:
    failures: list[str] = []
    for label, report in (("reference", reference), ("candidate", candidate)):
        if report.get("schema") != "guandan.repro-report/1":
            failures.append(f"{label}_schema_invalid")
        if report.get("deterministic") is not True:
            failures.append(f"{label}_not_deterministic")
        if report.get("repeat_count") != 20 or _nested(report, "repeatability", "repeatable") is not True:
            failures.append(f"{label}_not_repeatable_20_of_20")
        if _nested(report, "support", "sha256") != expected_support_sha256:
            failures.append(f"{label}_support_hash_mismatch")
        if _nested(report, "truth", "eligible_for_fix_verification") is not True:
            failures.append(f"{label}_truth_missing")
        if _nested(report, "truth", "correct_runs") != 20:
            failures.append(f"{label}_not_correct_20_of_20")
        if _nested(report, "probes", "fresh_child_deterministic", "status") != "PASS":
            failures.append(f"{label}_fresh_child_not_passed")
    if reference.get("mode") != "source":
        failures.append("reference_not_source")
    if candidate.get("mode") != "frozen":
        failures.append("candidate_not_frozen")
    if _nested(candidate, "runner", "build_id") != expected_candidate_build_id:
        failures.append("candidate_build_id_mismatch")
    reference_build = _nested(reference, "runner", "build_id")
    candidate_build = _nested(candidate, "runner", "build_id")
    if reference_build == candidate_build:
        failures.append("runner_builds_not_distinct")
    if _nested(reference, "truth_identity", "sha256") != _nested(
        candidate, "truth_identity", "sha256"
    ):
        failures.append("truth_hash_mismatch")
    if _nested(reference, "truth_identity", "input_sequence_sha256") != _nested(
        candidate, "truth_identity", "input_sequence_sha256"
    ):
        failures.append("input_sequence_hash_mismatch")
    reference_outputs = _normalized_output_fingerprints(reference)
    candidate_outputs = _normalized_output_fingerprints(candidate)
    if len(reference_outputs) != 1 or reference_outputs != candidate_outputs:
        failures.append("normalized_outputs_not_equivalent")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures}


def _validate_formal_manifest_source(
    manifest: Mapping[str, object],
    expected: Mapping[str, object] | None,
) -> dict[str, object]:
    failures: list[str] = []
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        failures.append("manifest_source_missing")
        source = {}
    if not isinstance(expected, Mapping):
        failures.append("expected_source_missing")
        expected = {}
    for field in ("commit", "tree", "dirty", "status_sha256"):
        if source.get(field) != expected.get(field):
            failures.append(f"manifest_source_{field}_mismatch")
    if source.get("dirty") is not False:
        failures.append("manifest_source_not_clean")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "commit": source.get("commit"),
        "tree": source.get("tree"),
        "dirty": source.get("dirty"),
    }


def _normalized_output_fingerprints(
    report: Mapping[str, object],
) -> set[str]:
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        return set()
    return {
        str(item.get("output_fingerprint"))
        for item in outcomes
        if isinstance(item, Mapping) and item.get("output_fingerprint")
    }


def _validate_repro_inputs(
    support_path: Path,
    truth_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    failures: list[str] = []
    try:
        support = verify_support_archive(support_path)
    except Exception as exc:
        return {
            "status": "FAIL",
            "failures": ["support_invalid"],
            "error_type": type(exc).__name__,
        }
    truth = _read_json(truth_path)
    reference = _read_json(reference_path)
    if truth.get("schema") != "guandan.repro-truth/1":
        failures.append("truth_schema_invalid")
    if truth.get("support_sha256") != support.sha256:
        failures.append("truth_support_hash_mismatch")
    if not truth.get("expected_level") and not truth.get("expected_hand"):
        failures.append("truth_has_no_independent_expectation")
    if reference.get("schema") != "guandan.repro-report/1":
        failures.append("reference_schema_invalid")
    if _nested(reference, "support", "sha256") != support.sha256:
        failures.append("reference_support_hash_mismatch")
    if reference.get("mode") != "source":
        failures.append("reference_not_source")
    if reference.get("repeat_count") != 20 or _nested(
        reference, "repeatability", "repeatable"
    ) is not True:
        failures.append("reference_not_repeatable_20_of_20")
    if _nested(reference, "truth", "correct_runs") != 20:
        failures.append("reference_not_correct_20_of_20")
    if _nested(reference, "truth_identity", "sha256") != sha256_file(truth_path):
        failures.append("reference_truth_hash_mismatch")
    return {
        "status": "PASS" if not failures else "FAIL",
        "support_sha256": support.sha256,
        "truth_sha256": sha256_file(truth_path),
        "reference_sha256": sha256_file(reference_path),
        "failures": failures,
    }


def _nested(value: Mapping[str, object], *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"qualification output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _post_report_artifact_hashes(
    *,
    bundle_root: Path | None,
    archive: Path | None,
    expected_bundle: str | None,
    expected_archive: str | None,
) -> dict[str, object]:
    if expected_bundle is None and expected_archive is None:
        return {
            "status": "NOT_RUN",
            "reason": "candidate artifacts were not produced",
            "bundle_before": None,
            "bundle_after": None,
            "archive_before": None,
            "archive_after": None,
        }
    actual_bundle = _tree_hash(bundle_root) if bundle_root is not None and bundle_root.is_dir() else None
    actual_archive = sha256_file(archive) if archive is not None and archive.is_file() else None
    passed = bool(
        expected_bundle is not None
        and expected_archive is not None
        and actual_bundle == expected_bundle
        and actual_archive == expected_archive
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "bundle_before": expected_bundle,
        "bundle_after": actual_bundle,
        "archive_before": expected_archive,
        "archive_after": actual_archive,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--repro-support", type=Path, required=True)
    parser.add_argument("--repro-truth", type=Path, required=True)
    parser.add_argument("--reference-repro-report", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--baseline-auth", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return Qualification(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
