from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import shutil
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "qualify_release.py"
SPEC = importlib.util.spec_from_file_location("qualify_release", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["qualify_release"] = MODULE
SPEC.loader.exec_module(MODULE)


def _formal_arguments() -> list[str]:
    return [
        "--release-root",
        r"C:\outside\candidate",
        "--wheelhouse",
        r"C:\outside\wheelhouse",
        "--work-root",
        r"C:\outside\work",
        "--output",
        r"C:\outside\qualification.json",
        "--session",
        r"C:\inputs\session",
        "--baseline-summary",
        r"C:\inputs\baseline-summary.json",
        "--repro-support",
        r"C:\inputs\support.zip",
        "--repro-truth",
        r"C:\inputs\truth.json",
        "--reference-repro-report",
        r"C:\inputs\reference-repro.json",
        "--baseline-bundle",
        r"C:\inputs\baseline-bundle",
        "--baseline-auth",
        r"C:\inputs\baseline-auth.json",
    ]


def test_qualification_parser_requires_all_formal_inputs():
    parser = MODULE.build_parser()
    args = parser.parse_args(_formal_arguments())
    assert args.release_root.name == "candidate"
    assert args.wheelhouse.name == "wheelhouse"
    assert args.session.name == "session"
    assert args.baseline_summary.name == "baseline-summary.json"

    value_options = {
        "--session",
        "--baseline-summary",
        "--repro-support",
        "--repro-truth",
        "--reference-repro-report",
        "--baseline-bundle",
        "--baseline-auth",
        "--work-root",
        "--output",
    }
    for option in value_options:
        values = _formal_arguments()
        index = values.index(option)
        del values[index : index + 2]
        with pytest.raises(SystemExit):
            parser.parse_args(values)


def test_tree_hash_is_order_independent(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = MODULE._tree_hash(tmp_path)
    (tmp_path / "a.txt").touch()
    second = MODULE._tree_hash(tmp_path)
    assert first == second


def test_clean_runtime_environment_removes_host_pollution(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "private")
    monkeypatch.setenv("JAVA_HOME", "jdk")
    environment = MODULE._clean_runtime_environment(tmp_path / "data")
    assert environment["DAGUANDAN_DATA_ROOT"].endswith("data")
    assert "PYTHONPATH" not in environment
    assert "JAVA_HOME" not in environment


def test_stage_failure_is_machine_readable():
    failure = MODULE._StageFailure("boom", 7, ["cmd"], Path("x.log"), {"foo": "bar"})
    assert str(failure) == "boom"
    assert failure.exit_code == 7
    assert failure.command == ["cmd"]
    assert failure.evidence == {"foo": "bar"}


def test_formal_rollback_stage_runs_real_legacy_write_and_relaunch_probes():
    source = SCRIPT.read_text(encoding="utf-8")
    method = source[source.index("def _install_rollback") : source.index("def _finish")]

    assert 'label="before-candidate"' in method
    assert "require_runtime_write=True" in method
    assert 'label="after-rollback"' in method
    assert 'command = [str(executable), "--help"]' in method
    assert "qualification_probe_" in method
    assert method.count("verify_baseline_auth(") >= 2
    assert method.count("release_status(runtime)") >= 2
    assert method.count("_verify_active_launcher(") >= 4
    assert "-VerifyOnly" in method
    assert "timeout_seconds=60.0" in method
    assert "_short_qualification_runtime_root()" in method
    assert "runtime_root_is_short_external" in method
    assert 'self.work_root / "install-runtime"' not in method
    assert '"preserve_for_audit"' in method


def test_short_formal_runtime_ignores_long_work_root_and_projects_under_240(
    tmp_path,
    monkeypatch,
):
    short_temp = tmp_path / "t"
    short_temp.mkdir()
    monkeypatch.setattr(MODULE.tempfile, "gettempdir", lambda: str(short_temp))
    runtime = MODULE._short_qualification_runtime_root()
    auth = {
        "executable": {"sha256": "a" * 64},
        "artifact": {
            "files": [
                {"path": "_internal/PySide6/Qt/qml/Deep/asset.bin"},
                {"path": "DaguandanAssistant.exe"},
            ]
        },
    }
    long_work = tmp_path / ("very-long-work-root-" * 8)

    projected = MODULE._projected_legacy_path_length(runtime, auth)

    assert not runtime.exists()
    assert projected < 240
    assert str(long_work) not in str(runtime)
    assert runtime.name.startswith("dga-q-")


def test_legacy_probe_real_launch_writes_only_mutable_run_tree(tmp_path):
    run_root = tmp_path / "run" / "DaguandanAssistant"
    run_root.mkdir(parents=True)
    tar = shutil.which("tar")
    if tar is None:
        pytest.skip("Windows tar.exe is unavailable")
    executable = run_root / "probe.exe"
    shutil.copy2(tar, executable)
    qualification = object.__new__(MODULE.Qualification)
    qualification.work_root = tmp_path / "work"
    qualification.work_root.mkdir()
    release = SimpleNamespace(executable=executable)

    first = qualification._legacy_baseline_probe(
        release,
        require_runtime_write=True,
        label="test-write",
    )
    second = qualification._legacy_baseline_probe(
        release,
        require_runtime_write=False,
        label="test-relaunch",
    )

    written = run_root / str(first["runtime_write_relative"])
    assert first["exit_code"] == 0
    assert first["runtime_write_observed"] is True
    assert written.is_file()
    assert first["runtime_write_sha256"] == MODULE.sha256_file(written)
    assert second["exit_code"] == 0
    assert second["runtime_write_relative"] is None


def test_host_summary_gate_requires_exact_formal_seven_scenario_pass(tmp_path):
    run = tmp_path / "runs" / "formal"
    source_path = run / "source" / "summary.json"
    bundle_path = run / "bundle" / "summary.json"
    source_path.parent.mkdir(parents=True)
    bundle_path.parent.mkdir(parents=True)
    scenarios = {
        name: {"passed": True}
        for name in MODULE.FULL_SCENARIOS
    }
    source_path.write_text(
        json.dumps(
            {
                "schema": "guandan.window-e2e-summary/1",
                "run_kind": "source",
                "execution_ok": True,
                "acceptance_eligible": True,
                "acceptance_passed": True,
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )
    bundle_path.write_text(
        json.dumps(
            {
                "schema": "guandan.window-e2e-summary/1",
                "run_kind": "frozen_exe",
                "execution_ok": True,
                "acceptance_eligible": True,
                "acceptance_passed": True,
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )
    host_path = run / "host_summary.json"
    host_path.write_text(
        json.dumps(
            {
                "schema": "guandan.window-e2e-host-summary/1",
                "execution_ok": True,
                "acceptance_eligible": True,
                "acceptance_passed": True,
                "complete_source_and_bundle_matrix": True,
                "source_only_debug_run": False,
                "source_exit_code": 0,
                "package_exit_code": 0,
                "bundle_exit_code": 0,
                "same_hwnd": True,
                "integrity_unchanged": True,
                "forced_simulator_termination": False,
                "source_summary": str(source_path),
                "bundle_summary": str(bundle_path),
            }
        ),
        encoding="utf-8",
    )

    result = MODULE._validate_formal_window_e2e(host_path)

    assert result["status"] == "PASS"
    assert result["source_scenarios"] == list(MODULE.FULL_SCENARIOS)
    assert result["frozen_scenarios"] == list(MODULE.FULL_SCENARIOS)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["scenarios"]["dpi"]["passed"] = False
    source_path.write_text(json.dumps(source), encoding="utf-8")
    failed = MODULE._validate_formal_window_e2e(host_path)
    assert failed["status"] == "FAIL"
    assert "source_scenario_dpi_not_passed" in failed["failures"]

    source["scenarios"]["dpi"]["passed"] = True
    source_path.write_text(json.dumps(source), encoding="utf-8")
    host = json.loads(host_path.read_text(encoding="utf-8"))
    host["source_summary"] = str(tmp_path / "unrelated-passing-summary.json")
    host_path.write_text(json.dumps(host), encoding="utf-8")
    wrong_path = MODULE._validate_formal_window_e2e(host_path)
    assert wrong_path["status"] == "FAIL"
    assert "source_summary_path_not_exact" in wrong_path["failures"]


def test_release_repro_equivalence_requires_both_correct_and_distinct_builds():
    reference = {
        "schema": "guandan.repro-report/1",
        "mode": "source",
        "deterministic": True,
        "verification_role": "reference",
        "repeat_count": 20,
        "support": {"sha256": "a" * 64},
        "repeatability": {"repeatable": True},
        "truth": {"eligible_for_fix_verification": True, "correct_runs": 20},
        "truth_identity": {"sha256": "b" * 64, "input_sequence_sha256": "c" * 64},
        "runner": {"build_id": "source"},
        "probes": {"fresh_child_deterministic": {"status": "PASS"}},
        "outcomes": [{"output_fingerprint": "d" * 64} for _ in range(20)],
    }
    candidate = {
        **reference,
        "mode": "frozen",
        "verification_role": "candidate",
        "runner": {"build_id": "BUILD-candidate"},
    }

    result = MODULE._validate_release_repro_equivalence(
        reference,
        candidate,
        expected_support_sha256="a" * 64,
        expected_candidate_build_id="BUILD-candidate",
    )

    assert result == {"status": "PASS", "failures": []}

    candidate["truth_identity"] = {
        "sha256": "e" * 64,
        "input_sequence_sha256": "c" * 64,
    }
    candidate["outcomes"] = [
        {"output_fingerprint": "f" * 64} for _ in range(20)
    ]
    failed = MODULE._validate_release_repro_equivalence(
        reference,
        candidate,
        expected_support_sha256="a" * 64,
        expected_candidate_build_id="BUILD-candidate",
    )

    assert failed["status"] == "FAIL"
    assert "truth_hash_mismatch" in failed["failures"]
    assert "normalized_outputs_not_equivalent" in failed["failures"]


def test_managed_roots_must_be_pairwise_disjoint_and_output_new(tmp_path):
    project = tmp_path / "project"
    release = tmp_path / "release"
    work = release / "work"
    wheelhouse = tmp_path / "wheelhouse"
    output = tmp_path / "qualification.json"

    failures = MODULE._root_overlap_failures(
        project_root=project,
        release_root=release,
        work_root=work,
        wheelhouse_root=wheelhouse,
        output_path=output,
    )

    assert "release_root overlaps work_root" in failures


def test_qualification_output_is_exclusive_and_post_publish_hash_gate_detects_change(
    tmp_path,
):
    output = tmp_path / "qualification.json"
    MODULE._write_new_json(output, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        MODULE._write_new_json(output, {"status": "FAIL"})

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "app.bin").write_bytes(b"before")
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"archive")
    expected_bundle = MODULE._tree_hash(bundle)
    expected_archive = MODULE.sha256_file(archive)
    (bundle / "app.bin").write_bytes(b"after!")

    result = MODULE._post_report_artifact_hashes(
        bundle_root=bundle,
        archive=archive,
        expected_bundle=expected_bundle,
        expected_archive=expected_archive,
    )

    assert result["status"] == "FAIL"
    assert result["bundle_before"] != result["bundle_after"]


def test_formal_manifest_source_must_equal_clean_preflight_head():
    expected = {
        "commit": "1" * 40,
        "tree": "2" * 40,
        "branch": "fix_dif_computer",
        "dirty": False,
        "status_sha256": None,
    }
    manifest = {"source": dict(expected)}

    assert MODULE._validate_formal_manifest_source(manifest, expected)["status"] == "PASS"

    manifest["source"]["tree"] = "3" * 40
    manifest["source"]["dirty"] = True
    manifest["source"]["status_sha256"] = "4" * 64
    failed = MODULE._validate_formal_manifest_source(manifest, expected)

    assert failed["status"] == "FAIL"
    assert "manifest_source_tree_mismatch" in failed["failures"]
    assert "manifest_source_not_clean" in failed["failures"]
