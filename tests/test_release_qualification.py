from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

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


def test_repro_gate_rejects_non_frozen_or_missing_independent_truth():
    reference = {
        "schema": "guandan.repro-report/1",
        "mode": "frozen",
        "deterministic": True,
        "repeat_count": 20,
        "support": {"sha256": "a" * 64},
        "repeatability": {"repeatable": True},
        "truth": {"eligible_for_fix_verification": True, "correct_runs": 0},
    }
    candidate = {
        **reference,
        "mode": "source",
        "runner": {"build_id": "BUILD-candidate"},
        "truth": {"eligible_for_fix_verification": False, "correct_runs": 20},
    }
    gate = {
        "schema": "guandan.repro-gate/1",
        "status": "PASS",
        "failures": [],
        "support_sha256": "a" * 64,
    }

    result = MODULE._validate_frozen_repro_gate(
        reference,
        candidate,
        gate,
        expected_support_sha256="a" * 64,
        expected_candidate_build_id="BUILD-candidate",
    )

    assert result["status"] == "FAIL"
    assert "candidate_not_frozen" in result["failures"]
    assert "candidate_truth_missing" in result["failures"]


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
