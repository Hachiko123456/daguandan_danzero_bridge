from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_portability_matrix.py"
_SPEC = importlib.util.spec_from_file_location("run_portability_matrix", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["run_portability_matrix"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _fake_runner(executable, bundle_root, data_root, read_only_bundle):
    del executable, bundle_root
    return {
        "doctor_exit_code": 0,
        "failed_checks": [],
        "passed": data_root.name != "bad",
        "read_only_bundle": read_only_bundle,
    }


def test_matrix_covers_unicode_and_readonly_without_fake_dpi_cases(tmp_path):
    bundle = tmp_path / "中文 bundle"
    bundle.mkdir()
    (bundle / "DaguandanAssistant.exe").write_bytes(b"exe")
    report = _MODULE.run_matrix(
        executable=bundle / "DaguandanAssistant.exe",
        bundle_root=bundle,
        output_root=tmp_path / "out",
        doctor_runner=_fake_runner,
    )

    assert report["status"] == "PASS"
    assert report["bundle_unchanged"] is True
    assert {item["case_id"] for item in report["cases"]} == {
        "unicode-space-path",
        "readonly-bundle",
    }
    assert report["physical_dpi_validation"] == "NOT_RUN"
    assert report["formal_acceptance_contribution"] is False


def test_matrix_fails_when_a_case_fails(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    exe = bundle / "DaguandanAssistant.exe"
    exe.write_bytes(b"exe")

    def failing_runner(_exe, _bundle, data_root, read_only_bundle):
        return {
            "passed": data_root.name != "readonly data",
            "read_only_bundle": read_only_bundle,
        }

    report = _MODULE.run_matrix(
        executable=exe,
        bundle_root=bundle,
        output_root=tmp_path / "out",
        doctor_runner=failing_runner,
    )

    assert report["status"] == "FAIL"


def test_matrix_fails_if_doctor_writes_any_bundle_file(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    exe = bundle / "DaguandanAssistant.exe"
    exe.write_bytes(b"exe")

    def writing_runner(_exe, selected_bundle, _data_root, _read_only):
        (selected_bundle / "unexpected.log").write_text("write", encoding="utf-8")
        return {"passed": True}

    report = _MODULE.run_matrix(
        executable=exe,
        bundle_root=bundle,
        output_root=tmp_path / "out",
        doctor_runner=writing_runner,
    )

    assert report["status"] == "FAIL"
    assert report["bundle_unchanged"] is False
