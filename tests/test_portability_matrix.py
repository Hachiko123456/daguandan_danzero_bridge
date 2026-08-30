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


def _fake_runner(executable, bundle_root, data_root, dpi):
    del executable, bundle_root
    return {
        "doctor_exit_code": 0,
        "failed_checks": [],
        "passed": data_root.name != "bad",
        "simulated_dpi": dpi,
    }


def test_matrix_covers_unicode_readonly_and_three_simulated_dpis(tmp_path):
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
        "dpi-96",
        "dpi-120",
        "dpi-144",
    }
    assert all(item["physical_dpi_validated"] is False for item in report["cases"])


def test_matrix_fails_when_a_case_fails(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    exe = bundle / "DaguandanAssistant.exe"
    exe.write_bytes(b"exe")

    def failing_runner(_exe, _bundle, data_root, dpi):
        return {"passed": data_root.name != "readonly data", "simulated_dpi": dpi}

    report = _MODULE.run_matrix(
        executable=exe,
        bundle_root=bundle,
        output_root=tmp_path / "out",
        doctor_runner=failing_runner,
    )

    assert report["status"] == "FAIL"
