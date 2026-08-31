from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_bootstrap_python.py"
SPEC = importlib.util.spec_from_file_location("audit_bootstrap_python", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["audit_bootstrap_python"] = MODULE
SPEC.loader.exec_module(MODULE)


def test_bootstrap_sys_path_audit_rejects_external_search_root():
    base = Path(sys.base_prefix)
    report = MODULE.audit_bootstrap_environment(
        base,
        PROJECT_ROOT / "python_runtime.lock.json",
        search_paths=[str(base / "Lib"), str(PROJECT_ROOT / "external-site-packages")],
    )

    codes = {item["code"] for item in report["errors"]}
    assert report["status"] == "FAIL"
    assert "BOOTSTRAP-SYSPATH-EXTERNAL" in codes


def test_no_site_bootstrap_ignores_pth_and_sitecustomize_sentinels(tmp_path: Path):
    bootstrap = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    isolated = tmp_path / "bootstrap"
    created = subprocess.run(
        [str(bootstrap), "-I", "-S", "-m", "venv", str(isolated)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    python = isolated / "Scripts" / "python.exe"
    site_packages = isolated / "Lib" / "site-packages"
    pth_sentinel = tmp_path / "pth-executed.txt"
    site_sentinel = tmp_path / "sitecustomize-executed.txt"
    (site_packages / "host-injection.pth").write_text(
        "import pathlib; pathlib.Path(" + repr(str(pth_sentinel)) + ").write_text('executed')\n",
        encoding="utf-8",
    )
    (site_packages / "sitecustomize.py").write_text(
        "import pathlib\npathlib.Path(" + repr(str(site_sentinel)) + ").write_text('executed')\n",
        encoding="utf-8",
    )
    base = subprocess.run(
        [str(python), "-I", "-S", "-c", "import sys; print(sys.base_prefix)"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()
    output = tmp_path / "audit.json"

    audited = subprocess.run(
        [
            str(python),
            "-I",
            "-S",
            str(SCRIPT),
            "--python-root",
            base,
            "--runtime-lock",
            str(PROJECT_ROOT / "python_runtime.lock.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert audited.returncode == 0, audited.stdout + audited.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["flags"]["no_site"] == 1
    assert report["site_modules_loaded"] == []
    assert not pth_sentinel.exists()
    assert not site_sentinel.exists()

    control = subprocess.run(
        [str(python), "-I", "-c", "pass"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert control.returncode == 0, control.stdout + control.stderr
    assert pth_sentinel.is_file()
    assert site_sentinel.is_file()
