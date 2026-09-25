from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_every_python_script_has_a_working_help_entrypoint() -> None:
    failures: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        completed = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        output = completed.stdout + "\n" + completed.stderr
        if completed.returncode != 0 or "usage:" not in output.lower():
            failures.append(f"{path.name}: exit={completed.returncode}; output={output[:240]!r}")
    assert not failures, "\n".join(failures)


def test_every_powershell_script_supports_standard_help_switch() -> None:
    failures: list[str] = []
    for path in sorted(SCRIPTS.glob("*.ps1")):
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path), "-?"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        if completed.returncode != 0:
            failures.append(f"{path.name}: exit={completed.returncode}; output={(completed.stdout + completed.stderr)[:240]!r}")
    assert not failures, "\n".join(failures)


def test_batch_packaging_entrypoint_supports_help_switch() -> None:
    completed = subprocess.run(
        ["cmd.exe", "/c", str(ROOT / "package_release.bat"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0
    assert "Usage: package_release.bat" in completed.stdout
