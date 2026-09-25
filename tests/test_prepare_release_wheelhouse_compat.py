from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_release_wheelhouse.ps1"


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell")
def test_prepare_release_wheelhouse_legacy_hash_fallback_matches_sha256(tmp_path: Path) -> None:
    source = SCRIPT.read_text(encoding="utf-8-sig")
    start = source.index("function Get-ReleaseFileHash")
    end = source.index("$candidateRequirements", start)
    function_source = source[start:end]

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"legacy PowerShell hash compatibility\x00")
    payload_literal = str(payload).replace("'", "''")
    harness = tmp_path / "hash_fallback.ps1"
    harness.write_text(
        function_source
        + "\n"
        + "function Get-Command { param($Name, $CommandType, $ErrorAction) return $null }\n"
        + f"Get-ReleaseFileHash -LiteralPath '{payload_literal}'\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == hashlib.sha256(payload.read_bytes()).hexdigest().upper()


def test_prepare_release_wheelhouse_contains_compatibility_fallback_and_unchanged_lock_checks() -> None:
    source = SCRIPT.read_text(encoding="utf-8-sig")

    assert "Get-Command -Name Get-FileHash -CommandType Cmdlet" in source
    assert "[System.Security.Cryptography.SHA256]::Create()" in source
    assert "[System.IO.File]::OpenRead($LiteralPath)" in source
    assert "[System.BitConverter]::ToString($sha256.ComputeHash($stream)).Replace('-', '')" in source
    assert "Get-ReleaseFileHash -LiteralPath $candidateRequirements" in source
    assert "Get-ReleaseFileHash -LiteralPath $candidateWheelhouse" in source
    assert "Downloaded wheels do not match the committed requirements lock" in source
    assert "Downloaded wheels do not match the committed wheelhouse lock" in source
    assert "Get-FileHash -LiteralPath $candidateRequirements" not in source
    assert "Get-FileHash -LiteralPath $candidateWheelhouse" not in source


@pytest.mark.skipif(os.name != "nt", reason="requires Windows PowerShell")
def test_prepare_release_wheelhouse_help_entrypoint_parses() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-?",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
