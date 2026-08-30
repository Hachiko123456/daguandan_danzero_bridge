from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_help_exposes_support_and_repro_routes():
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    for option in (
        "--export-support",
        "--include-support-images",
        "--repro-support",
        "--annotate-repro-truth",
        "--compare-repro",
    ):
        assert option in completed.stdout


def test_run_can_annotate_truth_without_mutating_support_zip(tmp_path):
    # A malformed ZIP is enough to prove the CLI routes through the hostile
    # archive verifier rather than writing over the input.
    support = tmp_path / "support.zip"
    support.write_bytes(b"not a zip")
    truth = tmp_path / "truth.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "run.py"),
            "--annotate-repro-truth",
            str(support),
            "--truth-output",
            str(truth),
            "--expected-level",
            "7",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not truth.exists()
    assert support.read_bytes() == b"not a zip"
