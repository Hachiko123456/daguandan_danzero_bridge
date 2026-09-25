from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


HELP_CASES = [
    (
        "scripts/build_saveable_truthlogs.py",
        ["--source-batch", "--output-batch", "--session-root", "read", "write"],
    ),
    (
        "scripts/run_selected_unverified_rescan.py",
        ["--session-root", "--profile-root", "--output-root", "--max-workers", "read", "write"],
    ),
    (
        "scripts/run_selected_unverified_rescan_quiet.py",
        ["--session-root", "--profile-root", "--output-root", "--max-workers", "read", "write"],
    ),
    (
        "scripts/run_video_scan.py",
        ["session_dir", "--output-root", "read", "write"],
    ),
    (
        "scripts/verify_terminal_placement_inference.py",
        ["--sessions-root", "read", "writes no files"],
    ),
]


def test_selected_scripts_help_contract() -> None:
    for script, expected_fragments in HELP_CASES:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert "usage:" in output.lower(), output
        for fragment in expected_fragments:
            assert fragment.lower() in output.lower(), f"{script} help missing {fragment!r}:\n{output}"
