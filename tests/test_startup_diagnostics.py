from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def _run(code: str, diagnostics_root: Path) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["DAGUANDAN_DIAGNOSTICS_ROOT"] = str(diagnostics_root)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )


def test_initialization_is_idempotent_and_noconsole_thread_errors_are_persisted(tmp_path):
    root = tmp_path / "diagnostics"
    completed = _run(
        "\n".join(
            (
                "import sys, threading",
                "sys.stdout = None",
                "sys.stderr = None",
                "from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics",
                "first = initialize_startup_diagnostics()",
                "second = initialize_startup_diagnostics()",
                "assert first is second",
                "assert sys.stdout is not None and sys.stderr is not None",
                "def fail(): raise RuntimeError('thread-marker-42')",
                "worker = threading.Thread(target=fail, name='doctor-worker')",
                "worker.start()",
                "worker.join()",
            )
        ),
        root,
    )

    assert completed.returncode == 0
    run_directories = tuple((root / "runs").iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert (run_directory / "startup.log").is_file()
    assert "thread-marker-42" in (run_directory / "exceptions.log").read_text(
        encoding="utf-8"
    )
    assert "worker_thread" in (run_directory / "startup.jsonl").read_text(
        encoding="utf-8"
    )
    assert (run_directory / "faulthandler.log").is_file()


def test_uncaught_main_thread_exception_is_persisted(tmp_path):
    root = tmp_path / "diagnostics"
    completed = _run(
        "\n".join(
            (
                "from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics",
                "initialize_startup_diagnostics()",
                "raise ValueError('main-marker-99')",
            )
        ),
        root,
    )

    assert completed.returncode != 0
    run_directory = next((root / "runs").iterdir())
    assert "main-marker-99" in (run_directory / "exceptions.log").read_text(
        encoding="utf-8"
    )


def test_unwritable_diagnostics_target_is_fail_open(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    completed = _run(
        "\n".join(
            (
                "from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics",
                "state = initialize_startup_diagnostics()",
                "assert state.enabled is False",
                "assert state.error",
            )
        ),
        blocked,
    )

    assert completed.returncode == 0


def test_run_entrypoint_writes_a_sanitized_atomic_runtime_identity_snapshot(tmp_path):
    root = tmp_path / "diagnostics"
    environment = dict(os.environ)
    environment["DAGUANDAN_DIAGNOSTICS_ROOT"] = str(root)
    environment["USERNAME"] = "snapshot-secret-user"

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run.py"), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0
    run_directory = next((root / "runs").iterdir())
    snapshot = run_directory / "runtime_identity.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["schema"] == "guandan.runtime-identity/1"
    assert payload["run_id"] == run_directory.name
    assert "snapshot-secret-user" not in snapshot.read_text(encoding="utf-8")
    assert not tuple(run_directory.glob(".runtime_identity.json.*.tmp"))
