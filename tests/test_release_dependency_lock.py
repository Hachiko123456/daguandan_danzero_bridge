from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.release_lock import verify_release_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = Path(
    r"C:\Users\yhx\AppData\Local\DaguandanAssistant\release-inputs\wheelhouse-cp312-win_amd64"
)


def test_runtime_requirements_are_exactly_pinned():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "torch==2.13.0" in pyproject
    assert "torch==2.13.0" in requirements
    assert "torch>=" not in pyproject
    assert "torch>=" not in requirements


def test_release_lock_has_hash_for_every_wheel():
    wheel_lock = json.loads((PROJECT_ROOT / "wheelhouse.lock.json").read_text(encoding="utf-8"))
    requirements = (PROJECT_ROOT / "requirements-release.lock").read_text(encoding="utf-8")

    assert wheel_lock["schema"] == "guandan.wheelhouse-lock/1"
    assert wheel_lock["files"]
    for record in wheel_lock["files"]:
        assert f"--hash=sha256:{record['sha256']}" in requirements


def test_prepared_release_wheelhouse_matches_all_committed_locks():
    if not WHEELHOUSE.is_dir():
        import pytest

        pytest.skip("external release wheelhouse has not been prepared")

    report = verify_release_inputs(
        project_root=PROJECT_ROOT,
        wheelhouse_root=WHEELHOUSE,
        python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    )

    assert report["status"] == "PASS", report["errors"]
