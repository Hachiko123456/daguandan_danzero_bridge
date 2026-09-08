from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from daguandan_bridge.diagnostic_budget import DiagnosticBudget


def test_budget_counts_other_runs_and_staging_without_entering_links(tmp_path):
    runs = tmp_path / "runs"
    old = runs / "old" / "opening" / "incidents" / "OPEN-1-a"
    pending = runs / "current" / "opening" / "incidents" / ".OPEN-2-b.tmp"
    old.mkdir(parents=True)
    pending.mkdir(parents=True)
    (old / "proof.png").write_bytes(b"a" * 100)
    (old / "incident.json").write_bytes(b"{}")
    (pending / "proof.png").write_bytes(b"b" * 50)
    (pending / "incident.json").write_bytes(b"{}")
    budget = DiagnosticBudget(
        runs / "current", run_media_bytes=100,
        total_media_bytes=175, run_text_bytes=100,
        total_text_bytes=100, run_incidents=5, total_incidents=6,
    )
    with budget.transaction() as available:
        assert available.media_bytes == 25
        assert available.text_bytes == 96
        assert available.incidents_remaining == 4


def test_budget_scope_rejects_run_outside_explicit_root(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        DiagnosticBudget(tmp_path / "unrelated", runs_root=tmp_path / "runs")


def test_budget_lock_serializes_competing_processes(tmp_path):
    runs = tmp_path / "runs"
    program = """
import sys
from pathlib import Path
from daguandan_bridge.diagnostic_budget import DiagnosticBudget
root = Path(sys.argv[1])
budget = DiagnosticBudget(root, total_media_bytes=100, run_media_bytes=100)
with budget.transaction() as allowance:
    if allowance.media_bytes >= 70:
        opening = root / 'opening'
        opening.mkdir(parents=True, exist_ok=True)
        (opening / 'proof.png').write_bytes(b'x' * 70)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    processes = [subprocess.Popen([sys.executable, "-c", program, str(runs / str(index))], env=env)
                 for index in range(3)]
    for process in processes:
        assert process.wait(timeout=15) == 0
    assert sum(path.stat().st_size for path in runs.rglob("*.png")) == 70


_WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell 5.1 cleanup")
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "Clean-DaguandanDiagnostics.ps1"


def _cleanup_fixture(tmp_path):
    runs = tmp_path / "DaguandanAssistant" / "diagnostics" / "runs"
    incidents = runs / "fixture-run" / "opening" / "incidents"
    for index in range(6):
        root = incidents / f"OPEN-{index}-a"
        (root / "frames").mkdir(parents=True)
        (root / "roi").mkdir()
        (root / "incident.json").write_text("{}", encoding="utf-8")
        (root / "opening_evidence.json").write_text('{"frames":[]}', encoding="utf-8")
        (root / "frames" / "proof.png").write_bytes(b"evidence" * 10)
        (root / "roi" / "hand.png").write_bytes(b"evidence" * 5)
        (root / "frames" / "keep.json").write_text("{}", encoding="utf-8")
        os.utime(root, (1000 + index, 1000 + index))
    (incidents / ".OPEN-pending.tmp" / "frames").mkdir(parents=True)
    (incidents / ".OPEN-pending.tmp" / "frames" / "proof.png").write_bytes(b"pending")
    incomplete = incidents / "OPEN-7-incomplete"
    (incomplete / "frames").mkdir(parents=True)
    (incomplete / "incident.json").write_text("{}", encoding="utf-8")
    (incomplete / "frames" / "proof.png").write_bytes(b"incomplete")
    for name in ("session.mp4", "model.ckpt", "support.zip", "config.json"):
        (runs / name).write_bytes(b"protected")
    return runs


def _run_cleanup(runs, *, apply=False, busy=False, command_line=None):
    powershell = Path(os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    def literal(value):
        return "'" + str(value).replace("'", "''") + "'"
    processes = "@([pscustomobject]@{Name='DaguandanAssistant.exe';CommandLine='assistant'})" if busy else "@()"
    if command_line is not None:
        processes = "@([pscustomobject]@{Name='python.exe';CommandLine=" + literal(command_line) + "})"
    command = (
        "function Get-CimInstance { " + processes + " }; & " + literal(_SCRIPT)
        + " -RunsRoot " + literal(runs) + (" -Apply" if apply else "")
    )
    return subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True, text=True, timeout=30,
    )


@_WINDOWS
def test_cleanup_defaults_to_preview_then_only_removes_old_completed_incident_media(tmp_path):
    runs = _cleanup_fixture(tmp_path)
    before = {path: path.read_bytes() for path in runs.rglob("*") if path.is_file()}
    preview = _run_cleanup(runs)
    assert preview.returncode == 0, preview.stderr
    assert "PREVIEW ONLY" in preview.stdout
    assert all(path.read_bytes() == content for path, content in before.items())
    result = _run_cleanup(runs, apply=True)
    assert result.returncode == 0, result.stderr
    removed = {path for path in before if not path.exists()}
    assert len(removed) == 6
    assert all(path.suffix == ".png" and path.parent.parent.name in {"OPEN-0-a", "OPEN-1-a", "OPEN-2-a"} for path in removed)
    assert all(path.read_bytes() == content for path, content in before.items() if path not in removed)
    assert "Deleted permanently (not recoverable here): 6 files" in result.stdout


@_WINDOWS
def test_cleanup_rejects_running_app_and_broad_root_without_deletion(tmp_path):
    runs = _cleanup_fixture(tmp_path)
    before = sorted(runs.rglob("*.png"))
    result = _run_cleanup(runs, apply=True, busy=True)
    assert result.returncode != 0
    assert "Close the assistant" in result.stderr
    assert sorted(runs.rglob("*.png")) == before
    result = _run_cleanup(tmp_path, apply=True)
    assert result.returncode != 0
    assert "Refusing root" in result.stderr
    assert sorted(runs.rglob("*.png")) == before


@_WINDOWS
def test_cleanup_preserves_old_owner_of_recent_shared_media(tmp_path):
    runs = _cleanup_fixture(tmp_path)
    incidents = runs / "fixture-run" / "opening" / "incidents"
    newest = incidents / "OPEN-5-a"
    (newest / "opening_evidence.json").write_text(json.dumps({"frames": [{
        "media_status": "shared", "media_reference": {"incident_id": "OPEN-0-a"},
    }]}), encoding="utf-8")
    result = _run_cleanup(runs, apply=True)
    assert result.returncode == 0, result.stderr
    assert (incidents / "OPEN-0-a" / "frames" / "proof.png").is_file()
    assert not (incidents / "OPEN-1-a" / "frames" / "proof.png").exists()


@_WINDOWS
def test_cleanup_failed_deletion_does_not_report_success(tmp_path):
    runs = _cleanup_fixture(tmp_path)
    locked = runs / "fixture-run" / "opening" / "incidents" / "OPEN-0-a" / "frames" / "proof.png"
    with locked.open("rb"):
        result = _run_cleanup(runs, apply=True)
    assert result.returncode != 0
    assert locked.is_file()
    assert "5 files" in result.stdout
    assert "failed/skipped: 1" in result.stdout


@_WINDOWS
def test_cleanup_skips_junction_run_and_budget_does_not_follow_it(tmp_path):
    runs = _cleanup_fixture(tmp_path)
    outside = tmp_path / "outside"
    (outside / "opening" / "incidents" / "OPEN-1-a").mkdir(parents=True)
    sentinel = outside / "opening" / "incidents" / "OPEN-1-a" / "proof.png"
    sentinel.write_bytes(b"outside" * 100)
    junction = runs / "linked-run"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "New-Item -ItemType Junction -Path '" + str(junction).replace("'", "''")
         + "' -Target '" + str(outside).replace("'", "''") + "' | Out-Null"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    try:
        result = _run_cleanup(runs, apply=True)
        assert result.returncode == 0, result.stderr
        assert sentinel.read_bytes() == b"outside" * 100
        budget = DiagnosticBudget(runs / "new", total_media_bytes=500)
        with budget.transaction() as available:
            assert available.media_bytes > 0
    finally:
        # Remove the verified junction itself, never recurse into its target.
        junction.rmdir()


def test_release_cleanup_copy_matches_source():
    release = _SCRIPT.parents[1] / "release_assets" / _SCRIPT.name
    assert release.read_bytes() == _SCRIPT.read_bytes()


@_WINDOWS
@pytest.mark.parametrize("command_line", [
    "python run.py", 'python "C:\\project\\python_project\\daguandan_danzero_bridge\\run.py"',
    '"C:\\Python312\\python.exe" .\\run.py',
])
def test_cleanup_blocks_real_project_entrypoint(tmp_path, command_line):
    runs = _cleanup_fixture(tmp_path)
    before = sorted(runs.rglob("*.png"))
    result = _run_cleanup(runs, apply=True, command_line=command_line)
    assert result.returncode != 0
    assert "Close the assistant" in result.stderr
    assert sorted(runs.rglob("*.png")) == before


@_WINDOWS
def test_cleanup_does_not_confuse_pure_pytest_process_with_gui(tmp_path):
    runs = _cleanup_fixture(tmp_path)
    result = _run_cleanup(runs, apply=True, command_line="python -m pytest tests/test_live_recorder.py -q")
    assert result.returncode == 0, result.stderr


def test_incomplete_slow_quota_scan_never_grants_write_allowance(tmp_path, monkeypatch):
    from daguandan_bridge import diagnostic_budget

    root = tmp_path / "runs" / "run"
    (root / "opening").mkdir(parents=True)
    file = root / "opening" / "proof.png"
    file.write_bytes(b"evidence")
    original = diagnostic_budget.plain_files

    def slow(path, *, deadline=None):
        time.sleep(0.03)
        yield from original(path, deadline=deadline)

    monkeypatch.setattr(diagnostic_budget, "plain_files", slow)
    admitted = False
    with pytest.raises(TimeoutError):
        with DiagnosticBudget(root).transaction(timeout=0.01):
            admitted = True
    assert admitted is False
    assert file.read_bytes() == b"evidence"
