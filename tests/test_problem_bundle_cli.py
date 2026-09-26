from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from daguandan_bridge import runtime_layout, startup_diagnostics, runtime_identity
from daguandan_bridge import problem_bundle

pytestmark = pytest.mark.unit
PROJECT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def cli(tmp_path, monkeypatch):
    root = tmp_path / "diagnostics"
    current = root / "runs/CLI-new-empty"
    _write(current / "startup_report.json", {"marker": "CLI_EMPTY_RUN"})
    bundle = tmp_path / "application"
    profiles = tmp_path / "data/profiles"
    monkeypatch.setattr(startup_diagnostics, "initialize_startup_diagnostics",
                        lambda: SimpleNamespace(root=root, run_directory=current))
    monkeypatch.setattr(startup_diagnostics, "record_startup_event", lambda *a: None)
    monkeypatch.setattr(runtime_identity, "write_runtime_identity_snapshot", lambda *a: None)
    monkeypatch.setattr(runtime_layout, "resolve_log_diagnostics_root", lambda: (root, "test"))
    monkeypatch.setattr(runtime_layout, "resolve_application_root", lambda: bundle)
    monkeypatch.setattr(runtime_layout, "resolve_runtime_layout",
                        lambda: SimpleNamespace(bundle_root=bundle, profiles_root=profiles))
    monkeypatch.setattr(runtime_layout, "ensure_runtime_layout", lambda: pytest.fail("not offline"))
    monkeypatch.delenv("DAGUANDAN_SESSIONS_ROOT", raising=False)
    spec = importlib.util.spec_from_file_location("problem_cli_test", PROJECT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, root=root, current=current, bundle=bundle, profiles=profiles)


def _archive(result):
    with zipfile.ZipFile(result["archive_path"]) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_no_argument_export_uses_resolver_and_excludes_just_created_cli_run(cli, capsys):
    previous = cli.root / "runs/previous"
    _write(previous / "startup_report.json", {"marker": "PREVIOUS_VALID_RUN"})
    assert cli.module.main(["--export-problem", "--problem-no-images"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert Path(result["archive_path"]).parent == cli.root / "exports"
    data = _archive(result)
    assert b"PREVIOUS_VALID_RUN" in data["run/startup_report.json"]
    assert b"CLI_EMPTY_RUN" not in b"".join(data.values())
    assert result["include_images"] is False


def test_only_new_cli_run_is_not_evidence(cli, capsys):
    assert cli.module.main(["--export-problem"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["run_id"] is None
    assert "run/startup_report.json" not in _archive(result)


def test_newest_case_exact_legacy_run_and_session_trace_after_restart(cli, capsys):
    case = cli.root / "cases/case_latest"
    legacy = cli.bundle / "logs/diagnostics/runs/case_run"
    profile = cli.profiles / "chosen_profile"
    session = profile / "sessions/game"
    _write(case / "case.json", {"case_id": case.name, "run_id": "case_run",
                                "profile_name": profile.name, "session_id": "game",
                                "session_directory": "Z:/attacker/arbitrary-private"})
    _write(cli.root / "cases/case_old/case.json", {"case_id": "case_old", "run_id": "wrong_run"})
    os.utime(cli.root / "cases/case_old/case.json", (1, 1))
    _write(legacy / "startup_report.json", {"run_id": "case_run", "marker": "MATCHED_LEGACY"})
    _write(cli.root / "runs/wrong_run/startup_report.json", {"marker": "WRONG_CASE"})
    _write(profile / "profile.json", {"active": True})
    _write(session / "manifest.json", {"session_id": "game", "sealed": False})
    (session / "recognition_trace.jsonl").write_text('{"marker":"MATCHED_TRACE"}\n', encoding="utf-8")
    assert cli.module.main(["--export-problem", "--problem-no-images"]) == 0
    result = json.loads(capsys.readouterr().out)
    data = _archive(result)
    assert result["case_id"] == case.name
    assert b"MATCHED_LEGACY" in data["run/startup_report.json"]
    assert b"MATCHED_TRACE" in data["session/recognition_trace.jsonl"]
    assert b"WRONG_CASE" not in b"".join(data.values())


def test_explicit_case_does_not_use_other_case_run(cli, capsys):
    case = cli.root / "cases/case_explicit"
    _write(case / "case.json", {"case_id": case.name, "run_id": "not_present"})
    _write(cli.root / "runs/wrong_run/startup_report.json", {"marker": "WRONG_CASE"})
    assert cli.module.main(["--export-problem", "--problem-case", str(case)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["case_id"] == case.name and result["run_id"] is None
    assert "run/startup_report.json" not in _archive(result)


@pytest.mark.parametrize("arguments", [["--problem-no-images"], ["--problem-case", "foo"],
                                         ["--export-problem", "--doctor"],
                                         ["--export-problem", "--export-support", "old.zip"]])
def test_problem_options_are_validated(cli, arguments):
    with pytest.raises(SystemExit) as exc:
        cli.module.main(arguments)
    assert exc.value.code == 2


def test_problem_failure_is_nonzero_and_no_false_archive(cli, monkeypatch, capsys):
    def fail(request):
        raise PermissionError("private filesystem path")
    monkeypatch.setattr(problem_bundle, "export_problem_bundle", fail)
    assert cli.module.main(["--export-problem"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAILED" and result["archive_path"] is None
    assert "private filesystem path" not in str(result)


def test_existing_export_support_branch_unchanged(cli, monkeypatch, capsys, tmp_path):
    from daguandan_bridge import support_export
    calls = []
    def export(request):
        calls.append(request)
        return SimpleNamespace(bundle=SimpleNamespace(manifest={"schema": "legacy-support"}))
    monkeypatch.setattr(support_export, "export_collected_support_bundle", export)
    destination = tmp_path / "legacy.zip"
    assert cli.module.main(["--export-support", str(destination)]) == 0
    assert calls[0].destination == destination
    assert calls[0].diagnostics_run_directory == cli.current
    assert calls[0].include_frames is False and calls[0].include_recognition_trace is False
    assert json.loads(capsys.readouterr().out)["schema"] == "legacy-support"


def test_real_headless_cli_uses_existing_run_not_bootstrap(tmp_path):
    root = tmp_path / "中文 diagnostic"
    run = root / "runs/run_before_cli"
    _write(run / "startup_report.json", {"marker": "REAL_PROCESS_OLD_RUN"})
    environment = dict(os.environ)
    environment["DAGUANDAN_DIAGNOSTICS_ROOT"] = str(root)
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, str(PROJECT / "run.py"), "--export-problem", "--problem-no-images"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["run_id"] == run.name
    assert b"REAL_PROCESS_OLD_RUN" in _archive(result)["run/startup_report.json"]
