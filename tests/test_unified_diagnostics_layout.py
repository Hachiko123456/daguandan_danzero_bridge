from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from daguandan_bridge import build_manifest as bm
from daguandan_bridge import runtime_layout as rl
from daguandan_bridge import startup_diagnostics as sd


pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE = "case_20260926_141516_a1b2c3d4"
CASE_ROOT = f"diagnostics/cases/{CASE}"
ARCHIVE = "DaguandanAssistant_problem_20260926_141516_12345678.zip"
SOURCE = {"commit": "a" * 40, "tree": "b" * 40, "branch": "test",
          "dirty": False, "status_sha256": None}


def _write_manifest(root):
    return bm.write_build_manifest(
        root, root, source_identity=SOURCE,
        python_identity={"version": "3.12.0", "implementation": "CPython", "architecture": "AMD64"},
        dependency_versions={},
    )


def _bundle(root):
    root.mkdir()
    (root / "DaguandanAssistant.exe").write_bytes(b"test executable")
    _write_manifest(root)
    return root


def _add(root, relative, content=b"evidence"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.mark.parametrize("frozen", [False, True])
def test_default_diagnostics_root_is_shared_without_changing_user_data(tmp_path, monkeypatch, frozen):
    app = tmp_path / "application with spaces"
    monkeypatch.setattr(rl, "__file__", str(app / "src" / "daguandan_bridge" / "runtime_layout.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "unrelated"), raising=False)
    env = {"LOCALAPPDATA": str(tmp_path / "local")}
    layout = rl.resolve_runtime_layout(frozen=frozen, executable_path=app / "app.exe", environ=env)
    root, source = rl.resolve_log_diagnostics_root(frozen=frozen, application_root=app, environ=env)
    early = sd.resolve_diagnostics_root(frozen=frozen, bundle_root=app, environ=env)
    assert root == early.path == layout.diagnostics_root == app / "diagnostics"
    assert source == early.source == layout.diagnostics_root_source
    assert layout.logs_root == app / "logs"
    if frozen:
        assert layout.runtime_root == tmp_path / "local" / rl.APP_DIRECTORY_NAME
        assert layout.profiles_root.is_relative_to(layout.runtime_root / "data")
    else:
        assert layout.profiles_root == app / "data" / "profiles"
    assert not app.exists()  # resolution has no write, migration or fallback
    assert not (tmp_path / "local").exists()


@pytest.mark.parametrize("key,suffix", [
    ("DAGUANDAN_DIAGNOSTICS_ROOT", "diagnostics"),
    ("DAGUANDAN_DIAGNOSTICS_DIR", "diagnostics"),
    ("DAGUANDAN_DIAGNOSTICS_ROOT", "logs/diagnostics"),
    ("DAGUANDAN_DIAGNOSTICS_ROOT", "logs/custom"),
])
def test_explicit_local_roots_keep_precedence(tmp_path, key, suffix):
    app = tmp_path / "app"
    env = {"DAGUANDAN_DATA_ROOT": str(tmp_path / "data"), key: str(app / suffix)}
    layout = rl.resolve_runtime_layout(frozen=True, bundle_root=app, environ=env)
    assert layout.diagnostics_root == app / suffix
    assert layout.runtime_root == tmp_path / "data"
    assert sd.resolve_diagnostics_root(frozen=True, bundle_root=app, environ=env).path == app / suffix


@pytest.mark.parametrize("suffix", [".", "data", "_internal", "diagnostics-other", "diagnostics/custom"])
def test_new_local_exception_does_not_allow_other_bundle_roots(tmp_path, suffix):
    app = tmp_path / "app"
    with pytest.raises(rl.RuntimeLayoutError, match="outside the bundle"):
        rl.resolve_log_diagnostics_root(frozen=True, application_root=app,
            environ={"DAGUANDAN_DIAGNOSTICS_ROOT": str(app / suffix)})


@pytest.mark.parametrize("relative", [
    "diagnostics/runs/run-1/startup.log",
    "diagnostics/runs/run-1/startup.log.1",
    "diagnostics/runs/run-1/startup.jsonl.2",
    "diagnostics/runs/20260926T141516.123456Z-123456abcdef/startup_report.json",
    "diagnostics/runs/run-1/exceptions.log",
    "diagnostics/runs/run-1/faulthandler.log",
    "diagnostics/runs/run-1/runtime_identity.json",
    "diagnostics/runs/run-1/readme.txt",
    "diagnostics/runs/_explicit_run/startup.log",
    "diagnostics/runs/.opening-budget.lock",
    "diagnostics/runs/run-1/dependency-probes/dependency-numpy-0123456789abcdef0123456789abcdef.json",
    "diagnostics/runs/run-1/opening/latest.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/incident.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/opening_evidence.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/repro.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/recognition_trace.jsonl",

    f"{CASE_ROOT}/case.json",
    f"{CASE_ROOT}/config/profile.json",
    f"{CASE_ROOT}/config/regions_config.json",
    f"{CASE_ROOT}/config/templates_config.json",
    f"{CASE_ROOT}/frames/000001.png",
    f"{CASE_ROOT}/frames/1000000.png",
    f"{CASE_ROOT}/frames/000001.json",
    f"{CASE_ROOT}/frames/incident_12345678.json",
    f"{CASE_ROOT}/frames/incident_capture-failed_1.json",
    "diagnostics/cases/case_20260926T141516.123456Z_1234abcd/case.json",
    f"diagnostics/exports/{ARCHIVE}",
    "diagnostics/exports/DaguandanAssistant_problem_20260926_141516_0123456789abcdef0123456789abcdef.zip",
    f"logs/diagnostics/cases/{CASE}/frames/000001.png",
    f"logs/custom/cases/{CASE}/frames/incident_12345678.json",
    f"logs/custom/exports/{ARCHIVE}",
    "diagnostics/exports/DaguandanAssistant_problem_20260926T141516Z_a1b2c3d4e5f6.zip",
])
def test_manifest_accepts_only_unmanifested_evidence_at_approved_shapes(tmp_path, relative):
    app = _bundle(tmp_path / "app")
    _add(app, relative)
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert result.ok, result.errors
    assert result.unexpected_files == (relative,)
    assert result.mutable_differences == ()


@pytest.mark.parametrize("relative", [
    "diagnostics/case.json",
    "diagnostics/startup.log",
    "diagnostics/runs/startup.log",
    "diagnostics/runs/run-1/nested/startup.log",
    "diagnostics/runs/run-1/python.exe",
    "diagnostics/runs/run-1/native.dll",
    "diagnostics/runs/run-1/plugin.py",
    "diagnostics/runs/run-1/000001.png",
    "diagnostics/runs/run-1/model.bin",
    "diagnostics/runs/run-1/startup.log.exe",
    "diagnostics/runs/run-1/startup.log.tmp",
    "diagnostics/runs/run-1/startup.log.0",
    "diagnostics/runs/run-1/startup.log.1000000",
    "diagnostics/.opening-budget.lock",
    "diagnostics/runs/.anything.lock",
    "diagnostics/runs/run-1/.opening-budget.lock",
    "diagnostics/runs/.opening-budget.lock.exe",
    "diagnostics/runs/run-1/dependency-probes/arbitrary.json",
    "diagnostics/runs/run-1/dependency-probes/dependency-unknown-0123456789abcdef0123456789abcdef.json",
    "diagnostics/runs/run-1/dependency-probes/dependency-numpy-12345678.json",
    "diagnostics/runs/run-1/dependency-probes/dependency-numpy-0123456789abcdef0123456789abcdef.exe",
    "diagnostics/runs/run-1/dependency-probes/nested/dependency-numpy-0123456789abcdef0123456789abcdef.json",
    "diagnostics/runs/run-1/dependency-probes/plugin.py",
    "diagnostics/runs/run-1/opening/other.json",
    "diagnostics/runs/run-1/opening/latest.png",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/unknown.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/incident.json.exe",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/runtime.dll",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/frames/000001.png",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/roi/hand.png",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/nested/incident.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-nonnumeric-a1b2c3d4/incident.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-123456789/incident.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-nothex00/incident.json",
    "diagnostics/runs/run-1/opening/incidents/not-open-12345678-a1b2c3d4/incident.json",
    "diagnostics/runs/run-1/opening/incidents/.OPEN-12345678-a1b2c3d4.temporary.tmp/incident.json",

    "diagnostics/cases/invalid/case.json",
    "diagnostics/cases/case_20260926_141516_nothex00/case.json",
    "diagnostics/cases/case_20260926_141516_123456789/case.json",
    f"{CASE_ROOT}/profile.json",
    f"{CASE_ROOT}/metadata.json",
    f"{CASE_ROOT}/config_snapshot.json",
    f"{CASE_ROOT}/frames/photo.png",
    f"{CASE_ROOT}/frames/1.png",
    f"{CASE_ROOT}/frames/1000000000000000000.png",
    f"{CASE_ROOT}/frames/000001.jpg",
    f"{CASE_ROOT}/frames/000001.json.exe",
    f"{CASE_ROOT}/frames/incident_.json",
    f"{CASE_ROOT}/frames/incident_test.png",
    f"{CASE_ROOT}/frames/nested/000001.png",
    f"{CASE_ROOT}/frames/plugin.py",
    f"{CASE_ROOT}/frames/binary.dll",
    f"{CASE_ROOT}/frames/model.bin",
    f"{CASE_ROOT}/config/settings.json",
    f"{CASE_ROOT}/config/plugin.py",
    f"{CASE_ROOT}/config/nested/profile.json",
    "diagnostics/exports/arbitrary.zip",
    "diagnostics/exports/DaguandanAssistant_problem_wrong_12345678.zip",
    "diagnostics/exports/DaguandanAssistant_problem_20260926_141516_NOTHEX00.zip",
    "diagnostics/exports/DaguandanAssistant_problem_20260926_141516_12345678.exe",
    f"diagnostics/exports/nested/{ARCHIVE}",
    f"diagnostics/exports/{ARCHIVE}.tmp",
    f"diagnostics/other/{ARCHIVE}",
    "diagnostics/other/anything.bin",
    "_internal/diagnostics/runs/one/startup.log",
    "data/diagnostics/runs/one/startup.log",
    "logs/custom/photo.png",
    f"logs/custom/cases/{CASE}/frames/000001.exe",
    f"logs/custom/cases/{CASE}/frames/plugin.py",
    f"logs/custom/cases/{CASE}/other/000001.png",
    "logs/custom/exports/arbitrary.zip",
])
def test_manifest_rejects_code_arbitrary_binaries_and_unbounded_layouts(tmp_path, relative):
    app = _bundle(tmp_path / "app")
    _add(app, relative)
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert not result.ok
    assert f"unexpected bundle file: {relative}" in result.errors


@pytest.mark.parametrize("relative", [
    "diagnostics/runs/run-1/startup.log",
    "diagnostics/runs/.opening-budget.lock",
    "diagnostics/runs/run-1/dependency-probes/dependency-numpy-0123456789abcdef0123456789abcdef.json",
    "diagnostics/runs/run-1/opening/latest.json",
    "diagnostics/runs/run-1/opening/incidents/OPEN-12345678-a1b2c3d4/incident.json",
    f"{CASE_ROOT}/case.json",
    f"{CASE_ROOT}/config/profile.json",
    f"{CASE_ROOT}/frames/000001.png",
    f"{CASE_ROOT}/frames/incident_12345678.json",
    f"diagnostics/exports/{ARCHIVE}",
])
@pytest.mark.parametrize("change", ["hash", "missing"])
def test_manifested_diagnostics_remain_immutable(tmp_path, relative, change):
    app = _bundle(tmp_path / "app")
    path = _add(app, relative, b"original")
    _write_manifest(app)
    if change == "hash":
        path.write_bytes(b"modified")  # same size; hashing cannot be bypassed
    else:
        path.unlink()
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert not result.ok
    assert any(relative in error for error in result.errors)
    assert not result.mutable_differences


@pytest.mark.parametrize("relative", ["diagnostics", "diagnostics/cases", CASE_ROOT,
    f"{CASE_ROOT}/frames", f"{CASE_ROOT}/frames/000001.png", "diagnostics/exports"])
def test_diagnostics_reparse_entries_are_rejected_before_traversal(tmp_path, monkeypatch, relative):
    app = _bundle(tmp_path / "app")
    _add(app, f"{CASE_ROOT}/frames/000001.png")
    _add(app, f"diagnostics/exports/{ARCHIVE}")
    target = app / relative
    original = bm._directory_entry_is_link_or_reparse
    monkeypatch.setattr(bm, "_directory_entry_is_link_or_reparse",
                        lambda entry, path: path == target or original(entry, path))
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert not result.ok
    assert any("reparse" in error and relative in error for error in result.errors)


@pytest.mark.parametrize("ancestor", ["app", "diagnostics"])
def test_resolver_rejects_reparse_chain_without_redirecting(tmp_path, monkeypatch, ancestor):
    app = tmp_path / "app"
    target = app if ancestor == "app" else app / "diagnostics"
    original = rl._path_is_reparse
    monkeypatch.setattr(rl, "_path_is_reparse", lambda path: path == target or original(path))
    with pytest.raises(rl.RuntimeLayoutError, match="reparse"):
        rl.resolve_log_diagnostics_root(frozen=True, application_root=app, environ={})
    assert not app.exists()


def _clean_environment():
    return {key: value for key, value in os.environ.items() if not key.startswith("DAGUANDAN_")}


def test_new_startup_leaves_legacy_logs_and_user_files_untouched(tmp_path):
    app = tmp_path / "app"
    legacy = _add(app, "logs/diagnostics/runs/old/startup.log", b"old evidence")
    user = _add(app, "data/profiles/user/sessions/session.txt", b"user evidence")
    env = _clean_environment()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    code = f"""
import sys
sys.frozen = True
sys.executable = {str(app / 'DaguandanAssistant.exe')!r}
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
state = initialize_startup_diagnostics(run_id='new-run')
assert state.enabled, state.error
"""
    completed = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                               capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert (app / "diagnostics/runs/new-run/startup_report.json").is_file()
    assert legacy.read_bytes() == b"old evidence"
    assert user.read_bytes() == b"user evidence"
    assert list(legacy.parent.parent.iterdir()) == [legacy.parent]


def test_explicit_unwritable_root_reports_path_and_has_no_fallback(tmp_path):
    app = tmp_path / "app"
    selected = tmp_path / "explicit diagnostics"
    env = _clean_environment()
    env.update(PYTHONPATH=str(PROJECT_ROOT / "src"), DAGUANDAN_DIAGNOSTICS_ROOT=str(selected))
    code = f"""
import sys
from pathlib import Path
sys.frozen = True
sys.executable = {str(app / 'DaguandanAssistant.exe')!r}
original = Path.mkdir
def denied(path, *args, **kwargs):
    if path.is_relative_to(Path({str(selected)!r})):
        raise PermissionError('explicit-root denied')
    return original(path, *args, **kwargs)
Path.mkdir = denied
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
state = initialize_startup_diagnostics(run_id='denied')
assert not state.enabled
assert state.root == Path({str(selected)!r})
assert str(state.root) in state.error
assert 'explicit-root denied' in state.error and 'no fallback' in state.error
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert b"explicit-root denied" in result.stderr
    assert not app.exists()
    assert not selected.exists()


@pytest.fixture(scope="module")
def launcher_executable(tmp_path_factory):
    if os.name != "nt":
        pytest.skip("Windows batch launcher execution")
    compiler = Path(os.environ.get("WINDIR", "C:/Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if not compiler.is_file():
        pytest.skip(".NET Framework compiler needed for a local CLI test double")
    root = tmp_path_factory.mktemp("diagnostics-launcher-stub")
    code = root / "Recorder.cs"
    code.write_text(r'''using System;
using System.IO;
class Recorder {
    static int Main(string[] args) {
        string output = Environment.GetEnvironmentVariable("TEST_LAUNCHER_CALL");
        File.WriteAllLines(output, args);
        File.WriteAllLines(output + ".env", new string[] {
            Environment.GetEnvironmentVariable("DAGUANDAN_DIAGNOSTICS_ROOT") ?? "",
            Environment.GetEnvironmentVariable("DAGUANDAN_DIAGNOSTICS_DIR") ?? "",
            Environment.GetEnvironmentVariable("DAGUANDAN_DATA_ROOT") ?? ""
        });
        return Int32.Parse(Environment.GetEnvironmentVariable("TEST_LAUNCHER_EXIT") ?? "0");
    }
}
''', encoding="utf-8")
    exe = root / "Recorder.exe"
    result = subprocess.run([str(compiler), "/nologo", "/target:exe", "/out:" + str(exe), str(code)],
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return exe


def _launcher(tmp_path, *, executable=None):
    app = tmp_path / "Chinese 路径 & space"
    app.mkdir()
    launcher = app / "Collect_Diagnostics.bat"
    shutil.copyfile(PROJECT_ROOT / "release_assets/Collect_Diagnostics.bat", launcher)
    if executable is not None:
        shutil.copyfile(executable, app / "DaguandanAssistant.exe")
    return launcher


def _run_launcher(launcher, args, *, env, answer=""):
    # cmd.exe does not accept list2cmdline's C-runtime backslash quote escaping.
    # These are fixed test arguments, not arbitrary shell input.
    cmd = os.environ.get("COMSPEC", "cmd.exe")
    return subprocess.run(
        f'"{cmd}" /d /s /c ""{launcher}" {args}"',
        env=env, cwd=launcher.parent.parent, input=answer.encode("ascii"),
        capture_output=True, timeout=20,
    )


@pytest.mark.parametrize("arguments,answer,expected", [
    ("--problem-no-images", "", ["--export-problem", "--problem-no-images"]),
    ("", "N\r\n", ["--export-problem", "--problem-no-images"]),
    ("", "Y\r\n", ["--export-problem"]),
])
def test_windows_launcher_calls_headless_export_and_preserves_overrides(
        tmp_path, launcher_executable, arguments, answer, expected):
    launcher = _launcher(tmp_path, executable=launcher_executable)
    call = tmp_path / "called.txt"
    env = _clean_environment()
    overrides = [str(tmp_path / name) for name in ("selected root", "alias", "data")]
    env.update(TEST_LAUNCHER_CALL=str(call), DAGUANDAN_DIAGNOSTICS_ROOT=overrides[0],
               DAGUANDAN_DIAGNOSTICS_DIR=overrides[1], DAGUANDAN_DATA_ROOT=overrides[2])
    result = _run_launcher(launcher, arguments, env=env, answer=answer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert call.read_text(encoding="utf-8-sig").splitlines() == expected
    assert Path(str(call) + ".env").read_text(encoding="utf-8-sig").splitlines() == overrides
    assert b"Nothing is uploaded" in result.stdout
    assert not (launcher.parent / "diagnostics").exists()
    assert not (launcher.parent / "logs").exists()
    assert not list(tmp_path.rglob("*.zip"))  # the script cannot fabricate an archive


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher execution")
@pytest.mark.parametrize("argument,code", [("--help", 0), ("/?", 0), ("--unknown", 2)])
def test_windows_launcher_help_does_not_require_exe_or_create_evidence(tmp_path, argument, code):
    launcher = _launcher(tmp_path)
    result = _run_launcher(launcher, argument, env=_clean_environment())
    assert result.returncode == code, result.stdout + result.stderr
    assert b"--help" in result.stdout
    assert list(launcher.parent.iterdir()) == [launcher]


@pytest.mark.skipif(os.name != "nt", reason="Windows batch launcher execution")
@pytest.mark.parametrize("broken", [False, True])
def test_windows_launcher_missing_or_unexecutable_exe_suggests_copy_not_fake_zip(tmp_path, broken):
    launcher = _launcher(tmp_path)
    if broken:
        (launcher.parent / "DaguandanAssistant.exe").write_bytes(b"not an executable")
    result = _run_launcher(launcher, "--problem-no-images", env=_clean_environment())
    assert result.returncode != 0, result.stdout + result.stderr
    assert b"manually copy the existing diagnostics folder" in result.stdout
    assert b"No problem ZIP was created by this script" in result.stdout
    assert not list(tmp_path.rglob("*.zip"))
    assert not (launcher.parent / "diagnostics").exists()


def test_windows_launcher_propagates_export_failure(tmp_path, launcher_executable):
    launcher = _launcher(tmp_path, executable=launcher_executable)
    env = _clean_environment()
    env.update(TEST_LAUNCHER_CALL=str(tmp_path / "called.txt"), TEST_LAUNCHER_EXIT="7")
    result = _run_launcher(launcher, "--problem-no-images", env=env)
    assert result.returncode == 7, result.stdout + result.stderr
    assert b"Exit code: 7" in result.stdout
    assert b"manually copy" in result.stdout
    assert b"Export completed" not in result.stdout
    assert not list(tmp_path.rglob("*.zip"))


@pytest.mark.parametrize("has_case", [False, True])
def test_real_headless_export_uses_existing_evidence_and_preserves_bundle_integrity(tmp_path, has_case):
    app = _bundle(tmp_path / "app")
    _add(app, "diagnostics/runs/problem-run/startup_report.json", b'{"run_id":"problem-run"}')
    _add(app, "diagnostics/runs/problem-run/startup.jsonl", b'{"event":"original_problem"}\n')
    if has_case:
        _add(app, f"{CASE_ROOT}/case.json", json.dumps({"case_id": CASE, "run_id": "problem-run"}).encode())
    env = _clean_environment()
    env.update(PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONIOENCODING="utf-8",
               LOCALAPPDATA=str(tmp_path / "local"))
    code = f"""
import importlib.abc
import runpy
import sys
class BlockGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'PySide6', 'cv2', 'numpy', 'torch'}}:
            raise RuntimeError('GUI/native libraries deliberately unavailable')
sys.meta_path.insert(0, BlockGui())
sys.frozen = True
sys.executable = {str(app / 'DaguandanAssistant.exe')!r}
entry = runpy.run_path({str(PROJECT_ROOT / 'run.py')!r}, run_name='headless_test')
assert entry['main'](['--export-problem', '--problem-no-images']) == 0
assert not any(name in sys.modules for name in ('PySide6', 'cv2', 'numpy', 'torch'))
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                            capture_output=True, timeout=30)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, stdout + stderr
    report = json.loads(stdout)
    archive = Path(report["archive_path"])
    assert archive.parent == app / "diagnostics/exports"
    assert report["run_id"] == "problem-run"
    assert report["case_id"] == (CASE if has_case else None)
    assert report["status"] == "PARTIAL"
    assert report["missing"]
    with zipfile.ZipFile(archive) as zipped:
        assert "problem_manifest.json" in zipped.namelist()
        assert "run/startup.jsonl" in zipped.namelist()
        assert not any(name.endswith(".png") for name in zipped.namelist())
        assert b"original_problem" in zipped.read("run/startup.jsonl")
    integrity = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert integrity.ok, integrity.errors
    assert not (app / "logs/diagnostics").exists()
    assert not (tmp_path / "local").exists()


@pytest.mark.skipif(os.name != "nt", reason="Real Windows directory junction")
@pytest.mark.parametrize("relative", ["diagnostics", f"{CASE_ROOT}/frames"])
def test_real_windows_junction_cannot_bypass_evidence_namespace(tmp_path, relative):
    app = _bundle(tmp_path / "app")
    outside = tmp_path / "outside evidence"
    outside.mkdir()
    sentinel = outside / "000001.png"
    sentinel.write_bytes(b"external evidence must remain untouched")
    junction = app / relative
    junction.parent.mkdir(parents=True, exist_ok=True)
    created = subprocess.run(
        [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True, timeout=20,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    try:
        assert os.path.isjunction(junction)
        result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
        assert not result.ok
        assert any("reparse" in error for error in result.errors)
        if relative == "diagnostics":
            with pytest.raises(rl.RuntimeLayoutError, match="reparse"):
                rl.resolve_log_diagnostics_root(frozen=True, application_root=app, environ={})
        assert sentinel.read_bytes() == b"external evidence must remain untouched"
    finally:
        # Remove only the junction itself, never recursively touch its target.
        assert junction.absolute().is_relative_to(tmp_path.absolute())
        junction.rmdir()


def test_manifest_probe_allowlist_covers_exact_current_doctor_dependencies(tmp_path):
    from daguandan_bridge.doctor import DEPENDENCIES

    app = _bundle(tmp_path / "app")
    for dependency in DEPENDENCIES:
        _add(app, "diagnostics/runs/doctor/dependency-probes/"
             + dependency.check_id.lower() + "-" + "a" * 32 + ".json", b"{}")
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert result.ok, result.errors
    assert len(result.unexpected_files) == len(DEPENDENCIES)


def test_two_real_doctor_runs_keep_generated_opening_and_probe_evidence_integrity_valid(tmp_path):
    app = _bundle(tmp_path / "app")
    configs = {
        "profile.json": {"name": "tencent_daguandan"},
        "regions_config.json": {"schema_version": 2, "regions": []},
        "templates_config.json": {"schema_version": 2, "templates": [{"file": "templates/3.png"}]},
    }
    for name, content in configs.items():
        _add(app, "data/profiles/tencent_daguandan/" + name, json.dumps(content).encode())
    _add(app, "data/profiles/tencent_daguandan/templates/3.png", b"fixture-template")
    _add(app, "data/profiles/tencent_daguandan/models/best.npz", b"fixture-model")
    _add(app, "data/profiles/tencent_daguandan/models/danzero/q_network.ckpt", b"fixture-model")
    _write_manifest(app)
    env = _clean_environment()
    env.update(PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONIOENCODING="utf-8",
               DAGUANDAN_DIAGNOSTICS_ROOT=str(app / "diagnostics"),
               LOCALAPPDATA=str(tmp_path / "local"))
    for number in (1, 2):
        # Run the REAL collector, run_doctor publisher and import-probe child.
        # Restrict expensive dependency checks to numpy; a separate allowlist
        # test covers every current doctor dependency ID. Only cleanup is fault-
        # injected: failed deletion is an explicitly tolerated production path,
        # so the next strict check must accept a completed probe left behind.
        code = f"""
from functools import partial
from pathlib import Path
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
startup = initialize_startup_diagnostics(run_id='doctor-{number}')
assert startup.enabled, startup.error
from daguandan_bridge.opening_evidence import OpeningEvidenceMonitor
monitor = OpeningEvidenceMonitor(
    diagnostics_root=startup.run_directory,
    max_run_image_bytes=0, max_total_image_bytes=0, delivery_settle_seconds=0,
)
try:
    assert monitor.emit_incident('OPENING-MANIFEST-REGRESSION', field='test', reason='text only')
    assert monitor.flush(5), monitor.metrics()
finally:
    monitor.close()
from daguandan_bridge import doctor
collect = doctor.collect_doctor_report
doctor.collect_doctor_report = partial(
    collect, root=Path({str(app)!r}), frozen=True,
    dependencies=(doctor.DEPENDENCIES[0],),
    environ={{'LOCALAPPDATA': {str(tmp_path / 'local')!r}}},
)
doctor._remove_probe_output = lambda path: None
raise SystemExit(doctor.run_doctor())
"""
        result = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                                capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        checks = {item["id"]: item for item in report["checks"]}
        assert checks["BUILD-INTEGRITY"]["status"] == "PASS", checks["BUILD-INTEGRITY"]
        assert checks["DEPENDENCY-NUMPY"]["status"] == "PASS", checks["DEPENDENCY-NUMPY"]
        run = app / "diagnostics/runs" / f"doctor-{number}"
        assert (run / "doctor.json").is_file()
        probes = list((run / "dependency-probes").glob("*.json"))
        assert len(probes) == 1
        assert json.loads(probes[0].read_text(encoding="utf-8"))["schema"] == "guandan.doctor-import-probe/1"
        assert (run / "opening/latest.json").is_file()
        incidents = list((run / "opening/incidents").iterdir())
        assert len(incidents) == 1
        assert {path.name for path in incidents[0].iterdir()} == {
            "incident.json", "opening_evidence.json", "repro.json", "recognition_trace.jsonl",
        }
        assert not list(run.rglob("*.png"))
        assert (app / "diagnostics/runs/.opening-budget.lock").is_file()
        integrity = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
        assert integrity.ok, integrity.errors
    assert not (app / "logs/diagnostics").exists()
