from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from daguandan_bridge import runtime_layout as rl
from daguandan_bridge import startup_diagnostics as sd
from daguandan_bridge import build_manifest as bm


pytestmark = pytest.mark.unit
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bundle(root: Path) -> Path:
    profile = root / "data" / "profiles" / "tencent_daguandan"
    profile.mkdir(parents=True)
    (root / "DaguandanAssistant.exe").write_bytes(b"test-exe")
    config = {"capture_backend": "printwindow", "viewport_mode": "bottom_aspect",
              "viewport_aspect_ratio": 16 / 9, "base_size": [1280, 720]}
    for name, value in (("profile.json", config), ("regions_config.json", {}),
                        ("templates_config.json", {})):
        (profile / name).write_text(json.dumps(value), encoding="utf-8")
    (profile / "templates").mkdir()
    (profile / "templates" / "牌.png").write_bytes(b"template")
    (profile / "models").mkdir()
    (profile / "models" / "best.npz").write_bytes(b"model")
    bm.write_build_manifest(
        root, root,
        source_identity={"commit": "a" * 40, "tree": "b" * 40, "branch": "test",
                         "dirty": False, "status_sha256": None},
        python_identity={"version": "3.12.0", "implementation": "CPython", "architecture": "AMD64"},
        dependency_versions={},
    )
    return root


def _environment(tmp_path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("DAGUANDAN_")}
    env.update({"PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONIOENCODING": "utf-8",
                "LOCALAPPDATA": str(tmp_path / "app data"), "TEMP": str(tmp_path / "temp"),
                "TMP": str(tmp_path / "temp")})
    return env


def _process(code: str, tmp_path: Path, *, env: dict | None = None):
    cwd = tmp_path / "arbitrary 工作目录"
    cwd.mkdir(exist_ok=True)
    result = subprocess.run([sys.executable, "-c", code], cwd=cwd,
                            env=env or _environment(tmp_path), capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("name", ["computer one", "另一台 电脑"])
def test_local_roots_ignore_cwd_meipass_and_appdata(tmp_path, monkeypatch, frozen, name):
    app = tmp_path / name
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "build machine D"), raising=False)
    monkeypatch.setattr(rl, "__file__", str(app / "src" / "daguandan_bridge" / "runtime_layout.py"))
    env = {"LOCALAPPDATA": str(tmp_path / "appdata"), "TEMP": str(tmp_path / "temp")}
    executable = app / "DaguandanAssistant.exe"
    layout = rl.resolve_runtime_layout(environ=env, frozen=frozen, executable_path=executable)
    early = sd.resolve_diagnostics_root(environ=env, frozen=frozen, executable_path=executable)
    assert layout.logs_root == app / "logs"
    assert early.path == layout.diagnostics_root == app / "logs" / "diagnostics"
    assert early.source == layout.diagnostics_root_source
    assert not (tmp_path / "appdata").exists()
    assert not (tmp_path / "temp").exists()
    assert list(cwd.iterdir()) == []


def test_two_extractions_share_existing_state_but_not_logs(tmp_path):
    first = _bundle(tmp_path / "解压 A")
    second = _bundle(tmp_path / "解压 B")
    env = {"LOCALAPPDATA": str(tmp_path / "local")}
    before = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
    a = rl.ensure_runtime_layout(rl.resolve_runtime_layout(frozen=True, bundle_root=first, environ=env))
    customized = a.profiles_root / "tencent_daguandan" / "profile.json"
    customized.write_text('{"capture_backend":"screen"}', encoding="utf-8")
    (a.logs_root / "old.log").write_text("only A", encoding="utf-8")
    b = rl.ensure_runtime_layout(rl.resolve_runtime_layout(frozen=True, bundle_root=second, environ=env))
    assert a.generation_root == b.generation_root
    assert a.preferences_root == b.preferences_root
    assert customized.read_text(encoding="utf-8") == '{"capture_backend":"screen"}'
    assert a.logs_root != b.logs_root
    assert a.diagnostics_root != b.diagnostics_root
    assert not (b.logs_root / "old.log").exists()
    assert not (a.runtime_root / "logs").exists()
    assert not (a.runtime_root / "diagnostics").exists()
    assert all((first / p).read_bytes() == content for p, content in before.items())
    assert bm.verify_build_manifest(first, first / bm.BUILD_MANIFEST_FILENAME, strict=True).ok


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("override", ["DAGUANDAN_DATA_ROOT", "DAGUANDAN_DIAGNOSTICS_DIR", "DAGUANDAN_DIAGNOSTICS_ROOT"])
def test_explicit_overrides_match_early_and_runtime_without_moving_logs(tmp_path, frozen, override):
    app = tmp_path / "app"
    target = tmp_path / "指定 数据"
    env = {"LOCALAPPDATA": str(tmp_path / "appdata"), override: str(target)}
    layout = rl.resolve_runtime_layout(environ=env, frozen=frozen, bundle_root=app)
    early = sd.resolve_diagnostics_root(environ=env, frozen=frozen, bundle_root=app)
    expected = target / "diagnostics" if override == "DAGUANDAN_DATA_ROOT" else target
    assert early.path == layout.diagnostics_root == expected
    assert layout.logs_root == app / "logs"
    if not frozen:
        assert layout.data_dir == app / "data"


def test_diagnostics_override_precedence_and_allowed_local_namespace(tmp_path):
    app = tmp_path / "app"
    env = {"LOCALAPPDATA": str(tmp_path / "local"),
           "DAGUANDAN_DATA_ROOT": str(tmp_path / "state"),
           "DAGUANDAN_DIAGNOSTICS_DIR": str(tmp_path / "alias"),
           "DAGUANDAN_DIAGNOSTICS_ROOT": str(app / "logs" / "custom")}
    early = sd.resolve_diagnostics_root(environ=env, frozen=True, bundle_root=app)
    layout = rl.resolve_runtime_layout(environ=env, frozen=True, bundle_root=app)
    assert layout.diagnostics_root == early.path == app / "logs" / "custom"
    assert layout.runtime_root == tmp_path / "state"


def test_early_diagnostics_do_not_need_appdata_or_a_manifest(tmp_path):
    app = tmp_path / "fresh extraction"
    assert sd.resolve_diagnostics_root(environ={}, frozen=True, bundle_root=app).path == app / "logs" / "diagnostics"
    assert not app.exists()


def test_frozen_startup_writes_local_report_without_importing_native_modules(tmp_path):
    app = _bundle(tmp_path / "中文 application")
    result = _process(f'''
import sys
sys.frozen = True
sys.executable = {str(app / "DaguandanAssistant.exe")!r}
sys._MEIPASS = "D:/build-machine/private"
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
state = initialize_startup_diagnostics(run_id="portable-test")
assert state.enabled, state.error
assert initialize_startup_diagnostics() is state
assert not any(name in sys.modules for name in ("cv2", "numpy", "PySide6", "torch"))
''', tmp_path)
    run = app / "logs" / "diagnostics" / "runs" / "portable-test"
    report = json.loads((run / "startup_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((app / bm.BUILD_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert report["build_id"] == manifest["build_id"]
    assert report["build_manifest"]["source"]["commit"] == "a" * 40
    assert report["build_manifest"]["bundled_resource_fingerprints"]["models"]["sha256"]
    assert report["active_profile"]["status"] == "not_selected"
    assert report["capture"]["capture_backend"] is None
    assert Path(report["paths"]["bundle_root"]) == app
    assert Path(report["diagnostics"]["root"]) == app / "logs" / "diagnostics"
    assert Path(report["python"]["executable"]) == app / "DaguandanAssistant.exe"
    assert report["python"]["version"] and report["platform"]["system"]
    assert not (tmp_path / "app data").exists()
    assert not (tmp_path / "temp").exists()
    assert not list(run.glob("*.tmp"))
    assert bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True).ok


def test_read_only_application_parent_reports_reason_without_fallback(tmp_path):
    app = tmp_path / "read only 应用"
    app.mkdir()
    result = _process(f'''
import sys
from pathlib import Path
sys.frozen = True
sys.executable = {str(app / "DaguandanAssistant.exe")!r}
original = Path.mkdir
def denied(path, *args, **kwargs):
    if path.is_relative_to(Path({str(app)!r})):
        raise PermissionError("read-only application parent")
    return original(path, *args, **kwargs)
Path.mkdir = denied
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
state = initialize_startup_diagnostics()
assert not state.enabled
assert state.root == Path({str(app / "logs" / "diagnostics")!r})
assert "PermissionError" in state.error and "read-only application parent" in state.error
assert str(state.root) in state.error
assert "no fallback" in state.error
''', tmp_path)
    assert "read-only application parent" in result.stderr
    assert str(app / "logs" / "diagnostics") in result.stderr
    assert not (tmp_path / "app data").exists()
    assert not (tmp_path / "temp").exists()
    assert list(app.iterdir()) == []


def test_runtime_log_creation_failure_has_selected_path(tmp_path, monkeypatch):
    app = _bundle(tmp_path / "app")
    layout = rl.resolve_runtime_layout(frozen=True, bundle_root=app,
                                       environ={"LOCALAPPDATA": str(tmp_path / "local")})
    original = Path.mkdir
    def denied(path, *args, **kwargs):
        if path == layout.logs_root:
            raise PermissionError("read-only parent")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(rl.RuntimeLayoutError, match=re.escape(str(layout.logs_root))) as failure:
        rl.ensure_runtime_layout(layout)
    assert "PermissionError" in str(failure.value)
    assert "no fallback" in str(failure.value)
    assert not (layout.runtime_root / "logs").exists()


def test_selected_profile_report_uses_loaded_config_and_bounded_fingerprints(tmp_path, monkeypatch):
    app = _bundle(tmp_path / "app")
    layout = rl.ensure_runtime_layout(rl.resolve_runtime_layout(frozen=True, bundle_root=app,
        environ={"LOCALAPPDATA": str(tmp_path / "local")}))
    active = layout.profiles_root / "tencent_daguandan"
    (active / "profile.json").write_text('{"capture_backend":"screen"}', encoding="utf-8")
    monkeypatch.setattr(sd, "_STATE", sd.StartupDiagnosticsState("report", app / "logs" / "diagnostics",
                        "application_logs_frozen", app / "logs" / "diagnostics" / "runs" / "report", True))
    original = Path.open
    def bounded(path, *args, **kwargs):
        assert "templates" not in path.parts and "models" not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", bounded)
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: pytest.fail("must not scan resource trees"))
    config = SimpleNamespace(capture_backend="gdi_screen", viewport_mode="full", base_size=(800, 600))
    cached = {"status": "identified", "algorithm": "recognition-resources-v2", "sha256": "f" * 64,
              "profile_name": "tencent_daguandan", "files": ["large inventory not copied"]}
    output = sd.write_startup_report(layout=layout, profile_name="tencent_daguandan",
                                    profile_config=config, resource_identity=cached)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert Path(report["active_profile"]["root"]) == active
    assert Path(report["paths"]["data_dir"]) == layout.data_dir
    assert report["capture"]["capture_backend"] == "gdi_screen"
    assert report["capture"]["viewport_mode"] == "full"
    assert report["capture"]["base_size"] == [800, 600]
    assert report["capture_config_source"] == "loaded_config"
    assert report["active_profile"]["config_fingerprint"]["status"] == "complete"
    assert len(report["active_profile"]["config_fingerprint"]["files"]) == 3
    assert report["active_resource_identity"]["sha256"] == "f" * 64
    assert "files" not in report["active_resource_identity"]
    before = report["active_profile"]["config_fingerprint"]["sha256"]
    (active / "regions_config.json").write_text('{"changed":true}', encoding="utf-8")
    assert sd.build_startup_report(layout=layout, profile_name="tencent_daguandan")["active_profile"]["config_fingerprint"]["sha256"] != before


def test_report_marks_oversized_config_partial_and_does_not_claim_full_identity(tmp_path):
    app = _bundle(tmp_path / "source")
    layout = rl.resolve_runtime_layout(frozen=False, bundle_root=app, environ={})
    profile = layout.profiles_root / "tencent_daguandan"
    (profile / "regions_config.json").write_bytes(b" " * (1024 * 1024 + 1))
    report = sd.build_startup_report(layout=layout, profile_name="tencent_daguandan")
    assert report["active_profile"]["config_fingerprint"]["status"] == "partial"
    assert any("exceeds" in error for error in report["errors"])
    assert report["source"]["fingerprint_scope"] == "startup_diagnostics_module_only"
    assert report["source"]["sha256"]
    assert report["capture"]["capture_backend"] == "printwindow"


@pytest.mark.parametrize("relative", [
    "logs/startup.log", "logs/startup.log.1", "logs/diagnostics/runs/one/startup.jsonl.1",
    "logs/diagnostics/runs/one/startup_report.json", "logs/diagnostics/manual_20260925/doctor.json",
    "logs/diagnostics/support/support_20260925_123456.zip",
    "logs/custom/support/support_20260925_123456.zip",
])
def test_integrity_allows_only_unmanifested_log_evidence(tmp_path, relative):
    app = _bundle(tmp_path / "app")
    path = app / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"runtime evidence")
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert result.ok, result.errors
    assert relative in result.unexpected_files


@pytest.mark.parametrize("relative", [
    "rogue.log", "diagnostics/new.json", "logs-not/new.log", "logs/rogue.dll", "logs/rogue.py",
    "logs/best.npz", "logs/unknown.zip", "_internal/logs/a.json", "data/logs/a.json",
    "data/profiles/tencent_daguandan/templates/new.json",
    "data/profiles/tencent_daguandan/models/new.log",
])
def test_integrity_rejects_other_additions(tmp_path, relative):
    app = _bundle(tmp_path / "app")
    path = app / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not approved")
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert not result.ok
    assert f"unexpected bundle file: {relative}" in result.errors


def test_shipped_log_is_still_immutable(tmp_path):
    app = _bundle(tmp_path / "app")
    path = app / "logs" / "shipped.log"
    path.parent.mkdir()
    path.write_text("original", encoding="utf-8")
    manifest = json.loads((app / bm.BUILD_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    bm.write_build_manifest(app, app, source_identity=manifest["source"],
                            python_identity=manifest["python"], dependency_versions={})
    path.write_text("changed", encoding="utf-8")
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert not result.ok
    assert any("logs/shipped.log" in error for error in result.errors)


def test_logs_reparse_is_not_a_mutable_escape_hatch(tmp_path, monkeypatch):
    app = _bundle(tmp_path / "app")
    logs = app / "logs"
    logs.mkdir()
    original = rl._path_is_reparse
    monkeypatch.setattr(rl, "_path_is_reparse", lambda p: Path(p) == logs or original(p))
    with pytest.raises(rl.RuntimeLayoutError, match="reparse"):
        sd.resolve_diagnostics_root(frozen=True, bundle_root=app, environ={})
    original_manifest = bm._is_link_or_reparse
    monkeypatch.setattr(bm, "_is_link_or_reparse", lambda p: Path(p) == logs or original_manifest(p))
    result = bm.verify_build_manifest(app, app / bm.BUILD_MANIFEST_FILENAME, strict=True)
    assert not result.ok
    assert any("reparse" in error for error in result.errors)


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell launcher validation")
@pytest.mark.parametrize("suffix,valid", [("logs/diagnostics", True), ("data/diagnostics", False), ("_internal/diagnostics", False)])
def test_launcher_validation_uses_same_local_namespace(tmp_path, suffix, valid):
    launcher = (PROJECT_ROOT / "release_assets" / "Collect_Diagnostics.bat").read_text(encoding="utf-8")
    line = next(line for line in launcher.splitlines() if line.startswith('powershell.exe -NoProfile -Command "$ErrorActionPreference='))
    command = line.split(' -Command "', 1)[1].rsplit('" >nul', 1)[0]
    app = tmp_path / "应用 with spaces"
    app.mkdir()
    env = dict(os.environ, DIAG_APP_ROOT=str(app), DIAG_ROOT=str(app / suffix))
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], env=env,
                            capture_output=True, text=True, timeout=20)
    assert (result.returncode == 0) == valid, result.stderr
    assert list(app.iterdir()) == []


@pytest.mark.parametrize("override", ["DAGUANDAN_DATA_ROOT", "DAGUANDAN_DIAGNOSTICS_ROOT"])
def test_invalid_explicit_override_is_visible_and_never_redirected(tmp_path, override):
    env = _environment(tmp_path)
    env[override] = "relative-invalid"
    result = _process('''
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
state = initialize_startup_diagnostics()
assert state.enabled is False
assert state.root is None
assert "absolute path" in state.error
assert "relative-invalid" in state.error
assert "no fallback" in state.error
''', tmp_path, env=env)
    assert "relative-invalid" in result.stderr
    assert not (tmp_path / "app data").exists()
    assert not (tmp_path / "temp").exists()


def test_source_startup_logs_belong_to_imported_checkout_not_process_cwd(tmp_path):
    app = tmp_path / "source 中文"
    _process(f'''
from pathlib import Path
import daguandan_bridge.runtime_layout as layout
layout.__file__ = {str(app / "src" / "daguandan_bridge" / "runtime_layout.py")!r}
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
state = initialize_startup_diagnostics(run_id="source-test")
assert state.enabled, state.error
assert state.root == Path({str(app / "logs" / "diagnostics")!r})
''', tmp_path)
    report = json.loads((app / "logs" / "diagnostics" / "runs" / "source-test" / "startup_report.json").read_text(encoding="utf-8"))
    assert report["build_id"] == "source"
    assert Path(report["paths"]["data_dir"]) == app / "data"
    assert Path(report["active_data_root"]) == app / "data"
    assert Path(report["diagnostics_root"]) == app / "logs" / "diagnostics"
    assert report["source"]["sha256"]
    assert not (tmp_path / "app data").exists()


def test_atomic_report_failure_preserves_previous_report_and_exposes_permission_reason(tmp_path, monkeypatch, capsys):
    app = tmp_path / "source"
    layout = rl.resolve_runtime_layout(frozen=False, bundle_root=app, environ={})
    run = app / "logs" / "diagnostics" / "runs" / "report"
    run.mkdir(parents=True)
    destination = run / "startup_report.json"
    destination.write_text('{"previous":true}', encoding="utf-8")
    monkeypatch.setattr(sd, "_STATE", sd.StartupDiagnosticsState("report", run.parent.parent,
                        "application_logs_source", run, True))
    original = os.replace
    def denied(source, target):
        if Path(target) == destination:
            raise PermissionError("report publish access denied")
        return original(source, target)
    monkeypatch.setattr(os, "replace", denied)
    assert sd.write_startup_report(layout=layout) is None
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
    assert not list(run.glob("*.tmp"))
    failure = capsys.readouterr().err
    assert str(destination) in failure and "PermissionError" in failure
    assert "report publish access denied" in failure
    events = (run / "startup.jsonl").read_text(encoding="utf-8")
    assert "startup_report_failed" in events


def test_fault_log_reparse_is_rejected_before_open(tmp_path):
    app = tmp_path / "app"
    run = app / "logs" / "diagnostics" / "runs" / "fixed"
    run.mkdir(parents=True)
    fault = run / "faulthandler.log"
    fault.write_text("do not touch", encoding="utf-8")
    _process(f'''
import sys
from pathlib import Path
sys.frozen = True
sys.executable = {str(app / "DaguandanAssistant.exe")!r}
import daguandan_bridge.runtime_layout as layout
original = layout._path_is_reparse
layout._path_is_reparse = lambda p: p == Path({str(fault)!r}) or original(p)
from daguandan_bridge.startup_diagnostics import initialize_startup_diagnostics
state = initialize_startup_diagnostics(run_id="fixed")
assert not state.enabled
assert "reparse" in state.error
''', tmp_path)
    assert fault.read_text(encoding="utf-8") == "do not touch"
