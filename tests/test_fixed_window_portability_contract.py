from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_fixed_window_portability.py"


def _module():
    spec = importlib.util.spec_from_file_location("validate_fixed_window_portability", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path, *, allow_resize=False, backend="printwindow", fallback=False, absolute_path=None):
    profile_root = root / "data" / "profiles" / "demo"
    profile_root.mkdir(parents=True)
    profile = {
        "name": "demo",
        "allow_resize": allow_resize,
        "capture_backend": backend,
        "allow_screen_fallback": fallback,
        "base_size": [1280, 720],
        "target_client_size": [1280, 764],
        "viewport_mode": "bottom_aspect",
    }
    if absolute_path is not None:
        profile["model_path"] = absolute_path
    (profile_root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (root / "runtime_layout.json").write_text(
        json.dumps({"schema": "guandan.user-data-generation/1", "generation_id": "demo"}),
        encoding="utf-8",
    )
    return profile_root / "profile.json"


def test_strict_fixture_passes_without_touching_files(tmp_path: Path):
    profile = _write_fixture(tmp_path)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    report = _module().build_report(project_root=tmp_path, profile_path=profile)

    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert report["schema"] == "guandan.fixed-window-portability-dry-run/1"
    assert report["summary"]["status"] == "PASS"
    assert report["side_effects"]["controls_wechat"] is False
    assert before == after


def test_missing_allow_resize_and_auto_backend_are_explicit_failures(tmp_path: Path):
    profile = _write_fixture(tmp_path, backend="auto")
    data = json.loads(profile.read_text(encoding="utf-8"))
    data.pop("allow_resize")
    profile.write_text(json.dumps(data), encoding="utf-8")

    report = _module().build_report(project_root=tmp_path, profile_path=profile)
    codes = {item["code"] for item in report["findings"]}
    assert report["summary"]["status"] == "FAIL"
    assert "profile.allow_resize_missing_or_invalid" in codes
    assert "profile.capture_backend_not_printwindow" in codes


def test_absolute_and_traversing_relative_paths_are_reported(tmp_path: Path):
    profile = _write_fixture(tmp_path, absolute_path=r"C:\old-machine\model.npz")
    data = json.loads(profile.read_text(encoding="utf-8"))
    data["template_path"] = "../templates/card.png"
    profile.write_text(json.dumps(data), encoding="utf-8")

    report = _module().build_report(project_root=tmp_path, profile_path=profile)
    risks = report["path_risks"]
    assert any(item["kind"] == "absolute_path" and item["severity"] == "blocker" for item in risks)
    assert any(item["kind"] == "relative_path" and item["severity"] == "blocker" for item in risks)
    assert report["summary"]["status"] == "FAIL"


def test_report_can_be_emitted_as_auditable_json(tmp_path: Path, capsys):
    _write_fixture(tmp_path)
    exit_code = _module().main(["--root", str(tmp_path)])
    output = capsys.readouterr().out
    report = json.loads(output)
    assert exit_code == 0
    assert report["summary"]["status"] == "PASS"
    assert report["report_type"] == "read_only_dry_run"
