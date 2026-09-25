from __future__ import annotations

import json
from pathlib import Path

import pytest

from daguandan_bridge.application.window_debug_storage import (
    ScreenshotOptInRequired,
    WindowDebugStorage,
    WindowDebugStorageError,
)


def test_storage_derives_window_debug_root_and_keeps_json_defaults(tmp_path: Path):
    storage = WindowDebugStorage(tmp_path / "diagnostics")
    run = storage.create_run("run-001")
    report = storage.write_report(
        run,
        {
            "ok": True,
            "secret": "hide",
            "path": r"C:\\private\\app.exe",
            "relative_path": "profiles/private.json",
        },
    )
    events = storage.append_event(run, {"event": "started"})

    persisted = json.loads(report.read_text(encoding="utf-8"))
    assert report == tmp_path / "diagnostics" / "window_debug" / "run-001" / "report.json"
    assert events == report
    assert sorted(p.name for p in run.path.iterdir()) == ["report.json"]
    assert persisted["secret"] == "[REDACTED]"
    assert persisted["path"] == "[REDACTED_PATH]"
    assert persisted["relative_path"] == "[REDACTED_PATH]"


def test_screenshots_are_explicit_opt_in(tmp_path: Path):
    storage = WindowDebugStorage(tmp_path / "diagnostics")
    run = storage.create_run("no-image")
    with pytest.raises(ScreenshotOptInRequired):
        storage.write_screenshot(run, "frame.png", b"png")
    assert not list(run.path.glob("*.png"))

    opted_in = storage.create_run("with-image", allow_screenshots=True)
    image = storage.write_screenshot(opted_in, "frame.png", b"png")
    assert image.name == "report.json"
    persisted = json.loads(image.read_text(encoding="utf-8"))
    assert persisted["media"]["media_saved"] is True
    assert persisted["media"]["items"]["frame.png"]["encoding"] == "base64"


def test_retention_by_count_is_deterministic_and_does_not_touch_siblings(tmp_path: Path):
    storage = WindowDebugStorage(tmp_path / "diagnostics", max_runs=2)
    for run_id in ("run-a", "run-b", "run-c"):
        run = storage.create_run(run_id)
        storage.write_report(run, {"run": run_id})
    sibling = storage.root.parent / "sessions" / "keep.json"
    sibling.parent.mkdir()
    sibling.write_text("keep", encoding="utf-8")

    result = storage.cleanup()
    assert result.removed_run_ids == ("run-a",)
    assert [p.name for p in storage.root.iterdir()] == ["run-b", "run-c"]
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_retention_by_bytes_removes_oldest_until_within_budget(tmp_path: Path):
    storage = WindowDebugStorage(tmp_path / "diagnostics", max_bytes=25)
    for run_id in ("run-a", "run-b", "run-c"):
        run = storage.create_run(run_id)
        storage.append_event(run, {"run": run_id})
    result = storage.cleanup()
    assert result.removed_run_ids == ("run-a", "run-b")
    assert [p.name for p in storage.root.iterdir()] == ["run-c"]
    # The newest run is protected even when it exceeds the byte budget.
    assert result.remaining_bytes > 25


def test_extremely_small_byte_budget_still_keeps_latest_safe_run(tmp_path: Path):
    storage = WindowDebugStorage(tmp_path / "diagnostics", max_bytes=1)
    for run_id in ("run-a", "run-b", "run-c"):
        run = storage.create_run(run_id)
        storage.append_event(run, {"run": run_id, "detail": "larger than budget"})

    result = storage.cleanup()

    assert result.removed_run_ids == ("run-a", "run-b")
    assert [p.name for p in storage.root.iterdir()] == ["run-c"]
    assert result.remaining_bytes > 1


def test_single_latest_run_is_kept_when_it_exceeds_byte_budget(tmp_path: Path):
    storage = WindowDebugStorage(tmp_path / "diagnostics", max_bytes=0)
    run = storage.create_run("run-only")
    storage.write_report(run, {"detail": "larger than zero bytes"})

    result = storage.cleanup()

    assert result.removed_run_ids == ()
    assert (run.path / "report.json").is_file()
    assert result.remaining_runs == 1
    assert result.remaining_bytes > 0


def test_path_traversal_and_executable_directory_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    storage = WindowDebugStorage(tmp_path / "diagnostics")
    with pytest.raises(WindowDebugStorageError):
        storage.create_run("..\\outside")
    run = storage.create_run("safe")
    with pytest.raises(WindowDebugStorageError):
        storage.write_json(run, "..\\outside.json", {})

    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    monkeypatch.setattr("sys.executable", str(exe_dir / "assistant.exe"))
    with pytest.raises(WindowDebugStorageError):
        WindowDebugStorage(exe_dir)
