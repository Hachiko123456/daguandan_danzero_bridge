from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_test_layers.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_test_layers_under_test", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load runner module from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_layers_are_registered_as_pytest_markers() -> None:
    """The runner's choices and pytest's marker registry must stay in lockstep."""
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    configured_markers = {
        entry.split(":", 1)[0].strip()
        for entry in pyproject["tool"]["pytest"]["ini_options"]["markers"]
    }

    runner = _load_runner()
    assert set(runner.LAYERS) <= configured_markers


def test_pytest_collection_scope_excludes_analysis_copy_and_keeps_real_tests() -> None:
    """The copied analysis tree must not become a second test module tree."""
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert pytest_options["testpaths"] == ["tests"]
    assert "codex_project_analysis" in pytest_options["norecursedirs"]

    runner = _load_runner()
    assert runner.PYTEST_COLLECTION_IGNORE == "--ignore=codex_project_analysis"


def test_required_layers_collect_at_least_one_test_through_public_cli(
    tmp_path: Path,
) -> None:
    """Every required layer must collect through the public CLI."""
    runner = _load_runner()
    layer_paths = {
        "unit": "tests/test_run_test_layers.py",
        "contract": "tests/test_game_action_invariant_contract.py",
        "visual_fixture": "tests/test_visual_fixture_layers.py",
        "integration": "tests/test_live_end_to_end.py",
        "replay": "tests/test_live_replay.py",
        "data_quality": "tests/test_fabledan_training_data.py",
    }

    for layer, test_path in layer_paths.items():
        summary_path = tmp_path / f"{layer}.json"
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER_PATH),
                "--layer",
                layer,
                "--pytest-arg=--collect-only",
                "--pytest-arg=-q",
                f"--pytest-arg={test_path}",
                "--summary-json",
                str(summary_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["layers"] == [layer]
        assert summary["pytest_exit_code"] == 0
        assert summary["selected_count"] >= 1
        assert summary["empty_layer"] is False
        assert runner._marker_expression([layer]) == summary["marker_expression"]

def test_empty_selection_fails_explicitly_and_nonzero(
    tmp_path: Path,
) -> None:
    """A layer with no collected tests must not be reported as a pseudo-pass."""
    summary_path = tmp_path / "empty.json"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--layer",
            "unit",
            "--pytest-arg=--collect-only",
            "--pytest-arg=-k=__run_test_layers_no_such_test__",
            "--summary-json",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "empty_layer"
    assert summary["empty_layer"] is True
    assert summary["pytest_exit_code"] != 0
    assert "no tests collected" in (result.stdout + result.stderr).lower()


def test_repeated_layers_are_deduplicated_and_composed_in_order() -> None:
    runner = _load_runner()

    assert runner._active_layers(["unit", "contract", "unit"]) == [
        "unit",
        "contract",
    ]
    assert runner._marker_expression(["unit", "contract"]) == "unit or contract"


def test_unknown_layer_is_rejected_by_cli_parser() -> None:
    runner = _load_runner()

    with pytest.raises(SystemExit) as exc_info:
        runner.main(["--layer", "not-a-layer"])

    assert exc_info.value.code == 2


def test_dry_run_writes_summary_without_invoking_pytest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _load_runner()
    summary_path = tmp_path / "summary.json"

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"pytest must not run during --dry-run: {args!r} {kwargs!r}")

    monkeypatch.setattr(runner.subprocess, "run", fail_if_called)

    assert (
        runner.main(
            [
                "--dry-run",
                "--layer",
                "unit",
                "--layer",
                "contract",
                "--pytest-arg=tests/test_example.py",
                "--summary-json",
                str(summary_path),
            ]
        )
        == 0
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert summary["dry_run"] is True
    assert summary["layers"] == ["unit", "contract"]
    assert summary["marker_expression"] == "unit or contract"
    assert summary["pytest_args"] == ["tests/test_example.py"]
    assert summary["pytest_exit_code"] is None
    assert "pytest command:" in capsys.readouterr().out


def test_main_forwards_composed_marker_and_pytest_args_to_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    summary_path = tmp_path / "summary.json"
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Completed:
        returncode = 0

    def fake_run(command: list[str], **kwargs: object) -> Completed:
        calls.append((command, kwargs))
        plugin_path = Path(kwargs["env"][runner.PLUGIN_SUMMARY_ENV])
        plugin_path.write_text(
            json.dumps({"selected_count": 1, "empty_layer": False}),
            encoding="utf-8",
        )
        return Completed()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert (
        runner.main(
            [
                "--layer",
                "replay",
                "--layer",
                "unit",
                "--pytest-arg=-q",
                "--pytest-arg",
                "tests/test_contract.py",
                "--summary-json",
                str(summary_path),
            ]
        )
        == 0
    )

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        runner.sys.executable,
        "-m",
        "pytest",
        "-p",
        "run_test_layers",
        "--ignore=codex_project_analysis",
        "-m",
        "replay or unit",
        "-q",
        "tests/test_contract.py",
    ]
    assert kwargs["cwd"] == ROOT

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["pytest_exit_code"] == 0


def test_default_selection_excludes_explicitly_non_default_layers() -> None:
    runner = _load_runner()

    assert runner._active_layers(None) == [
        layer for layer in runner.LAYERS if layer not in runner.DEFAULT_EXCLUDED
    ]
    assert runner.DEFAULT_EXCLUDED == {
        "windows_integration",
        "benchmark",
        "legacy_compat",
    }


def test_list_reports_available_layers_and_default_state(capsys: pytest.CaptureFixture[str]) -> None:
    runner = _load_runner()

    assert runner.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "Available pytest layers:" in output
    for layer in runner.LAYERS:
        expected = "no" if layer in runner.DEFAULT_EXCLUDED else "yes"
        assert f"{layer:20} default={expected} marker={layer}" in output










