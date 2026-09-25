from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_windows_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("run_windows_smoke_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_accepts_smoke_cli_options_and_rejects_invalid_duration() -> None:
    smoke = _load_smoke()
    args = smoke.build_parser().parse_args(
        [
            "--profile-root",
            "profiles",
            "--output",
            "report.json",
            "--duration",
            "1.5",
            "--session",
            "session-1",
        ]
    )

    assert args.profile_root == Path("profiles")
    assert args.output == Path("report.json")
    assert args.duration == pytest.approx(1.5)
    assert args.session == Path("session-1")

    with pytest.raises(SystemExit) as exc_info:
        smoke.build_parser().parse_args(["--duration", "-1"])
    assert exc_info.value.code == 2


def test_non_windows_environment_is_explicitly_not_run(tmp_path: Path) -> None:
    smoke = _load_smoke()
    report = smoke.run_smoke(
        profile_root=tmp_path,
        desktop_checker=lambda: (
            False,
            {
                "reason": "non_windows_platform",
                "message": "Windows desktop smoke test is only runnable on Windows",
            },
        ),
    )

    assert report["status"] == smoke.STATUS_NOT_RUN
    assert report["exit_code"] == 3
    assert report["environment"]["reason"] == "non_windows_platform"
    assert report["errors"][0]["code"] == "ENVIRONMENT-NOT-RUN"


def test_invisible_desktop_does_not_construct_capture_service(tmp_path: Path) -> None:
    smoke = _load_smoke()
    constructed = False

    def service_factory(_profiles_root: Path):
        nonlocal constructed
        constructed = True
        raise AssertionError("capture must not run without an interactive desktop")

    report = smoke.run_smoke(
        profile_root=tmp_path,
        desktop_checker=lambda: (
            False,
            {
                "reason": "visible_desktop_unavailable",
                "message": "The Windows input desktop is not available or visible",
            },
        ),
        service_factory=service_factory,
    )

    assert report["status"] == "NOT_RUN"
    assert report["exit_code"] == 3
    assert constructed is False


def test_cli_writes_not_run_report_and_returns_exit_code_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke = _load_smoke()
    output = tmp_path / "windows_smoke.json"
    monkeypatch.setattr(
        smoke,
        "_check_visible_desktop",
        lambda: (
            False,
            {
                "reason": "visible_desktop_unavailable",
                "message": "No visible Windows desktop is available",
            },
        ),
    )

    exit_code = smoke.main(["--output", str(output)])

    assert exit_code == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "NOT_RUN"
    assert report["exit_code"] == 3
    assert report["errors"][0]["code"] == "ENVIRONMENT-NOT-RUN"


def test_cli_parser_error_returns_exit_code_2() -> None:
    smoke = _load_smoke()
    assert smoke.main(["--duration", "not-a-number"]) == 2
