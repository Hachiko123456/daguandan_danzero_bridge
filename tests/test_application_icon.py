from __future__ import annotations

import os
from pathlib import Path
import shutil
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from daguandan_bridge import application_icon


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _qapplication():
    return QApplication.instance() or QApplication([])


def _runtime(*, frozen: bool, meipass: Path | None, executable: Path) -> SimpleNamespace:
    return SimpleNamespace(
        frozen=frozen,
        _MEIPASS=str(meipass) if meipass is not None else None,
        executable=str(executable),
    )


def _source_module_file(root: Path) -> Path:
    return root / "src" / "daguandan_bridge" / "application_icon.py"


def test_resolves_source_app_icon(tmp_path):
    root = tmp_path / "source"
    icon_path = root / "app.ico"
    root.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "app.ico", icon_path)

    assert (
        application_icon.resolve_application_icon_path(
            module_file=_source_module_file(root),
            runtime=_runtime(
                frozen=False,
                meipass=None,
                executable=tmp_path / "python.exe",
            ),
        )
        == icon_path
    )


def test_resolves_frozen_meipass_before_executable_sibling(tmp_path):
    meipass = tmp_path / "meipass"
    exe_root = tmp_path / "exe"
    meipass.mkdir()
    exe_root.mkdir()
    meipass_icon = meipass / "app.ico"
    exe_icon = exe_root / "app.ico"
    shutil.copy2(PROJECT_ROOT / "app.ico", meipass_icon)
    shutil.copy2(PROJECT_ROOT / "app.ico", exe_icon)

    result = application_icon.load_application_icon(
        module_file=_source_module_file(tmp_path / "source"),
        runtime=_runtime(
            frozen=True,
            meipass=meipass,
            executable=exe_root / "DaguandanAssistant.exe",
        ),
    )

    assert result.ok
    assert result.path == meipass_icon
    assert result.source == "frozen_meipass"
    assert result.available_sizes


def test_frozen_mode_falls_back_to_executable_sibling(tmp_path):
    exe_root = tmp_path / "exe"
    exe_root.mkdir()
    exe_icon = exe_root / "app.ico"
    shutil.copy2(PROJECT_ROOT / "app.ico", exe_icon)

    result = application_icon.load_application_icon(
        runtime=_runtime(
            frozen=True,
            meipass=tmp_path / "missing-meipass",
            executable=exe_root / "DaguandanAssistant.exe",
        )
    )

    assert result.ok
    assert result.path == exe_icon
    assert result.source == "executable_directory"


def test_missing_and_invalid_resources_are_recorded_as_startup_diagnostics(
    tmp_path, monkeypatch
):
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        "daguandan_bridge.startup_diagnostics.record_startup_event",
        lambda event, evidence=None: events.append((event, dict(evidence or {}))),
    )
    runtime = _runtime(
        frozen=False,
        meipass=None,
        executable=tmp_path / "python.exe",
    )
    module_file = _source_module_file(tmp_path)

    missing = application_icon.load_application_icon(
        module_file=module_file,
        runtime=runtime,
    )
    assert not missing.ok
    assert missing.attempts and not missing.attempts[0].exists

    icon_path = tmp_path / "app.ico"
    icon_path.write_bytes(b"not an ico")
    invalid = application_icon.load_application_icon(
        module_file=module_file,
        runtime=runtime,
    )
    assert not invalid.ok
    assert invalid.attempts[0].exists
    assert invalid.attempts[0].is_null is True
    assert invalid.attempts[0].available_sizes == ()
    assert [event for event, _ in events] == [
        "application_icon_load_failed",
        "application_icon_load_failed",
    ]
    assert events[-1][1]["resource"] == "app.ico"


def test_installs_same_validated_icon_on_application_and_window(tmp_path):
    root = tmp_path / "source"
    icon_path = root / "app.ico"
    root.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "app.ico", icon_path)
    runtime = _runtime(
        frozen=False,
        meipass=None,
        executable=tmp_path / "python.exe",
    )

    app = QApplication.instance() or QApplication([])
    window = QWidget()
    result = application_icon.install_application_icon(
        app,
        window,
        result=application_icon.load_application_icon(
            module_file=_source_module_file(root),
            runtime=runtime,
        ),
    )

    assert result.ok
    assert not app.windowIcon().isNull()
    assert not window.windowIcon().isNull()
    assert tuple(
        (size.width(), size.height()) for size in window.windowIcon().availableSizes()
    ) == result.available_sizes
    window.close()
