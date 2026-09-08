from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from daguandan_bridge.infrastructure.live_v2_advice_service_factory import (
    ADVICE_WORKER_REFERENCE,
)
from daguandan_bridge.infrastructure.live_v2_vision_service_factory import (
    DEFAULT_VISION_WORKER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = PROJECT_ROOT / "run.py"
PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_release.ps1"
COLLECTION_PATH = PROJECT_ROOT / "scripts" / "pyinstaller_live_v2_collection.json"

REQUIRED_LIVE_V2_MODULES = {
    "daguandan_bridge.application.recording_process_protocol",
    "daguandan_bridge.application.live_v2_advice_protocol",
    "daguandan_bridge.application.live_v2_advice_runtime",
    "daguandan_bridge.application.live_v2_frame_pipeline",
    "daguandan_bridge.application.live_v2_session_runtime",
    "daguandan_bridge.application.live_v2_vision_protocol",
    "daguandan_bridge.application.live_v2_vision_runtime",
    "daguandan_bridge.application.live_v2_worker_protocol",
    "daguandan_bridge.fabledan.advisor",
    "daguandan_bridge.infrastructure.live_v2_advice_service_factory",
    "daguandan_bridge.infrastructure.live_v2_advice_worker",
    "daguandan_bridge.infrastructure.live_v2_rule_session",
    "daguandan_bridge.infrastructure.live_v2_vision_service_factory",
    "daguandan_bridge.infrastructure.live_v2_vision_worker",
    "daguandan_bridge.infrastructure.live_v2_worker_host",
    "daguandan_bridge.infrastructure.live_v2_worker_process",
    "daguandan_bridge.infrastructure.process_session_recorder",
    "daguandan_bridge.infrastructure.recording_process",
    "daguandan_bridge.infrastructure.recording_process_forensics",
    "daguandan_bridge.infrastructure.recording_process_worker",
    "daguandan_bridge.recognition_service",
    "multiprocessing.popen_spawn_win32",
    "multiprocessing.reduction",
    "multiprocessing.spawn",
}


def _call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return ""
    return node.value.func.id if isinstance(node.value.func, ast.Name) else ""


def test_freeze_support_runs_before_any_application_import_or_startup_side_effect() -> None:
    source = RUN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUN_PATH))
    freeze_index = next(
        index for index, node in enumerate(tree.body) if _call_name(node) == "freeze_support"
    )
    first_project_import = next(
        index
        for index, node in enumerate(tree.body)
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("daguandan_bridge")
        ) or (
            isinstance(node, ast.Import)
            and any(name.name.startswith("daguandan_bridge") for name in node.names)
        )
    )
    assert freeze_index < first_project_import
    assert source.index("freeze_support()") < source.index("initialize_startup_diagnostics()")


def test_live_v2_collection_inventory_is_versioned_unique_and_importable() -> None:
    document = json.loads(COLLECTION_PATH.read_text(encoding="utf-8"))
    assert document["schema"] == "guandan.pyinstaller-live-v2-collection/1"
    modules = document["hidden_imports"]
    assert isinstance(modules, list)
    assert modules == sorted(set(modules))
    assert REQUIRED_LIVE_V2_MODULES <= set(modules)
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    assert missing == []


def test_dynamic_worker_references_resolve_to_functions_in_collected_modules() -> None:
    collected = set(json.loads(COLLECTION_PATH.read_text(encoding="utf-8"))["hidden_imports"])
    for reference in (DEFAULT_VISION_WORKER, ADVICE_WORKER_REFERENCE):
        assert reference.module_path in collected
        worker = reference.resolve()
        assert callable(worker)
        assert worker.__module__ == reference.module_path
        assert worker.__name__ == reference.function_name


def test_package_script_validates_and_applies_collection_before_entry_script() -> None:
    source = PACKAGE_SCRIPT.read_text(encoding="utf-8")
    assert "scripts\\pyinstaller_live_v2_collection.json" in source
    assert "guandan.pyinstaller-live-v2-collection/1" in source
    assert "ConvertFrom-Json" in source
    assert "hidden-import inventory is empty" in source
    assert "hidden-import inventory contains duplicates" in source
    append_imports = '$pyinstallerArguments += @("--hidden-import", $module)'
    append_entry = '$pyinstallerArguments += (Join-Path $projectRoot "run.py")'
    invoke = "Invoke-CleanPython $buildPython (Join-Path $buildEnvPath \"Scripts\") @pyinstallerArguments"
    assert source.index(append_imports) < source.index(append_entry) < source.index(invoke)
    assert '"--collect-submodules", "daguandan_bridge"' in source


def test_run_help_still_works_after_early_freeze_support() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUN_PATH), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--doctor" in completed.stdout
