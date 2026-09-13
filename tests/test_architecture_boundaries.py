from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "daguandan_bridge"

# These command-oriented validation modules are composition roots: they wire
# UI/infrastructure adapters together but do not contain reusable application
# use cases.  Keep the list explicit so a new boundary exception cannot appear
# unnoticed.
#
# ``live_v2_recorded_replay.py`` is intentionally an exception even though it
# lives under ``application``: it is the recorded-replay entry point that must
# construct the same production LiveV2 runtime used by the live controller.
# Its infrastructure imports are therefore explicit and reviewable here,
# rather than hidden behind a dynamic import or a broad boundary exemption.
APPLICATION_COMPOSITION_ROOTS = {
    "live_v2_recorded_replay.py",
    "session_replay_audit.py",
    "shadow_live_replay.py",
    "simulated_game_window.py",
    "window_e2e_validation.py",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add("." * node.level + (node.module or ""))
    return result


def test_domain_and_application_layers_do_not_import_ui_or_infrastructure():
    forbidden = ("PySide6", "qfluentwidgets", "cv2", "daguandan_bridge.gui", "..gui", "..infrastructure")
    for layer in ("domain", "application"):
        for path in (SOURCE / layer).rglob("*.py"):
            if layer == "application" and path.name in APPLICATION_COMPOSITION_ROOTS:
                continue
            imports = _imports(path)
            assert not any(name.startswith(forbidden) for name in imports), (path, imports)


def test_application_composition_root_exceptions_are_exact_and_present():
    application = SOURCE / "application"
    assert APPLICATION_COMPOSITION_ROOTS == {
        path.name
        for path in application.glob("*.py")
        if path.name in APPLICATION_COMPOSITION_ROOTS
    }


def test_recorded_replay_exception_documents_production_composition_boundary():
    """Recorded replay is exempt because it deliberately wires production adapters."""
    imports = _imports(SOURCE / "application" / "live_v2_recorded_replay.py")
    assert "..infrastructure.live_session" in imports
    assert "..infrastructure.live_v2_composition" in imports


def test_orchestrator_depends_on_ports_not_live_concrete_adapters():
    source = (SOURCE / "live" / "orchestrator.py").read_text(encoding="utf-8")
    for concrete in (
        "ScreenshotRecognitionService",
        "LiveSessionStore",
        "SessionRecorder",
        "danzero.advisor",
    ):
        assert concrete not in source
    assert "RecognitionPort" in source
    assert "SessionPersistencePort" in source
    assert "RecordingPort" in source
    assert "AdvicePort" in source


def test_live_controller_and_window_do_not_construct_live_adapters():
    controller = (SOURCE / "gui" / "live_controller.py").read_text(encoding="utf-8")
    window = (SOURCE / "gui" / "main_window.py").read_text(encoding="utf-8")
    for concrete in ("LiveSessionStore(", "SessionRecorder(", "LiveOrchestrator(", "ScreenshotRecognitionService(", "DanzeroAdvisor("):
        assert concrete not in controller
        assert concrete not in window
    assert "session_factory.start_session" in controller

def test_video_scan_application_use_case_does_not_import_media_runtime_packages():
    imports = _imports(SOURCE / "application" / "video_scan.py")
    assert not any(name.startswith(("cv2", "numpy")) for name in imports), imports
