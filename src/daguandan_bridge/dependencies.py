from __future__ import annotations

import importlib
from typing import Any


_LIVE_WORKER_DEPENDENCIES_PRELOADED = False


class DependencyError(RuntimeError):
    """缺少运行依赖时抛出的错误。"""


def import_required(module_name: str, pip_name: str) -> Any:
    """延迟导入第三方库，便于无需 GUI 的单元测试运行。"""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DependencyError(
            f"缺少依赖模块 {module_name!r}，请先执行：pip install {pip_name}"
        ) from exc


def preload_danzero_rule_dependencies() -> None:
    """Import and touch the rule-engine stack before worker threads start.

    Frozen Python 3.12 builds have shown import/GC crashes when the GUI thread
    lazily imports the Danzero/RLCard rule stack while the capture worker is
    also lazily importing Win32 capture modules.  Keep this as a startup-only
    preload: it does not lock or wrap per-frame work.
    """

    from daguandan_bridge.danzero.rules import actions_for_cards

    actions_for_cards(("5C", "5H"), "6")


def preload_win32_capture_dependencies() -> None:
    """Import capture backends once, synchronously, before capture threads run."""

    import_required("cv2", "opencv-python")
    import_required("numpy", "numpy")
    import_required("win32gui", "pywin32")
    import_required("win32ui", "pywin32")


def preload_live_worker_dependencies() -> None:
    """Preload imports shared by live capture and rule evaluation workers."""

    global _LIVE_WORKER_DEPENDENCIES_PRELOADED
    if _LIVE_WORKER_DEPENDENCIES_PRELOADED:
        return
    preload_danzero_rule_dependencies()
    preload_win32_capture_dependencies()
    _LIVE_WORKER_DEPENDENCIES_PRELOADED = True
