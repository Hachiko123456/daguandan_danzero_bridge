from __future__ import annotations

import importlib
from typing import Any


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
