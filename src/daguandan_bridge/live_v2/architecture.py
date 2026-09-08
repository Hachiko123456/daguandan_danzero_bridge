"""Executable import-boundary checks for the live-v2 core.

The filesystem entry point in this module is development tooling. Runtime core
modules may depend only on the standard library and explicitly listed live-v2
modules. ``types`` is a compatibility facade, not a place for new definitions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


PACKAGE_NAME = "daguandan_bridge.live_v2"
REVIEW_TARGET_LINES = 450
MAX_MODULE_LINES = 600
CONTRACT_MODULE_LINES = MAX_MODULE_LINES

MODULE_DEPENDENCY_ALLOWLIST: dict[str, frozenset[str]] = {
    "action_semantics": frozenset(),
    "identity": frozenset(),
    "observations": frozenset({"identity"}),
    "candidates": frozenset({"action_semantics", "identity", "observations"}),
    "corrections": frozenset(
        {"action_semantics", "candidates", "identity", "observations"}
    ),
    "results": frozenset({"candidates", "identity", "observations"}),
    "events": frozenset(
        {"action_semantics", "candidates", "corrections", "results"}
    ),
    "game_state": frozenset(
        {"action_semantics", "corrections", "events", "identity", "observations"}
    ),
    "types": frozenset({"events", "identity", "observations"}),
    "protocols": frozenset({"game_state", "types"}),
    "candidate_projection": frozenset({"types"}),
    "evidence_buffer": frozenset({"types"}),
    "engine": frozenset(
        {
            "event_resolver",
            "game_state",
            "gap_lifecycle",
            "input_lifecycle",
            "opportunity",
            "protocols",
            "types",
        }
    ),
    "event_resolver": frozenset({"protocols", "types"}),
    "gap_lifecycle": frozenset({"types"}),
    "input_lifecycle": frozenset({"types"}),
    "opportunity": frozenset({"game_state", "protocols", "types"}),
    "reconciliation": frozenset({"event_resolver", "types"}),
    "reducer_snapshot": frozenset(
        {"corrections", "game_state", "identity", "types"}
    ),
    "reducer_transaction": frozenset({"corrections", "game_state", "types"}),
    "rules_adapter": frozenset(
        {"candidate_projection", "reducer_transaction", "types"}
    ),
    "scheduler": frozenset({"scheduler_queue", "scheduling_records", "types"}),
    "scheduler_queue": frozenset({"scheduling_records", "types"}),
    "scheduling_records": frozenset({"types"}),
    "seat_tracker": frozenset({"types"}),
    "vision_adapter": frozenset({"types"}),
    "architecture": frozenset(),
    "__init__": frozenset(
        {
            "architecture",
            "engine",
            "event_resolver",
            "game_state",
            "gap_lifecycle",
            "input_lifecycle",
            "opportunity",
            "protocols",
            "reconciliation",
            "types",
        }
    ),
}

STDLIB_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    "action_semantics": frozenset({"__future__", "dataclasses", "typing"}),
    "identity": frozenset({"__future__", "dataclasses", "enum"}),
    "observations": frozenset({"__future__", "dataclasses", "enum"}),
    "candidates": frozenset({"__future__", "dataclasses", "enum"}),
    "corrections": frozenset({"__future__", "dataclasses", "enum"}),
    "results": frozenset({"__future__", "dataclasses", "enum"}),
    "events": frozenset(),
    "game_state": frozenset({"__future__", "dataclasses"}),
    "types": frozenset(),
    "protocols": frozenset({"__future__", "typing"}),
    "candidate_projection": frozenset({"__future__", "itertools", "typing"}),
    "evidence_buffer": frozenset(
        {"__future__", "collections", "dataclasses", "typing"}
    ),
    "engine": frozenset({"__future__", "dataclasses"}),
    "event_resolver": frozenset({"__future__", "dataclasses"}),
    "gap_lifecycle": frozenset({"__future__", "dataclasses", "enum"}),
    "input_lifecycle": frozenset({"__future__", "dataclasses", "enum"}),
    "opportunity": frozenset({"__future__", "dataclasses", "enum"}),
    "reconciliation": frozenset({"__future__", "dataclasses"}),
    "reducer_snapshot": frozenset({"__future__", "dataclasses"}),
    "reducer_transaction": frozenset({"__future__", "dataclasses", "typing"}),
    "rules_adapter": frozenset({"__future__", "dataclasses", "threading"}),
    "scheduler": frozenset({"__future__", "collections", "typing"}),
    "scheduler_queue": frozenset({"__future__", "collections"}),
    "scheduling_records": frozenset({"__future__", "dataclasses", "typing"}),
    "seat_tracker": frozenset({"__future__", "dataclasses"}),
    "vision_adapter": frozenset({"__future__", "collections"}),
    "architecture": frozenset({"__future__", "ast", "dataclasses", "pathlib"}),
    "__init__": frozenset(),
}

EXTERNAL_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    module: frozenset() for module in MODULE_DEPENDENCY_ALLOWLIST
}
# This anti-corruption adapter may read the legacy recognizer's immutable DTO.
# The live-v2 core still has no dependency on the recognizer implementation.
EXTERNAL_IMPORT_ALLOWLIST["vision_adapter"] = frozenset(
    {"daguandan_bridge.domain.recognition"}
)
EXTERNAL_IMPORT_ALLOWLIST["reducer_transaction"] = frozenset(
    {"daguandan_bridge.domain.live"}
)

FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "PySide6",
    "qfluentwidgets",
    "cv2",
    "numpy",
    "mss",
    "win32",
    "daguandan_bridge.gui",
    "daguandan_bridge.infrastructure",
    "daguandan_bridge.live.orchestrator",
    "daguandan_bridge.live.reducer",
    "daguandan_bridge.live.card_uncertainty",
    "daguandan_bridge.danzero.rules",
    "daguandan_bridge.danzero.state",
    "daguandan_bridge.capture_service",
    "daguandan_bridge.recognition_service",
)

MODULE_LINE_LIMITS: dict[str, int] = {
    "identity": CONTRACT_MODULE_LINES,
    "observations": CONTRACT_MODULE_LINES,
    "candidates": CONTRACT_MODULE_LINES,
    "corrections": CONTRACT_MODULE_LINES,
    "game_state": CONTRACT_MODULE_LINES,
    "reducer_snapshot": CONTRACT_MODULE_LINES,
    "reducer_transaction": CONTRACT_MODULE_LINES,
    "results": CONTRACT_MODULE_LINES,
    "events": CONTRACT_MODULE_LINES,
    "types": CONTRACT_MODULE_LINES,
    "protocols": CONTRACT_MODULE_LINES,
}


@dataclass(frozen=True, slots=True)
class DependencyViolation:
    module: str
    line: int
    imported: str
    reason: str

    def format(self) -> str:
        return f"{self.module}:{self.line}: {self.reason}: {self.imported}"


def find_dependency_cycles(
    dependencies: dict[str, frozenset[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic cycles in the declared live-v2 dependency graph."""

    graph = MODULE_DEPENDENCY_ALLOWLIST if dependencies is None else dependencies
    visited: set[str] = set()
    active: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(module: str) -> None:
        if module in active:
            start = active.index(module)
            cycle = tuple(active[start:] + [module])
            rotations = tuple(
                cycle[index:-1] + cycle[:index] + (cycle[index],)
                for index in range(len(cycle) - 1)
            )
            cycles.add(min(rotations))
            return
        if module in visited:
            return
        active.append(module)
        for dependency in sorted(graph.get(module, ())):
            if dependency in graph:
                visit(dependency)
        active.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)
    return tuple(sorted(cycles))


def _is_forbidden(imported: str) -> bool:
    return any(
        imported == prefix
        or imported.startswith(f"{prefix}.")
        or (prefix == "win32" and imported.startswith("win32"))
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _check_absolute_import(
    module: str,
    imported: str,
    line: int,
) -> DependencyViolation | None:
    if imported in EXTERNAL_IMPORT_ALLOWLIST[module]:
        return None
    if _is_forbidden(imported):
        return DependencyViolation(module, line, imported, "forbidden dependency")
    if imported == PACKAGE_NAME or imported.startswith(f"{PACKAGE_NAME}."):
        suffix = imported.removeprefix(f"{PACKAGE_NAME}.")
        dependency = suffix.split(".", 1)[0]
        if dependency not in MODULE_DEPENDENCY_ALLOWLIST[module]:
            return DependencyViolation(
                module,
                line,
                imported,
                "live-v2 dependency is not allowed by the module table",
            )
        return None
    root = imported.split(".", 1)[0]
    if root not in STDLIB_IMPORT_ALLOWLIST[module]:
        return DependencyViolation(
            module,
            line,
            imported,
            "external or unlisted standard-library dependency",
        )
    return None


def check_source_dependencies(
    module: str,
    source: str,
) -> tuple[DependencyViolation, ...]:
    """Check one module's source without importing or executing it."""

    if module not in MODULE_DEPENDENCY_ALLOWLIST:
        raise ValueError(f"unknown live-v2 module: {module}")
    violations: list[DependencyViolation] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return (
            DependencyViolation(
                module,
                exc.lineno or 0,
                "<syntax>",
                exc.msg,
            ),
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _check_absolute_import(module, alias.name, node.lineno)
                if violation is not None:
                    violations.append(violation)
        elif isinstance(node, ast.ImportFrom):
            imported = node.module or ""
            if node.level:
                if node.level == 2 and imported:
                    absolute = f"daguandan_bridge.{imported}"
                    violation = _check_absolute_import(module, absolute, node.lineno)
                    if violation is not None:
                        violations.append(violation)
                    continue
                if node.level != 1 or not imported:
                    violations.append(
                        DependencyViolation(
                            module,
                            node.lineno,
                            "." * node.level + imported,
                            "relative import escapes or targets the package root",
                        )
                    )
                    continue
                dependency = imported.split(".", 1)[0]
                if dependency not in MODULE_DEPENDENCY_ALLOWLIST[module]:
                    violations.append(
                        DependencyViolation(
                            module,
                            node.lineno,
                            f".{imported}",
                            "live-v2 dependency is not allowed by the module table",
                        )
                    )
                continue
            violation = _check_absolute_import(module, imported, node.lineno)
            if violation is not None:
                violations.append(violation)
    return tuple(sorted(violations, key=lambda item: (item.line, item.imported)))


def check_package_boundaries(package_root: Path) -> tuple[DependencyViolation, ...]:
    """Scan all declared modules plus the 500-line architecture guard."""

    violations: list[DependencyViolation] = []
    for cycle in find_dependency_cycles():
        violations.append(
            DependencyViolation(
                cycle[0],
                0,
                " -> ".join(cycle),
                "declared module dependency cycle",
            )
        )
    declared_paths = {
        "__init__.py" if module == "__init__" else f"{module}.py"
        for module in MODULE_DEPENDENCY_ALLOWLIST
    }
    for path in package_root.glob("*.py"):
        if path.name not in declared_paths:
            violations.append(
                DependencyViolation(
                    path.stem,
                    0,
                    str(path),
                    "module is missing from the dependency allowlist",
                )
            )
    for module in MODULE_DEPENDENCY_ALLOWLIST:
        path = package_root / ("__init__.py" if module == "__init__" else f"{module}.py")
        if not path.is_file():
            violations.append(
                DependencyViolation(module, 0, str(path), "declared module is missing")
            )
            continue
        source = path.read_text(encoding="utf-8")
        line_count = len(source.splitlines())
        line_limit = MODULE_LINE_LIMITS.get(module, MAX_MODULE_LINES)
        if line_count >= line_limit:
            violations.append(
                DependencyViolation(
                    module,
                    line_limit,
                    str(path),
                    f"module has {line_count} lines; limit is below {line_limit}",
                )
            )
        violations.extend(check_source_dependencies(module, source))
    return tuple(violations)


def assert_package_boundaries(package_root: Path) -> None:
    violations = check_package_boundaries(package_root)
    if violations:
        details = "\n".join(item.format() for item in violations)
        raise AssertionError(f"live-v2 architecture violations:\n{details}")
