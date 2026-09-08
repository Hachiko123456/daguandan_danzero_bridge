from __future__ import annotations

import ast
from pathlib import Path

import pytest

from daguandan_bridge.live_v2.architecture import (
    CONTRACT_MODULE_LINES,
    MAX_MODULE_LINES,
    MODULE_DEPENDENCY_ALLOWLIST,
    MODULE_LINE_LIMITS,
    REVIEW_TARGET_LINES,
    assert_package_boundaries,
    check_package_boundaries,
    check_source_dependencies,
    find_dependency_cycles,
)
from daguandan_bridge.live_v2 import (
    RuleStateProvider,
    TrustedGameSnapshot,
    LiveEngine,
    UnifiedActionResolver,
    assert_package_boundaries as public_assert_package_boundaries,
    check_package_boundaries as public_check_package_boundaries,
)


PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "daguandan_bridge" / "live_v2"
INFRASTRUCTURE_ROOT = PACKAGE_ROOT.parent / "infrastructure"


def test_current_package_obeys_declared_boundaries() -> None:
    assert check_package_boundaries(PACKAGE_ROOT) == ()
    assert_package_boundaries(PACKAGE_ROOT)


def test_dependency_table_is_explicit_and_acyclic_for_runtime_core() -> None:
    assert MODULE_DEPENDENCY_ALLOWLIST == {
        "action_semantics": frozenset(),
        "identity": frozenset(),
        "observations": frozenset({"identity"}),
        "candidates": frozenset(
            {"action_semantics", "identity", "observations"}
        ),
        "corrections": frozenset(
            {"action_semantics", "candidates", "identity", "observations"}
        ),
        "results": frozenset({"candidates", "identity", "observations"}),
        "events": frozenset(
            {"action_semantics", "candidates", "corrections", "results"}
        ),
        "game_state": frozenset(
            {
                "action_semantics",
                "corrections",
                "events",
                "identity",
                "observations",
            }
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
        "scheduler": frozenset(
            {"scheduler_queue", "scheduling_records", "types"}
        ),
        "scheduler_queue": frozenset({"scheduling_records", "types"}),
        "scheduling_records": frozenset({"types"}),
        "seat_tracker": frozenset({"types"}),
        "vision_adapter": frozenset({"types"}),
        "turn_core": frozenset({"candidates", "identity"}),
        "opening_core": frozenset({"identity", "turn_core"}),
        "single_turn_observer": frozenset({"candidates", "identity", "turn_core"}),
        "simple_advice_gate": frozenset({"turn_core"}),
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
                "opening_core",
                "simple_advice_gate",
                "single_turn_observer",
                "types",
                "turn_core",
            }
        ),
    }
    assert "protocols" not in MODULE_DEPENDENCY_ALLOWLIST["events"]
    assert find_dependency_cycles() == ()


def test_cycle_checker_rejects_an_indirect_module_cycle() -> None:
    graph = {
        "one": frozenset({"two"}),
        "two": frozenset({"three"}),
        "three": frozenset({"one"}),
    }
    assert find_dependency_cycles(graph) == (("one", "two", "three", "one"),)


@pytest.mark.parametrize(
    "source",
    [
        "import PySide6\n",
        "import cv2\n",
        "import win32gui\n",
        "from daguandan_bridge.gui import live_controller\n",
        "from daguandan_bridge.infrastructure import live_session\n",
        "from daguandan_bridge.live.orchestrator import LiveOrchestrator\n",
        "from ..gui import live_controller\n",
    ],
)
def test_checker_rejects_gui_native_image_and_legacy_dependencies(source: str) -> None:
    violations = check_source_dependencies("types", source)
    assert len(violations) == 1
    assert violations[0].reason in {
        "forbidden dependency",
        "relative import escapes or targets the package root",
    }


def test_checker_rejects_unlisted_third_party_and_internal_dependencies() -> None:
    assert check_source_dependencies("types", "import torch\n")[0].imported == "torch"
    violation = check_source_dependencies("types", "from .protocols import RuleProjector\n")[0]
    assert "not allowed" in violation.reason


def test_checker_accepts_protocol_dependencies_without_reverse_coupling() -> None:
    source = "from __future__ import annotations\nfrom typing import Protocol\nfrom .types import Seat\n"
    assert check_source_dependencies("protocols", source) == ()
    game_state_source = "from .protocols import RuleStateProvider\n"
    assert check_source_dependencies("game_state", game_state_source)


def test_vision_adapter_has_one_explicit_legacy_dto_exception() -> None:
    source = (
        "from __future__ import annotations\n"
        "from ..domain.recognition import PlayRegionResult\n"
        "from .types import SeatObservation\n"
    )
    assert check_source_dependencies("vision_adapter", source) == ()
    forbidden = "from ..recognition_service import RecognitionService\n"
    assert check_source_dependencies("vision_adapter", forbidden)


def test_pure_core_rejects_every_legacy_rule_import() -> None:
    imports = (
        "from ..danzero.rules import actions_for_cards\n"
        "from ..danzero.state import GameStateError\n"
        "from ..live.card_uncertainty import feasible_action_variants\n"
        "from ..live.reducer import LiveReducer\n"
    )
    for module in MODULE_DEPENDENCY_ALLOWLIST:
        assert check_source_dependencies(module, imports)


def test_one_infrastructure_gateway_owns_all_live_v2_legacy_imports() -> None:
    forbidden = {
        "daguandan_bridge.danzero.rules",
        "daguandan_bridge.danzero.state",
        "daguandan_bridge.live.card_uncertainty",
        "daguandan_bridge.live.reducer",
    }
    offenders: set[str] = set()
    for path in INFRASTRUCTURE_ROOT.glob("live_v2*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                absolute = "daguandan_bridge." + module.removeprefix("..")
                if absolute in forbidden:
                    offenders.add(path.name)
    assert offenders == {"live_v2_legacy_gateway.py"}


def test_checker_reports_syntax_errors_without_executing_source() -> None:
    violation = check_source_dependencies("types", "def broken(:\n")[0]
    assert violation.imported == "<syntax>"


def test_declared_modules_remain_below_hard_size_limit() -> None:
    for module in MODULE_DEPENDENCY_ALLOWLIST:
        path = PACKAGE_ROOT / ("__init__.py" if module == "__init__" else f"{module}.py")
        limit = MODULE_LINE_LIMITS.get(module, MAX_MODULE_LINES)
        assert len(path.read_text(encoding="utf-8").splitlines()) < limit


def test_readability_target_is_advisory_and_hard_limit_is_relaxed() -> None:
    assert REVIEW_TARGET_LINES == 450
    assert MAX_MODULE_LINES == 600
    assert CONTRACT_MODULE_LINES == MAX_MODULE_LINES
    assert MODULE_LINE_LIMITS["candidates"] == MAX_MODULE_LINES
    assert MODULE_LINE_LIMITS["results"] == MAX_MODULE_LINES
    assert MODULE_LINE_LIMITS["turn_core"] == 650


def test_package_checker_does_not_silently_ignore_a_new_module(tmp_path: Path) -> None:
    (tmp_path / "surprise.py").write_text("from PySide6 import QtCore\n", encoding="utf-8")
    violations = check_package_boundaries(tmp_path)
    assert any("missing from the dependency allowlist" in item.reason for item in violations)


def test_boundary_checker_names_are_part_of_the_public_package_api() -> None:
    assert public_check_package_boundaries is check_package_boundaries
    assert public_assert_package_boundaries is assert_package_boundaries
    assert LiveEngine.__module__.endswith(".engine")
    assert UnifiedActionResolver.__module__.endswith(".event_resolver")
    assert TrustedGameSnapshot.__module__.endswith(".game_state")
    assert RuleStateProvider.__module__.endswith(".protocols")
