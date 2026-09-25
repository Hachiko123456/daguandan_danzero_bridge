"""Small, JSON-backed visual state-transition fixtures.

The fixtures deliberately contain no images, Win32 handles, model objects, or
session paths.  A layer test can load a scenario and feed ``frame_result`` to
``OpeningTracker`` or another pure seam while asserting the adjacent
``expected`` contract.  Keeping the source data in JSON makes every scenario
inspectable and serializable in CI diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

_FIXTURE_PATH = Path(__file__).with_name("opening_scenarios.json")


@dataclass(frozen=True)
class VisualFrame:
    """One artificial captured-frame observation and its stable identity."""

    frame_id: str
    monotonic_ms: int
    generation: object
    anchor_score: float
    page: Mapping[str, Any]
    result: Mapping[str, Any]

    @property
    def observation_id(self) -> str:
        return self.frame_id


@dataclass(frozen=True)
class VisualScenario:
    """A named sequence plus explicit expected state/event projections."""

    name: str
    description: str
    frames: tuple[VisualFrame, ...]
    expected: Mapping[str, Any]

    @property
    def expected_states(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.expected.get("states", ()))

    @property
    def expected_reasons(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.expected.get("reasons", ()))

    @property
    def expected_events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.expected.get("events", ()))

    @property
    def expected_tracker_statuses(self) -> tuple[str, ...]:
        """Exact ``OpeningTracker`` status projection, when supplied."""

        return tuple(str(value) for value in self.expected.get("tracker_statuses", ()))

    @property
    def expected_tracker_reasons(self) -> tuple[str, ...]:
        """Exact pre-layer reason projection for timeout scenarios."""

        return tuple(str(value) for value in self.expected.get("tracker_reasons_before_layer_termination", ()))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy suitable for test diagnostics."""

        return {
            "name": self.name,
            "description": self.description,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "monotonic_ms": frame.monotonic_ms,
                    "generation": frame.generation,
                    "anchor_score": frame.anchor_score,
                    "page": _plain(frame.page),
                    "result": _plain(frame.result),
                }
                for frame in self.frames
            ],
            "expected": _plain(self.expected),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _expand_hand(value: Any, hand: tuple[str, ...]) -> Any:
    if value == "$HAND":
        return list(hand)
    if isinstance(value, Mapping):
        return {key: _expand_hand(item, hand) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_hand(item, hand) for item in value]
    return value


def _namespace(value: Any) -> Any:
    """Recursively turn fixture mappings into attribute-readable test values."""

    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(key): _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_namespace(item) for item in value)
    return value


def _load_document() -> dict[str, Any]:
    document = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported visual fixture schema: {document.get('schema_version')!r}")
    hand = tuple(str(card) for card in document.get("hand", ()))
    if len(hand) != 27 or len(set(hand)) != 27:
        raise ValueError("fixture hand must contain exactly 27 unique cards")
    document["hand"] = hand
    document["scenarios"] = [
        _expand_hand(scenario, hand) for scenario in document.get("scenarios", ())
    ]
    return document


def load_scenarios() -> tuple[VisualScenario, ...]:
    """Load all JSON scenarios in deterministic file order."""

    document = _load_document()
    scenarios: list[VisualScenario] = []
    for raw in document["scenarios"]:
        frames = tuple(
            VisualFrame(
                frame_id=str(frame["frame_id"]),
                monotonic_ms=int(frame["monotonic_ms"]),
                generation=frame["generation"],
                anchor_score=float(frame["anchor_score"]),
                page=dict(frame.get("page", {})),
                result=dict(frame.get("result", {})),
            )
            for frame in raw.get("frames", ())
        )
        scenarios.append(
            VisualScenario(
                name=str(raw["name"]),
                description=str(raw.get("description", "")),
                frames=frames,
                expected=dict(raw.get("expected", {})),
            )
        )
    return tuple(scenarios)


def scenario(name: str) -> VisualScenario:
    """Return one named scenario with a useful error for layer selection."""

    for item in load_scenarios():
        if item.name == name:
            return item
    available = ", ".join(item.name for item in load_scenarios())
    raise KeyError(f"unknown visual fixture {name!r}; available: {available}")


def scenario_names() -> tuple[str, ...]:
    return tuple(item.name for item in load_scenarios())


def frame_result(frame: VisualFrame) -> SimpleNamespace:
    """Build an attribute-readable recognized-result stand-in for one frame."""

    return _namespace(frame.result)


def page_signal(frame: VisualFrame) -> SimpleNamespace:
    """Build a tiny page signal stand-in without importing Qt or Win32 code."""

    return _namespace(frame.page)


def tracker_inputs(scenario_or_name: VisualScenario | str) -> Iterator[tuple[object, float, object, int, str]]:
    """Yield ``OpeningTracker.observe`` keyword values in fixture order."""

    selected = scenario(scenario_or_name) if isinstance(scenario_or_name, str) else scenario_or_name
    for frame in selected.frames:
        yield (
            frame_result(frame),
            frame.anchor_score,
            frame.generation,
            frame.monotonic_ms,
            frame.observation_id,
        )


__all__ = [
    "VisualFrame",
    "VisualScenario",
    "frame_result",
    "load_scenarios",
    "page_signal",
    "scenario",
    "scenario_names",
    "tracker_inputs",
]
