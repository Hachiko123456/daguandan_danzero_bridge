from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any


class StrategyExecutionTrace:
    """Thread-safe progress data retained when a request fails or times out."""

    def __init__(
        self,
        request_id: str = "",
    ) -> None:
        self.request_id = str(request_id)
        self._lock = Lock()
        self._phase = ""
        self._phase_started = perf_counter()
        self._timings: dict[str, float] = {}
        self._engine_input: dict[str, object] | None = None

    def begin(self, phase: str) -> None:
        with self._lock:
            self._close_phase()
            self._phase = str(phase)
            self._phase_started = perf_counter()

    def end(self) -> None:
        with self._lock:
            self._close_phase()
            self._phase = ""

    def set_engine_input(self, value: dict[str, object]) -> None:
        with self._lock:
            safe = self._make_json_safe(value)
            self._engine_input = safe if isinstance(safe, dict) else None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            current_phase = self._phase
            phase_elapsed = (
                (perf_counter() - self._phase_started) * 1000
                if current_phase
                else 0.0
            )
            timings = dict(self._timings)
            if current_phase:
                timings[current_phase] = timings.get(current_phase, 0.0) + phase_elapsed
            return {
                "request_id": self.request_id,
                "current_phase": current_phase or None,
                "phase_elapsed_ms": round(phase_elapsed, 3),
                "timings": {key: round(value, 3) for key, value in timings.items()},
                "engine_input": self._engine_input,
            }

    def _close_phase(self) -> None:
        if self._phase:
            self._timings[self._phase] = (
                self._timings.get(self._phase, 0.0)
                + (perf_counter() - self._phase_started) * 1000
            )

    @staticmethod
    def _make_json_safe(value: Any) -> object:
        if isinstance(value, dict):
            return {
                str(key): StrategyExecutionTrace._make_json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [StrategyExecutionTrace._make_json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


@dataclass(frozen=True)
class AdviceResult:
    strategy: str
    cards: tuple[str, ...]
    play_type: str
    is_pass: bool
    state_revision: int
    elapsed_ms: float
    request_id: str = ""
    engine_input: dict[str, object] | None = None
    timings: dict[str, float] = field(default_factory=dict)


LocalAdvice = AdviceResult
