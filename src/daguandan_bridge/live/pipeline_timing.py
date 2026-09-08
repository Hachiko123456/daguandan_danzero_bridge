"""Bounded, thread-safe processing-clock telemetry (never image timestamps)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from time import monotonic_ns


_MAX_COUNT = (1 << 63) - 1


@dataclass
class _Stage:
    count: int = 0
    total_ms: float = 0.0
    maximum_ms: float = 0.0
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=128))


class PipelineTiming:
    """Fixed-cardinality totals and last-128 timing samples, with no disk I/O.

    ``observe(name, elapsed_ms)`` accepts a real processing-clock duration;
    ``increment(name)`` counts failures, no-result and discarded work as well
    as successes. Unknown high-cardinality names collapse into ``other``.
    Quantiles describe only the retained tail, not all-session percentiles.
    """

    MAX_STAGES = 32
    MAX_COUNTERS = 64

    def __init__(self) -> None:
        self._lock = Lock()
        self._stages: dict[str, _Stage] = {}
        self._counters: dict[str, int] = {}

    @staticmethod
    def now_ns() -> int:
        return monotonic_ns()

    @staticmethod
    def _key(name: str, values: dict, limit: int) -> str:
        key = str(name)[:64]
        if key not in values and len(values) >= limit - 1:
            return "other"
        return key

    def increment(self, name: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        with self._lock:
            key = self._key(name, self._counters, self.MAX_COUNTERS)
            self._counters[key] = min(_MAX_COUNT, self._counters.get(key, 0) + amount)

    def observe(self, name: str, elapsed_ms: float) -> None:
        value = float(elapsed_ms)
        if value < 0 or not isfinite(value):
            self.increment("invalid_timing")
            return
        # A corrupt clock/input must not make JSON non-finite or unbounded.
        value = min(value, float(_MAX_COUNT))
        with self._lock:
            key = self._key(name, self._stages, self.MAX_STAGES)
            stage = self._stages.setdefault(key, _Stage())
            stage.count = min(_MAX_COUNT, stage.count + 1)
            stage.total_ms = min(float(_MAX_COUNT), stage.total_ms + value)
            stage.maximum_ms = max(stage.maximum_ms, value)
            stage.samples.append(value)

    def elapsed(self, name: str, started_ns: int) -> None:
        self.observe(name, (self.now_ns() - started_ns) / 1_000_000)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            stages = {}
            for name, stage in self._stages.items():
                values = sorted(stage.samples)
                stages[name] = {
                    "count": stage.count,
                    "total_ms": round(stage.total_ms, 3),
                    "max_ms": round(stage.maximum_ms, 3),
                    "recent_count": len(values),
                    "recent_p50_ms": round(values[(len(values) - 1) // 2], 3),
                    "recent_p95_ms": round(values[min(len(values) - 1, int(len(values) * .95))], 3),
                }
            return {
                "schema": "guandan.pipeline-timing/1",
                "clock": "processing_monotonic_ns",
                "percentile_scope": "last_128_samples_per_stage_not_all_session",
                "stages": stages,
                "counters": dict(self._counters),
            }
