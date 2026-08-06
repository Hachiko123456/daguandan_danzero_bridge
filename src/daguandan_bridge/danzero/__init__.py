"""Manual-state DanZero advice API for Tencent DaGuandan."""

from .advisor import LocalAdvice, LocalGuandanAdvisor, StrategyExecutionTrace
from .state import GameStateError, GuanDanState


class DanzeroAdvisor:
    """Provide DanZero advice from a manually confirmed game state."""

    def __init__(self) -> None:
        self._advisor = LocalGuandanAdvisor("danzero")

    def initialize(self) -> None:
        """Load the DanZero model so the first advice call stays responsive."""
        self._advisor.initialize()

    def recommend(
        self,
        state: GuanDanState,
        *,
        request_id: str = "",
        trace: StrategyExecutionTrace | None = None,
    ) -> LocalAdvice:
        return self._advisor.recommend(
            state.local_snapshot(),
            request_id=request_id,
            trace=trace,
        )


__all__ = [
    "DanzeroAdvisor",
    "GameStateError",
    "GuanDanState",
    "LocalAdvice",
]
