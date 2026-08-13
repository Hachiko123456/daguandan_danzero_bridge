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

    def audit_info(self) -> dict[str, object]:
        """Return immutable checkpoint identity without exposing model internals."""

        checkpoint = self._advisor._danzero_checkpoint_path()
        import hashlib

        return {
            "backend": "danzero",
            "status": "loaded" if self._advisor._agent is not None else "available",
            "path": str(checkpoint),
            "digest": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "schema": "danzero-adapter/v1",
        }

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
