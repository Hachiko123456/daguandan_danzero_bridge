"""Manual-state DanZero advice API for Tencent DaGuandan."""

from pathlib import Path

from ..config import PROFILES_ROOT
from .advisor import LocalAdvice, LocalGuandanAdvisor, StrategyExecutionTrace
from .state import GameStateError, GuanDanState


class DanzeroAdvisor:
    """Provide DanZero advice from a manually confirmed game state."""

    strategy_id = "danzero"
    display_name = "DanZero"

    def __init__(
        self,
        profiles_root: Path | str = PROFILES_ROOT,
        profile_name: str = "tencent_daguandan",
    ) -> None:
        self.profiles_root = Path(profiles_root)
        self.profile_name = str(profile_name)
        self._advisor = LocalGuandanAdvisor(
            "danzero",
            danzero_checkpoint=(
                self.profiles_root
                / self.profile_name
                / "models"
                / "danzero"
                / "q_network.ckpt"
            ),
        )

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
