"""Projection of a verified visual cannot-beat control into local advice."""

from enum import Enum

from ..domain.advice import AdviceResult
from ..domain.live_runtime import AdviceRequestKey, LiveAdvice
from ..live_v2.game_state import TrustedGameSnapshot
from ..live_v2.results import AdviceOpportunity


class LocalPassOpportunityPhase(str, Enum):
    WAITING_HINT = "waiting_hint"
    MODEL_SUBMITTED = "model_submitted"
    TERMINAL = "terminal"


def local_pass_advice(
    snapshot: TrustedGameSnapshot,
    opportunity: AdviceOpportunity,
) -> LiveAdvice:
    key = AdviceRequestKey(
        snapshot.version.session_id,
        snapshot.version.turn_index + 1,
        snapshot.version.state_revision,
    )
    return LiveAdvice(
        key=key,
        status="ready",
        advice=AdviceResult(
            strategy="local_cannot_beat",
            cards=(),
            play_type="Pass",
            is_pass=True,
            state_revision=snapshot.version.state_revision,
            elapsed_ms=0.0,
            request_id=opportunity.opportunity_id,
        ),
        visible=True,
    )


__all__ = ["LocalPassOpportunityPhase", "local_pass_advice"]
