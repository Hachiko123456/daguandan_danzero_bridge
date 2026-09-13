"""Production composition root for one live-v2 session runtime."""

from __future__ import annotations

from typing import Callable, Literal

from ..advisor_strategy import normalize_advisor_strategy
from ..application.live_v2_advice_protocol import AdviceWorkerConfig
from ..application.live_v2_session_runtime import LiveV2SessionRuntime
from ..application.live_v2_vision_protocol import VisionWorkerConfig
from ..application.live_v2_frame_types import FramePipelineConfig
from ..application.ports import RecognitionPort, RecordingPort, SessionPersistencePort
from ..domain.live_runtime import LiveUpdate
from ..live_v2.identity import VersionIdentity
from .live_v2_advice_service_factory import create_live_v2_advice_runtime
from .live_v2_rule_session import ProductionRuleSession
from .live_v2_vision_service_factory import build_live_v2_vision_runtime


def build_production_live_v2_runtime(
    *,
    store: SessionPersistencePort,
    recorder: RecordingPort,
    recognizer: RecognitionPort,
    profiles_root: object,
    profile_name: str,
    advisor_backend: str,
    on_update: Callable[[LiveUpdate], None] | None,
    vision_factory_override: Callable[[VersionIdentity], object] | None = None,
    advice_runtime_factory_override: Callable[[VersionIdentity], object] | None = None,
    vision_delivery: Literal["latest", "synchronous"] = "latest",
) -> LiveV2SessionRuntime:
    """Wire durable rules, isolated vision and isolated FableDan advice."""

    if vision_delivery not in {"latest", "synchronous"}:
        raise ValueError("vision_delivery must be latest or synchronous")
    if getattr(store, "is_started", False) is not True:
        raise RuntimeError("production live-v2 composition requires an already-started store")

    vision_config = VisionWorkerConfig(
        profile_root=str(profiles_root),
        profile_name=profile_name,
        pipeline=FramePipelineConfig(strict_current_seat_only=True),
    )
    backend = normalize_advisor_strategy(advisor_backend)
    advice_config = AdviceWorkerConfig(
        profile_root=str(profiles_root),
        advisor_backend=backend,
        profile_name=profile_name,
        fabledan_runtime_policy="model_required",
    )

    def vision_factory(version):
        if vision_factory_override is not None:
            return vision_factory_override(version)
        return build_live_v2_vision_runtime(
            vision_config,
            session_id=version.session_id,
            capture_generation=version.capture_generation,
            state_revision=version.state_revision,
        )

    def advice_factory(version):
        if advice_runtime_factory_override is not None:
            return advice_runtime_factory_override(version)
        return create_live_v2_advice_runtime(advice_config, version)

    return LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store),
        store=store,
        recorder=recorder,
        recognition_service=recognizer,
        vision_factory=vision_factory,
        advice_runtime_factory=advice_factory,
        on_update=on_update,
        synchronous_vision=vision_delivery == "synchronous",
    )


__all__ = ["build_production_live_v2_runtime"]
