"""Persistent worker entry point for live-v2 screenshot recognition."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

from ..annotation_service import AnnotationService
from ..application.live_v2_frame_pipeline import LiveV2FramePipeline
from ..application.live_v2_frame_types import FramePipelineConfig, SurfaceProbeConfig
from ..application.live_v2_vision_protocol import (
    VisionWorkerConfig,
    VisionWorkerPayload,
    VisionWorkerSuccess,
)
from ..application.live_v2_worker_protocol import WorkerRequest
from ..live_v2.identity import FrameIdentity, Seat
from ..recognition_service import ScreenshotRecognitionService
from ..template_service import TemplateService


@dataclass(frozen=True, slots=True)
class _PipelineKey:
    profile_root: str
    profile_name: str
    session_id: str
    capture_generation: int
    roi_version: str
    source_id: str
    pipeline: FramePipelineConfig
    surface: SurfaceProbeConfig
    diagnostic_tracing: bool


_PIPELINES: dict[_PipelineKey, LiveV2FramePipeline] = {}


def run_live_v2_vision_worker(request: WorkerRequest) -> VisionWorkerSuccess:
    """Recognize one full frame using a stream-bound cached pipeline."""

    started = perf_counter()
    payload = request.payload
    if not isinstance(payload, VisionWorkerPayload):
        raise TypeError("vision worker requires VisionWorkerPayload")
    identity = payload.identity
    version, frame = identity.version, identity.frame
    if (
        request.session_id,
        request.capture_generation,
        request.state_revision,
        request.request_sequence,
    ) != (
        version.session_id,
        version.capture_generation,
        version.state_revision,
        identity.request_sequence,
    ):
        raise ValueError("worker request identity does not match vision payload")

    key = _key(payload.config, frame)
    pipeline = _PIPELINES.get(key)
    cache_hit = pipeline is not None
    if pipeline is None:
        pipeline = _build_pipeline(payload.config)
        _PIPELINES[key] = pipeline
    result = pipeline.process_frame(
        payload.image,
        frame=frame,
        version=version,
        wild_rank=payload.wild_rank,
        expected_seat=payload.expected_seat,
        opening_lead_seat=payload.opening_lead_seat,
        formal_action_boundary=payload.formal_action_boundary,
        repair_seats=payload.repair_seats,
    )
    return VisionWorkerSuccess(
        identity=identity,
        pipeline_result=result,
        worker_pid=os.getpid(),
        elapsed_ms=(perf_counter() - started) * 1000.0,
        cache_hit=cache_hit,
    )


def _key(config: VisionWorkerConfig, frame: FrameIdentity) -> _PipelineKey:
    return _PipelineKey(
        str(Path(config.profile_root).resolve()),
        config.profile_name,
        frame.session_id,
        frame.capture_generation,
        frame.roi_version,
        frame.source_id,
        config.pipeline,
        config.surface,
        config.diagnostic_tracing,
    )


def _build_pipeline(config: VisionWorkerConfig) -> LiveV2FramePipeline:
    root = Path(config.profile_root)
    annotation = AnnotationService(root, config.profile_name)
    templates = TemplateService(root, config.profile_name)
    recognition = ScreenshotRecognitionService(
        annotation,
        templates,
        diagnostic_tracing=config.diagnostic_tracing,
    )
    return LiveV2FramePipeline(
        recognition,
        config=config.pipeline,
        surface_config=config.surface,
    )


__all__ = ["run_live_v2_vision_worker"]
