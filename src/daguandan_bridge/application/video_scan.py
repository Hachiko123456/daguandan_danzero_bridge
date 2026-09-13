"""Pure AVI scanner that emits frame evidence for the testing workbench.

The video is the only game-data input.  The optional frame index supplies
timestamps only; absent metadata falls back to ``frame_number / FPS``.  This
module does not inspect any pre-existing session analysis or answer files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from ..storage import atomic_write_json
from ..video_scan_runtime import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_POS_FRAMES,
    Frame,
    copy_frame,
    crop_annotation_record,
    has_pixels,
    default_video_capture,
    is_frame,
    update_array_sha256,
)
from .action_trace_projection import ActionTraceProjector, SEATS
from .action_trace_reconciliation import reconcile_action_trace
from .turn_slot_projection import project_turn_slots


@dataclass(frozen=True)
class VideoScanRequest:
    """All explicit inputs for one scan.

    ``output_directory`` is mandatory and must be outside the source video
    directory.  This prevents an analysis run from adding generated files to a
    user's immutable session-data tree.
    """

    video_path: Path | str
    output_directory: Path | str
    frame_index_path: Path | str | None = None
    session_id: str | None = None

    @property
    def video(self) -> Path:
        return Path(self.video_path).expanduser().resolve()

    @property
    def output(self) -> Path:
        return Path(self.output_directory).expanduser().resolve()

    @property
    def frame_index(self) -> Path | None:
        return None if self.frame_index_path is None else Path(self.frame_index_path).expanduser().resolve()


@dataclass(frozen=True)
class VideoScanResult:
    output_directory: Path
    manifest_path: Path
    observations_path: Path
    action_trace_path: Path
    opening_path: Path
    summary_path: Path
    turn_slots_path: Path
    status: str
    decoded_frames: int
    observation_count: int
    failed_frames: int
    action_count: int
    needs_review_count: int
    fresh_recognitions: int = 0
    reused_recognitions: int = 0
    frame_cache_hits: int = 0
    roi_cache_hits: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "output_directory": str(self.output_directory),
            "manifest_path": str(self.manifest_path),
            "observations_path": str(self.observations_path),
            "action_trace_path": str(self.action_trace_path),
            "opening_path": str(self.opening_path),
            "summary_path": str(self.summary_path),
            "turn_slots_path": str(self.turn_slots_path),
            "status": self.status,
            "decoded_frames": self.decoded_frames,
            "observation_count": self.observation_count,
            "failed_frames": self.failed_frames,
            "action_count": self.action_count,
            "needs_review_count": self.needs_review_count,
            "fresh_recognitions": self.fresh_recognitions,
            "reused_recognitions": self.reused_recognitions,
            "frame_cache_hits": self.frame_cache_hits,
            "roi_cache_hits": self.roi_cache_hits,
        }


@dataclass
class _ScanCache:
    """Run-local exact-input cache.

    The cache is deliberately scoped to one scan.  It never crosses a video,
    profile, or process boundary, so a stale recognition result cannot leak
    into a later run.  Keys are SHA-256 hashes of the exact ndarray bytes (and
    shape/dtype), never perceptual hashes or a multi-frame consensus.
    """

    frame_results: dict[str, tuple[int, dict[str, object]]] | None = None
    context_results: dict[str, tuple[int, dict[str, object]]] | None = None
    region_results: dict[tuple[str, str, str], tuple[int, dict[str, object]]] | None = None
    fresh_recognitions: int = 0
    reused_recognitions: int = 0
    frame_cache_hits: int = 0
    roi_cache_hits: int = 0

    def __post_init__(self) -> None:
        if self.frame_results is None:
            self.frame_results = {}
        if self.context_results is None:
            self.context_results = {}
        if self.region_results is None:
            self.region_results = {}


# Hard cap on retained full-resolution suit-reread frames.  A 1280x720 BGR
# frame is ~2.6 MB, so 24 frames ≈ 64 MB — enough to keep the common short
# unknown-suit window fast, while evicted frames fall back to a targeted,
# pixel-identical seek reread.
_MAX_RETAINED_SUIT_FRAMES = 24


@dataclass
class _PendingSuitFrameStore:
    """Keep a **bounded** window of frames that may be needed for a suit reread.

    A frame is retained from the first hidden-suit observation in a seat's
    visible display until that display becomes empty.  The scanner later uses
    this short in-memory window for ``allow_unknown_suit=False`` rereads.

    Retention is hard-capped at ``_MAX_RETAINED_SUIT_FRAMES``: a full
    1280x720 BGR frame is ~2.6 MB, and an unbounded window used to hold the
    whole run in memory (measured: 245 frames ≈ 0.65 GB per session, doubled
    again by the two parallel batch workers).  When the cap is exceeded the
    OLDEST frame is evicted, and any later request for it is served by the
    targeted seek reread instead — verified pixel-identical to a sequential
    read on both corpus videos, so eviction cannot change scan results.
    """

    frames: dict[int, Frame] | None = None
    pending_seats: set[str] | None = None
    evicted: int = 0

    def __post_init__(self) -> None:
        if self.frames is None:
            self.frames = {}
        if self.pending_seats is None:
            self.pending_seats = set()

    def release(self) -> None:
        """Drop every retained frame (the reread window is over)."""
        self.frames.clear()

    def observe(self, frame_index: int, frame: Frame, observation: dict[str, object]) -> None:
        regions = observation.get("regions")
        if not isinstance(regions, dict):
            return
        for seat, region in regions.items():
            if not isinstance(region, dict) or bool(region.get("is_pass", False)):
                continue
            if _has_unknown_suit(_cards(region.get("cards"))):
                self.pending_seats.add(str(seat))
        if self.pending_seats:
            # Raw BGR pixels are retained exactly.  This is a per-scan,
            # in-memory optimization; no generated input is written beside the
            # source session.
            self.frames[frame_index] = copy_frame(frame)
            while len(self.frames) > _MAX_RETAINED_SUIT_FRAMES:
                self.frames.pop(min(self.frames))
                self.evicted += 1
        for seat in tuple(self.pending_seats):
            region = regions.get(seat)
            cards = _cards(region.get("cards")) if isinstance(region, dict) else ()
            passed = bool(region.get("is_pass", False)) if isinstance(region, dict) else False
            if not cards or passed:
                self.pending_seats.discard(seat)


class VideoActionScanner:
    """Read and recognize every available AVI frame without consensus gating."""

    schema = "guandan.video-scan/1"

    def __init__(
        self,
        recognition_service: Any,
        *,
        projector: ActionTraceProjector | None = None,
        video_capture_factory: Callable[[str], Any] = default_video_capture,
    ) -> None:
        self._recognition = recognition_service
        self._projector = projector or ActionTraceProjector()
        self._video_capture_factory = video_capture_factory

    def scan(
        self,
        request: VideoScanRequest,
        *,
        stop_requested: Callable[[], bool] | None = None,
        on_progress: Callable[[int, int, int], None] | None = None,
        on_action: Callable[[dict[str, object]], None] | None = None,
    ) -> VideoScanResult:
        video = request.video
        output = request.output
        if not video.is_file():
            raise FileNotFoundError(f"找不到对局视频：{video}")
        self._validate_output(video, output)
        output.mkdir(parents=True, exist_ok=True)
        indexed, warnings = _load_index(request.frame_index)
        observations_path = output / "frame_observations.jsonl.gz"
        raw_action_trace_path = output / "raw_action_trace.jsonl"
        action_trace_path = output / "action_trace.jsonl"
        opening_path = output / "opening_candidates.json"
        turn_slots_path = output / "turn_slots.json"
        summary_path = output / "scan_summary.json"
        manifest_path = output / "scan_manifest.json"
        observations: list[dict[str, object]] = []
        decoded_frames = 0
        failed_frames = 0
        cache = _ScanCache()
        pending_suit_frames = _PendingSuitFrameStore()
        frame_ordinals: dict[int, int] = {}
        capture = self._video_capture_factory(str(video))
        fps = 10.0
        stopped = False
        try:
            if not bool(capture.isOpened()):
                raise RuntimeError(f"无法打开对局视频：{video}")
            fps = _positive(capture.get(CAP_PROP_FPS), default=10.0)
            # Resolve the frame total once: reading it per frame is wasted work.
            total_frames = _total_frames(capture, indexed)
            with gzip.open(observations_path, "wt", encoding="utf-8", newline="\n") as stream:
                ordinal = 0
                while True:
                    if stop_requested is not None and stop_requested():
                        stopped = True
                        warnings.append("scan_stopped_by_request")
                        break
                    try:
                        decoded, frame = capture.read()
                    except Exception as exc:
                        warnings.append(f"decode_exception:{type(exc).__name__}:{exc}")
                        break
                    if not decoded or frame is None:
                        break
                    frame_index, timestamp_ms, time_source = _metadata(ordinal, indexed, fps)
                    observation = self._observe(frame, frame_index, timestamp_ms, time_source, cache)
                    _write_line(stream, observation)
                    observations.append(observation)
                    frame_ordinals[frame_index] = ordinal
                    pending_suit_frames.observe(frame_index, frame, observation)
                    decoded_frames += 1
                    failed_frames += int(bool(observation["recognition_errors"]))
                    if on_progress is not None:
                        on_progress(decoded_frames, total_frames, frame_index)
                    ordinal += 1
                for ordinal in range(decoded_frames, len(indexed)) if not stopped else ():
                    frame_index, timestamp_ms, time_source = _metadata(ordinal, indexed, fps)
                    failed = _decode_failure(frame_index, timestamp_ms, time_source)
                    _write_line(stream, failed)
                    observations.append(failed)
                    failed_frames += 1
                if decoded_frames < len(indexed):
                    warnings.append("video_shorter_than_frame_index")
                if indexed and decoded_frames > len(indexed):
                    warnings.append("video_longer_than_frame_index")
        finally:
            capture.release()
        trace = self._projector.project(observations)
        actions = trace["actions"]
        opening = trace["opening"]
        reviews = trace["needs_review"]
        assert isinstance(actions, list) and isinstance(opening, dict) and isinstance(reviews, list)
        # The projector deliberately only describes action windows.  A card
        # whose suit is hidden by a button must not cause a second action to be
        # invented; use the observations already belonging to that window to
        # reread the suit and amend the projected action in place.  This is
        # intentionally a scan-only operation.  It neither reads old session
        # logs nor changes the raw frame-observation artifact.
        try:
            actions = repair_scan_actions(
                actions,
                observations,
                reread=self._suit_reread_callback(
                    video=video,
                    observations=observations,
                    retained_frames=pending_suit_frames.frames,
                    frame_ordinals=frame_ordinals,
                ),
            )
        finally:
            # The reread window is over.  Release the retained frames now rather
            # than holding them until the whole scan returns.
            pending_suit_frames.release()
        raw_actions = [copy.deepcopy(action) for action in actions]
        _write_lines(raw_action_trace_path, raw_actions)
        actions = reconcile_action_trace(raw_actions, observations)
        opening = _refresh_opening(opening, actions)
        reviews = _scan_review_items(actions)
        turn_slots = project_turn_slots(observations, actions)
        if on_action is not None:
            for action in actions:
                on_action(dict(action))
        _write_lines(action_trace_path, actions)
        atomic_write_json(opening_path, opening)
        atomic_write_json(turn_slots_path, turn_slots)
        status = "stopped" if stopped else "complete" if decoded_frames else "failed"
        summary = {
            "schema": "guandan.video-scan-summary/1", "status": status,
            "session_id": request.session_id, "decoded_frames": decoded_frames,
            "observation_count": len(observations), "failed_frames": failed_frames,
            "raw_action_count": len(raw_actions), "action_count": len(actions),
            "needs_review_count": len(reviews) + 1,
            "turn_slot_counts": turn_slots.get("counts", {}),
            "opening_status": opening.get("status"), "warnings": warnings,
            "recognition_cache": _cache_summary(cache),
        }
        atomic_write_json(summary_path, summary)
        atomic_write_json(manifest_path, {
            "schema": self.schema, "status": status,
            "source": {
                "video_path": str(video), "video_sha256": _sha256(video),
                "frame_index_path": str(request.frame_index) if request.frame_index else None,
                "timestamp_mode": "frame_index" if indexed else "video_fps", "fps": fps,
            },
            "artifacts": {"frame_observations": observations_path.name,
                          "raw_action_trace": raw_action_trace_path.name,
                          "action_trace": action_trace_path.name,
                          "opening_candidates": opening_path.name,
                          "turn_slots": turn_slots_path.name,
                          "summary": summary_path.name},
            "counts": {"decoded_frames": decoded_frames, "observations": len(observations),
                        "failed_frames": failed_frames, "raw_actions": len(raw_actions),
                        "actions": len(actions),
                        "needs_review": len(reviews) + 1,
                        "turn_slots": turn_slots.get("counts", {}),
                        "recognition_cache": _cache_summary(cache)},
            "warnings": warnings,
            "input_policy": {"video_only": True, "frame_index_optional": True,
                              "recognition_consensus": "none",
                              "raw_frame_retention": "every_decoded_frame",
                              "cache_policy": "exact_frame_and_play_roi_sha256"},
        })
        return VideoScanResult(output, manifest_path, observations_path, action_trace_path,
                               opening_path, summary_path, turn_slots_path, status, decoded_frames,
                               len(observations), failed_frames, len(actions), len(reviews) + 1,
                               cache.fresh_recognitions, cache.reused_recognitions,
                               cache.frame_cache_hits, cache.roi_cache_hits)

    def _observe(self, frame: Frame, frame_index: int, timestamp_ms: int,
                 timestamp_source: str, cache: _ScanCache) -> dict[str, object]:
        """Recognize one decoded frame, reusing only byte-for-byte identical input.

        Every decoded frame still returns an independent observation row.  A
        reuse merely avoids sending the exact same image or play-region crop to
        a deterministic recognizer again; it is never used to confirm, reject,
        or discard an action.
        """

        frame_hash = _array_sha256(frame)
        cached_frame = cache.frame_results.get(frame_hash)
        if cached_frame is not None:
            source_frame, cached = cached_frame
            cache.frame_cache_hits += 1
            # One broad recognition plus one play recognition per seat were
            # avoided.  Only fully successful observations enter this cache.
            cache.reused_recognitions += 1 + len(SEATS)
            return _reuse_frame_observation(
                cached,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                timestamp_source=timestamp_source,
                frame_hash=frame_hash,
                source_frame=source_frame,
            )

        errors: list[str] = []
        broad: Any | None = None
        context_hash, context_hash_source = self._context_hash(frame, frame_hash)
        cached_context = cache.context_results.get(context_hash)
        recognize = getattr(self._recognition, "recognize", None)
        broad_reused_from: int | None = None
        if cached_context is not None:
            broad_reused_from, broad = cached_context
            broad = copy.deepcopy(broad)
            cache.reused_recognitions += 1
        elif callable(recognize):
            try:
                cache.fresh_recognitions += 1
                broad = recognize(frame, allow_unknown_suit=True)
                broad = _broad_payload(broad)
                cache.context_results[context_hash] = (frame_index, copy.deepcopy(broad))
            except Exception as exc:
                errors.append(f"frame_recognition_error:{type(exc).__name__}:{exc}")
        else:
            errors.append("frame_recognition_error:recognize_not_available")
        wild_rank = _value(broad, "round_level")
        recognize_region = getattr(self._recognition, "recognize_play_region", None)
        regions: dict[str, dict[str, object]] = {}
        region_cache: dict[str, dict[str, object]] = {}
        for seat in SEATS:
            roi_hash, roi_hash_source = self._play_roi_hash(frame, seat, frame_hash)
            cache_key = (seat, str(wild_rank or ""), roi_hash)
            cached_region = cache.region_results.get(cache_key)
            if cached_region is not None:
                source_frame, result = cached_region
                region = copy.deepcopy(result)
                region["recognition_mode"] = "reused"
                region["reused_from_frame"] = source_frame
                region["roi_sha256"] = roi_hash
                region["roi_hash_source"] = roi_hash_source
                regions[seat] = region
                region_cache[seat] = {
                    "mode": "reused", "reused_from_frame": source_frame,
                    "roi_sha256": roi_hash, "roi_hash_source": roi_hash_source,
                }
                cache.reused_recognitions += 1
                cache.roi_cache_hits += 1
                continue
            if not callable(recognize_region):
                region = _empty_region("recognize_play_region_not_available")
                region.update({"recognition_mode": "fresh", "reused_from_frame": None,
                               "roi_sha256": roi_hash, "roi_hash_source": roi_hash_source})
                regions[seat] = region
                region_cache[seat] = {
                    "mode": "fresh", "reused_from_frame": None,
                    "roi_sha256": roi_hash, "roi_hash_source": roi_hash_source,
                }
                errors.append(f"region_recognition_error:{seat}:method_not_available")
                continue
            try:
                cache.fresh_recognitions += 1
                result = recognize_region(frame, seat, wild_rank=wild_rank,
                                          allow_unknown_suit=True, allow_pass=True)
                cards = [str(card) for card in (_value(result, "cards") or ())]
                passed = bool(_value(result, "is_pass"))
                region = {
                    "visible": bool(cards or passed), "is_pass": passed, "cards": cards,
                    "confidence": _positive(_value(result, "confidence"), default=0.0),
                    "source": str(_value(result, "source") or ""),
                    "diagnostics": [str(item) for item in (_value(result, "diagnostics") or ())],
                    "suit_options": [list(item) for item in (_value(result, "suit_options") or ())],
                    "recognition_mode": "fresh", "reused_from_frame": None,
                    "roi_sha256": roi_hash, "roi_hash_source": roi_hash_source,
                }
                regions[seat] = region
                # Cache only a successful recognizer call.  In particular, an
                # exception must not turn into a permanent cached empty result.
                cache.region_results[cache_key] = (frame_index, copy.deepcopy(region))
                region_cache[seat] = {
                    "mode": "fresh", "reused_from_frame": None,
                    "roi_sha256": roi_hash, "roi_hash_source": roi_hash_source,
                }
            except Exception as exc:
                region = _empty_region("recognition_exception")
                region.update({"recognition_mode": "fresh", "reused_from_frame": None,
                               "roi_sha256": roi_hash, "roi_hash_source": roi_hash_source})
                regions[seat] = region
                region_cache[seat] = {
                    "mode": "fresh", "reused_from_frame": None,
                    "roi_sha256": roi_hash, "roi_hash_source": roi_hash_source,
                }
                errors.append(f"region_recognition_error:{seat}:{type(exc).__name__}:{exc}")
        observation = {
            "schema": "guandan.video-frame-observation/1", "frame_index": frame_index,
            "timestamp_ms": timestamp_ms, "timestamp_source": timestamp_source,
            "decode_ok": True, "recognition_errors": errors,
            "frame_sha256": frame_hash, "recognition_mode": "fresh",
            "reused_from_frame": None,
            "roi_hashes": {seat: str(region.get("roi_sha256") or "")
                           for seat, region in regions.items()},
            "recognition_cache": {
                "frame": {"mode": "fresh", "reused_from_frame": None,
                          "sha256": frame_hash},
                "context": {"mode": "reused" if broad_reused_from is not None else "fresh",
                             "reused_from_frame": broad_reused_from,
                             "sha256": context_hash,
                             "hash_source": context_hash_source},
                "regions": region_cache,
            },
            "opening": {"round_level": str(wild_rank) if wild_rank else None,
                         "my_hand": [str(card) for card in (_value(broad, "my_hand") or ())],
                        "lead_player_signal": _value(broad, "lead_player"),
                        "current_player_signal": _value(broad, "current_player"),
                        "unresolved_fields": [str(item) for item in (_value(broad, "unresolved_fields") or ())],
                        "diagnostics": [str(item) for item in (_value(broad, "diagnostics") or ())]},
            "buttons": [str(item) for item in (_value(broad, "buttons") or ())], "regions": regions,
        }
        # A whole-frame hit is safe only when all recognitions completed.  If a
        # decoder/service error is transient, the next identical frame retries
        # recognition instead of silently replaying the failure forever.
        if not errors:
            cache.frame_results[frame_hash] = (frame_index, copy.deepcopy(observation))
        return observation

    def _suit_reread_callback(
        self,
        *,
        video: Path,
        observations: list[dict[str, object]],
        retained_frames: dict[int, Frame] | None,
        frame_ordinals: dict[int, int],
    ) -> Callable[[str, int], tuple[tuple[str, ...], float] | None]:
        """Return a local, AVI-only full-suit reread for pending action frames.

        Normal scanning intentionally permits ``?`` so an occluded rank is not
        discarded.  This callback performs the second read with that permission
        disabled, but only after projection identifies an unknown-suit action
        and only for frames in its own evidence window.
        """

        saved_frames = retained_frames or {}
        source_frames: dict[int, Frame] = {}
        levels = {
            int(raw["frame_index"]): _value(raw.get("opening"), "round_level")
            for raw in observations
            if isinstance(raw, dict) and isinstance(raw.get("opening"), dict)
            and _safe_frame_index(raw) is not None
        }
        recognize_region = getattr(self._recognition, "recognize_play_region", None)

        def reread(seat: str, frame_index: int) -> tuple[tuple[str, ...], float] | None:
            if not callable(recognize_region):
                return None
            frame = saved_frames.get(frame_index)
            if frame is None:
                frame = source_frames.get(frame_index)
            if frame is None:
                ordinal = frame_ordinals.get(frame_index)
                if ordinal is None:
                    return None
                frame = _read_source_frame(self._video_capture_factory, video, ordinal)
                if frame is None:
                    return None
                source_frames[frame_index] = frame
            try:
                result = recognize_region(
                    frame,
                    seat,
                    wild_rank=levels.get(frame_index),
                    allow_unknown_suit=False,
                    allow_pass=True,
                )
            except Exception:
                # The primary pass is still retained as evidence. A failed
                # optional reread must not make a completed AVI scan fail.
                return None
            if bool(_value(result, "is_pass")):
                return None
            cards = _cards(_value(result, "cards"))
            return cards, _positive(_value(result, "confidence"), default=0.0)

        return reread

    def _play_roi_hash(self, frame: Frame, seat: str, frame_hash: str) -> tuple[str, str]:
        """Return the exact play-ROI hash, falling back safely for adapters.

        ``ScreenshotRecognitionService`` exposes ``play_roi``.  Lightweight
        test/different-profile adapters might not, in which case full-frame
        caching remains correct (but naturally less selective) rather than
        risking an incorrectly cropped cache key.
        """

        cropper = getattr(self._recognition, "play_roi", None)
        if not callable(cropper):
            return frame_hash, "frame_fallback"
        try:
            roi = cropper(frame, seat)
            if not has_pixels(roi):
                return frame_hash, "frame_fallback"
            return _array_sha256(roi), "play_roi"
        except Exception:
            return frame_hash, "frame_fallback"

    def _context_hash(self, frame: Frame, frame_hash: str) -> tuple[str, str]:
        """Hash non-play configured regions for the broad initial-state read.

        The generic ``recognize`` call historically scans the whole screenshot,
        including play regions.  When a recognizer exposes configured regions,
        the stable context signature lets us reuse that broad *initial-state*
        payload while play ROIs change.  It does not claim that the broad
        recognizer itself is ROI-aware; adapters without region metadata fall
        back to the exact full-frame hash, where reuse is still unquestionably
        safe.
        """

        service = getattr(self._recognition, "annotation_service", None)
        list_regions = getattr(service, "list_regions", None)
        if not callable(list_regions):
            return frame_hash, "frame_fallback"
        try:
            records = tuple(list_regions())
        except Exception:
            return frame_hash, "frame_fallback"
        parts: list[tuple[str, str]] = []
        for record in records:
            if str(getattr(record, "role", "")).lower() == "play":
                continue
            name = str(getattr(record, "name", ""))
            crop = crop_annotation_record(frame, record)
            if crop is None:
                continue
            parts.append((name, _array_sha256(crop)))
        if not parts:
            return frame_hash, "frame_fallback"
        digest = hashlib.sha256()
        for name, value in sorted(parts):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest(), "non_play_rois"

    @staticmethod
    def _validate_output(video: Path, output: Path) -> None:
        if output.exists() and output.is_file():
            raise ValueError(f"扫描输出路径不是目录：{output}")
        if output == video.parent or output in video.parents or video.parent in output.parents:
            raise ValueError("扫描输出目录必须位于源视频目录之外")


def scan_video(
    request: VideoScanRequest,
    recognition_service: Any,
    *,
    projector: ActionTraceProjector | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_action: Callable[[dict[str, object]], None] | None = None,
) -> VideoScanResult:
    """Convenience entrypoint for UI and corpus callers."""
    return VideoActionScanner(recognition_service, projector=projector).scan(
        request,
        stop_requested=stop_requested,
        on_progress=on_progress,
        on_action=on_action,
    )


def repair_scan_actions(
    actions: Iterable[dict[str, object]],
    observations: Iterable[dict[str, object]],
    *,
    reread: Callable[[str, int], tuple[tuple[str, ...], float] | None] | None = None,
) -> list[dict[str, object]]:
    """Resolve unknown suits using only evidence already assigned to an action.

    This is deliberately a *post-projection* scan repair.  The projection owns
    action boundaries; this function never searches beyond its
    ``evidence_frames``, never adds an action, and never changes raw frame
    observations.  It also deliberately limits itself to suit correction: a
    candidate has to have the same number and multiset of ranks as the pending
    action.  When ``reread`` is supplied, every frame of a pending action
    window is recognized again with unknown suits disabled; one complete frame
    is enough to repair because this is a second read of a clearer image, not
    a multi-frame consensus rule.
    """

    readings = _window_readings(observations)
    repaired: list[dict[str, object]] = []
    for raw_action in actions:
        action = copy.deepcopy(raw_action)
        actor = str(action.get("actor") or "")
        cards = _cards(action.get("cards"))
        if bool(action.get("is_pass", False)) or not actor or not cards:
            repaired.append(action)
            continue

        window = _action_window_readings(action, actor, readings)
        # A projector may already have resolved an earlier hidden-suit frame
        # while choosing a complete candidate. Preserve that result and its
        # provenance; this pass is only for actions still marked pending.
        had_unknown_evidence = (
            _has_unknown_suit(cards)
            or bool(action.get("cards_before_repair"))
            or any(
                _has_unknown_suit(_cards(variant.get("cards")))
                for variant in action.get("observed_variants", ())
                if isinstance(variant, dict)
            )
        )
        pending = had_unknown_evidence or action.get("repair_status") == "unresolved"
        if not pending:
            repaired.append(action)
            continue

        unknown_readings = [item for item in window if _has_unknown_suit(item[1])]

        reference = _repair_reference_cards(cards, unknown_readings)
        evidence_frames = _action_evidence_frames(action)
        reread_attempts: list[int] = []
        if reread is None:
            complete = [
                item for item in window
                if _is_complete_suit_repair(item[1], reference)
            ]
        else:
            complete = []
            for frame in evidence_frames:
                reread_attempts.append(frame)
                candidate = reread(actor, frame)
                if candidate is None:
                    continue
                candidate_cards, confidence = candidate
                if _is_complete_suit_repair(candidate_cards, reference):
                    complete.append((frame, candidate_cards, confidence))
        if not complete and reread is not None and not _has_unknown_suit(cards):
            # The first pass may already have found a complete frame. Keep it
            # as a safe fallback if the optional strict reread fails, while
            # still recording that the second pass was attempted.
            action["repair_status"] = "resolved"
            action["repair_reason"] = "fallback_first_pass_complete_suits"
            action["repair_evidence_frames"] = list(dict.fromkeys(
                [frame for frame, _cards, _confidence in unknown_readings] + reread_attempts
            ))
            action["suit_reread_attempt_frames"] = reread_attempts
            action["suit_reread_mode"] = "allow_unknown_suit_false"
            _clear_unknown_suit_review(action)
            repaired.append(action)
            continue
        if not complete:
            action["cards_before_repair"] = list(reference)
            action["repair_status"] = "unresolved"
            action["repair_reason"] = "no_complete_suit_reading_in_action_window"
            action["repair_frame"] = None
            action["repair_evidence_frames"] = [frame for frame, _cards, _confidence in unknown_readings]
            if reread is not None:
                action["suit_reread_attempt_frames"] = reread_attempts
                action["suit_reread_mode"] = "allow_unknown_suit_false"
            _mark_unknown_suit_for_review(action)
            repaired.append(action)
            continue

        # A clearer frame is preferred by confidence.  When confidence is
        # equal, prefer the later frame because buttons/animations normally
        # move away as the display settles.  All such frames remain available
        # verbatim in frame_observations; this choice only sets action metadata.
        repair_frame, resolved_cards, _confidence = max(
            complete,
            key=lambda item: (item[2], item[0]),
        )
        action["cards_before_repair"] = list(reference)
        action["cards"] = sorted(resolved_cards)
        action["best_frame"] = repair_frame
        action["repair_status"] = "resolved"
        action["repair_reason"] = (
            "second_pass_complete_suits" if reread is not None else "later_frame_complete_suits"
        )
        action["repair_frame"] = repair_frame
        action["repair_evidence_frames"] = list(dict.fromkeys(
            [frame for frame, _cards, _confidence in unknown_readings] + [repair_frame]
        ))
        if reread is not None:
            action["suit_reread_attempt_frames"] = reread_attempts
            action["suit_reread_mode"] = "allow_unknown_suit_false"
        _clear_unknown_suit_review(action)
        repaired.append(action)
    return repaired


def _window_readings(
    observations: Iterable[dict[str, object]],
) -> dict[int, dict[str, tuple[tuple[str, ...], float]]]:
    """Index raw play-region results by frame and seat without modifying them."""

    indexed: dict[int, dict[str, tuple[tuple[str, ...], float]]] = {}
    for raw in observations:
        if not isinstance(raw, dict) or not raw.get("decode_ok", False):
            continue
        try:
            frame = int(raw["frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        regions = raw.get("regions")
        if not isinstance(regions, dict):
            continue
        seats: dict[str, tuple[tuple[str, ...], float]] = {}
        for seat, region in regions.items():
            if not isinstance(region, dict) or bool(region.get("is_pass", False)):
                continue
            cards = _cards(region.get("cards"))
            if cards:
                seats[str(seat)] = (cards, _positive(region.get("confidence"), default=0.0))
        if seats:
            indexed[frame] = seats
    return indexed


def _action_window_readings(
    action: dict[str, object],
    actor: str,
    readings: dict[int, dict[str, tuple[tuple[str, ...], float]]],
) -> list[tuple[int, tuple[str, ...], float]]:
    frames = action.get("evidence_frames")
    if not isinstance(frames, list):
        return []
    result: list[tuple[int, tuple[str, ...], float]] = []
    for raw_frame in frames:
        try:
            frame = int(raw_frame)
        except (TypeError, ValueError):
            continue
        seat_reading = readings.get(frame, {}).get(actor)
        if seat_reading is not None:
            cards, confidence = seat_reading
            result.append((frame, cards, confidence))
    return result


def _action_evidence_frames(action: dict[str, object]) -> list[int]:
    frames = action.get("evidence_frames")
    if not isinstance(frames, list):
        return []
    result: list[int] = []
    for value in frames:
        try:
            frame = int(value)
        except (TypeError, ValueError):
            continue
        if frame not in result:
            result.append(frame)
    return result


def _repair_reference_cards(
    action_cards: tuple[str, ...],
    unknown_readings: list[tuple[int, tuple[str, ...], float]],
) -> tuple[str, ...]:
    """Use the pending action's cards, or its strongest unknown reading.

    Projection normally selects the strongest variant already.  The fallback is
    useful for compatible projectors that retain a previous partial display as
    ``cards`` while exposing a fuller unknown-suit result in the same evidence
    window.
    """

    candidates = [action_cards, *(cards for _frame, cards, _confidence in unknown_readings)]
    return max(candidates, key=lambda cards: (len(cards), -_unknown_suit_count(cards)))


def _is_complete_suit_repair(candidate: tuple[str, ...], reference: tuple[str, ...]) -> bool:
    return (
        bool(candidate)
        and not _has_unknown_suit(candidate)
        and len(candidate) == len(reference)
        and _rank_multiset(candidate) == _rank_multiset(reference)
    )


def _cards(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()


def _has_unknown_suit(cards: tuple[str, ...]) -> bool:
    return any(card.endswith("?") for card in cards)


def _unknown_suit_count(cards: tuple[str, ...]) -> int:
    return sum(card.endswith("?") for card in cards)


def _rank_multiset(cards: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for card in cards:
        if card in {"small_joker", "big_joker"}:
            rank = card
        elif card.endswith(("S", "H", "C", "D", "?")):
            rank = card[:-1]
        else:
            rank = card
        counts[rank] = counts.get(rank, 0) + 1
    return tuple(sorted(counts.items()))


def _mark_unknown_suit_for_review(action: dict[str, object]) -> None:
    uncertainty = [str(item) for item in action.get("uncertainty", ())]
    if "unknown_suit" not in uncertainty:
        uncertainty.append("unknown_suit")
    action["uncertainty"] = uncertainty
    action["review_status"] = "needs_review"


def _clear_unknown_suit_review(action: dict[str, object]) -> None:
    uncertainty = [str(item) for item in action.get("uncertainty", ()) if str(item) != "unknown_suit"]
    action["uncertainty"] = uncertainty
    action["review_status"] = "needs_review" if uncertainty else "unverified"


def _refresh_opening(opening: dict[str, object], actions: list[dict[str, object]]) -> dict[str, object]:
    """Keep opening candidates aligned with repaired first-action card data."""

    refreshed = copy.deepcopy(opening)
    candidates = refreshed.get("candidates")
    if not isinstance(candidates, list):
        return refreshed
    by_start: dict[tuple[str, int], dict[str, object]] = {}
    for action in actions:
        if bool(action.get("is_pass", False)):
            continue
        key = _opening_action_key(action, actor_field="actor", frame_field="frame_start")
        if key is not None:
            by_start[key] = action
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = _opening_action_key(candidate, actor_field="seat", frame_field="first_action_frame")
        if key is None:
            continue
        action = by_start.get(key)
        if action is not None:
            candidate["cards"] = list(_cards(action.get("cards")))
    return refreshed


def _opening_action_key(
    item: dict[str, object],
    *,
    actor_field: str,
    frame_field: str,
) -> tuple[str, int] | None:
    try:
        return str(item.get(actor_field) or ""), int(item.get(frame_field))
    except (TypeError, ValueError):
        return None


def _scan_review_items(actions: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "type": "action_uncertainty",
            "action_id": action.get("action_id"),
            "actor": action.get("actor"),
            "frames": action.get("evidence_frames", []),
            "reasons": action.get("uncertainty", []),
        }
        for action in actions
        if action.get("uncertainty")
    ]


def _load_index(path: Path | None) -> tuple[list[dict[str, int]], list[str]]:
    if path is None:
        return [], []
    if not path.is_file():
        return [], ["frame_index_unavailable"]
    records: list[dict[str, int]] = []
    warnings: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for ordinal, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    records.append({"frame_index": int(raw.get("frame_index", ordinal)),
                                    "timestamp_ms": int(raw.get("monotonic_ms", raw.get("timestamp_ms", 0)))})
                except (TypeError, ValueError, json.JSONDecodeError):
                    warnings.append(f"frame_index_parse_error:{ordinal + 1}")
    except OSError as exc:
        return [], [f"frame_index_read_error:{type(exc).__name__}"]
    return records, warnings


def _metadata(ordinal: int, records: list[dict[str, int]], fps: float) -> tuple[int, int, str]:
    if ordinal < len(records):
        record = records[ordinal]
        return record["frame_index"], record["timestamp_ms"], "frame_index"
    return ordinal, round(ordinal * 1000 / fps), "video_fps"


def _total_frames(capture: Any, indexed: list[dict[str, int]]) -> int:
    if indexed:
        return len(indexed)
    try:
        return max(0, int(capture.get(CAP_PROP_FRAME_COUNT)))
    except Exception:
        return 0


def _decode_failure(frame_index: int, timestamp_ms: int, source: str) -> dict[str, object]:
    return {"schema": "guandan.video-frame-observation/1", "frame_index": frame_index,
            "timestamp_ms": timestamp_ms, "timestamp_source": source, "decode_ok": False,
            "recognition_errors": ["decode_error:video_ended_before_frame_index"],
            "opening": {"round_level": None, "my_hand": [], "lead_player_signal": None,
                        "current_player_signal": None, "unresolved_fields": [], "diagnostics": []},
            "buttons": [], "regions": {seat: _empty_region("decode_unavailable") for seat in SEATS}}


def _empty_region(reason: str) -> dict[str, object]:
    return {"visible": False, "is_pass": False, "cards": [], "confidence": 0.0,
            "source": reason, "diagnostics": [reason], "suit_options": []}


def _broad_payload(value: Any) -> dict[str, object]:
    """Keep only fields consumed by the pure scan's opening observation.

    This intentionally omits the broad recognizer's legacy event list.  Play
    evidence is produced by the independently cached per-seat calls below,
    so retaining broad events would both duplicate work and risk presenting a
    stale play event when only a play ROI changes.
    """

    return {
        "round_level": _value(value, "round_level"),
        "my_hand": tuple(_value(value, "my_hand") or ()),
        "lead_player": _value(value, "lead_player"),
        "current_player": _value(value, "current_player"),
        "unresolved_fields": tuple(_value(value, "unresolved_fields") or ()),
        "diagnostics": tuple(_value(value, "diagnostics") or ()),
        "buttons": tuple(_value(value, "buttons") or ()),
    }




def _safe_frame_index(raw: dict[str, object]) -> int | None:
    try:
        return int(raw["frame_index"])
    except (KeyError, TypeError, ValueError):
        return None


def _read_source_frame(
    capture_factory: Callable[[str], Any],
    video: Path,
    ordinal: int,
) -> Any | None:
    """Read one missing evidence frame only when memory did not retain it."""

    capture = capture_factory(str(video))
    try:
        if not bool(capture.isOpened()):
            return None
        setter = getattr(capture, "set", None)
        if not callable(setter):
            return None
        setter(CAP_PROP_POS_FRAMES, ordinal)
        decoded, frame = capture.read()
        return frame if decoded and is_frame(frame) else None
    except Exception:
        return None
    finally:
        try:
            capture.release()
        except Exception:
            pass


def _array_sha256(array: Frame) -> str:
    """Hash the exact decoded frame through the media runtime adapter."""

    digest = hashlib.sha256()
    update_array_sha256(array, digest)
    return digest.hexdigest()


def _reuse_frame_observation(
    cached: dict[str, object],
    *,
    frame_index: int,
    timestamp_ms: int,
    timestamp_source: str,
    frame_hash: str,
    source_frame: int,
) -> dict[str, object]:
    """Copy cached recognition payload while preserving this frame's identity."""

    observation = copy.deepcopy(cached)
    observation.update({
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "timestamp_source": timestamp_source,
        "frame_sha256": frame_hash,
        "recognition_mode": "reused",
        "reused_from_frame": source_frame,
    })
    cache_info = observation.get("recognition_cache")
    if not isinstance(cache_info, dict):
        cache_info = {}
    cache_info["frame"] = {
        "mode": "reused", "reused_from_frame": source_frame,
        "sha256": frame_hash,
    }
    regions = observation.get("regions")
    region_info: dict[str, object] = {}
    if isinstance(regions, dict):
        for seat, raw_region in regions.items():
            if not isinstance(raw_region, dict):
                continue
            raw_region["recognition_mode"] = "reused"
            raw_region["reused_from_frame"] = source_frame
            region_info[str(seat)] = {
                "mode": "reused", "reused_from_frame": source_frame,
                "roi_sha256": raw_region.get("roi_sha256"),
                "roi_hash_source": raw_region.get("roi_hash_source"),
            }
    cache_info["regions"] = region_info
    observation["recognition_cache"] = cache_info
    return observation


def _cache_summary(cache: _ScanCache) -> dict[str, int]:
    return {
        "fresh_recognitions": cache.fresh_recognitions,
        "reused_recognitions": cache.reused_recognitions,
        "frame_cache_hits": cache.frame_cache_hits,
        "roi_cache_hits": cache.roi_cache_hits,
    }


def _value(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None) if value is not None else None


def _positive(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _write_line(handle: Any, value: dict[str, object]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _write_lines(path: Path, values: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            _write_line(handle, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "VideoActionScanner",
    "VideoScanRequest",
    "VideoScanResult",
    "repair_scan_actions",
    "scan_video",
]
