"""Real-process Advice latency acceptance under Vision/recording load."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from daguandan_bridge.application.live_v2_advice_protocol import (
    AdviceRuntimeStatus, AdviceWorkerConfig,
)
from daguandan_bridge.application.live_v2_vision_protocol import (
    VisionRuntimeStatus, VisionWorkerConfig,
)
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.infrastructure.live_v2_advice_service_factory import (
    create_live_v2_advice_runtime,
)
from daguandan_bridge.infrastructure.live_v2_vision_service_factory import (
    build_live_v2_vision_runtime,
)
from daguandan_bridge.infrastructure.process_session_recorder import ProcessSessionRecorder
from daguandan_bridge.live_v2.events import ActionKind
from daguandan_bridge.live_v2.game_state import (
    GameAction, SeatCardCount, TrustedGameSnapshot,
)
from daguandan_bridge.live_v2.identity import (
    FrameIdentity, Seat, StateVersion, VersionIdentity,
)
from daguandan_bridge.live_v2.results import (
    AdviceOpportunity, OpportunityReason, OpportunityStatus,
)


VIDEO = PROFILES_ROOT / (
    "tencent_daguandan/sessions/game_20260814_004447_aab3dc/video/game.avi"
)
HAND = (
    "10S", "3C", "4H", "4H", "5C", "5D", "6C", "6H", "7D",
    "8C", "8D", "8H", "9C", "9H", "9H", "9S", "AH", "AS",
    "JC", "JC", "JD", "JH", "KC", "KS", "QC", "QH", "small_joker",
)
PREFIX = (
    (Seat.LEFT, ActionKind.PLAY, ("5C", "5H")),
    (Seat.SELF, ActionKind.PLAY, ("9C", "9H", "9H", "9S")),
    (Seat.RIGHT, ActionKind.PASS, ()),
    (Seat.OPPOSITE, ActionKind.PASS, ()),
    (Seat.LEFT, ActionKind.PASS, ()),
)
ROUNDS, ADVICE_DEADLINE_S, TOTAL_BUDGET_S = 10, 3.0, 40.0


@dataclass(frozen=True)
class AdviceTiming:
    sequence: int
    public_turn: int
    status: str
    worker_pid: int | None
    worker_generation: int
    advisor_cache_hit: bool
    total_ms: float
    submit_ms: float
    child_model_ms: float
    pre_model_ms: float
    failure_stage: str
    failure_code: str
    message: str
    worker_timing: dict[str, int | None]


def _action(session: str, index: int, seat: Seat, kind: ActionKind,
            cards: tuple[str, ...]) -> GameAction:
    first = FrameIdentity(
        session, 1, index * 2, 10_000 + index * 100 - 20, "roi", "truth",
    )
    last = FrameIdentity(
        session, 1, index * 2 + 1, 10_000 + index * 100, "roi", "truth",
    )
    return GameAction(
        f"exact-prefix-action-{index}",
        StateVersion(session, index - 1, index - 1),
        StateVersion(session, index, index),
        seat, kind, cards, tuple((card,) for card in cards), 1,
        (f"exact-evidence-{index}",), first, last, 1.0, last.captured_ms,
    )


def _snapshot(public_turn: int) -> TrustedGameSnapshot:
    if public_turn not in {2, 6}:
        raise ValueError("only exact turn-2/turn-6 prefixes are supported")
    session, count = "real-concurrent-advice", public_turn - 1
    actions = tuple(
        _action(session, index, *spec)
        for index, spec in enumerate(PREFIX[:count], 1)
    )
    hand = list(HAND)
    remaining = Counter({seat: 27 for seat in Seat})
    for action in actions:
        if action.kind is not ActionKind.PLAY:
            continue
        remaining[action.seat] -= len(action.cards)
        if action.seat is Seat.SELF:
            for card in action.cards:
                hand.remove(card)
    return TrustedGameSnapshot(
        VersionIdentity(session, 1, count, count, count), "6", "6",
        1 if public_turn == 2 else 2, Seat.SELF,
        Seat.LEFT if public_turn == 2 else Seat.SELF, tuple(hand), actions,
        actions if public_turn == 2 else (),
        tuple(SeatCardCount(seat, remaining[seat]) for seat in Seat),
        (), True, False, 10_000 + count * 100,
    )


def _opportunity(snapshot: TrustedGameSnapshot, sequence: int) -> AdviceOpportunity:
    turn = snapshot.version.turn_index + 1
    return AdviceOpportunity(
        f"concurrent-turn-{turn}-request-{sequence}", snapshot.version, Seat.SELF,
        OpportunityStatus.READY, OpportunityReason.TRUSTED_STATE,
        snapshot.captured_ms, snapshot.captured_ms,
    )


def _frames(count: int = 12) -> tuple[np.ndarray, ...]:
    if not VIDEO.is_file():
        pytest.skip(f"missing historical AVI: {VIDEO}")
    capture, frames = cv2.VideoCapture(str(VIDEO)), []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 220)
        for _ in range(count):
            ok, image = capture.read()
            if not ok or image is None:
                break
            frames.append(image)
    finally:
        capture.release()
    if len(frames) < count:
        pytest.skip(f"historical AVI decoded only {len(frames)}/{count} frames")
    assert {frame.shape for frame in frames} == {(720, 1280, 3)}
    return tuple(frames)


def _stage(worker_timing: dict[str, int | None]) -> str:
    if worker_timing.get("send_finished_ms") is None:
        return "send_not_completed"
    if worker_timing.get("child_started_ms") is None:
        return "child_not_started"
    return "model_time"


def test_real_advice_under_vision_and_recording_load(
    tmp_path: Path, record_property,
) -> None:
    frames = _frames()
    snapshots = {turn: _snapshot(turn) for turn in (2, 6)}
    advice = create_live_v2_advice_runtime(
        AdviceWorkerConfig(
            str(PROFILES_ROOT), "fabledan", profile_name="tencent_daguandan",
            fabledan_runtime_policy="model_required",
        ), snapshots[2].version,
    )
    vision = build_live_v2_vision_runtime(
        VisionWorkerConfig(str(PROFILES_ROOT)),
        session_id="real-concurrent-vision", capture_generation=1,
    )
    recording_dir = tmp_path / "recording"
    recording_dir.mkdir()
    recorder = ProcessSessionRecorder(recording_dir, size=(1280, 720), fps=10)
    stop, vision_ready, recording_ready = (
        threading.Event(), threading.Event(), threading.Event(),
    )
    load_failures: list[dict[str, str]] = []
    load_stats = {"vision_submitted": 0, "vision_completed": 0, "recorded": 0}
    timings: list[AdviceTiming] = []
    threads: tuple[threading.Thread, ...] = ()
    ready = recording_result = None
    vision_pid = None
    recording_pid = recorder.worker_pid
    load_started = time.monotonic()

    def vision_load() -> None:
        next_tick, sequence = time.monotonic(), 0
        try:
            while not stop.is_set():
                sequence += 1
                identity = FrameIdentity(
                    "real-concurrent-vision", 1, sequence, sequence * 100,
                    "historical-1280x720", "historical-avi",
                )
                vision.submit(
                    frames[(sequence - 1) % len(frames)], frame=identity,
                    version=VersionIdentity(
                        "real-concurrent-vision", 1, 0, sequence, 0,
                    ),
                    expected_seat=Seat.LEFT, visual_self_opportunity=False,
                    wild_rank="6", request_sequence=sequence, timeout_ms=20_000,
                )
                load_stats["vision_submitted"] += 1
                completed = sum(
                    result.status is VisionRuntimeStatus.FRAME
                    for result in vision.drain_results()
                )
                load_stats["vision_completed"] += completed
                vision_ready.set()
                next_tick += 0.1
                stop.wait(max(0.0, next_tick - time.monotonic()))
        except BaseException as exc:
            load_failures.append({
                "source": "vision", "type": type(exc).__name__,
                "message": str(exc),
            })
            vision_ready.set()

    def recording_load() -> None:
        next_tick, sequence = time.monotonic(), 0
        try:
            while not stop.is_set():
                sequence += 1
                recorder.write_frame(
                    frames[(sequence - 1) % len(frames)], sequence * 100,
                    f"2026-09-07T00:00:{sequence % 60:02d}+08:00",
                )
                load_stats["recorded"] += 1
                recording_ready.set()
                next_tick += 0.1
                stop.wait(max(0.0, next_tick - time.monotonic()))
        except BaseException as exc:
            load_failures.append({
                "source": "recording", "type": type(exc).__name__,
                "message": str(exc),
            })
            recording_ready.set()

    try:
        ready = advice.start(timeout=20)
        vision.start(timeout=20)
        vision_pid = vision.worker_pid
        threads = (
            threading.Thread(target=vision_load, daemon=True),
            threading.Thread(target=recording_load, daemon=True),
        )
        for thread in threads:
            thread.start()
        assert vision_ready.wait(5) and recording_ready.wait(5)
        assert not load_failures, f"load startup failed: {load_failures!r}"
        loop_deadline = time.monotonic() + TOTAL_BUDGET_S
        for sequence in range(1, ROUNDS + 1):
            public_turn = 2 if sequence % 2 else 6
            snapshot = snapshots[public_turn]
            started = time.perf_counter()
            immediate = advice.submit(
                snapshot, _opportunity(snapshot, sequence),
                request_sequence=sequence,
                timeout_ms=int(ADVICE_DEADLINE_S * 1_000),
            )
            submitted = time.perf_counter()
            try:
                result = immediate[-1] if immediate else advice.get_result(timeout=5)
                status, model_ms = result.status.value, result.elapsed_ms
                pid, generation = result.worker_pid, result.worker_generation
                cache_hit = result.advisor_cache_hit
                code, message = result.failure_code, result.message
            except TimeoutError as exc:
                status, model_ms, pid, generation, cache_hit = (
                    "no_result", 0.0, advice.worker_pid,
                    advice.worker_generation, False,
                )
                code, message = "get_result_timeout", str(exc)
            total_ms = (time.perf_counter() - started) * 1_000
            submit_ms = (submitted - started) * 1_000
            pre_model_ms = max(0.0, total_ms - submit_ms - model_ms)
            host_timing = next((
                item for item in reversed(advice._host.recent_request_timings)
                if item.worker_generation == generation
                and item.request_sequence == sequence
            ), None)
            worker_timing = {} if host_timing is None else asdict(host_timing)
            failed = (
                total_ms >= ADVICE_DEADLINE_S * 1_000
                or status != AdviceRuntimeStatus.ADVICE.value
                or not cache_hit or pid != ready.worker_pid
            )
            timings.append(AdviceTiming(
                sequence, public_turn, status, pid, generation, cache_hit,
                total_ms, submit_ms, model_ms, pre_model_ms,
                _stage(worker_timing) if failed else "", code, message,
                worker_timing,
            ))
            if failed or time.monotonic() >= loop_deadline:
                break
    finally:
        stop.set()
        for thread in threads:
            thread.join(5)
        for result in vision.drain_results():
            load_stats["vision_completed"] += (
                result.status is VisionRuntimeStatus.FRAME
            )
        advice.close()
        vision.close()
        recording_result = recorder.close()

    evidence = {
        "schema": "guandan.real-concurrent-advice-acceptance/1",
        "rounds_required": ROUNDS,
        "deadline_ms": int(ADVICE_DEADLINE_S * 1_000),
        "elapsed_ms": (time.monotonic() - load_started) * 1_000,
        "parent_pid": os.getpid(),
        "advice_pid": None if ready is None else ready.worker_pid,
        "vision_pid": vision_pid,
        "recording_pid": recording_pid,
        "load_stats": load_stats,
        "load_failures": load_failures,
        "recording_integrity": recording_result.integrity,
        "advice_timings": [asdict(item) for item in timings],
    }
    path = tmp_path / "real_concurrent_advice_evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), "utf-8")
    record_property("real_concurrent_advice_evidence", json.dumps(evidence))
    assert not load_failures, f"concurrent load failed; evidence={path}: {load_failures!r}"
    assert load_stats["vision_submitted"] >= 2 and load_stats["recorded"] >= 2
    assert len({os.getpid(), ready.worker_pid, vision_pid, recording_pid}) == 4
    assert recording_result.integrity["status"] == "PASS", f"evidence={path}"
    assert len(timings) == ROUNDS, f"bounded loop stopped early; evidence={path}"
    failed = [timing for timing in timings if timing.failure_stage]
    assert not failed, f"Advice acceptance failed; evidence={path}: {failed!r}"
