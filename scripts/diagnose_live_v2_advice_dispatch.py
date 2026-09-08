from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import threading
import time

import cv2

from daguandan_bridge.application.live_v2_advice_protocol import AdviceWorkerConfig
from daguandan_bridge.application.live_v2_vision_protocol import VisionWorkerConfig
from daguandan_bridge.infrastructure.live_v2_advice_service_factory import create_live_v2_advice_runtime
from daguandan_bridge.infrastructure.live_v2_vision_service_factory import build_live_v2_vision_runtime
from daguandan_bridge.infrastructure.process_session_recorder import ProcessSessionRecorder
from daguandan_bridge.live_v2.events import ActionKind
from daguandan_bridge.live_v2.game_state import GameAction, SeatCardCount, TrustedGameSnapshot
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, StateVersion, VersionIdentity
from daguandan_bridge.live_v2.results import AdviceOpportunity, OpportunityReason, OpportunityStatus


ROOT = Path(__file__).parents[1]
PROFILES = ROOT / "data" / "profiles"
VIDEO = PROFILES / "tencent_daguandan" / "sessions" / "game_20260814_004447_aab3dc" / "video" / "game.avi"
HAND = (
    "small_joker", "6H", "6C", "AS", "AH", "KS", "KC", "QH", "QC",
    "JH", "JC", "JC", "JD", "10S", "9S", "9H", "9H", "9C", "8H",
    "8C", "8D", "7D", "5C", "5D", "4H", "4H", "3C",
)


def exact_turn2_snapshot() -> TrustedGameSnapshot:
    session_id = "dispatch-exact-turn2"
    before = StateVersion(session_id, 2, 0)
    after = StateVersion(session_id, 3, 1)
    first = FrameIdentity(session_id, 1, 83, 1_000, "roi", "source-video")
    last = FrameIdentity(session_id, 1, 86, 1_100, "roi", "source-video")
    action = GameAction(
        "truth-left-pair", before, after, Seat.LEFT, ActionKind.PLAY,
        ("5H", "5C"), (("5H",), ("5C",)), 1,
        ("source-frame-83-left", "source-frame-86-left"),
        first, last, 0.99, 1_100,
    )
    return TrustedGameSnapshot(
        VersionIdentity.from_state(after, capture_generation=1, update_sequence=3),
        "6", "6", 1, Seat.SELF, Seat.LEFT, HAND, (action,), (action,),
        tuple(SeatCardCount(seat, 25 if seat is Seat.LEFT else 27) for seat in Seat),
        (), True, False, 1_100,
    )


def opportunity(snapshot: TrustedGameSnapshot, sequence: int) -> AdviceOpportunity:
    return AdviceOpportunity(
        f"turn2-round-{sequence}", snapshot.version, Seat.SELF,
        OpportunityStatus.READY, OpportunityReason.TRUSTED_STATE,
        snapshot.captured_ms, snapshot.captured_ms,
    )


def read_frame():
    capture = cv2.VideoCapture(str(VIDEO))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 83)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read source frame from {VIDEO}")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--backend", choices=("fabledan", "danzero"), default="fabledan")
    parser.add_argument("--without-load", action="store_true")
    parser.add_argument("--restart-each-round", action="store_true")
    args = parser.parse_args()
    snapshot = exact_turn2_snapshot()
    frame = read_frame()
    prewarm_version = VersionIdentity(
        snapshot.version.session_id, snapshot.version.capture_generation, 0, 0, 0,
    )
    advice = create_live_v2_advice_runtime(
        AdviceWorkerConfig(str(PROFILES), args.backend), prewarm_version,
    )
    vision = recorder = load_thread = None
    stop = threading.Event()
    failures: list[str] = []
    try:
        advice.start(timeout=20)
        if not args.without_load:
            vision = build_live_v2_vision_runtime(
                VisionWorkerConfig(str(PROFILES)),
                session_id=snapshot.version.session_id,
                capture_generation=1,
                state_revision=snapshot.version.state_revision,
            )
            vision.start(timeout=20)
            temp_root = Path(tempfile.mkdtemp(prefix="advice-dispatch-load-"))
            recorder = ProcessSessionRecorder(
                temp_root, size=(frame.shape[1], frame.shape[0]), fps=10,
            )

            def load() -> None:
                sequence = 0
                while not stop.is_set():
                    sequence += 1
                    try:
                        recorder.write_frame(frame, sequence * 100, f"wall-{sequence}")
                        identity = FrameIdentity(
                            snapshot.version.session_id, 1, sequence, sequence * 100,
                            "roi-v1", "source-video",
                        )
                        version = VersionIdentity(
                            snapshot.version.session_id, 1,
                            snapshot.version.state_revision, sequence,
                            snapshot.version.turn_index,
                        )
                        vision.submit(
                            frame, frame=identity, version=version,
                            expected_seat=Seat.LEFT,
                            visual_self_opportunity=False, wild_rank="6",
                            request_sequence=sequence, timeout_ms=5_000,
                        )
                        try:
                            vision.get_result(timeout=5)
                        except TimeoutError:
                            failures.append("vision result timeout")
                    except BaseException as exc:
                        failures.append(f"{type(exc).__name__}: {exc}")
                        return

            load_thread = threading.Thread(target=load, daemon=True)
            load_thread.start()
        timings = []
        for sequence in range(1, args.rounds + 1):
            if args.restart_each_round and sequence > 1:
                advice.restart(
                    prewarm_version, request_sequence=10_000 + sequence,
                    timeout=20,
                )
            started = time.perf_counter()
            advice.submit(
                snapshot, opportunity(snapshot, sequence),
                request_sequence=sequence, timeout_ms=3_000,
            )
            result = advice.get_result(timeout=5)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            print(
                f"round={sequence} status={result.status.value} "
                f"elapsed={elapsed:.3f}s child={result.elapsed_ms:.1f}ms "
                f"trace={advice.recent_request_timings[-1]}"
            )
        print(
            f"backend={args.backend} rounds={len(timings)} "
            f"max={max(timings):.3f}s avg={sum(timings) / len(timings):.3f}s "
            f"load_failures={failures}"
        )
        return 0 if max(timings) < 3 and not failures else 1
    finally:
        stop.set()
        if load_thread is not None:
            load_thread.join(10)
        if recorder is not None:
            recorder.close()
        if vision is not None:
            vision.close()
        advice.close()


if __name__ == "__main__":
    raise SystemExit(main())
