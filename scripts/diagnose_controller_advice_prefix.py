from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.advisor_strategy import build_advisor
from daguandan_bridge.application.ports import LiveSessionConstruction
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.gui.live_controller import LiveAssistantController
from daguandan_bridge.infrastructure.live_v2_composition import build_production_live_v2_runtime
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.session_store import LiveSessionStore


HAND = (
    "10S", "3C", "4H", "4H", "5C", "5D", "6C", "6H", "7D",
    "8C", "8D", "8H", "9C", "9H", "9H", "9S", "AH", "AS",
    "JC", "JC", "JD", "JH", "KC", "KS", "QC", "QH", "small_joker",
)
PREFIX = (
    ("left", ("5C", "5H"), False),
    ("self", ("9C", "9H", "9H", "9S"), False),
    ("right", (), True),
    ("opposite", (), True),
    ("left", (), True),
)


class _Source:
    def close(self) -> None:
        pass


class _Capture:
    profiles_root = PROFILES_ROOT


class _Evidence:
    def mark_session_started(self) -> None:
        pass


class _Factory:
    def __init__(self, root: Path) -> None:
        self.root = root

    def start_session(self, *, round_level, hand, lead_player,
                      recognition_strategy, on_update):
        del recognition_strategy
        store = LiveSessionStore(self.root, "diag", session_id="controller-prefix")
        store.start({"schema": "controller-prefix-diagnostic/1"})
        runtime = build_production_live_v2_runtime(
            store=store,
            recorder=InMemorySessionRecorder(store.directory),
            recognizer=SimpleNamespace(),
            profiles_root=PROFILES_ROOT,
            profile_name="tencent_daguandan",
            advisor_backend="fabledan",
            on_update=on_update,
        )
        initial = runtime.start(
            round_level=round_level,
            hand=hand,
            lead_player=lead_player,
            monotonic_ms=time.monotonic_ns() // 1_000_000,
        )
        return LiveSessionConstruction(runtime, _Source(), initial)


def _wait(app: QApplication, predicate, timeout: float = 4.0) -> float:
    started = time.perf_counter()
    deadline = started + timeout
    while time.perf_counter() < deadline:
        app.processEvents()
        if predicate():
            return (time.perf_counter() - started) * 1000
        time.sleep(0.002)
    app.processEvents()
    return (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-main-warmup", action="store_true")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    temp = Path(tempfile.mkdtemp(prefix="controller-prefix-diag-"))
    advisor = build_advisor(
        "fabledan", profiles_root=PROFILES_ROOT,
        profile_name="tencent_daguandan", fabledan_diagnostics="off",
    )
    controller = LiveAssistantController(
        _Capture(), recognition_service=SimpleNamespace(), advisor=advisor,
        session_factory=_Factory(temp), opening_evidence_monitor=_Evidence(),
    )
    controller._start_analysis_worker = lambda: None
    controller._start_capture_worker = lambda: None
    if args.suppress_main_warmup:
        controller._start_danzero_warmup = lambda: None
    updates = []
    controller.update_ready.connect(updates.append)
    started = time.perf_counter()
    ok = controller.start_session(round_level="6", hand=HAND, lead_player="left")
    start_ms = (time.perf_counter() - started) * 1000
    if not ok:
        print(json.dumps({"start_ok": False, "start_ms": start_ms}))
        return 2
    runtime = controller.orchestrator
    pump = runtime._advice_pump
    advice = pump._runtime
    ready = advice.advisor_ready
    rows = []
    for index, (actor, cards, is_pass) in enumerate(PREFIX, 1):
        update = runtime.commit_trusted_action(
            actor=actor, cards=cards, is_pass=is_pass,
            monotonic_ms=time.monotonic_ns() // 1_000_000,
            confidence=1.0,
        )
        if index in {1, 5}:
            expected_turn = update.snapshot.turn_id
            elapsed = _wait(
                app,
                lambda: (
                    runtime.latest_advice is not None
                    and runtime.latest_advice.key.turn_id == expected_turn
                    and runtime.latest_advice.status != "requested"
                ),
            )
            rows.append({
                "after_action": index,
                "public_turn": update.snapshot.turn_id,
                "elapsed_ms": elapsed,
                "latest_advice": None if runtime.latest_advice is None else {
                    "status": runtime.latest_advice.status,
                    "turn_id": runtime.latest_advice.key.turn_id,
                    "request_id": runtime.latest_advice.key.request_id,
                    "error": runtime.latest_advice.error,
                },
                "worker_pid": advice.worker_pid,
                "worker_generation": advice.worker_generation,
                "timings": [str(item) for item in advice.recent_request_timings],
            })
    payload = {
        "parent_pid": os.getpid(),
        "suppress_main_warmup": args.suppress_main_warmup,
        "start_ms": start_ms,
        "advisor_ready": None if ready is None else {
            "worker_pid": ready.worker_pid,
            "worker_generation": ready.worker_generation,
            "elapsed_ms": ready.elapsed_ms,
            "cache_hit": ready.cache_hit,
            "runtime_backend": ready.runtime_backend,
            "model_path": ready.model_path,
            "model_digest": ready.model_digest,
            "model_status": ready.model_status,
        },
        "main_warmup_running": controller._danzero_warmup_running,
        "main_warmup_complete": controller._danzero_warmup_complete,
        "rows": rows,
        "update_count": len(updates),
        "pipeline_timing": controller._active_live_token.pipeline_timing.snapshot(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    controller._invalidate_live_token()
    runtime.finish()
    controller.opening_evidence.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
