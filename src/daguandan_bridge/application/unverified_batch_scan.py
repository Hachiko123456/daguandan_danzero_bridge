"""Bounded parallel scanning for sessions without verified TruthLogs."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

from ..live.truth_log import TruthInitialState, TruthLog, load_truth_log
from ..storage import atomic_write_json
from .session_corpus_validation import SessionCorpusValidationService
from .session_workbench import SessionDescriptor
from .truth_log_from_scan import build_truth_log_from_scan
from .video_scan import VideoScanRequest

# Progress is reported as ONE aggregate figure for the whole batch.  Two workers
# finish frames independently, so without aggregation the bar visibly jumps
# between sessions.  Emissions are throttled so a per-frame callback cannot
# flood the UI event loop.
_PROGRESS_MIN_DELTA_PERCENT = 1
_PROGRESS_MIN_INTERVAL_SEC = 0.2
# Below this much free RAM a second full-video worker risks paging the desktop.
_MIN_FREE_MEMORY_GB = 2.0


def _available_memory_gb() -> float | None:
    """Free physical memory in GB, or ``None`` when it cannot be determined."""
    try:
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPhys / 1024**3
    except Exception:
        return None


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class UnverifiedBatchScanResult:
    output_directory: Path
    selected_count: int
    completed_count: int
    failed_count: int
    cancelled: bool
    summary_path: Path
    sessions: tuple[dict[str, object], ...]


class UnverifiedBatchScanService:
    """Scan only non-verified sessions with bounded session-level parallelism.

    A single video always stays sequential: turn reconstruction needs ordered
    frames. Parallelism is only across independent sessions, and every worker
    creates its own recognizer/scanner/cache.
    """

    def __init__(self, *, scanner_service: SessionCorpusValidationService | None = None) -> None:
        self._scanner_service = scanner_service or SessionCorpusValidationService()

    @staticmethod
    def recommended_workers() -> int:
        cpu = os.cpu_count() or 1
        # Full-video scans retain evidence frames; two workers are the safe
        # desktop ceiling even on machines with many cores.
        workers = max(1, min(2, cpu // 2 or 1))
        free_gb = _available_memory_gb()
        if free_gb is not None and free_gb < _MIN_FREE_MEMORY_GB:
            # A second concurrent scan would push the desktop into paging.
            return 1
        return workers

    def scan(
        self,
        descriptors: Iterable[SessionDescriptor],
        *,
        profile_root: Path | str,
        output_root: Path | str,
        max_workers: int | None = None,
        stop_requested: Callable[[], bool] | None = None,
        on_progress: Callable[[str, int, int, int, int], None] | None = None,
    ) -> UnverifiedBatchScanResult:
        selected = tuple(
            item for item in descriptors
            if item.truth_status in {"draft", "missing", "invalid"} and item.has_video
        )
        output = Path(output_root).expanduser().resolve() / (
            f"unverified_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        )
        output.mkdir(parents=True, exist_ok=False)
        stop = stop_requested or (lambda: False)
        worker_count = max(1, min(max_workers or self.recommended_workers(), 2, len(selected) or 1))
        rows: list[dict[str, object]] = []
        lock = threading.Lock()
        progress_by_session: dict[str, tuple[int, int]] = {}
        finished_ids: set[str] = set()
        throttle = {"percent": -1, "at": 0.0}

        def _overall_percent() -> int:
            """Whole-batch completion: finished sessions count 1, running ones their fraction."""
            started = sum(
                min(1.0, done / total) if total else 0.0
                for session_id, (done, total) in progress_by_session.items()
                if session_id not in finished_ids
            )
            return int(((len(finished_ids) + started) / max(1, len(selected))) * 100)

        def emit(session_id: str, done: int, total: int, frame: int, *, finished: bool = False) -> None:
            """Report ONE aggregate figure for the whole batch (monotonic)."""
            with lock:
                if total:
                    progress_by_session[session_id] = (int(done), int(total))
                if finished:
                    finished_ids.add(session_id)
                percent = _overall_percent()
            now = time.monotonic()
            if percent <= throttle["percent"] and (now - throttle["at"]) < _PROGRESS_MIN_INTERVAL_SEC:
                return
            throttle["percent"] = percent
            throttle["at"] = now
            if on_progress is not None:
                on_progress(session_id, done, total, frame, len(finished_ids), percent)

        def one(item: SessionDescriptor) -> dict[str, object]:
            session_output = output / item.session_id
            session_output.mkdir(parents=True, exist_ok=False)
            scanner = self._scanner_service._make_scanner(profile_root)
            request = VideoScanRequest(
                video_path=item.video_path,
                frame_index_path=item.frame_index_path if item.has_frame_index else None,
                output_directory=session_output,
                session_id=item.session_id,
            )
            try:
                result = scanner.scan(
                    request,
                    stop_requested=stop,
                    on_progress=lambda done, total, frame: emit(item.session_id, done, total, frame),
                )
                row = dict(_json_safe(result.to_dict()))
                row.update({"session_id": item.session_id, "status": result.status})
                actions = [json.loads(line) for line in result.action_trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if item.truth_log_path.is_file():
                    baseline = load_truth_log(item.truth_log_path, session_id=item.session_id)
                    draft = build_truth_log_from_scan(baseline, actions)
                    draft_path = session_output / "truth_log.draft.json"
                    review_path = session_output / "truth_log_review.json"
                    atomic_write_json(draft_path, draft.truth_log.to_dict())
                    atomic_write_json(review_path, {"schema": "guandan.truth-log-review/1", "review_items": list(draft.review_items)})
                    row["truth_draft"] = {"status": "generated", "path": str(draft_path), "review_count": len(draft.review_items)}
                else:
                    row["truth_draft"] = {"status": "not_generated", "reason": "initial_state_requires_manual_review"}
                return row
            except Exception as exc:
                return {"session_id": item.session_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        if selected:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="unverified-video-scan") as executor:
                futures = {executor.submit(one, item): item for item in selected}
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    emit(str(row.get("session_id")), 0, 0, 0, finished=True)
                    if stop():
                        # Running scans observe the same stop callback; do not
                        # schedule more work after the current executor batch.
                        break
        rows.sort(key=lambda row: str(row.get("session_id", "")))
        summary = {
            "schema": "guandan.unverified-batch-scan/1",
            "selected_count": len(selected),
            "completed_count": sum(row.get("status") in {"complete", "completed"} for row in rows),
            "failed_count": sum(row.get("status") == "failed" for row in rows),
            "cancelled": bool(stop()),
            "max_workers": worker_count,
            "sessions": rows,
        }
        summary_path = output / "summary.json"
        atomic_write_json(summary_path, summary)
        return UnverifiedBatchScanResult(
            output, len(selected), int(summary["completed_count"]), int(summary["failed_count"]), bool(summary["cancelled"]), summary_path, tuple(rows)
        )
