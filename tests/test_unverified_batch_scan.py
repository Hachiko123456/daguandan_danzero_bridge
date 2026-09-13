from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.application.session_corpus_validation import SessionCorpusValidationService
from daguandan_bridge.application.session_workbench import SessionDescriptor
from daguandan_bridge.application.unverified_batch_scan import UnverifiedBatchScanService
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, save_truth_log


def _descriptor(root: Path, name: str, status: str) -> SessionDescriptor:
    session = root / name
    (session / "video").mkdir(parents=True)
    video = session / "video" / "game.avi"
    video.write_bytes(b"video")
    index = session / "video" / "frame_index.jsonl"
    index.write_text('{"frame_index": 0}\n', encoding="utf-8")
    manifest = session / "manifest.json"
    manifest.write_text(json.dumps({"session_id": name, "frame_count": 1}), encoding="utf-8")
    truth_path = session / "truth_log.json"
    if status != "missing":
        save_truth_log(
            truth_path,
            TruthLog(name, TruthInitialState("2", "self", ("2S",)), (), label_status=status),
        )
    return SessionDescriptor(
        root=session,
        session_id=name,
        manifest_path=manifest,
        video_path=video,
        frame_index_path=index,
        timeline_path=session / "timeline.jsonl",
        truth_log_path=truth_path,
        manifest_readable=True,
        has_video=True,
        has_frame_index=True,
        has_timeline=False,
        truth_status=status,
        truth_error="",
        frame_count=1,
        timeline_event_count=0,
    )


class _Result:
    def __init__(self, output: Path) -> None:
        self.status = "complete"
        self.action_trace_path = output / "action_trace.jsonl"
        self.action_trace_path.write_text(
            json.dumps({"actor": "self", "is_pass": False, "cards": ["2S"], "frame_start": 1}) + "\n",
            encoding="utf-8",
        )

    def to_dict(self):
        return {
            "status": self.status,
            "action_trace_path": self.action_trace_path,
            "decoded_frames": 1,
            "action_count": 1,
        }


class _Scanner:
    def __init__(self) -> None:
        self.requests = []

    def scan(self, request, *, stop_requested, on_progress):
        self.requests.append(request)
        on_progress(1, 1, 0)
        return _Result(Path(request.output_directory))


def test_batch_scans_only_nonverified_sessions_and_writes_isolated_artifacts(tmp_path: Path):
    verified = _descriptor(tmp_path, "verified", "verified")
    draft = _descriptor(tmp_path, "draft", "draft")
    missing = _descriptor(tmp_path, "missing", "missing")
    scanner = _Scanner()
    backing = SessionCorpusValidationService(scan_factory=lambda **_kwargs: scanner)
    service = UnverifiedBatchScanService(scanner_service=backing)
    progress = []

    result = service.scan(
        (verified, draft, missing),
        profile_root=tmp_path,
        output_root=tmp_path / "reports",
        max_workers=1,
        on_progress=lambda *args: progress.append(args),
    )

    assert result.selected_count == 2
    assert result.completed_count == 2
    assert result.failed_count == 0
    assert {Path(request.video_path).parent.parent.name for request in scanner.requests} == {"draft", "missing"}
    assert progress
    assert (result.output_directory / "summary.json").is_file()
    draft_row = next(row for row in result.sessions if row["session_id"] == "draft")
    missing_row = next(row for row in result.sessions if row["session_id"] == "missing")
    assert Path(draft_row["truth_draft"]["path"]).is_file()
    assert missing_row["truth_draft"]["status"] == "not_generated"
    assert not (verified.root / "turn_slots.json").exists()


def test_batch_progress_is_one_monotonic_aggregate_across_workers(tmp_path: Path):
    """Two concurrent workers must never make the reported bar jump backwards."""
    import time

    draft = _descriptor(tmp_path, "draft", "draft")
    missing = _descriptor(tmp_path, "missing", "missing")
    steps = 20

    class _SteppedScanner:
        def scan(self, request, *, stop_requested, on_progress):
            for done in range(1, steps + 1):
                on_progress(done, steps, done)
                time.sleep(0.002)
            return _Result(Path(request.output_directory))

    backing = SessionCorpusValidationService(scan_factory=lambda **_kwargs: _SteppedScanner())
    service = UnverifiedBatchScanService(scanner_service=backing)
    samples = []

    result = service.scan(
        (draft, missing),
        profile_root=tmp_path,
        output_root=tmp_path / "reports",
        max_workers=2,
        on_progress=lambda *args: samples.append(args),
    )

    assert result.completed_count == 2
    percents = [args[5] for args in samples]
    assert percents, "batch must report progress"
    assert percents == sorted(percents), f"aggregate progress regressed: {percents}"
    assert percents[-1] == 100
    assert all(0 <= percent <= 100 for percent in percents)
    # At least one in-flight sample shows a partial aggregate before any session ends.
    assert any(percent > 0 for percent, args in zip(percents, samples) if args[4] == 0)


def test_recommended_workers_drops_to_one_when_free_memory_is_low(monkeypatch):
    """A second full-video worker must not be started when RAM is scarce."""
    import os

    from daguandan_bridge.application import unverified_batch_scan as module

    monkeypatch.setattr(module, "_available_memory_gb", lambda: 0.5)
    assert module.UnverifiedBatchScanService.recommended_workers() == 1

    monkeypatch.setattr(module, "_available_memory_gb", lambda: 64.0)
    assert module.UnverifiedBatchScanService.recommended_workers() == max(
        1, min(2, (os.cpu_count() or 1) // 2 or 1)
    )
