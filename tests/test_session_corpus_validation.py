from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from daguandan_bridge.application.session_corpus_validation import SessionCorpusValidationService
from daguandan_bridge.application.session_locator import SessionLocator
from daguandan_bridge.application.session_replay_audit import TruthAuditReference, _truth_metadata
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn, save_truth_log


def _make_session(root: Path, name: str, *, truth_status: str | None = None, timeline: bool = True) -> Path:
    session = root / name
    (session / "video").mkdir(parents=True)
    (session / "manifest.json").write_text(json.dumps({"session_id": name}), encoding="utf-8")
    (session / "video" / "game.avi").write_bytes(b"avi")
    (session / "video" / "frame_index.jsonl").write_text('{"frame_index": 0}\n', encoding="utf-8")
    if timeline:
        (session / "timeline.jsonl").write_text('{"event_type":"initial_state_confirmed"}\n', encoding="utf-8")
    if truth_status is not None:
        (session / "truth_log.json").write_text(
            json.dumps(
                {
                    "schema": "guandan.truth/4",
                    "source_session_id": name,
                    "initial_state": {"round_level": "2", "lead_player": "left", "my_hand": ["2S"]},
                    "turns": [],
                    "label_status": truth_status,
                    "provenance": {"source": "test", "annotator": "test"},
                }
            ),
            encoding="utf-8",
        )
    return session


def test_locator_uses_one_canonical_workbench_descriptor(tmp_path: Path):
    sessions = tmp_path / "arbitrary" / "recordings"
    sessions.mkdir(parents=True)
    one = _make_session(sessions, "game_one")
    selection = SessionLocator().discover(sessions)
    assert [item.root for item in selection.sessions] == [one]
    assert selection.sessions[0].truth_status == "missing"
    assert selection.sessions[0].is_playable


def test_runner_inventory_only_distinguishes_truth_levels_and_is_read_only(tmp_path: Path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _make_session(sessions, "game_verified", truth_status="verified")
    _make_session(sessions, "game_draft", truth_status="draft")
    _make_session(sessions, "manual_missing", timeline=False)
    before = {path.relative_to(sessions).as_posix(): path.read_bytes() for path in sessions.rglob("*") if path.is_file()}

    run = SessionCorpusValidationService().validate(
        sessions,
        mode="corpus",
        output=tmp_path / "reports",
        execute_replay=False,
        run_id="inventory",
        trusted_sessions=("manual_missing",),
    )
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert run.passed
    assert summary["counts"] == {
        "discovered": 3,
        "selected": 3,
        "structural_passed": 3,
        "structural_failed": 0,
        "semantic_eligible": 2,
        "truth_missing": 1,
        "truth_draft": 1,
        "truth_verified": 1,
        "truth_invalid": 0,
        "semantic_failed": 0,
    }
    assert summary["sessions"][0]["semantic_status"] in {"eligible", "not_eligible"}
    scan = json.loads((run.output_directory / "scan_manifest.json").read_text(encoding="utf-8"))
    assert scan["status"] == "not_run"
    assert scan["counts"]["truth_missing"] == 1
    assert all((run.output_directory / "scan" / row["session_id"] / "scan_summary.json").is_file() for row in scan["sessions"])
    after = {path.relative_to(sessions).as_posix(): path.read_bytes() for path in sessions.rglob("*") if path.is_file()}
    assert before == after


def test_single_mode_accepts_an_arbitrary_session_path(tmp_path: Path):
    source = tmp_path / "copied" / "deep" / "session"
    source.parent.mkdir(parents=True)
    session = _make_session(source.parent, source.name, timeline=False)
    run = SessionCorpusValidationService().validate(
        session,
        mode="single",
        output=tmp_path / "reports",
        execute_replay=False,
        run_id="single",
    )
    assert len(run.selection) == 1
    assert run.selection[0].root == session
    assert run.summary_path.is_file()


def test_trusted_draft_is_explicitly_labeled_without_promoting_source(tmp_path: Path):
    session = _make_session(tmp_path, "game_draft", truth_status=None)
    truth_path = session / "truth_log.json"
    save_truth_log(
        truth_path,
        TruthLog("game_draft", TruthInitialState("2", "left", ("2S",)), ()),
    )
    from daguandan_bridge.live.truth_log import load_truth_log

    truth = load_truth_log(truth_path, session_id="game_draft")
    metadata = _truth_metadata(
        TruthAuditReference("canonical", truth_path), truth, trusted=True
    )
    assert metadata["kind"] == "trusted_draft"
    assert metadata["reference_kind"] == "canonical"
    assert metadata["trusted_session"] is True
    assert truth.label_status == "draft"


def test_replay_failure_is_not_reported_as_passed(tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _make_session(sessions, "game_replay_failure", timeline=True)
    service = SessionCorpusValidationService()

    def fail(*args, **kwargs):
        raise RuntimeError("decoder failed")

    monkeypatch.setattr(service, "_run_audit", fail)
    run = service.validate(
        sessions,
        mode="corpus",
        output=tmp_path / "reports",
        execute_replay=True,
        run_id="failure",
    )
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert run.passed is False
    assert summary["status"] == "failed"
    assert summary["replay_audit"]["status"] == "error"


def test_semantic_truth_failure_is_not_reported_as_passed(tmp_path: Path, monkeypatch):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _make_session(sessions, "game_semantic_failure", timeline=True)
    service = SessionCorpusValidationService()
    fake_audit = SimpleNamespace(
        execution_ok=True,
        sessions=({
            "session_id": "game_semantic_failure",
            "execution_status": "completed",
            "truth_quality": "failed",
            "frames_processed": 1,
            "indexed_frames": 1,
            "visual": {"actions": {"rows": [], "count": 0}},
            "evidence_paths": {},
        },),
        summary_path=tmp_path / "audit.json",
        verification_path=None,
    )
    monkeypatch.setattr(service, "_run_audit", lambda *args, **kwargs: fake_audit)
    run = service.validate(
        sessions,
        mode="single",
        output=tmp_path / "reports",
        execute_replay=True,
        trusted_sessions=("game_semantic_failure",),
        run_id="semantic-failure",
    )
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert run.passed is False
    assert summary["semantic_status"] == "failed"
    assert summary["semantic_failed"] == 1
    assert summary["status"] == "failed"


def test_trusted_session_is_included_in_smoke_selection(tmp_path: Path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for index in range(10):
        _make_session(sessions, f"game_{index:02d}", timeline=True)
    target = _make_session(sessions, "game_trusted", truth_status="draft", timeline=True)
    run = SessionCorpusValidationService().validate(
        sessions,
        mode="smoke",
        output=tmp_path / "reports",
        execute_replay=False,
        trusted_sessions=(target,),
        run_id="priority",
    )
    assert "game_trusted" in {item.session_id for item in run.selection}


class _PureVideoScanner:
    def __init__(self, actions: list[dict[str, object]] | None = None) -> None:
        self.requests: list[object] = []
        self.actions = actions or []

    def scan(self, request: object) -> dict[str, object]:
        self.requests.append(request)
        output = Path(getattr(request, "output_directory"))
        trace = output / "action_trace.jsonl"
        trace.write_text(
            "".join(json.dumps(row) + "\n" for row in self.actions),
            encoding="utf-8",
        )
        return {
            "status": "complete",
            "decoded_frames": 7,
            "action_count": len(self.actions),
            "observations_path": output / "frame_observations.jsonl.gz",
            "action_trace_path": trace,
            "summary_path": output / "scan_summary.json",
        }


def test_video_only_session_scans_without_manifest_index_timeline_or_truth(tmp_path: Path):
    session = tmp_path / "copied" / "game_video_only"
    (session / "video").mkdir(parents=True)
    video = session / "video" / "game.avi"
    video.write_bytes(b"AVI only")
    scanner = _PureVideoScanner()
    service = SessionCorpusValidationService(scan_factory=lambda **_: scanner)

    run = service.validate(session, mode="single", output=tmp_path / "reports", run_id="video-only")
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))

    assert run.passed
    assert len(scanner.requests) == 1
    request = scanner.requests[0]
    assert Path(getattr(request, "video_path")) == video
    assert getattr(request, "frame_index_path") is None
    assert not hasattr(request, "timeline_path")
    assert not hasattr(request, "truth_log_path")
    assert summary["sessions"][0]["scan"]["status"] == "completed"
    assert summary["sessions"][0]["structural"]["warnings"] == [
        "frame_index_missing_timestamps_will_be_derived_from_avi",
        "manifest_missing_session_id_uses_directory_name",
    ]


def test_truth_is_compared_only_after_pure_scan_and_reports_first_difference(tmp_path: Path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session = _make_session(sessions, "game_truth_compare", truth_status="draft", timeline=True)
    save_truth_log(
        session / "truth_log.json",
        TruthLog(
            "game_truth_compare",
            TruthInitialState("2", "left", ("2S",)),
            (TruthTurn(1, "left", False, ("2H",), frame_index=11),),
        ),
    )
    scanner = _PureVideoScanner(
        [{"actor": "left", "is_pass": False, "cards": ["2C"], "frame_start": 11}]
    )
    before = {path.relative_to(session).as_posix(): path.read_bytes() for path in session.rglob("*") if path.is_file()}
    service = SessionCorpusValidationService(scan_factory=lambda **_: scanner)

    run = service.validate(
        session,
        mode="single",
        output=tmp_path / "reports",
        run_id="first-difference",
        trusted_sessions=(session,),
    )
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))

    assert run.passed is False
    semantic = summary["sessions"][0]["semantic"]
    assert semantic["status"] == "failed"
    assert semantic["first_difference"]["action_index"] == 1
    assert semantic["first_difference"]["expected"]["cards"] == ["2H"]
    assert semantic["first_difference"]["actual"]["cards"] == ["2C"]
    after = {path.relative_to(session).as_posix(): path.read_bytes() for path in session.rglob("*") if path.is_file()}
    assert before == after


def test_user_confirmed_session_is_semantically_eligible_without_source_promotion(tmp_path: Path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session = _make_session(sessions, "game_20260816_125402_1687ea", truth_status="draft")
    scanner = _PureVideoScanner()
    service = SessionCorpusValidationService(scan_factory=lambda **_: scanner)

    run = service.validate(session, mode="single", output=tmp_path / "reports", run_id="trusted-user")
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))

    assert summary["truth_policy"]["trusted_session_ids"] == ["game_20260816_125402_1687ea"]
    assert summary["sessions"][0]["semantic"]["eligible"] is True
    assert json.loads((session / "truth_log.json").read_text(encoding="utf-8"))["label_status"] == "draft"


def test_scan_writes_external_draft_truth_and_review_from_canonical_trace_only(tmp_path: Path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session_id = "game_20260816_125402_1687ea"
    session = _make_session(sessions, session_id, truth_status=None)
    baseline_path = session / "truth_log.json"
    baseline = TruthLog(
        session_id,
        TruthInitialState("2", "left", ("2S", "2H")),
        (),
        source_video="video/game.avi",
        frame_index_path="video/frame_index.jsonl",
    )
    save_truth_log(baseline_path, baseline)
    baseline_before = baseline_path.read_bytes()
    scanner = _PureVideoScanner(
        [
            {
                "action_id": 7,
                "actor": "left",
                "is_pass": False,
                "cards": ["A?", "KS"],
                "frame_index": 184,
                "evidence_frames": [165, 184],
                "review_status": "needs_review",
                "repair_status": "unresolved",
                "uncertainty": ["unknown_suit"],
            },
            {
                "action_id": 8,
                "actor": "self",
                "is_pass": True,
                "cards": [],
                "frame_index": 185,
            },
        ]
    )
    service = SessionCorpusValidationService(scan_factory=lambda **_: scanner)

    run = service.validate(
        session,
        mode="single",
        output=tmp_path / "reports",
        run_id="known-game-draft",
    )
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    scan_row = summary["sessions"][0]["scan"]
    draft_info = scan_row["truth_draft"]
    draft_path = Path(draft_info["truth_log_path"])
    review_path = Path(draft_info["review_path"])
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assert len(scanner.requests) == 1
    request = scanner.requests[0]
    assert not hasattr(request, "truth_log_path")
    assert scan_row["action_trace_path"].endswith("action_trace.jsonl")
    assert draft_info["status"] == "generated"
    assert draft_info["review_count"] == 1
    assert draft_path == run.output_directory / "scan" / session_id / "truth_log.draft.json"
    assert review_path == run.output_directory / "scan" / session_id / "truth_log_review.json"
    assert draft["label_status"] == "draft"
    assert draft["provenance"]["source"] == "video_scan+canonical_reconciliation"
    assert draft["source_session_id"] == session_id
    assert draft["initial_state"] == baseline.to_dict()["initial_state"]
    assert draft["turns"][0]["cards"] == ["A?"] + ["KS"]
    assert draft["turns"][0]["trick_id"] is None
    assert draft["turns"][1]["cards"] == []
    assert review["label_status"] == "draft"
    assert review["provenance"]["source"] == "video_scan+canonical_reconciliation"
    assert review["canonical_action_trace_path"].endswith("action_trace.jsonl")
    assert review["review_items"][0]["provenance"]["source"] == "video_scan+canonical_reconciliation"
    assert baseline_path.read_bytes() == baseline_before


class _MissingCanonicalTraceScanner:
    def scan(self, request: object) -> dict[str, object]:
        output = Path(getattr(request, "output_directory"))
        return {
            "status": "complete",
            "decoded_frames": 1,
            "action_count": 0,
            "action_trace_path": output / "action_trace.jsonl",
            "summary_path": output / "scan_summary.json",
        }


def test_scan_does_not_write_draft_when_canonical_trace_is_absent(tmp_path: Path):
    session = _make_session(tmp_path, "game_missing_canonical_trace", truth_status="draft")
    service = SessionCorpusValidationService(
        scan_factory=lambda **_: _MissingCanonicalTraceScanner()
    )

    run = service.validate(
        session,
        mode="single",
        output=tmp_path / "reports",
        run_id="missing-canonical-trace",
    )
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    draft_info = summary["sessions"][0]["scan"]["truth_draft"]

    assert draft_info == {
        "status": "not_generated",
        "truth_log_path": None,
        "review_path": None,
        "review_count": 0,
        "reason": "canonical_action_trace_missing",
    }
    assert not (run.output_directory / "scan" / "game_missing_canonical_trace" / "truth_log.draft.json").exists()
    assert not (run.output_directory / "scan" / "game_missing_canonical_trace" / "truth_log_review.json").exists()

class _PostScanOnlyScanner(_PureVideoScanner):
    def __init__(self) -> None:
        super().__init__([{"actor": "left", "is_pass": False, "cards": ["2S"]}])
        self.finished = False

    def scan(self, request: object) -> dict[str, object]:
        result = super().scan(request)
        self.finished = True
        return result


def test_invalid_baseline_is_not_loaded_until_after_scan_completion(
    tmp_path: Path,
    monkeypatch,
):
    import daguandan_bridge.application.session_corpus_validation as validation_module
    import daguandan_bridge.application.session_workbench as workbench_module

    session = _make_session(tmp_path, "game_deferred_baseline", truth_status=None)
    (session / "truth_log.json").write_text("{ not valid json", encoding="utf-8")
    scanner = _PostScanOnlyScanner()
    reads_after_scan: list[Path] = []
    original_load = validation_module.load_truth_log

    def _tracked_load(path: Path, *, session_id: str | None = None):
        assert scanner.finished, "baseline TruthLog was opened before scan completion"
        reads_after_scan.append(Path(path))
        return original_load(path, session_id=session_id)

    # This is the loader called by SessionLocator.inspect_session. If scan
    # discovery regresses to SessionLocator.discover, the pre-scan assertion
    # fails before the scanner can be invoked.
    monkeypatch.setattr(workbench_module, "load_truth_log", _tracked_load)
    monkeypatch.setattr(validation_module, "load_truth_log", _tracked_load)
    service = SessionCorpusValidationService(scan_factory=lambda **_: scanner)

    run = service.validate(
        session,
        mode="single",
        output=tmp_path / "reports",
        run_id="deferred-invalid-baseline",
    )
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    draft_info = summary["sessions"][0]["scan"]["truth_draft"]

    assert scanner.finished
    assert reads_after_scan == [session / "truth_log.json", session / "truth_log.json"]
    assert summary["sessions"][0]["truth_status"] == "invalid"
    assert draft_info["status"] == "failed"
    assert draft_info["truth_log_path"] is None
    assert "ConfigFileError" in str(draft_info["reason"])
