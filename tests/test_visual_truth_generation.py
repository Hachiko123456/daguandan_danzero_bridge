from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.application.visual_truth_generation import (
    VISUAL_SCAN_PROVENANCE,
    VisualTruthGenerationService,
    write_visual_truth_generation_report,
)
from daguandan_bridge.live.replay import ReplayComparison, ReplayWarning, VisualPipelineReplayResult
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    load_truth_log,
    save_truth_log,
)


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _session(root: Path, name: str = "game-batch") -> Path:
    session = root / name
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"session_id": name}), encoding="utf-8"
    )
    reducer = LiveReducer(name)
    initial = reducer.confirm_initial_state(
        round_level="2", hand=HAND, lead_player="self"
    )
    (session / "timeline.jsonl").write_text(
        json.dumps(initial.to_dict(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    video = session / "video"
    video.mkdir()
    (video / "frame_index.jsonl").write_text(
        "".join(
            json.dumps(
                {"frame_index": index, "monotonic_ms": index * 100, "wall_time": f"t{index}"}
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    return session


def _runner(
    *,
    warning: ReplayWarning | None = None,
    processed: int = 3,
    finished_actor: str | None = None,
    unresolved_event: bool = False,
    terminal_history_gap: bool = False,
):
    def replay(_session, _recognition, *, output_root, on_turn, **_kwargs):
        on_turn(
            {
                "kind": "action",
                "turn_id": 1,
                "trick_id": 1,
                "frame_index": 1,
                "actor": "self",
                "recognized_cards": ["2S"],
                "recognized_pass": False,
            }
        )
        output = Path(output_root) / "visual_replay.jsonl"
        events = [
            {
                "event_id": "EVT-000001",
                "event_type": "player_played",
                "actor": "self",
                "payload": {
                    "cards": ["2S"],
                    "is_pass": False,
                    **(
                        {"integrity_warnings": ["wildcard_interpretation_ambiguous"]}
                        if unresolved_event
                        else {}
                    ),
                },
            }
        ]
        if finished_actor is not None:
            events.append(
                {
                    "event_id": "EVT-000002",
                    "event_type": "player_finished",
                    "actor": finished_actor,
                    "payload": {"placement": 3, "frame_index": 2},
                }
            )
        if terminal_history_gap:
            events.append(
                {
                    "event_id": "EVT-000003",
                    "event_type": "terminal_history_gap",
                    "actor": None,
                    "payload": {},
                }
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "events": events,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return VisualPipelineReplayResult(
            output_path=output,
            comparison_path=Path(output_root) / "comparison.json",
            frame_count=processed,
            warnings=(warning,) if warning is not None else (),
            comparison=ReplayComparison((), (), (), (), ()),
        )

    return replay


def _service(
    *,
    warning: ReplayWarning | None = None,
    processed: int = 3,
    finished_actor: str | None = None,
    unresolved_event: bool = False,
    terminal_history_gap: bool = False,
):
    return VisualTruthGenerationService(
        recognition_factory=lambda _session: object(),
        replay=_runner(
            warning=warning,
            processed=processed,
            finished_actor=finished_actor,
            unresolved_event=unresolved_event,
            terminal_history_gap=terminal_history_gap,
        ),
    )


def test_discovery_is_runtime_root_scoped_and_ignores_non_sessions(tmp_path: Path):
    root = tmp_path / "sessions"
    first = _session(root, "game-first")
    (root / "not-a-session").mkdir()

    assert VisualTruthGenerationService.discover_sessions([root]) == (first.resolve(),)


def test_generation_stages_missing_truth_without_publishing(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)

    run = _service().generate([root], run_id="batch_stage")
    result = run.sessions[0]

    assert result.status == "staged"
    assert result.published is False
    assert not (session / "truth_log.json").exists()
    assert result.stage_directory is not None
    staged = load_truth_log(result.stage_directory / "truth_log.json", session_id="game-batch")
    assert staged.label_status == "draft"
    assert staged.provenance.source == VISUAL_SCAN_PROVENANCE
    assert all(turn.provenance.source == VISUAL_SCAN_PROVENANCE for turn in staged.turns)
    assert result.receipt_path is not None and result.receipt_path.is_file()
    assert (result.stage_directory / "replay" / "visual_replay.jsonl").is_file()
    manifest = json.loads((result.stage_directory / "manifest.json").read_text(encoding="utf-8"))
    assert "replay/" in manifest["write_scope"]["allowed"]
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["stage"]["replay"] == "replay"

    report = write_visual_truth_generation_report(tmp_path / "reports", run)
    report_data = json.loads(report.read_text(encoding="utf-8"))
    assert report_data["summary"]["staged"] == 1


def test_generation_never_replaces_existing_truth_even_with_publish(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)
    existing = TruthLog(
        "game-batch",
        TruthInitialState("2", "self", HAND),
        (),
    )
    truth_path = session / "truth_log.json"
    save_truth_log(truth_path, existing)
    before = truth_path.read_bytes()

    result = _service().generate([root], publish=True, run_id="batch_existing").sessions[0]

    assert result.status == "staged_existing"
    assert result.published is False
    assert truth_path.read_bytes() == before
    assert result.stage_directory is not None
    assert (result.stage_directory / "comparison.json").is_file()


def test_generation_only_missing_skips_existing_canonical_log(tmp_path: Path):
    root = tmp_path / "sessions"
    existing_session = _session(root, "game-existing")
    missing_session = _session(root, "game-missing")
    save_truth_log(
        existing_session / "truth_log.json",
        TruthLog("game-existing", TruthInitialState("2", "self", HAND), ()),
    )

    run = _service().generate([root], only_missing=True, run_id="batch_missing")

    assert [item.session.name for item in run.sessions] == ["game-missing"]
    assert not (existing_session / "derived" / "truth_scan_drafts" / "batch_missing").exists()
    assert (missing_session / "derived" / "truth_scan_drafts" / "batch_missing").is_dir()


def test_generation_blocks_publish_when_existing_staged_truth_differs(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)
    trusted = session / "derived" / "truth_scan_drafts" / "trusted"
    trusted.mkdir(parents=True)
    save_truth_log(
        trusted / "truth_log.json",
        TruthLog("game-batch", TruthInitialState("2", "self", HAND), ()),
    )

    result = _service().generate([root], publish=True, run_id="batch_compare").sessions[0]

    assert result.status == "blocked"
    assert result.published is False
    assert not (session / "truth_log.json").exists()
    assert result.stage_directory is not None
    receipt = json.loads(
        (result.stage_directory / "visual_scan_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["comparison_reference"]["kind"] == "staged_draft"
    manifest = json.loads((result.stage_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical"]["available"] is False
    assert manifest["comparison_reference"]["path"] == (
        "derived/truth_scan_drafts/trusted/truth_log.json"
    )
    gates = {str(gate["name"]): bool(gate["passed"]) for gate in result.gates}
    assert gates["trusted_staged_draft_match"] is False


def test_generation_can_publish_when_staged_truth_is_semantically_exact(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)
    trusted = session / "derived" / "truth_scan_drafts" / "trusted"
    trusted.mkdir(parents=True)
    save_truth_log(
        trusted / "truth_log.json",
        TruthLog(
            "game-batch",
            TruthInitialState("2", "self", HAND),
            (),
        ),
    )
    # The runner's one action is the reviewed reference's action semantics.
    save_truth_log(
        trusted / "truth_log.json",
        TruthLog(
            "game-batch",
            TruthInitialState("2", "self", HAND),
            (TruthTurn(1, 1, "self", False, ("2S",)),),
        ),
    )

    result = _service().generate(
        [root],
        publish=True,
        run_id="batch_match",
        trusted_staged_references=[trusted / "truth_log.json"],
    ).sessions[0]

    assert result.status == "published"
    assert result.published is True
    gates = {str(gate["name"]): bool(gate["passed"]) for gate in result.gates}
    assert gates["trusted_staged_draft_match"] is True
    assert (session / "truth_log.json").is_file()


def test_trusted_stage_can_corroborate_only_unresolved_visual_semantics(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)
    trusted = session / "derived" / "truth_scan_drafts" / "trusted"
    trusted.mkdir(parents=True)
    save_truth_log(
        trusted / "truth_log.json",
        TruthLog(
            "game-batch",
            TruthInitialState("2", "self", HAND),
            (TruthTurn(1, 1, "self", False, ("2S",)),),
        ),
    )

    result = _service(unresolved_event=True).generate(
        [root],
        publish=True,
        run_id="batch_trusted_uncertainty",
        trusted_staged_references=[trusted / "truth_log.json"],
    ).sessions[0]

    assert result.status == "published"
    assert (session / "truth_log.json").is_file()
    gates = {str(gate["name"]): bool(gate["passed"]) for gate in result.gates}
    assert gates["resolved_cards_and_wildcards"] is False
    assert gates["trusted_staged_draft_match"] is True
    assert gates["trusted_staged_draft_corroborates_uncertainty"] is True
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["scan"]["trusted_staged_draft_corroborates_uncertainty"] == {
        "applied": True,
        "comparison_reference": "derived/truth_scan_drafts/trusted/truth_log.json",
        "comparison_reference_trusted": True,
    }


def test_unresolved_semantics_still_blocks_without_trusted_stage(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)

    result = _service(unresolved_event=True).generate(
        [root], publish=True, run_id="batch_unresolved"
    ).sessions[0]

    assert result.status == "blocked"
    assert not (session / "truth_log.json").exists()
    gates = {str(gate["name"]): bool(gate["passed"]) for gate in result.gates}
    assert gates["resolved_cards_and_wildcards"] is False
    assert "trusted_staged_draft_corroborates_uncertainty" not in gates


def test_generation_blocks_partial_or_warned_scan_before_publish(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)
    warning = ReplayWarning("missing_video_frames", "only two frames decoded")

    result = _service(warning=warning, processed=2).generate(
        [root], publish=True, run_id="batch_blocked"
    ).sessions[0]

    assert result.status == "blocked"
    assert not (session / "truth_log.json").exists()
    gates = {str(gate["name"]): bool(gate["passed"]) for gate in result.gates}
    assert gates["full_indexed_frames"] is False
    assert gates["replay_warnings"] is False


def test_terminal_history_gap_gate_exposes_detected_evidence(tmp_path: Path):
    root = tmp_path / "sessions"
    _session(root)

    result = _service(terminal_history_gap=True).generate(
        [root], publish=True, run_id="batch_gap"
    ).sessions[0]

    gate = next(gate for gate in result.gates if gate["name"] == "terminal_history_gap")
    assert gate == {
        "name": "terminal_history_gap",
        "passed": False,
        "detail": "terminal_history_gap detected in replay evidence",
    }


def test_generation_blocks_visual_finish_without_all_cards_in_draft(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)

    result = _service(finished_actor="self").generate(
        [root], publish=True, run_id="batch_missing_final_action"
    ).sessions[0]

    assert result.status == "blocked"
    assert not (session / "truth_log.json").exists()
    gates = {str(gate["name"]): bool(gate["passed"]) for gate in result.gates}
    assert gates["visual_finish_card_accounting"] is False
    assert result.receipt_path is not None
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    finish = receipt["scan"]["visual_finish_evidence"]
    assert finish == [
        {
            "event_id": "EVT-000002",
            "actor": "self",
            "rank": 3,
            "frame_index": 2,
            "draft_remaining": len(HAND) - 1,
            "matches_draft_remaining": False,
        }
    ]


def test_generation_publish_creates_valid_visual_truth_once(tmp_path: Path):
    root = tmp_path / "sessions"
    session = _session(root)

    result = _service().generate([root], publish=True, run_id="batch_publish").sessions[0]

    assert result.status == "published"
    assert result.published
    published = load_truth_log(session / "truth_log.json", session_id="game-batch")
    assert [(turn.actor, turn.cards) for turn in published.turns] == [("self", ("2S",))]
    assert published.provenance.source == VISUAL_SCAN_PROVENANCE
    assert published.turns[0].provenance.source == VISUAL_SCAN_PROVENANCE
