from __future__ import annotations


import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from daguandan_bridge.application.session_workbench import SessionDescriptor
from scripts import run_listener_regression as command


def _descriptor(
    root: Path,
    name: str,
    status: str,
    *,
    video: bool = True,
    manifest: dict[str, object] | None = None,
) -> SessionDescriptor:
    session = root / name
    if manifest is not None:
        session.mkdir(parents=True, exist_ok=True)
        (session / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return SessionDescriptor(
        root=session,
        session_id=name,
        manifest_path=session / "manifest.json",
        video_path=session / "video" / "game.avi",
        frame_index_path=session / "video" / "frame_index.jsonl",
        timeline_path=session / "timeline.jsonl",
        truth_log_path=session / "truth_log.json",
        manifest_readable=True,
        has_video=video,
        has_frame_index=True,
        has_timeline=True,
        truth_status=status,
        truth_error="",
        frame_count=10,
        timeline_event_count=3,
    )


def _strict_descriptor(root: Path, name: str, status: str = "verified") -> SessionDescriptor:
    return _descriptor(
        root,
        name,
        status,
        manifest={
            "source_observability": "strict",
            "resource_identity_match": True,
        },
    )


def _result_row(
    session_id: str = "strict",
    *,
    execution_status: str = "completed",
    listener_status: str = "complete",
    opening_status: str = "recognized",
    truth_quality: str = "passed",
    visual_quality: str = "passed",
    comparison_status: str = "passed",
    truth_qualification: str = "verified_label",
    truth_kind: str = "canonical",
    source_observability: object = "strict",
    resource_identity_match: object = True,
    first_divergence: object = None,
    fabledan_quality: str = "passed",
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "truth_log": {"kind": truth_kind},
        "truth_qualification": truth_qualification,
        "execution_status": execution_status,
        "frame_replay_status": "complete",
        "listener_status": listener_status,
        "opening_status": opening_status,
        "truth_quality": truth_quality,
        "visual_quality": visual_quality,
        "comparison_status": comparison_status,
        "fabledan_quality": fabledan_quality,
        "source_observability": source_observability,
        "resource_identity_match": resource_identity_match,
        "first_divergence": first_divergence,
    }


def test_console_progress_renders_phase_and_frame_counts(capsys):
    progress = command._ConsoleProgress(2)
    progress("session_start", "game-a", 0, 2, "prepare")
    progress("visual", "game-a", 50, 100, "frame 50")
    progress.finish()

    output = capsys.readouterr().out
    assert "12.50%" in output
    assert "\u89c6\u89c9\u76d1\u542c" in output
    assert "frame 50" in output


def test_parser_accepts_multiple_sessions_and_diagnostic_flag():
    args = command.build_parser().parse_args(
        ["--session", "game-a", "game-b", "--session", "game-c", "--include-ineligible"]
    )

    assert args.session == ["game-a", "game-b", "game-c"]
    assert args.include_ineligible is True


def test_help_explains_strict_gate_and_diagnostic_selection():
    help_text = command.build_parser().format_help()

    assert "--random-count" in help_text
    assert "--seed" in help_text
    assert "--include-ineligible" in help_text
    assert "\u4e25\u683c" in help_text
    assert "TruthLog" in help_text
    assert "LiveV2SessionRuntime" in help_text


def test_select_descriptors_accepts_multiple_session_filters(tmp_path: Path):
    descriptors = tuple(
        _descriptor(tmp_path, f"game-{index}", "verified") for index in range(3)
    )

    selected = command.select_descriptors(
        descriptors, session_filters=("game-2", "game-0", "game-2")
    )

    assert [item.session_id for item in selected] == ["game-2", "game-0"]


def test_worker_count_is_bounded_and_defaults_to_three():
    args = command.build_parser().parse_args([])
    assert args.workers == 3
    assert command._validate_worker_count(1) == 1
    assert command._validate_worker_count(3) == 3
    with pytest.raises(ValueError):
        command._validate_worker_count(4)


def test_concurrent_progress_is_aggregate_and_never_moves_backwards(capsys):
    progress = command._ConsoleProgress(2)
    progress("session_start", "game-a", 0, 2, "\u51c6\u5907")
    progress("session_start", "game-b", 0, 2, "\u51c6\u5907")
    progress("visual", "game-a", 80, 100, "\u5e27 80")
    progress("visual", "game-b", 20, 100, "\u5e27 20")
    progress("fabledan_truth", "game-a", 5, 10, "\u63a8\u8350 5/10")
    progress("visual", "game-b", 10, 100, "\u8fdf\u5230\u7684\u65e7\u8fdb\u5ea6")
    progress("session_done", "game-a", 1, 2, "\u5b8c\u6210")
    progress("session_done", "game-b", 2, 2, "\u5b8c\u6210")
    progress.finish()

    import re

    percentages = [
        float(value)
        for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)%", capsys.readouterr().out)
    ]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100.0


def test_default_selection_keeps_status_filter_but_random_requires_strict_eligible(tmp_path: Path):
    selected = command.select_descriptors(
        (
            _strict_descriptor(tmp_path, "verified"),
            _descriptor(tmp_path, "legacy", "verified"),
            _descriptor(tmp_path, "draft", "draft"),
            _descriptor(tmp_path, "missing", "missing"),
            _descriptor(tmp_path, "no-video", "verified", video=False),
        )
    )
    assert [item.session_id for item in selected] == ["verified", "legacy"]

    sampled = command.select_descriptors(
        (
            _strict_descriptor(tmp_path, "verified-0"),
            _strict_descriptor(tmp_path, "verified-1"),
            _descriptor(tmp_path, "legacy-0", "verified"),
        ),
        random_count=2,
        seed=20260912,
    )
    assert [item.session_id for item in sampled] == ["verified-0", "verified-1"]


def test_random_selection_is_reproducible_and_limited(tmp_path: Path):
    descriptors = tuple(
        _strict_descriptor(tmp_path, f"game-{index}") for index in range(8)
    )

    first = command.select_descriptors(descriptors, random_count=3, seed=20260912)
    second = command.select_descriptors(descriptors, random_count=3, seed=20260912)

    assert len(first) == 3
    assert [item.session_id for item in first] == [item.session_id for item in second]
    assert all(command.qualify_descriptor(item)["strict_eligible"] for item in first)


def test_include_ineligible_allows_diagnostic_random_sample_without_changing_qualification(tmp_path: Path):
    descriptors = (
        _strict_descriptor(tmp_path, "strict"),
        _descriptor(tmp_path, "legacy", "verified"),
    )

    selected = command.select_descriptors(
        descriptors,
        include_ineligible=True,
        random_count=2,
        seed=7,
    )

    assert [item.session_id for item in selected] == ["legacy", "strict"]
    assert command.qualify_descriptor(selected[0])["strict_eligible"] is False


def test_old_descriptor_missing_new_fields_is_unknown_and_ineligible_not_strict(tmp_path: Path):
    descriptor = _descriptor(tmp_path, "old", "verified")

    qualification = command.qualify_descriptor(descriptor)

    assert qualification["strict_eligible"] is False
    assert qualification["source_observability"] == "unknown"
    assert qualification["resource_identity"] == "unknown"
    assert "source_observability_unknown" in qualification["reasons"]
    assert "resource_identity_unknown" in qualification["reasons"]


def test_manual_or_nonobservable_source_requires_diagnostic_opt_in():
    report = command.evaluate_run(
        {"sessions": [_result_row("manual_001", source_observability=None, opening_status="opening_not_observable")]},
        include_ineligible=True,
    )

    result = report["results"][0]
    assert result["classification"] == "source_not_observable"
    assert result["strict_eligible"] is False
    assert result["first_divergence"] is None
    assert report["blocking_failures"] == []
    assert report["status"] == "PASS"


def test_resource_mismatch_is_never_a_strict_pass_and_diagnostic_mode_is_nonblocking():
    report = command.evaluate_run(
        {"sessions": [_result_row("wrong-resources", resource_identity_match=False)]},
        include_ineligible=True,
    )

    result = report["results"][0]
    assert result["classification"] == "resource_mismatch"
    assert result["resource_identity"] == "mismatch"
    assert result["strict_eligible"] is False
    assert report["strict_passes"] == []
    assert report["blocking_failures"] == []
    assert report["classification_counts"]["resource_mismatch"] == 1


def test_strict_pass_requires_all_quality_and_provenance_gates():
    report = command.evaluate_run({"sessions": [_result_row("strict-pass")]})

    assert report["status"] == "PASS"
    assert report["classification_counts"] == {
        "strict_pass": 1,
        "source_not_observable": 0,
        "resource_mismatch": 0,
        "listener_divergence": 0,
        "invalid": 0,
    }
    assert report["strict_passes"][0]["classification"] == "strict_pass"
    assert report["strict_passes"][0]["strict_eligible"] is True


def test_listener_divergence_is_blocking_and_first_divergence_is_preserved():
    divergence = {"frame_index": 42, "event": "play"}
    report = command.evaluate_run(
        {
            "sessions": [
                _result_row(
                    "bad",
                    visual_quality="failed",
                    comparison_status="failed",
                    first_divergence=divergence,
                    fabledan_quality="failed",
                )
            ]
        }
    )

    assert report["status"] == "FAIL"
    assert [item["session_id"] for item in report["blocking_failures"]] == ["bad"]
    assert report["blocking_failures"][0]["classification"] == "listener_divergence"
    assert report["blocking_failures"][0]["first_divergence"] == divergence


def test_unknown_legacy_result_is_advisory_invalid_not_guessed_strict():
    report = command.evaluate_run(
        {
            "sessions": [
                _result_row(
                    "legacy-summary",
                    source_observability=None,
                    resource_identity_match=None,
                )
            ]
        },
        include_ineligible=True,
    )

    result = report["results"][0]
    assert result["classification"] == "invalid"
    assert result["strict_eligible"] is False
    assert report["strict_passes"] == []
    assert report["blocking_failures"] == []
    assert result["failed"] is True


def test_fabledan_advisory_does_not_demote_a_strict_visual_pass():
    report = command.evaluate_run(
        {"sessions": [_result_row("verified-advisory", fabledan_quality="advisory")]}
    )

    assert report["status"] == "PASS"
    assert report["strict_passes"][0]["session_id"] == "verified-advisory"


def test_machine_readable_report_and_exit_status_for_strict_pass(tmp_path: Path, monkeypatch):
    descriptor = _strict_descriptor(tmp_path / "sessions", "strict")
    output = tmp_path / "reports"
    profile = tmp_path / "profile"
    sessions_root = tmp_path / "sessions"
    profile.mkdir()

    run_directory = output / "run"
    run_directory.mkdir(parents=True)
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps({"sessions": [_result_row("strict")], "fabledan": {}}),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self, *, profile_root):
            assert profile_root == profile

        def audit(self, *args, **kwargs):
            return SimpleNamespace(
                summary_path=summary_path,
                verification_path=None,
                run_directory=run_directory,
                execution_ok=True,
            )

    monkeypatch.setattr(command, "inspect_sessions", lambda _: (descriptor,))
    monkeypatch.setattr(command, "SessionReplayAuditService", FakeService)

    exit_code = command.main(
        [
            "--sessions-root",
            str(sessions_root),
            "--profile-root",
            str(profile),
            "--output",
            str(output),
            "--run-id",
            "run",
        ]
    )

    wrapper = json.loads((run_directory / "listener_core_regression.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert wrapper["schema"] == "guandan.listener-core-regression/1"
    assert wrapper["status"] == "PASS"
    assert wrapper["quality"]["classification_counts"]["strict_pass"] == 1
    assert wrapper["quality"]["results"][0]["classification"] == "strict_pass"


def test_main_returns_quality_failure_exit_status_for_listener_divergence(tmp_path: Path, monkeypatch):
    descriptor = _strict_descriptor(tmp_path / "sessions", "bad")
    output = tmp_path / "reports"
    profile = tmp_path / "profile"
    sessions_root = tmp_path / "sessions"
    profile.mkdir()
    run_directory = output / "run"
    run_directory.mkdir(parents=True)
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "sessions": [
                    _result_row(
                        "bad",
                        visual_quality="failed",
                        first_divergence={"frame_index": 9},
                    )
                ],
                "fabledan": {},
            }
        ),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self, *, profile_root):
            pass

        def audit(self, *args, **kwargs):
            return SimpleNamespace(
                summary_path=summary_path,
                verification_path=None,
                run_directory=run_directory,
                execution_ok=True,
            )

    monkeypatch.setattr(command, "inspect_sessions", lambda _: (descriptor,))
    monkeypatch.setattr(command, "SessionReplayAuditService", FakeService)

    exit_code = command.main(
        [
            "--sessions-root",
            str(sessions_root),
            "--profile-root",
            str(profile),
            "--output",
            str(output),
            "--run-id",
            "run",
        ]
    )

    wrapper = json.loads((run_directory / "listener_core_regression.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert wrapper["status"] == "FAIL"
    assert wrapper["quality"]["results"][0]["first_divergence"] == {"frame_index": 9}


def test_llm_summary_main_chain_is_live_v2_not_legacy(tmp_path: Path):
    command._write_llm_artifacts(
        tmp_path,
        raw_summary={"sessions": [], "fabledan": {}},
        quality={"status": "PASS", "session_count": 0},
    )

    summary = (tmp_path / "00_llm_summary.md").read_text(encoding="utf-8")
    main_chain = summary.split("```text\n", 1)[1].split("\n```", 1)[0]

    assert "LiveV2SessionRuntime" in main_chain
    assert "LiveOrchestrator" not in main_chain
    assert "\u65e7 LiveOrchestrator \u4ec5\u4fdd\u7559\u517c\u5bb9\u8def\u5f84" in summary
