from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_listener_regression as command
from daguandan_bridge.application.session_workbench import SessionDescriptor


def _descriptor(root: Path, name: str, status: str, *, video: bool = True) -> SessionDescriptor:
    session = root / name
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


def test_console_progress_renders_phase_and_frame_counts(capsys):
    progress = command._ConsoleProgress(2)
    progress("session_start", "game-a", 0, 2, "prepare")
    progress("visual", "game-a", 50, 100, "frame 50")
    progress.finish()

    output = capsys.readouterr().out
    assert "12.50%" in output
    assert "\u89c6\u89c9\u76d1\u542c" in output
    assert "frame 50" in output


def test_parser_accepts_multiple_session_ids_in_one_option():
    args = command.build_parser().parse_args(
        ["--session", "game-a", "game-b", "--session", "game-c"]
    )

    assert args.session == ["game-a", "game-b", "game-c"]


def test_select_descriptors_accepts_multiple_session_filters(tmp_path: Path):
    descriptors = tuple(
        _descriptor(tmp_path, f"game-{index}", "verified")
        for index in range(3)
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
    progress("session_start", "game-a", 0, 2, "准备")
    progress("session_start", "game-b", 0, 2, "准备")
    progress("visual", "game-a", 80, 100, "帧 80")
    progress("visual", "game-b", 20, 100, "帧 20")
    progress("fabledan_truth", "game-a", 5, 10, "推荐 5/10")
    progress("visual", "game-b", 10, 100, "迟到的旧进度")
    progress("session_done", "game-a", 1, 2, "完成")
    progress("session_done", "game-b", 2, 2, "完成")
    progress.finish()

    import re

    percentages = [
        float(value)
        for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)%", capsys.readouterr().out)
    ]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100.0


def test_default_selection_uses_only_verified_video_sessions(tmp_path: Path):
    selected = command.select_descriptors(
        (
            _descriptor(tmp_path, "verified", "verified"),
            _descriptor(tmp_path, "draft", "draft"),
            _descriptor(tmp_path, "missing", "missing"),
            _descriptor(tmp_path, "no-video", "verified", video=False),
        )
    )

    assert [item.session_id for item in selected] == ["verified"]


def test_random_selection_is_reproducible_and_limited(tmp_path: Path):
    descriptors = tuple(
        _descriptor(tmp_path, f"game-{index}", "verified")
        for index in range(8)
    )

    first = command.select_descriptors(
        descriptors, random_count=3, seed=20260912
    )
    second = command.select_descriptors(
        descriptors, random_count=3, seed=20260912
    )

    assert len(first) == 3
    assert [item.session_id for item in first] == [
        item.session_id for item in second
    ]


def test_help_explains_random_selection_and_seed():
    help_text = command.build_parser().format_help()

    assert "--random-count" in help_text
    assert "--seed" in help_text
    assert "--workers" in help_text
    assert "一次指定多个" in help_text
    assert "TruthLog" in help_text
    assert "LiveV2SessionRuntime" in help_text
    assert "LiveOrchestrator" in help_text


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
    assert "旧 LiveOrchestrator 仅保留兼容路径" in summary


def test_verified_label_is_strict_and_listener_incomplete_blocks():
    report = command.evaluate_run(
        {
            "sessions": [
                {
                    "session_id": "verified-label",
                    "truth_log": {"kind": "canonical"},
                    "truth_qualification": "verified_label",
                    "execution_status": "completed",
                    "listener_status": "incomplete",
                    "truth_quality": "passed",
                    "visual_quality": "incomplete",
                    "fabledan_quality": "passed",
                }
            ]
        }
    )

    assert report["status"] == "FAIL"
    assert [item["session_id"] for item in report["blocking_failures"]] == [
        "verified-label"
    ]
    assert report["advisory_results"] == []


@pytest.mark.parametrize("truth_kind", ("canonical", "verified"))
@pytest.mark.parametrize("qualification", ("verified_label", "verified", "trusted_for_run"))
def test_verified_truth_kinds_and_qualifications_are_all_strict(truth_kind, qualification):
    report = command.evaluate_run(
        {
            "sessions": [
                {
                    "session_id": f"{truth_kind}-{qualification}",
                    "truth_log": {"kind": truth_kind},
                    "truth_qualification": qualification,
                    "execution_status": "completed",
                    "frame_replay_status": "complete",
                    "listener_status": "incomplete",
                    "truth_quality": "passed",
                    "visual_quality": "failed",
                    "comparison_status": "failed",
                    "fabledan_quality": "passed",
                }
            ]
        }
    )

    assert report["status"] == "FAIL"
    assert [item["session_id"] for item in report["blocking_failures"]] == [
        f"{truth_kind}-{qualification}"
    ]
    assert report["advisory_results"] == []


def test_non_verified_results_remain_advisory_even_when_failed():
    report = command.evaluate_run(
        {
            "sessions": [
                {
                    "session_id": "draft",
                    "truth_log": {"kind": "canonical"},
                    "truth_qualification": "reference_only",
                    "execution_status": "incomplete",
                    "listener_status": "incomplete",
                    "visual_quality": "failed",
                }
            ]
        }
    )

    assert report["status"] == "PASS"
    assert report["blocking_failures"] == []
    assert report["advisory_results"][0]["failed"] is True


def test_selection_filters_and_optional_diagnostic_sessions(tmp_path: Path):
    selected = command.select_descriptors(
        (
            _descriptor(tmp_path, "verified", "verified"),
            _descriptor(tmp_path, "draft", "draft"),
            _descriptor(tmp_path, "missing", "missing"),
        ),
        session_filters=("draft", "missing"),
        include_draft=True,
        include_no_truth=True,
    )

    assert [item.session_id for item in selected] == ["draft", "missing"]


def test_fabledan_stale_visual_advice_is_advisory_not_failure():
    from daguandan_bridge.application.session_replay_audit import _advice_quality

    visual = {
        "available": True,
        "completed": True,
        "status": "completed_with_stale",
        "advice": {"requested": 26, "ready": 21, "failed": 0, "stale": 5, "timeout": 0},
    }
    assert _advice_quality({"available": False}, visual) == "advisory"
    assert _advice_quality(
        {
            "available": True,
            "completed": True,
            "status": "passed",
            "advice": {"requested": 26, "ready": 26, "failed": 0, "stale": 0, "timeout": 0},
        },
        visual,
    ) == "passed"


def test_verified_visual_or_comparison_failure_is_blocking_even_with_not_evaluated_fabledan():
    report = command.evaluate_run(
        {
            "sessions": [
                {
                    "session_id": "verified-advisory",
                    "truth_log": {"kind": "canonical"},
                    "truth_qualification": "verified_label",
                    "execution_status": "completed",
                    "frame_replay_status": "complete",
                    "listener_status": "complete",
                    "opening_status": "recognized",
                    "truth_quality": "passed",
                    "visual_quality": "passed",
                    "comparison_status": "passed",
                    "fabledan_quality": "advisory",
                    "fabledan": {
                        "truth_driven": {
                            "available": True,
                            "completed": True,
                            "status": "passed",
                            "advice": {"requested": 78, "ready": 78, "failed": 0, "timeout": 0},
                        },
                        "visual_driven": {
                            "available": True,
                            "completed": True,
                            "status": "completed_with_stale",
                            "advice": {"requested": 26, "ready": 21, "failed": 0, "stale": 5, "timeout": 0},
                        },
                    },
                }
            ]
        }
    )

    assert report["status"] == "PASS"
    assert report["blocking_failures"] == []
    assert report["strict_passes"][0]["session_id"] == "verified-advisory"


def test_verified_visual_or_comparison_failure_is_blocking_even_with_not_evaluated_fabledan():
    report = command.evaluate_run({
        "sessions": [{
            "session_id": "verified",
            "truth_log": {"kind": "canonical"},
            "truth_qualification": "verified_label",
            "execution_status": "completed",
            "frame_replay_status": "complete",
            "listener_status": "complete",
            "opening_status": "recognized",
            "truth_quality": "passed",
            "visual_quality": "failed",
            "comparison_status": "failed",
            "fabledan_quality": "not_evaluated",
        }]
    })
    assert report["status"] == "FAIL"
    assert [item["session_id"] for item in report["blocking_failures"]] == ["verified"]


def test_evaluate_run_blocks_verified_listener_or_fabledan_failure():
    report = command.evaluate_run(
        {
            "sessions": [
                {
                    "session_id": "good",
                    "truth_log": {"kind": "canonical"},
                    "truth_qualification": "verified",
                    "execution_status": "completed",
                    "truth_quality": "passed",
                    "visual_quality": "passed",
                    "fabledan_quality": "passed",
                },
                {
                    "session_id": "bad",
                    "truth_log": {"kind": "canonical"},
                    "truth_qualification": "verified",
                    "execution_status": "completed",
                    "truth_quality": "passed",
                    "visual_quality": "failed",
                    "fabledan_quality": "failed",
                    "first_divergence": {"frame_index": 42},
                },
            ]
        }
    )

    assert report["status"] == "FAIL"
    assert [item["session_id"] for item in report["blocking_failures"]] == ["bad"]
    assert report["blocking_failures"][0]["first_divergence"] == {"frame_index": 42}


def test_evaluate_run_keeps_unverified_sessions_advisory():
    report = command.evaluate_run(
        {
            "sessions": [
                {
                    "session_id": "verified",
                    "truth_log": {"kind": "canonical"},
                    "truth_qualification": "verified",
                    "execution_status": "completed",
                    "truth_quality": "passed",
                    "visual_quality": "passed",
                    "fabledan_quality": "passed",
                },
                {
                    "session_id": "draft",
                    "truth_log": {"kind": "canonical"},
                    "truth_qualification": "unverified",
                    "execution_status": "incomplete",
                    "truth_quality": "diagnostic",
                    "visual_quality": "incomplete",
                    "fabledan_quality": "failed",
                },
            ]
        }
    )

    assert report["status"] == "PASS"
    assert report["blocking_failures"] == []
    assert report["advisory_results"][0]["session_id"] == "draft"
    assert report["advisory_results"][0]["failed"] is True
