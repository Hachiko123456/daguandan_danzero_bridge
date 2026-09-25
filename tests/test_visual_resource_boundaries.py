from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.profiles import (
    profile_validation_report,
    validate_profile_directory,
    validate_profile_regions,
)
from daguandan_bridge.resource_fingerprint import (
    compare_resource_identities,
    recognition_resource_identity,
)


def _region(name: str, box: list[int], *, role: str = "play") -> dict[str, object]:
    x, y, w, h = box
    return {
        "name": name,
        "role": role,
        "abs_box": box,
        "ratio_box": [x / 1280, y / 720, w / 1280, h / 720],
    }


def test_validate_profile_regions_reports_bounds_and_critical_play_overlap():
    regions = [
        _region("left_play", [0, 0, 100, 100]),
        _region("opposite_play", [200, 0, 100, 100]),
        _region("right_play", [400, 0, 100, 100]),
        _region("my_play", [450, 0, 100, 100]),
        _region("level_rank", [-1, 0, 10, 10], role="generic"),
    ]

    issues = validate_profile_regions(regions)
    by_code = {issue.code: issue for issue in issues}

    assert by_code["roi.out_of_bounds"].region == "level_rank"
    assert by_code["roi.critical_play_overlap"].region == "right_play"
    assert by_code["roi.critical_play_overlap"].related_region == "my_play"
    assert by_code["roi.critical_play_overlap"].severity == "warning"


def test_validate_profile_directory_returns_explainable_report(tmp_path: Path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "profile.json").write_text(
        json.dumps({"base_size": [1280, 720]}), encoding="utf-8"
    )
    (root / "regions_config.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "regions": [
                    _region("left_play", [0, 0, 100, 100]),
                    _region("opposite_play", [200, 0, 100, 100]),
                    _region("right_play", [400, 0, 100, 100]),
                    _region("my_play", [700, 0, 100, 100]),
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_profile_directory(root)

    assert report["status"] == "pass"
    assert report["opening_blocking"] is False
    assert report["action_blocking"] is False
    assert report["profile_name"] == "p"
    assert report["issues"] == []


def test_resource_identity_includes_models_and_compare_explains_changes(tmp_path: Path):
    root = tmp_path / "profiles" / "p"
    (root / "templates").mkdir(parents=True)
    (root / "models").mkdir()
    for name in ("profile.json", "regions_config.json", "templates_config.json"):
        (root / name).write_text(name, encoding="utf-8")
    (root / "templates" / "card.png").write_bytes(b"template")
    (root / "models" / "recognizer.bin").write_bytes(b"model")

    first = recognition_resource_identity(root.parent, "p")
    assert first["algorithm"] == "recognition-resources-v2"
    assert {item["path"] for item in first["files"]} >= {
        "models/recognizer.bin",
        "templates/card.png",
    }

    (root / "models" / "recognizer.bin").write_bytes(b"changed")
    second = recognition_resource_identity(root.parent, "p")
    comparison = compare_resource_identities(first, second)

    assert comparison["status"] == "mismatch"
    assert comparison["changed_files"] == ["models/recognizer.bin"]


def test_play_overlap_is_action_blocking_but_does_not_fail_profile_report(tmp_path: Path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "profile.json").write_text(
        json.dumps({"base_size": [1280, 720]}), encoding="utf-8"
    )
    (root / "regions_config.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "regions": [
                    _region("left_play", [0, 0, 100, 100]),
                    _region("opposite_play", [200, 0, 100, 100]),
                    _region("right_play", [400, 0, 100, 100]),
                    _region("my_play", [450, 0, 100, 100]),
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_profile_directory(root)
    overlap = next(
        issue for issue in report["issues"]
        if issue["code"] == "roi.critical_play_overlap"
    )

    assert report["status"] == "pass"
    assert report["opening_blocking"] is False
    assert report["action_blocking"] is True
    assert overlap["severity"] == "warning"


def test_structural_roi_errors_remain_fatal_and_block_opening_and_actions():
    issues = validate_profile_regions(
        [
            _region("left_play", [-1, 0, 100, 100]),
            _region("opposite_play", [200, 0, 100, 100]),
            _region("right_play", [400, 0, 100, 100]),
            _region("my_play", [700, 0, 100, 100]),
        ]
    )

    assert any(issue.code == "roi.out_of_bounds" and issue.severity == "fatal" for issue in issues)
    report = profile_validation_report(
        [
            _region("left_play", [-1, 0, 100, 100]),
            _region("opposite_play", [200, 0, 100, 100]),
            _region("right_play", [400, 0, 100, 100]),
            _region("my_play", [700, 0, 100, 100]),
        ]
    )
    assert report["status"] == "fail"
    assert report["opening_blocking"] is True
    assert report["action_blocking"] is True

def test_missing_regions_config_blocks_opening_and_actions(tmp_path: Path):
    root = tmp_path / "missing-regions"
    root.mkdir()
    (root / "profile.json").write_text(
        json.dumps({"base_size": [1280, 720]}), encoding="utf-8"
    )

    report = validate_profile_directory(root)

    assert report["status"] == "fail"
    assert report["opening_blocking"] is True
    assert report["action_blocking"] is True
    assert any(
        issue["code"] == "profile.missing_regions_config"
        and issue["severity"] == "fatal"
        for issue in report["issues"]
    )


def test_unparseable_regions_config_blocks_opening_and_actions(tmp_path: Path):
    root = tmp_path / "invalid-regions"
    root.mkdir()
    (root / "profile.json").write_text(
        json.dumps({"base_size": [1280, 720]}), encoding="utf-8"
    )
    (root / "regions_config.json").write_text("{not-json", encoding="utf-8")

    report = validate_profile_directory(root)

    assert report["status"] == "fail"
    assert report["opening_blocking"] is True
    assert report["action_blocking"] is True
    assert any(
        issue["code"] == "profile.invalid_regions_config"
        and issue["severity"] == "fatal"
        for issue in report["issues"]
    )

