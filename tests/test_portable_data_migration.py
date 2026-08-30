from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from daguandan_bridge.build_manifest import write_build_manifest
from daguandan_bridge.portable_data_migration import (
    MIGRATION_RECEIPT_SCHEMA,
    migrate_portable_data,
)
from daguandan_bridge.runtime_layout import (
    RuntimeLayoutError,
    ensure_runtime_layout,
    resolve_runtime_layout,
)


def _candidate_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "candidate" / "DaguandanAssistant"
    profile = root / "data" / "profiles" / "tencent_daguandan"
    template = profile / "templates" / "rank" / "7_level.png"
    model = profile / "models" / "best.npz"
    template.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    template.write_bytes(b"new-template")
    model.write_bytes(b"new-model")
    (profile / "profile.json").write_text('{"name":"tencent_daguandan"}', encoding="utf-8")
    (profile / "regions_config.json").write_text('{"regions":[]}', encoding="utf-8")
    (profile / "templates_config.json").write_text(
        '{"templates":[{"file":"templates/rank/7_level.png"}]}',
        encoding="utf-8",
    )
    (root / "DaguandanAssistant.exe").write_bytes(b"candidate")
    write_build_manifest(
        root,
        root,
        source_identity={
            "commit": "a" * 40,
            "tree": "b" * 40,
            "branch": "candidate",
            "dirty": False,
            "status_sha256": None,
        },
        python_identity={
            "version": "3.12.0",
            "implementation": "CPython",
            "architecture": "AMD64",
        },
        dependency_versions={"PyInstaller": "test"},
    )
    return root


def _legacy_portable(tmp_path: Path) -> Path:
    root = tmp_path / "旧版 程序"
    profile = root / "data" / "profiles" / "tencent_daguandan"
    (profile / "templates" / "rank").mkdir(parents=True)
    (profile / "models").mkdir()
    session = profile / "sessions" / "game_old"
    session.mkdir(parents=True)
    (profile / "profile.json").write_text('{"name":"tencent_daguandan"}', encoding="utf-8")
    (profile / "regions_config.json").write_text('{"regions":[{"name":"legacy"}]}', encoding="utf-8")
    (profile / "templates_config.json").write_text('{"templates":[]}', encoding="utf-8")
    (profile / "templates" / "rank" / "7_level.png").write_bytes(b"legacy-template")
    (profile / "models" / "best.npz").write_bytes(b"legacy-model")
    (session / "timeline.jsonl").write_text('{"event":"old"}\n', encoding="utf-8")
    (profile / "hand_template_calibration.json").write_text("{}", encoding="utf-8")
    (profile / "diagnostics").mkdir()
    (profile / "diagnostics" / "debug.log").write_text("legacy log", encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _layout(bundle: Path, runtime: Path):
    return resolve_runtime_layout(
        frozen=True,
        bundle_root=bundle,
        environ={"DAGUANDAN_DATA_ROOT": str(runtime)},
    )


def test_explicit_migration_creates_new_generation_and_preserves_old_portable(tmp_path):
    bundle = _candidate_bundle(tmp_path)
    legacy = _legacy_portable(tmp_path)
    runtime = tmp_path / "user-data"
    base = ensure_runtime_layout(_layout(bundle, runtime))
    old_snapshot = _snapshot(legacy)

    result = migrate_portable_data(legacy, layout=base)

    assert result.source_preserved is True
    assert _snapshot(legacy) == old_snapshot
    assert result.generation_id != base.generation_id
    active = _layout(bundle, runtime)
    assert active.generation_id == result.generation_id
    profile = active.profiles_root / "tencent_daguandan"
    assert (profile / "models" / "best.npz").read_bytes() == b"legacy-model"
    assert (profile / "templates" / "rank" / "7_level.png").read_bytes() == b"legacy-template"
    assert (profile / "sessions" / "game_old" / "timeline.jsonl").is_file()
    assert not (profile / "hand_template_calibration.json").exists()
    assert not (profile / "diagnostics").exists()
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == MIGRATION_RECEIPT_SCHEMA
    assert receipt["source"]["preserved_in_place"] is True
    assert receipt["copy"]["excluded_count"] == 2
    assert str(legacy) not in result.receipt_path.read_text(encoding="utf-8")


def test_same_migration_is_idempotent_and_does_not_create_another_generation(tmp_path):
    bundle = _candidate_bundle(tmp_path)
    legacy = _legacy_portable(tmp_path)
    runtime = tmp_path / "runtime"
    first = migrate_portable_data(legacy, layout=_layout(bundle, runtime))
    generations = first.receipt_path.parent.parent / "data" / "v1" / "generations"
    before = sorted(path.name for path in generations.iterdir())

    second = migrate_portable_data(legacy, layout=_layout(bundle, runtime))

    assert second.migration_id == first.migration_id
    assert second.generation_id == first.generation_id
    assert sorted(path.name for path in generations.iterdir()) == before


def test_migration_rejects_source_overlap_unknown_layout_and_source_mode(tmp_path):
    bundle = _candidate_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    layout = ensure_runtime_layout(_layout(bundle, runtime))

    with pytest.raises(RuntimeLayoutError, match="overlaps the runtime root"):
        migrate_portable_data(runtime, layout=layout)

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    with pytest.raises(RuntimeLayoutError, match="exactly one"):
        migrate_portable_data(unknown, layout=layout)

    source_layout = resolve_runtime_layout(
        frozen=False,
        bundle_root=tmp_path / "checkout",
        environ={"LOCALAPPDATA": str(tmp_path / "local")},
    )
    with pytest.raises(RuntimeLayoutError, match="only from the frozen"):
        migrate_portable_data(_legacy_portable(tmp_path / "second"), layout=source_layout)
