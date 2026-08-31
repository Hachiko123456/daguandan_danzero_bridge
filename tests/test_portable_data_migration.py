from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from daguandan_bridge.build_manifest import write_build_manifest
from daguandan_bridge.portable_data_migration import (
    MIGRATION_RECEIPT_SCHEMA,
    migrate_portable_data,
    restore_portable_migration,
)
from daguandan_bridge.runtime_layout import (
    RuntimeLayoutError,
    activate_generation,
    copy_seed_resources,
    ensure_runtime_layout,
    layout_for_generation,
    resolve_runtime_layout,
    write_generation_marker,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    assert receipt["activation"]["mode"] == "immediate_next_start"
    assert receipt["activation"]["state"] == "ACTIVE"
    assert receipt["activation"]["previous_pointer"]["generation_id"] == (
        result.previous_generation_id
    )
    assert result.to_dict()["receipt_path"] == str(result.receipt_path.resolve())
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


def test_migration_restore_reinstates_exact_previous_pointer_and_is_idempotent(
    tmp_path,
):
    bundle = _candidate_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    legacy = _legacy_portable(tmp_path)
    before = ensure_runtime_layout(_layout(bundle, runtime))
    before_pointer = json.loads(
        (runtime / "data" / "v1" / "active.json").read_text(encoding="utf-8")
    )
    migrated = migrate_portable_data(legacy, layout=before)

    restored = restore_portable_migration(
        migrated.receipt_path,
        layout=_layout(bundle, runtime),
    )

    active = json.loads(
        (runtime / "data" / "v1" / "active.json").read_text(encoding="utf-8")
    )
    assert active == before_pointer
    assert restored.restored is True
    assert restored.already_restored is False
    assert restored.restored_generation_id == before.generation_id
    receipt = json.loads(migrated.receipt_path.read_text(encoding="utf-8"))
    assert receipt["restoration"]["status"] == "RESTORED"
    again = restore_portable_migration(
        migrated.receipt_path,
        layout=_layout(bundle, runtime),
    )
    assert again.already_restored is True


def test_migration_restore_refuses_to_overwrite_newer_generation(tmp_path):
    bundle = _candidate_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    migrated = migrate_portable_data(
        _legacy_portable(tmp_path),
        layout=_layout(bundle, runtime),
    )
    current = _layout(bundle, runtime)
    newer = layout_for_generation(current, f"{current.build_id}-newer")
    seed = copy_seed_resources(newer, newer.generation_root)
    write_generation_marker(newer.generation_root, newer, seed_summary=seed)
    activate_generation(newer, newer.generation_id)

    with pytest.raises(RuntimeLayoutError, match="newer selection"):
        restore_portable_migration(
            migrated.receipt_path,
            layout=_layout(bundle, runtime),
        )

    assert _layout(bundle, runtime).generation_id == newer.generation_id
    receipt = json.loads(migrated.receipt_path.read_text(encoding="utf-8"))
    assert "restoration" not in receipt


def test_data_migration_cli_reports_receipt_and_restores_it(tmp_path):
    bundle = _candidate_bundle(tmp_path)
    runtime = tmp_path / "runtime"
    legacy = _legacy_portable(tmp_path)
    script = PROJECT_ROOT / "scripts" / "manage_data_migration.py"
    common = [
        sys.executable,
        str(script),
        "--bundle-root",
        str(bundle),
        "--runtime-root",
        str(runtime),
    ]
    migrated = subprocess.run(
        [*common, "migrate", str(legacy)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert migrated.returncode == 0, migrated.stdout + migrated.stderr
    migration = json.loads(migrated.stdout)
    assert migration["activated"] is True
    receipt = Path(migration["receipt_path"])
    restored = subprocess.run(
        [*common, "restore", str(receipt)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert json.loads(restored.stdout)["restored"] is True
