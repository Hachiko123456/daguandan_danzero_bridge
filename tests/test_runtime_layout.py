from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import pytest

from daguandan_bridge.build_manifest import write_build_manifest
import daguandan_bridge.runtime_layout as runtime_layout
from daguandan_bridge.runtime_layout import (
    ACTIVE_GENERATION_SCHEMA,
    GENERATION_MARKER,
    RuntimeLayoutError,
    ensure_runtime_layout,
    prepare_runtime_layout,
    resolve_runtime_layout,
)


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "只读 程序包" / "DaguandanAssistant"
    profile = root / "data" / "profiles" / "tencent_daguandan"
    template = profile / "templates" / "rank" / "7_level.png"
    model = profile / "models" / "best.npz"
    danzero = profile / "models" / "danzero" / "q_network.ckpt"
    template.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    danzero.parent.mkdir(parents=True)
    template.write_bytes(b"template")
    model.write_bytes(b"model")
    danzero.write_bytes(b"checkpoint")
    (profile / "profile.json").write_text(
        json.dumps({"name": "tencent_daguandan"}), encoding="utf-8"
    )
    (profile / "regions_config.json").write_text(
        json.dumps({"regions": []}), encoding="utf-8"
    )
    (profile / "templates_config.json").write_text(
        json.dumps(
            {"templates": [{"file": "templates/rank/7_level.png"}]}
        ),
        encoding="utf-8",
    )
    (root / "DaguandanAssistant.exe").write_bytes(b"exe")
    _write_manifest(root)
    return root


def _write_manifest(root: Path) -> None:
    write_build_manifest(
        root,
        root,
        source_identity={
            "commit": "c" * 40,
            "tree": "d" * 40,
            "branch": "test",
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


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _layout(bundle: Path, user_root: Path):
    return resolve_runtime_layout(
        frozen=True,
        bundle_root=bundle,
        environ={"DAGUANDAN_DATA_ROOT": str(user_root)},
    )


def test_source_mode_keeps_repository_data_even_with_frozen_override(tmp_path):
    checkout = tmp_path / "checkout"
    layout = resolve_runtime_layout(
        frozen=False,
        bundle_root=checkout,
        environ={
            "DAGUANDAN_DATA_ROOT": str(tmp_path / "ignored"),
            "LOCALAPPDATA": str(tmp_path / "local"),
        },
    )

    assert layout.frozen is False
    assert layout.bundle_root == checkout
    assert layout.data_dir == checkout / "data"
    assert layout.profiles_root == checkout / "data" / "profiles"
    assert layout.runtime_root == checkout


def test_frozen_first_run_seeds_external_versioned_generation_without_bundle_writes(
    tmp_path,
):
    bundle = _bundle(tmp_path)
    user_root = tmp_path / "用户 数据"
    before = _snapshot(bundle)
    layout = ensure_runtime_layout(_layout(bundle, user_root))

    assert layout.runtime_root == user_root
    assert layout.generation_root.parent.name == "generations"
    assert layout.build_id in layout.generation_root.name
    assert (layout.generation_root / GENERATION_MARKER).is_file()
    assert (
        layout.profiles_root
        / "tencent_daguandan"
        / "templates"
        / "rank"
        / "7_level.png"
    ).read_bytes() == b"template"
    assert layout.logs_root.is_dir()
    assert layout.diagnostics_root.is_dir()
    assert layout.preferences_root.is_dir()
    assert layout.cache_root.is_dir()
    assert _snapshot(bundle) == before

    active = json.loads(layout.active_generation_path.read_text(encoding="utf-8"))
    assert active["schema"] == ACTIVE_GENERATION_SCHEMA
    assert active["build_id"] == layout.build_id
    assert active["generation_id"] == layout.generation_id


def test_prepare_runtime_layout_never_publishes_active_generation_pointer(tmp_path):
    bundle = _bundle(tmp_path)
    user_root = tmp_path / "prepared-runtime"

    prepared = prepare_runtime_layout(_layout(bundle, user_root))

    assert prepared.generation_root.is_dir()
    assert (prepared.generation_root / GENERATION_MARKER).is_file()
    assert prepared.active_generation_path is not None
    assert not prepared.active_generation_path.exists()


def test_existing_user_customizations_and_sessions_are_not_reseeded(tmp_path):
    bundle = _bundle(tmp_path)
    base = _layout(bundle, tmp_path / "runtime")
    first = ensure_runtime_layout(base)
    model = first.profiles_root / "tencent_daguandan" / "models" / "best.npz"
    session = (
        first.profiles_root
        / "tencent_daguandan"
        / "sessions"
        / "game_test"
        / "timeline.jsonl"
    )
    model.write_bytes(b"custom-model")
    session.parent.mkdir(parents=True)
    session.write_text("{}\n", encoding="utf-8")

    second = ensure_runtime_layout(_layout(bundle, tmp_path / "runtime"))

    assert second.generation_root == first.generation_root
    assert model.read_bytes() == b"custom-model"
    assert session.read_text(encoding="utf-8") == "{}\n"


def test_concurrent_first_run_publishes_one_complete_generation(tmp_path):
    bundle = _bundle(tmp_path)
    user_root = tmp_path / "concurrent-runtime"

    def initialize(_index: int) -> Path:
        return ensure_runtime_layout(_layout(bundle, user_root)).generation_root

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(initialize, range(8)))

    assert len(set(results)) == 1
    generation = results[0]
    assert (generation / GENERATION_MARKER).is_file()
    assert (generation / "data" / "profiles" / "tencent_daguandan").is_dir()
    assert not list(generation.parent.glob("*.tmp"))


def test_missing_manifest_never_falls_back_to_writing_beside_executable(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    layout = _layout(bundle, tmp_path / "runtime")

    assert layout.manifest_status == "unidentified"
    with pytest.raises(RuntimeLayoutError, match="cannot seed user data"):
        ensure_runtime_layout(layout)
    assert not (bundle / "logs").exists()
    assert not (bundle / "sessions").exists()


def test_runtime_artifacts_accidentally_packaged_as_seed_are_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    session = (
        bundle
        / "data"
        / "profiles"
        / "tencent_daguandan"
        / "sessions"
        / "game_bad"
        / "timeline.jsonl"
    )
    session.parent.mkdir(parents=True)
    session.write_text("{}\n", encoding="utf-8")
    _write_manifest(bundle)

    with pytest.raises(RuntimeLayoutError, match="runtime artifact was included"):
        ensure_runtime_layout(_layout(bundle, tmp_path / "runtime"))


def test_unowned_nonempty_override_is_rejected_but_early_diagnostics_is_adopted(
    tmp_path,
):
    bundle = _bundle(tmp_path)
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "personal.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(RuntimeLayoutError, match="refusing to adopt"):
        ensure_runtime_layout(_layout(bundle, unowned))

    owned_bootstrap = tmp_path / "bootstrap"
    (owned_bootstrap / "diagnostics" / "runs").mkdir(parents=True)
    initialized = ensure_runtime_layout(_layout(bundle, owned_bootstrap))
    assert initialized.generation_root.is_dir()


def test_data_override_must_be_absolute_external_and_not_reparse(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    with pytest.raises(RuntimeLayoutError, match="absolute path"):
        resolve_runtime_layout(
            frozen=True,
            bundle_root=bundle,
            environ={"DAGUANDAN_DATA_ROOT": "relative-data"},
        )
    with pytest.raises(RuntimeLayoutError, match="outside the bundle"):
        _layout(bundle, bundle / "runtime")
    with pytest.raises(RuntimeLayoutError, match="outside the bundle"):
        resolve_runtime_layout(
            frozen=True,
            bundle_root=bundle,
            environ={
                "DAGUANDAN_DATA_ROOT": str(tmp_path / "valid-runtime"),
                "DAGUANDAN_DIAGNOSTICS_ROOT": str(bundle / "diagnostics"),
            },
        )

    unsafe = tmp_path / "unsafe"
    original = runtime_layout._path_is_reparse

    def simulated_reparse(path: Path) -> bool:
        return path == unsafe or original(path)

    monkeypatch.setattr(runtime_layout, "_path_is_reparse", simulated_reparse)
    with pytest.raises(RuntimeLayoutError, match="reparse point"):
        ensure_runtime_layout(_layout(bundle, unsafe))


def test_corrupted_active_pointer_fails_closed(tmp_path):
    bundle = _bundle(tmp_path)
    initial = ensure_runtime_layout(_layout(bundle, tmp_path / "runtime"))
    initial.active_generation_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeLayoutError, match="active data pointer"):
        _layout(bundle, tmp_path / "runtime")
