from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from diagnostics.checkpoint_compare.register import (
    RegistrationError,
    register_checkpoint,
)


def _manifest(path: Path, checkpoints=None) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "fabledan-checkpoint-manifest/1",
                "checkpoints": list(checkpoints or []),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_register_infers_cycle_hashes_model_and_appends_manifest(tmp_path):
    model = (
        tmp_path
        / "data"
        / "profiles"
        / "tencent_daguandan"
        / "models"
        / "best1150.npz"
    )
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint-1150")
    manifest = _manifest(tmp_path / "checkpoints.json")
    validated = []

    result = register_checkpoint(
        name="best1150",
        total_samples=150_000_000,
        manifest_path=manifest,
        project_root=tmp_path,
        model_validator=lambda path: validated.append(path),
    )

    expected_hash = sha256(model.read_bytes()).hexdigest()
    assert result.changed is True
    assert result.entry == {
        "name": "best1150",
        "path": "data/profiles/tencent_daguandan/models/best1150.npz",
        "sha256": expected_hash,
        "cycle": 1150,
        "total_samples": 150_000_000,
    }
    assert validated == [model]
    assert json.loads(manifest.read_text("utf-8"))["checkpoints"] == [result.entry]

    repeated = register_checkpoint(
        name="best1150",
        total_samples=150_000_000,
        manifest_path=manifest,
        project_root=tmp_path,
        model_validator=lambda _path: None,
    )
    assert repeated.changed is False


def test_register_rejects_runtime_alias_and_unconfirmed_replacement(tmp_path):
    models = tmp_path / "data" / "profiles" / "tencent_daguandan" / "models"
    models.mkdir(parents=True)
    runtime_model = models / "best.npz"
    runtime_model.write_bytes(b"runtime")
    manifest = _manifest(tmp_path / "checkpoints.json")

    with pytest.raises(RegistrationError, match="运行时固定别名"):
        register_checkpoint(
            name="best",
            total_samples=1,
            manifest_path=manifest,
            project_root=tmp_path,
            model_validator=lambda _path: None,
        )

    candidate = models / "best1150.npz"
    candidate.write_bytes(b"first")
    register_checkpoint(
        name="best1150",
        total_samples=100,
        manifest_path=manifest,
        project_root=tmp_path,
        model_validator=lambda _path: None,
    )
    candidate.write_bytes(b"second")
    with pytest.raises(RegistrationError, match="确认替换时使用 --replace"):
        register_checkpoint(
            name="best1150",
            total_samples=200,
            manifest_path=manifest,
            project_root=tmp_path,
            model_validator=lambda _path: None,
        )


def test_register_validates_model_before_changing_manifest(tmp_path):
    models = tmp_path / "data" / "profiles" / "tencent_daguandan" / "models"
    models.mkdir(parents=True)
    (models / "best1150.npz").write_bytes(b"invalid")
    manifest = _manifest(tmp_path / "checkpoints.json")
    before = manifest.read_bytes()

    with pytest.raises(RegistrationError, match="无法由 NumpyModel 加载"):
        register_checkpoint(
            name="best1150",
            total_samples=100,
            manifest_path=manifest,
            project_root=tmp_path,
            model_validator=lambda _path: (_ for _ in ()).throw(ValueError("bad npz")),
        )

    assert manifest.read_bytes() == before
