from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from diagnostics.checkpoint_compare import ComparisonError, run_comparison


def _token_hash(tokens: list[int]) -> str:
    return sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def _feats_hash(feats: list[list[float]]) -> str:
    return sha256(
        np.ascontiguousarray(feats, dtype=np.float32).tobytes()
    ).hexdigest()


def _action(index: int, text: str, action_type: str, *, bomb: bool = False):
    return {
        "legal_index": index,
        "readable_action": text,
        "type": action_type,
        "type_id": 8 if bomb else 0,
        "is_bomb": bomb,
        "is_pass": action_type == "PASS",
        "physical_cards": [],
    }


def _trace_record(decision_id: str, turn_id: int, actions: list[dict]):
    tokens = [turn_id, 17, 18]
    feats = [
        [float(index), float(turn_id), 0.25, 0.5]
        for index in range(len(actions))
    ]
    return {
        "schema": "fabledan-trace/1",
        "run_id": "run-test",
        "request_id": f"request-{decision_id}",
        "decision_id": decision_id,
        "turn_id": turn_id,
        "encoding": {
            "tokens": {
                "token_length": len(tokens),
                "tokens": tokens,
                "tokens_sha256": _token_hash(tokens),
            },
            "features": {
                "feats_shape": [len(feats), 4],
                "feats_dtype": "float32",
                "feats": feats,
                "feats_sha256": _feats_hash(feats),
            },
        },
        "legal_actions": {
            "legal_count_after_cap": len(actions),
            "actions": actions,
        },
    }


def _write_trace(path: Path) -> None:
    t2 = _trace_record(
        "T0002",
        2,
        [
            _action(0, "PASS", "PASS"),
            _action(1, "88", "PAIR"),
            _action(2, "QQ", "PAIR"),
            _action(3, "9999", "BOMB", bomb=True),
        ],
    )
    t14 = _trace_record(
        "T0014",
        14,
        [
            _action(0, "PASS", "PASS"),
            _action(1, "JJJJ", "BOMB", bomb=True),
            _action(2, "JJQQKK", "PLATE"),
            _action(3, "QQKKAA", "PLATE"),
        ],
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in (t2, t14)) + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path) -> tuple[Path, list[Path]]:
    models = root / "models"
    models.mkdir()
    checkpoints = []
    paths = []
    for index, name in enumerate(("best675", "cycle990", "best1050"), start=1):
        path = models / f"{name}.npz"
        path.write_bytes(f"checkpoint-{name}".encode())
        paths.append(path)
        checkpoints.append(
            {
                "name": name,
                "path": str(path.relative_to(root)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "cycle": index,
                "total_samples": index * 100,
            }
        )
    manifest = root / "checkpoints.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "fabledan-checkpoint-manifest/1",
                "checkpoints": checkpoints,
            }
        ),
        encoding="utf-8",
    )
    return manifest, paths


class _FakeModel:
    calls: list[tuple[str, list[int], np.ndarray]] = []

    def __init__(self, path: Path):
        self.name = path.stem

    def q_values(self, tokens: np.ndarray, feats: np.ndarray) -> np.ndarray:
        self.calls.append((self.name, tokens.tolist(), feats.copy()))
        turn_id = int(tokens[0])
        values = {
            ("best675", 2): [0.1, 0.2, 0.15, 0.3],
            ("cycle990", 2): [0.4, 0.2, 0.3, 0.1],
            ("best1050", 2): [0.1, 0.5, 0.4, 0.2],
            ("best675", 14): [0.2, 0.5, 0.4, 0.3],
            ("cycle990", 14): [0.6, 0.2, 0.5, 0.4],
            ("best1050", 14): [0.1, 0.2, 0.7, 0.3],
        }
        return np.asarray(values[(self.name, turn_id)], dtype=np.float32)


def test_comparison_reuses_frozen_input_and_writes_complete_reports(tmp_path):
    trace = tmp_path / "fabledan_trace.jsonl"
    _write_trace(trace)
    manifest, _ = _write_manifest(tmp_path)
    results = tmp_path / "results"
    _FakeModel.calls = []

    comparison = run_comparison(
        trace_path=trace,
        manifest_path=manifest,
        results_directory=results,
        project_root=tmp_path,
        model_factory=_FakeModel,
    )

    assert comparison["experiment_contract"]["encode_decision_called"] is False
    assert comparison["strategy_changes"]["T0002"] == [
        {"checkpoint": "best675", "selected": "9999", "selected_q": pytest.approx(0.3)},
        {"checkpoint": "cycle990", "selected": "PASS", "selected_q": pytest.approx(0.4)},
        {"checkpoint": "best1050", "selected": "88", "selected_q": pytest.approx(0.5)},
    ]
    assert comparison["comparison_tables"]["T0014"][0]["selected"] == "JJJJ"
    assert len(comparison["turn_results"]["T0002"][0]["q_ranking"]) == 4
    assert (results / "comparison.json").is_file()
    markdown = (results / "comparison.md").read_text(encoding="utf-8")
    assert "## T0002" in markdown
    assert "## T0014" in markdown
    assert "## Complete Rankings" in markdown

    for turn_id in (2, 14):
        calls = [call for call in _FakeModel.calls if call[1][0] == turn_id]
        assert len(calls) == 3
        assert calls[0][1] == calls[1][1] == calls[2][1]
        assert np.array_equal(calls[0][2], calls[1][2])
        assert np.array_equal(calls[1][2], calls[2][2])


def test_checkpoint_hash_mismatch_fails_before_inference_or_output(tmp_path):
    trace = tmp_path / "fabledan_trace.jsonl"
    _write_trace(trace)
    manifest, paths = _write_manifest(tmp_path)
    paths[-1].write_bytes(b"unexpected checkpoint")
    called = False

    def model_factory(_path):
        nonlocal called
        called = True
        raise AssertionError("模型不应在 checkpoint 预检失败时加载")

    with pytest.raises(ComparisonError, match="best1050: SHA256 不匹配"):
        run_comparison(
            trace_path=trace,
            manifest_path=manifest,
            results_directory=tmp_path / "results",
            project_root=tmp_path,
            model_factory=model_factory,
        )

    assert called is False
    assert not (tmp_path / "results" / "comparison.json").exists()
    assert not (tmp_path / "results" / "comparison.md").exists()


def test_frozen_input_hash_mismatch_fails_before_inference(tmp_path):
    trace = tmp_path / "fabledan_trace.jsonl"
    _write_trace(trace)
    records = [json.loads(line) for line in trace.read_text("utf-8").splitlines()]
    records[0]["encoding"]["tokens"]["tokens"][0] = 99
    trace.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )
    manifest, _ = _write_manifest(tmp_path)

    with pytest.raises(ComparisonError, match="T0002: tokens SHA256 不匹配"):
        run_comparison(
            trace_path=trace,
            manifest_path=manifest,
            results_directory=tmp_path / "results",
            project_root=tmp_path,
            model_factory=lambda _path: pytest.fail("模型不应在 input 预检失败时加载"),
        )


def test_windows_utf8_bom_inputs_are_supported(tmp_path):
    trace = tmp_path / "fabledan_trace.jsonl"
    _write_trace(trace)
    trace.write_text(trace.read_text("utf-8"), encoding="utf-8-sig")
    manifest, _ = _write_manifest(tmp_path)
    manifest.write_text(manifest.read_text("utf-8"), encoding="utf-8-sig")

    comparison = run_comparison(
        trace_path=trace,
        manifest_path=manifest,
        results_directory=tmp_path / "results",
        project_root=tmp_path,
        model_factory=_FakeModel,
    )

    assert comparison["frozen_inputs"]["T0002"]["legal_count"] == 4
