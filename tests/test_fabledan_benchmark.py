from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from daguandan_bridge.application import fabledan_benchmark as benchmark
from daguandan_bridge.build_manifest import write_build_manifest
from daguandan_bridge.runtime_layout import ensure_runtime_layout, resolve_runtime_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_benchmark_is_seat_swapped_and_writes_reproducible_schema(tmp_path, monkeypatch):
    model_path = tmp_path / "profile" / "models" / "best.npz"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"test weights")

    class FakeModel:
        def __init__(self, path):
            self.path = path

    def fake_round(agents, rng=None, tribute_mode=None):
        assert tribute_mode is None
        return [3, -3, 3, -3], [0, 2, 1, 3], SimpleNamespace(events=[])

    monkeypatch.setattr(benchmark, "NumpyModel", FakeModel)
    monkeypatch.setattr(benchmark, "play_round", fake_round)
    result = benchmark.FableDanBenchmarkService(tmp_path).run_fixed("profile")

    protocol = result.payload["protocol"]
    summary = result.payload["result"]
    assert protocol["games"] == 200
    assert protocol["seat_swapping"] is True
    assert protocol["seat_configurations"] == {"model_team_0_2": 100, "model_team_1_3": 100}
    assert summary["wins"] == 100
    assert summary["losses"] == 100
    assert result.output_path.is_file()
    envelope = result.to_dict()
    assert envelope["schema"] == "guandan.fabledan-benchmark-cli/1"
    assert envelope["output_path"] == str(result.output_path.resolve())
    assert envelope["output_sha256"] == hashlib.sha256(
        result.output_path.read_bytes()
    ).hexdigest()


def test_fixed_benchmark_honors_exact_new_output_and_rejects_stale_file(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "profiles" / "profile" / "models" / "best.npz"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"test weights")
    monkeypatch.setattr(benchmark, "NumpyModel", lambda _path: object())
    monkeypatch.setattr(
        benchmark,
        "play_round",
        lambda _agents, rng=None, tribute_mode=None: (
            [3, -3, 3, -3],
            [0, 2, 1, 3],
            SimpleNamespace(events=[]),
        ),
    )
    output = tmp_path / "用户 输出" / "exact-result.json"
    service = benchmark.FableDanBenchmarkService(tmp_path / "profiles")

    result = service.run_fixed("profile", output_path=output)

    assert result.output_path == output.resolve()
    with pytest.raises(FileExistsError, match="已存在"):
        service.run_fixed("profile", output_path=output)


def test_frozen_layout_benchmark_writes_only_to_runtime_output(tmp_path, monkeypatch):
    bundle = tmp_path / "immutable bundle" / "DaguandanAssistant"
    profile = bundle / "data" / "profiles" / "tencent_daguandan"
    template = profile / "templates" / "rank" / "7.png"
    model = profile / "models" / "best.npz"
    template.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    (bundle / "DaguandanAssistant.exe").write_bytes(b"MZ")
    for name in ("profile.json", "regions_config.json", "templates_config.json"):
        (profile / name).write_text("{}", encoding="utf-8")
    template.write_bytes(b"template")
    model.write_bytes(b"weights")
    write_build_manifest(
        PROJECT_ROOT,
        bundle,
        source_identity={
            "commit": "1" * 40,
            "tree": "2" * 40,
            "branch": "test",
            "dirty": False,
            "status_sha256": None,
        },
        python_identity={
            "version": "3.12.0",
            "implementation": "CPython",
            "architecture": "AMD64",
        },
        dependency_versions={},
    )
    before = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    runtime = tmp_path / "runtime"
    layout = ensure_runtime_layout(
        resolve_runtime_layout(
            frozen=True,
            bundle_root=bundle,
            environ={"DAGUANDAN_DATA_ROOT": str(runtime)},
        )
    )
    monkeypatch.setattr(benchmark, "NumpyModel", lambda _path: object())
    monkeypatch.setattr(
        benchmark,
        "play_round",
        lambda _agents, rng=None, tribute_mode=None: (
            [3, -3, 3, -3],
            [0, 2, 1, 3],
            SimpleNamespace(events=[]),
        ),
    )
    output = runtime / "benchmarks" / "frozen-result.json"

    result = benchmark.FableDanBenchmarkService(layout.profiles_root).run_fixed(
        output_path=output
    )

    assert result.output_path == output.resolve()
    assert output.is_file()
    assert not output.is_relative_to(bundle)
    after = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_packaged_benchmark_launcher_uses_exact_output_and_propagates_exit(tmp_path):
    fake = tmp_path / "fake benchmark.cmd"
    fake.write_text(
        "@echo off\n"
        "set OUT=\n"
        ":loop\n"
        "if \"%~1\"==\"\" goto run\n"
        "if \"%~1\"==\"--benchmark-output\" set \"OUT=%~2\"\n"
        "shift\n"
        "goto loop\n"
        ":run\n"
        "> \"%OUT%\" echo {\"schema\":\"fabledan.fixed-benchmark/1\"}\n"
        "exit /b 0\n",
        encoding="ascii",
    )
    output = tmp_path / "结果 目录" / "exact.json"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(
                PROJECT_ROOT
                / "release_assets"
                / "Run_FableDan_Fixed_Benchmark.ps1"
            ),
            "-ExecutablePath",
            str(fake),
            "-OutputPath",
            str(output),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == (
        "fabledan.fixed-benchmark/1"
    )
    launcher = (
        PROJECT_ROOT / "release_assets" / "Run_FableDan_Fixed_Benchmark.ps1"
    ).read_text(encoding="utf-8")
    batch = (
        PROJECT_ROOT / "release_assets" / "Run_FableDan_Fixed_Benchmark.bat"
    ).read_text(encoding="utf-8")
    assert "Get-Content -LiteralPath $output" in launcher
    assert "--benchmark-output $output" in launcher
    assert "exit /b %BENCHMARK_EXIT%" in batch
