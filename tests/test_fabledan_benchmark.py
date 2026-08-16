from __future__ import annotations

from types import SimpleNamespace

from daguandan_bridge.application import fabledan_benchmark as benchmark


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
