from __future__ import annotations

"""Reproducible, fixed FableDan evaluation against the RuleAgent baseline."""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import math
from pathlib import Path
import random

from ..config import PROFILES_ROOT
from ..fabledan.advisor import UPSTREAM_COMMIT
from ..fabledan._vendor.fabledan.agents import NumpyAgent, RuleAgent
from ..fabledan._vendor.fabledan.engine import play_round
from ..fabledan._vendor.fabledan.model_np import NumpyModel
from ..profiles import normalize_profile_name
from ..storage import atomic_write_json


FIXED_BENCHMARK_ID = "fabledan-standard-no-tribute-v1"
FIXED_GAME_COUNT = 200
FIXED_SEED = 20260816


@dataclass(frozen=True)
class FableDanBenchmarkResult:
    output_path: Path
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "guandan.fabledan-benchmark-cli/1",
            "status": "PASS",
            "output_path": str(self.output_path.resolve()),
            "output_sha256": sha256(self.output_path.read_bytes()).hexdigest(),
            "benchmark": self.payload,
        }


class FableDanBenchmarkService:
    """Run only the published fixed benchmark; no runtime knobs are exposed."""

    def __init__(self, profiles_root: Path = PROFILES_ROOT) -> None:
        self.profiles_root = Path(profiles_root)

    def run_fixed(
        self,
        profile_name: str = "tencent_daguandan",
        *,
        output_path: Path | str | None = None,
    ) -> FableDanBenchmarkResult:
        profile = normalize_profile_name(profile_name)
        model_path = self.profiles_root / profile / "models" / "best.npz"
        if not model_path.is_file():
            raise FileNotFoundError(f"缺少 FableDan 模型：{model_path}")
        model_hash = sha256(model_path.read_bytes()).hexdigest()
        model = NumpyModel(model_path)
        rng = random.Random(FIXED_SEED)
        wins = losses = 0
        rewards: list[int] = []
        seat_configurations = {"model_team_0_2": 0, "model_team_1_3": 0}
        reward_histogram: dict[str, int] = {}

        for game_index in range(FIXED_GAME_COUNT):
            model_on_even_team = game_index % 2 == 0
            model_seats = {0, 2} if model_on_even_team else {1, 3}
            key = "model_team_0_2" if model_on_even_team else "model_team_1_3"
            seat_configurations[key] += 1
            model_agent = NumpyAgent(model)
            rule_agent = RuleAgent()
            agents = [model_agent if seat in model_seats else rule_agent for seat in range(4)]
            rewards_by_player, _ranking, round_state = play_round(
                agents,
                rng=rng,
                tribute_mode=None,
            )
            if any(event[0] in {"tribute", "return"} for event in round_state.events):
                raise RuntimeError("固定基准评测意外出现进贡/还贡事件")
            reward = int(rewards_by_player[min(model_seats)])
            rewards.append(reward)
            reward_histogram[str(reward)] = reward_histogram.get(str(reward), 0) + 1
            if reward > 0:
                wins += 1
            else:
                losses += 1

        low, high = _wilson_interval(wins, FIXED_GAME_COUNT)
        payload: dict[str, object] = {
            "schema": "fabledan.fixed-benchmark/1",
            "benchmark_id": FIXED_BENCHMARK_ID,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "profile": profile,
            "model": {
                "filename": model_path.name,
                "sha256": model_hash,
                "relative_path": "models/best.npz",
            },
            "implementation": {"fabledan_upstream_commit": UPSTREAM_COMMIT},
            "protocol": {
                "games": FIXED_GAME_COUNT,
                "seed": FIXED_SEED,
                "ruleset": "standard_no_tribute",
                "tribute_mode": None,
                "model_agent": "NumpyAgent(eps=0)",
                "opponent_agent": "RuleAgent",
                "seat_swapping": True,
                "seat_configurations": seat_configurations,
            },
            "result": {
                "wins": wins,
                "losses": losses,
                "win_rate": wins / FIXED_GAME_COUNT,
                "wilson_95": {"low": low, "high": high},
                "average_team_reward": sum(rewards) / FIXED_GAME_COUNT,
                "reward_histogram": reward_histogram,
            },
            "interpretation": (
                "这是 FableDan 对固定 RuleAgent 基线的离线胜率，"
                "不是与真人或 DouZero 对局的胜率，也不是单手 Q 值概率。"
            ),
        }
        output = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else (
                self.profiles_root
                / profile
                / "models"
                / "benchmarks"
                / f"{FIXED_BENCHMARK_ID}-{model_hash[:16]}.json"
            ).resolve()
        )
        if output_path is not None and output.exists():
            raise FileExistsError(f"基准输出文件已存在：{output.name}")
        if output.exists() and not output.is_file():
            raise OSError(f"基准输出路径不是普通文件：{output.name}")
        atomic_write_json(output, payload)
        return FableDanBenchmarkResult(output_path=output, payload=payload)


def _wilson_interval(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        raise ValueError("games must be positive")
    z = 1.959963984540054
    rate = wins / games
    denominator = 1.0 + z * z / games
    centre = (rate + z * z / (2.0 * games)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / games + z * z / (4.0 * games * games)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)
