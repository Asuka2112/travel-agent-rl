from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import yaml

from .combine import REWARD_COMPONENTS, combine_rewards
from .components import (
    answer_llm_judge_reward,
    final_answer_boundary_reward,
    stage_awareness_reward,
    tool_efficiency_reward,
    valid_agent_step_reward,
    valid_tool_call_reward,
)


@dataclass(frozen=True)
class LinearWeightSchedule:
    start: float
    end: float

    def value(self, progress: float) -> float:
        progress = min(1.0, max(0.0, float(progress)))
        return self.start + (self.end - self.start) * progress


@dataclass(frozen=True)
class RewardResult:
    components: dict[str, float]
    weights: dict[str, float]
    total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RewardStatistics:
    values: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def update(self, result: RewardResult) -> None:
        for name, value in result.components.items():
            self.values[name].append(value)
        self.values["total_reward"].append(result.total)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": len(values),
                "mean": fmean(values),
                "std": pstdev(values),
            }
            for name, values in sorted(self.values.items())
            if values
        }


class RewardEngine:
    def __init__(
        self,
        *,
        weights: Mapping[str, float],
        tool_schemas: Mapping[str, Mapping[str, Any]],
        schedules: Mapping[str, LinearWeightSchedule] | None = None,
        judge_fallback: float = 0.0,
    ) -> None:
        unknown = set(weights) - set(REWARD_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown reward weights: {sorted(unknown)}")
        self.base_weights = {name: float(weights.get(name, 0.0)) for name in REWARD_COMPONENTS}
        self.tool_schemas = dict(tool_schemas)
        self.schedules = dict(schedules or {})
        self.judge_fallback = float(judge_fallback)
        self.statistics = RewardStatistics()

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        tool_schemas: Mapping[str, Mapping[str, Any]],
    ) -> RewardEngine:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        schedules = {}
        for name, item in config.get("curriculum", {}).items():
            if item.get("schedule") != "linear":
                raise ValueError(f"unsupported reward schedule for {name}")
            schedules[name] = LinearWeightSchedule(
                start=float(item["start"]), end=float(item["end"])
            )
        return cls(
            weights=config["weights"],
            tool_schemas=tool_schemas,
            schedules=schedules,
            judge_fallback=float(config.get("judge_failure_fallback", 0.0)),
        )

    def weights_at(self, progress: float) -> dict[str, float]:
        weights = dict(self.base_weights)
        for name, schedule in self.schedules.items():
            weights[name] = schedule.value(progress)
        return weights

    def score(
        self,
        messages: Sequence[Any],
        *,
        judge_result: float | Mapping[str, Any] | None,
        progress: float,
    ) -> RewardResult:
        components = {
            "valid_agent_step_reward": valid_agent_step_reward(messages),
            "valid_tool_call_reward": valid_tool_call_reward(messages, self.tool_schemas),
            "final_answer_boundary_reward": final_answer_boundary_reward(messages),
            "stage_awareness_reward": stage_awareness_reward(messages),
            "tool_efficiency_reward": tool_efficiency_reward(messages),
            "answer_llm_judge_reward": answer_llm_judge_reward(
                judge_result, fallback=self.judge_fallback
            ),
        }
        if any(not math.isfinite(value) for value in components.values()):
            raise FloatingPointError("non-finite reward component")
        weights = self.weights_at(progress)
        result = RewardResult(
            components=components,
            weights=weights,
            total=combine_rewards(components, weights),
        )
        if not math.isfinite(result.total):
            raise FloatingPointError("non-finite total reward")
        self.statistics.update(result)
        return result
