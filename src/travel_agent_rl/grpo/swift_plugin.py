from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import requests
from swift.plugin.multi_turn import multi_turns
from swift.plugin.orm import ORM, orms

from travel_agent_rl.rewards import (
    answer_llm_judge_reward,
    final_answer_boundary_reward,
    stage_awareness_reward,
    tool_efficiency_reward,
    valid_agent_step_reward,
    valid_tool_call_reward,
)
from travel_agent_rl.sft.data import build_agent_tool_schemas

from .scheduler import TravelAgentRolloutScheduler

TOOL_SCHEMAS = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in build_agent_tool_schemas()
}


def _trajectory_messages(
    trajectory_inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    request_ids: Sequence[str] | None,
    count: int,
) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for index in range(count):
        request_id = request_ids[index] if request_ids and index < len(request_ids) else None
        candidates = trajectory_inputs.get(request_id, []) if trajectory_inputs and request_id else []
        messages = candidates[-1].get("messages", []) if candidates else []
        result.append(list(messages))
    return result


class _TrajectoryORM(ORM):
    scorer: Callable[[Sequence[Any]], float]

    def __call__(
        self,
        completions: list[str],
        *,
        trajectory_inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        request_id: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        del kwargs
        trajectories = _trajectory_messages(trajectory_inputs, request_id, len(completions))
        return [float(self.scorer(messages)) for messages in trajectories]


class ValidAgentStepORM(_TrajectoryORM):
    scorer = staticmethod(valid_agent_step_reward)


class ValidToolCallORM(_TrajectoryORM):
    @staticmethod
    def scorer(messages: Sequence[Any]) -> float:
        return valid_tool_call_reward(messages, TOOL_SCHEMAS)


class FinalAnswerBoundaryORM(_TrajectoryORM):
    scorer = staticmethod(final_answer_boundary_reward)


class StageAwarenessORM(_TrajectoryORM):
    scorer = staticmethod(stage_awareness_reward)


class ToolEfficiencyORM(_TrajectoryORM):
    scorer = staticmethod(tool_efficiency_reward)


class FixedLocalJudgeORM(ORM):
    """Call the fixed local Base judge; failures deterministically score zero."""

    def __call__(
        self,
        completions: list[str],
        *,
        trajectory_inputs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        request_id: Sequence[str] | None = None,
        trainer_state: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        del kwargs
        trajectories = _trajectory_messages(trajectory_inputs, request_id, len(completions))
        endpoint = os.getenv("FIXED_JUDGE_ENDPOINT", "http://127.0.0.1:8019/judge")
        try:
            response = requests.post(
                endpoint,
                json={"trajectories": trajectories},
                timeout=float(os.getenv("FIXED_JUDGE_TIMEOUT", "300")),
            )
            response.raise_for_status()
            results = response.json()["scores"]
        except Exception:
            results = [None] * len(completions)
        raw_scores = [answer_llm_judge_reward(result, fallback=0.0) for result in results]
        step = float(getattr(trainer_state, "global_step", 0) or 0)
        max_steps = float(
            os.getenv("GRPO_SCHEDULE_MAX_STEPS")
            or getattr(trainer_state, "max_steps", 1)
            or 1
        )
        progress = min(1.0, max(0.0, step / max_steps))
        weight = 0.2 + (1.0 - 0.2) * progress
        return [score * weight for score in raw_scores]


multi_turns["travel_agent_replay"] = TravelAgentRolloutScheduler
orms.update(
    {
        "travel_valid_agent_step": ValidAgentStepORM,
        "travel_valid_tool_call": ValidToolCallORM,
        "travel_final_boundary": FinalAnswerBoundaryORM,
        "travel_stage_awareness": StageAwarenessORM,
        "travel_tool_efficiency": ToolEfficiencyORM,
        "travel_fixed_judge": FixedLocalJudgeORM,
    }
)
