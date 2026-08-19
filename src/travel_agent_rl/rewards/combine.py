from __future__ import annotations

from collections.abc import Mapping

REWARD_COMPONENTS = (
    "valid_agent_step_reward",
    "valid_tool_call_reward",
    "final_answer_boundary_reward",
    "stage_awareness_reward",
    "tool_efficiency_reward",
    "answer_llm_judge_reward",
)


def combine_rewards(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    unknown = (set(scores) | set(weights)) - set(REWARD_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown reward components: {sorted(unknown)}")
    return sum(scores.get(name, 0.0) * weights.get(name, 0.0) for name in REWARD_COMPONENTS)
