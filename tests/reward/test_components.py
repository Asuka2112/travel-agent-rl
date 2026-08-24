from __future__ import annotations

import math

import pytest

from travel_agent_rl.rewards import (
    LinearWeightSchedule,
    RewardEngine,
    answer_llm_judge_reward,
    final_answer_boundary_reward,
    stage_awareness_reward,
    tool_efficiency_reward,
    valid_agent_step_reward,
    valid_tool_call_reward,
)

SCHEMAS = {
    "poi_search": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
}


def call(city: str = "Paris") -> dict[str, str]:
    return {
        "role": "assistant",
        "content": (
            '<tool-call>{"name":"poi_search","arguments":{"city":'
            f'"{city}"' + "}}</tool-call>"
        ),
    }


def good() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Plan Paris"},
        call(),
        {"role": "tool", "content": '{"status":"ok","data":{"items":[]}}'},
        {"role": "assistant", "content": "<answer>Grounded itinerary.</answer>"},
    ]


def test_valid_agent_step_reward_is_independent() -> None:
    assert valid_agent_step_reward(good()) == 1.0
    broken = [good()[0], {"role": "tool", "content": "orphan"}]
    assert valid_agent_step_reward(broken) < 1.0


def test_valid_tool_call_reward_validates_name_and_json_schema() -> None:
    assert valid_tool_call_reward(good(), SCHEMAS) == 1.0
    invalid = [good()[0], call(city=""), good()[2], good()[3]]
    assert valid_tool_call_reward(invalid, SCHEMAS) == 1.0
    invalid[1] = {
        "role": "assistant",
        "content": '<tool-call>{"name":"poi_search","arguments":{}}</tool-call>',
    }
    assert valid_tool_call_reward(invalid, SCHEMAS) == 0.0


def test_final_answer_boundary_requires_one_terminal_answer() -> None:
    assert final_answer_boundary_reward(good()) == 1.0
    assert final_answer_boundary_reward(good()[:-1]) == 0.0
    assert final_answer_boundary_reward(good() + [call("Lyon")]) == 0.0
    forced = good()[:-1] + [
        {
            "role": "assistant",
            "content": (
                "Maximum rollout turns reached before a final answer. "
                "This deterministic terminal marker is excluded from policy loss."
            ),
        }
    ]
    assert final_answer_boundary_reward(forced) == 0.0
    assert valid_agent_step_reward(forced) < 1.0
    assert valid_tool_call_reward(forced, SCHEMAS) == 1.0
    assert tool_efficiency_reward(forced) < tool_efficiency_reward(good())


def test_stage_awareness_requires_observation_before_final() -> None:
    assert stage_awareness_reward(good()) == 1.0
    premature = [good()[0], good()[-1]]
    assert stage_awareness_reward(premature) < stage_awareness_reward(good())


def test_tool_efficiency_penalizes_exact_repetition() -> None:
    loop = [good()[0], call(), good()[2], call(), good()[2], good()[3]]
    assert tool_efficiency_reward(good()) == 1.0
    assert tool_efficiency_reward(loop) < tool_efficiency_reward(good())


def test_judge_reward_has_deterministic_failure_and_timeout_fallback() -> None:
    assert answer_llm_judge_reward({"score": 8}) == pytest.approx(0.8)
    assert answer_llm_judge_reward({"timed_out": True}, fallback=0.1) == 0.1
    assert answer_llm_judge_reward(float("nan"), fallback=0.2) == 0.2


def test_curriculum_and_disabled_components_are_deterministic() -> None:
    engine = RewardEngine(
        weights={
            "valid_agent_step_reward": 0.0,
            "valid_tool_call_reward": 1.0,
            "final_answer_boundary_reward": 1.0,
            "stage_awareness_reward": 0.5,
            "tool_efficiency_reward": 0.5,
            "answer_llm_judge_reward": 0.2,
        },
        tool_schemas=SCHEMAS,
        schedules={"answer_llm_judge_reward": LinearWeightSchedule(0.2, 1.0)},
    )
    start = engine.score(good(), judge_result=0.5, progress=0.0)
    end = engine.score(good(), judge_result=0.5, progress=1.0)
    assert start.weights["valid_agent_step_reward"] == 0.0
    assert start.weights["answer_llm_judge_reward"] == 0.2
    assert end.weights["answer_llm_judge_reward"] == 1.0
    summary = engine.statistics.summary()
    assert all(math.isfinite(item["mean"]) for item in summary.values())
    assert all("std" in item for item in summary.values())
