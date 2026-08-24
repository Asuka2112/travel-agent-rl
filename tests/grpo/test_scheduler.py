from __future__ import annotations

import asyncio

from swift.llm.infer.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from swift.llm.template.template_inputs import RolloutInferRequest

from travel_agent_rl.agent.state import AgentSession
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.grpo import (
    TravelAgentRolloutScheduler,
    hydrate_request_tools,
    validate_rollout_alignment,
)
from travel_agent_rl.tools.base import BaseTool
from travel_agent_rl.tools.base import ToolExecutionError
from travel_agent_rl.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "echo"
    argument_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(self, arguments):
        return {"value": arguments["value"]}


class SnapshotMissTool(BaseTool):
    name = "web_search"
    description = "deterministic replay miss"
    argument_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def run(self, arguments):
        raise ToolExecutionError(
            f"no snapshot for {arguments['query']}",
            code="snapshot_miss",
            retryable=False,
            provider="web_snapshot",
        )


class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


class FakeEngine:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    async def infer_async(self, infer_request, request_config, **kwargs):
        del infer_request, request_config, kwargs
        content = next(self.outputs)
        token_ids = list(content.encode())
        return ChatCompletionResponse(
            model="fake",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                    token_ids=token_ids,
                    logprobs={"content": [{"logprob": -0.1} for _ in token_ids]},
                )
            ],
            usage=UsageInfo(0, len(token_ids), len(token_ids)),
        )


def session_factory():
    return AgentSession(TravelEnvironment(ToolRegistry([EchoTool()])), max_turns=2)


def test_multi_turn_rollout_aligns_actions_masks_and_logprobs() -> None:
    call = '<tool_call>{"name":"echo","arguments":{"value":"x"}}</tool_call>'
    scheduler = TravelAgentRolloutScheduler(
        infer_engine=FakeEngine([call, "grounded answer"]),
        max_turns=3,
        tokenizer=FakeTokenizer(),
        session_factory=session_factory,
    )
    output = asyncio.run(
        scheduler.run(
            RolloutInferRequest(messages=[{"role": "user", "content": "query"}]),
            object(),
        )
    )
    assert [message["role"] for message in output.messages] == [
        "user",
        "assistant",
        "tool_response",
        "assistant",
    ]
    assert output.rollout_infos["tool_sequence"] == ["echo"]
    assert output.rollout_infos["replay_misses"] == 0
    assert output.rollout_infos["logprob_source"] == "rollout_server"
    assert all(all(value == 1 for value in mask) for mask in output.response_loss_mask)
    assert len(output.response_token_ids) == 2
    # Tool observations are separate messages and never appear in policy response masks.
    assert len(output.response_token_ids) == sum(
        message["role"] == "assistant" for message in output.messages
    )
    validate_rollout_alignment(
        output.messages,
        output.response_token_ids,
        output.response_loss_mask,
        output.rollout_infos["rollout_logprobs"],
    )


def test_turn_limit_adds_zero_loss_terminal_without_faking_logprob() -> None:
    call = '<tool_call>{"name":"echo","arguments":{"value":"x"}}</tool_call>'
    scheduler = TravelAgentRolloutScheduler(
        infer_engine=FakeEngine([call]),
        max_turns=1,
        tokenizer=FakeTokenizer(),
        session_factory=session_factory,
    )
    output = asyncio.run(
        scheduler.run(
            RolloutInferRequest(messages=[{"role": "user", "content": "query"}]),
            object(),
        )
    )
    assert output.rollout_infos["forced_terminal"] is True
    assert output.response_loss_mask[-1]
    assert set(output.response_loss_mask[-1]) == {0}
    assert output.rollout_infos["rollout_logprobs"][-1] is None
    assert output.messages[-1]["role"] == "assistant"


def test_alignment_rejects_logprob_length_mismatch() -> None:
    try:
        validate_rollout_alignment(
            [{"role": "assistant", "content": "x"}], [[1]], [[1]], [[-0.1, -0.2]]
        )
    except AssertionError as exc:
        assert "logprob" in str(exc)
    else:
        raise AssertionError("mismatched logprobs were accepted")


def test_trainer_side_server_marker_does_not_require_tokenizer() -> None:
    scheduler = TravelAgentRolloutScheduler(max_turns=3)
    assert scheduler.infer_engine is None
    assert scheduler.tokenizer is None


def test_server_mode_hydrates_dataset_tools_before_prompt_rendering() -> None:
    request = RolloutInferRequest(
        messages=[{"role": "user", "content": "query"}],
        data_dict={"tools": '[{"type":"function","function":{"name":"echo"}}]'},
    )
    assert request.tools is None
    assert hydrate_request_tools(request) == "data_dict"
    assert request.tools == [{"type": "function", "function": {"name": "echo"}}]


def test_snapshot_miss_is_structured_and_rollout_recovers() -> None:
    miss = '<tool_call>{"name":"web_search","arguments":{"query":"uncached route"}}</tool_call>'
    fallback = '<tool_call>{"name":"echo","arguments":{"value":"fallback"}}</tool_call>'
    scheduler = TravelAgentRolloutScheduler(
        infer_engine=FakeEngine([miss, fallback, "grounded fallback answer"]),
        max_turns=3,
        tokenizer=FakeTokenizer(),
        session_factory=lambda: AgentSession(
            TravelEnvironment(ToolRegistry([SnapshotMissTool(), EchoTool()])),
            max_turns=3,
        ),
    )
    output = asyncio.run(
        scheduler.run(
            RolloutInferRequest(messages=[{"role": "user", "content": "query"}]),
            object(),
        )
    )
    miss_observation = next(
        message["content"]
        for message in output.messages
        if message["role"] == "tool_response" and "snapshot_miss" in message["content"]
    )
    payload = __import__("json").loads(miss_observation)
    assert payload["error"]["code"] == "snapshot_miss"
    assert payload["error"]["retryable"] is False
    assert payload["data"]["query"] == "uncached route"
    assert payload["data"]["results"] == []
    assert payload["data"]["public_network_used"] is False
    assert output.rollout_infos["replay_misses"] == 1
    assert output.rollout_infos["replay_miss_recovered"] == 1
    assert output.rollout_infos["replay_miss_final_failure"] == 0
    assert output.rollout_infos["final_answer_present"] is True
    assert output.rollout_infos["alignment_verified"] is True
    assert output.rollout_infos["tool_observations_excluded_from_policy_loss"] is True
