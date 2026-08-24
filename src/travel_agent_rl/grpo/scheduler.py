from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from swift.plugin.multi_turn import MultiTurnScheduler

from travel_agent_rl.agent.state import AgentSession, SessionStatus
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.evaluation.agent_eval import parse_native_agent_output
from travel_agent_rl.rewards.components import FORCED_TERMINAL_MARKER
from travel_agent_rl.sft.data import build_agent_tool_schemas
from travel_agent_rl.tools.base import BaseTool, ToolExecutionError
from travel_agent_rl.tools.registry import ToolRegistry, build_distillation_registry


class OfflineWeatherTool(BaseTool):
    name = "weather"
    description = "Frozen replay weather boundary; live public requests are disabled."
    argument_schema = next(
        item["function"]["parameters"]
        for item in build_agent_tool_schemas()
        if item["function"]["name"] == "weather"
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ToolExecutionError(
            "live weather is disabled in replay-only GRPO",
            code="offline_weather_unavailable",
            provider="frozen_grpo",
        )


def build_replay_session(max_turns: int) -> AgentSession:
    if os.getenv("WEB_MODE", "replay_only") != "replay_only":
        raise RuntimeError("GRPO requires WEB_MODE=replay_only")
    root = Path(os.getenv("TRAVEL_AGENT_RL_ROOT", ".")).resolve()
    registry = build_distillation_registry(
        root,
        web_mode="replay_only",
        web_backend=os.getenv("WEB_BACKEND", "selfhosted"),
        snapshot_path=root / "data/intermediate/web_snapshot",
        web_run_id=os.getenv("WEB_RUN_ID", "grpo_replay_v4"),
    )
    tools = [
        OfflineWeatherTool() if name == "weather" else registry.get(name)
        for name in registry.names
    ]
    if any(tool is None for tool in tools):
        raise AssertionError("the GRPO registry must expose all eight tools")
    environment = TravelEnvironment(
        ToolRegistry(tools),
        max_attempts=1,
        cache_dir=root / "outputs/grpo/tool_cache",
        cache_namespace=f"snapshot_v{os.getenv('WEB_SNAPSHOT_VERSION', '4')}",
    )
    return AgentSession(
        environment,
        max_turns=max_turns,
        max_public_web_actions=int(os.getenv("GRPO_MAX_WEB_ACTIONS", "6")),
    )


def _choice_logprobs(choice: Any, expected: int) -> list[float] | None:
    payload = getattr(choice, "logprobs", None)
    if not payload:
        return None
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list):
        return None
    values: list[float] = []
    for item in content:
        if isinstance(item, dict) and item.get("logprob") is not None:
            values.append(float(item["logprob"]))
        else:
            return None
    return values if len(values) == expected else None


def hydrate_request_tools(infer_request: Any) -> str:
    """Restore tools moved into ``data_dict`` by ms-swift server-mode GRPO.

    ms-swift 3.8.3 intentionally sends only a small metadata allow-list to the
    remote rollout server.  With ``vllm_server_pass_dataset``, dataset columns
    such as ``tools`` arrive in ``RolloutInferRequest.data_dict`` instead of the
    request's native ``tools`` field.  The template only renders tool schemas
    from the native field, so leaving this untouched silently removes all tool
    instructions from the model prompt.
    """
    if getattr(infer_request, "tools", None):
        return "request"
    data_dict = getattr(infer_request, "data_dict", None) or {}
    raw_tools = data_dict.get("tools")
    if raw_tools is None:
        return "absent"
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except json.JSONDecodeError as exc:
            raise ValueError("dataset tools field is not valid JSON") from exc
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("dataset tools field must contain a non-empty list")
    infer_request.tools = raw_tools
    return "data_dict"


def _append_rollout_telemetry(payload: dict[str, Any]) -> None:
    path_value = os.getenv("GRPO_TELEMETRY_PATH")
    if not path_value:
        return
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(os.getenv("TRAVEL_AGENT_RL_ROOT", ".")) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def validate_rollout_alignment(
    messages: list[dict[str, Any]],
    response_token_ids: list[list[int]],
    response_loss_mask: list[list[int]],
    rollout_logprobs: list[list[float] | None],
) -> None:
    assistant_count = sum(message.get("role") == "assistant" for message in messages)
    if assistant_count != len(response_token_ids):
        raise AssertionError(
            f"assistant/token turn mismatch: {assistant_count} != {len(response_token_ids)}"
        )
    if len(response_token_ids) != len(response_loss_mask):
        raise AssertionError("response token/mask outer lengths differ")
    if len(response_token_ids) != len(rollout_logprobs):
        raise AssertionError("response token/logprob outer lengths differ")
    for ids, mask, logprobs in zip(
        response_token_ids, response_loss_mask, rollout_logprobs
    ):
        if len(ids) != len(mask):
            raise AssertionError("response token/mask inner lengths differ")
        if any(value not in {0, 1} for value in mask):
            raise AssertionError("response loss mask is not binary")
        if logprobs is not None and len(ids) != len(logprobs):
            raise AssertionError("response token/logprob inner lengths differ")


class TravelAgentRolloutScheduler(MultiTurnScheduler):
    """Server-mode, replay-only, eight-tool scheduler for Agentic GRPO."""

    def __init__(
        self,
        infer_engine: Any = None,
        max_turns: int | None = None,
        tokenizer: Any = None,
        *,
        session_factory: Callable[[], AgentSession] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(infer_engine=infer_engine, max_turns=max_turns, **kwargs)
        if tokenizer is None and infer_engine is not None:
            raise ValueError("the rollout-server scheduler requires its tokenizer")
        self.tokenizer = tokenizer
        self.max_turns = int(max_turns or 13)
        self.require_eight_tools = session_factory is None
        self.session_factory = session_factory or (
            lambda: build_replay_session(max(1, self.max_turns - 1))
        )

    def _forced_terminal(self) -> tuple[str, list[int]]:
        text = (
            f"{FORCED_TERMINAL_MARKER} "
            "This deterministic terminal marker is excluded from policy loss."
        )
        return text, self.tokenizer.encode(text, add_special_tokens=False)

    async def run(self, infer_request: Any, request_config: Any, **kwargs: Any) -> Any:
        from swift.llm.infer.protocol import RolloutOutput

        started = time.perf_counter()
        tool_schema_source = hydrate_request_tools(infer_request)
        tool_schema_count = len(getattr(infer_request, "tools", None) or [])
        if self.require_eight_tools and tool_schema_count != 8:
            raise RuntimeError(
                "production GRPO rollout requires exactly eight visible tool schemas; "
                f"received {tool_schema_count} from {tool_schema_source}"
            )
        session = self.session_factory()
        messages = deepcopy(infer_request.messages)
        infer_request.messages = messages
        response_token_ids: list[list[int]] = []
        response_loss_mask: list[list[int]] = []
        rollout_logprobs: list[list[float] | None] = []
        tool_sequence: list[str] = []
        action_signatures: list[str] = []
        errors: list[str] = []
        replay_misses = 0
        forced_terminal = False
        final_answer_present = False
        response = None

        for current_turn in range(1, self.max_turns + 1):
            response = await self.infer_engine.infer_async(
                infer_request, request_config, **kwargs
            )
            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            token_ids = list(choice.token_ids or [])
            messages.append({"role": "assistant", "content": content})
            response_token_ids.append(token_ids)
            response_loss_mask.append([1] * len(token_ids))
            rollout_logprobs.append(_choice_logprobs(choice, len(token_ids)))

            if choice.finish_reason == "length":
                errors.append("generation_length_limit")
                break
            try:
                kind, payload = parse_native_agent_output(content)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid_agent_output:{type(exc).__name__}:{exc}")
                break
            if kind == "answer":
                final_answer_present = True
                break
            assert isinstance(payload, dict)
            tool_sequence.append(str(payload["name"]))
            action_signatures.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            protocol_action = (
                "<tool-call>\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n</tool-call>"
            )
            transition = session.step(protocol_action)
            if transition.error_code:
                errors.append(transition.error_code)
            if transition.observation is not None:
                replay_misses += int("snapshot_miss" in transition.observation)
                messages.append(
                    {"role": "tool_response", "content": transition.observation}
                )
            if current_turn >= self.max_turns:
                forced_terminal = True
                terminal, terminal_ids = self._forced_terminal()
                messages.append({"role": "assistant", "content": terminal})
                response_token_ids.append(terminal_ids)
                response_loss_mask.append([0] * len(terminal_ids))
                rollout_logprobs.append(None)
                break
            if transition.status is SessionStatus.FINISHED:
                break
            infer_request.messages = messages

        if response is None:
            raise RuntimeError("rollout produced no response")
        validate_rollout_alignment(
            messages, response_token_ids, response_loss_mask, rollout_logprobs
        )
        data_dict = getattr(infer_request, "data_dict", None) or {}
        telemetry = {
            "timestamp_unix": time.time(),
            "stage": os.getenv("GRPO_STAGE", "unknown"),
            "request_id": getattr(infer_request, "uuid", None),
            "query_id": data_dict.get("query_id"),
            "difficulty": data_dict.get("difficulty"),
            "rollout_latency_seconds": time.perf_counter() - started,
            "num_turns": len(response_token_ids) - int(forced_terminal),
            "assistant_response_tokens": sum(map(len, response_token_ids)),
            "tool_calls": len(tool_sequence),
            "tool_sequence": tool_sequence,
            "repeated_action": len(action_signatures) != len(set(action_signatures)),
            "observation_count": sum(
                message.get("role") == "tool_response" for message in messages
            ),
            "errors": errors,
            "replay_misses": replay_misses,
            "public_network_requests": 0,
            "forced_terminal": forced_terminal,
            "final_answer_present": final_answer_present,
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "tool_schema_source": tool_schema_source,
            "tool_schema_count": tool_schema_count,
        }
        _append_rollout_telemetry(telemetry)
        return RolloutOutput(
            response=response,
            messages=messages,
            response_token_ids=response_token_ids,
            response_loss_mask=response_loss_mask,
            rollout_infos={
                "num_turns": len(response_token_ids) - int(forced_terminal),
                "tool_calls": len(tool_sequence),
                "tool_sequence": tool_sequence,
                "repeated_action": len(action_signatures) != len(set(action_signatures)),
                "errors": errors,
                "replay_misses": replay_misses,
                "public_network_requests": 0,
                "tool_schema_source": tool_schema_source,
                "tool_schema_count": tool_schema_count,
                "forced_terminal": forced_terminal,
                "final_answer_present": final_answer_present,
                "rollout_logprobs": rollout_logprobs,
                "logprob_source": (
                    "rollout_server"
                    if all(values is not None for values in rollout_logprobs)
                    else "trainer_recomputed_for_missing_turns"
                ),
            },
        )
