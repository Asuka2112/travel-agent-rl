from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from travel_agent_rl.agent.protocol import FinalAnswer, ToolCall, parse_agent_output
from travel_agent_rl.evaluation.agent_eval import parse_native_agent_output

FORCED_TERMINAL_MARKER = "Maximum rollout turns reached before a final answer."


def _messages(messages: Sequence[Any]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role == "system":
            continue
        role = "tool" if role == "tool_response" else role
        normalized.append((str(role), "" if content is None else str(content)))
    return normalized


def _parse(content: str) -> ToolCall | FinalAnswer | None:
    if content.startswith(FORCED_TERMINAL_MARKER):
        return None
    try:
        return parse_agent_output(content)
    except (ValueError, TypeError, json.JSONDecodeError):
        try:
            kind, payload = parse_native_agent_output(content)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        if kind == "tool_call":
            assert isinstance(payload, dict)
            return ToolCall(name=payload["name"], arguments=payload["arguments"])
        return FinalAnswer(content=str(payload))


def valid_agent_step_reward(messages: Sequence[Any]) -> float:
    """Fraction of protocol/transition checks satisfied by a trajectory."""
    rows = _messages(messages)
    if not rows:
        return 0.0
    passed = int(rows[0][0] == "user")
    checks = 1
    previous_action: ToolCall | FinalAnswer | None = None
    for index, (role, content) in enumerate(rows):
        if index == 0:
            continue
        checks += 1
        previous_role = rows[index - 1][0]
        if role == "assistant":
            parsed = _parse(content)
            valid = previous_role in {"user", "tool"} and parsed is not None
            previous_action = parsed
        elif role == "tool":
            valid = previous_role == "assistant" and isinstance(previous_action, ToolCall)
        else:
            valid = False
        passed += int(valid)
    return passed / checks


def valid_tool_call_reward(
    messages: Sequence[Any],
    tool_schemas: Mapping[str, Mapping[str, Any]],
) -> float:
    """Return the fraction of emitted tool actions with valid names and arguments."""
    valid = 0
    candidates = 0
    for role, content in _messages(messages):
        if role != "assistant":
            continue
        if content.startswith(FORCED_TERMINAL_MARKER):
            continue
        parsed = _parse(content)
        if isinstance(parsed, FinalAnswer):
            continue
        candidates += 1
        if not isinstance(parsed, ToolCall) or parsed.name not in tool_schemas:
            continue
        try:
            Draft202012Validator(tool_schemas[parsed.name]).validate(parsed.arguments)
        except ValidationError:
            continue
        valid += 1
    return valid / candidates if candidates else 0.0


def final_answer_boundary_reward(messages: Sequence[Any]) -> float:
    """Reward exactly one final answer at the end of the trajectory."""
    rows = _messages(messages)
    final_indices = [
        index
        for index, (role, content) in enumerate(rows)
        if role == "assistant" and isinstance(_parse(content), FinalAnswer)
    ]
    return float(len(final_indices) == 1 and final_indices[0] == len(rows) - 1)


def stage_awareness_reward(messages: Sequence[Any]) -> float:
    """Score tool→observation→next-action progression and evidence-aware finalization."""
    rows = _messages(messages)
    if not rows:
        return 0.0
    passed = 0
    checks = 0
    tool_calls = 0
    for index, (role, content) in enumerate(rows):
        if role == "assistant":
            parsed = _parse(content)
            if isinstance(parsed, ToolCall):
                tool_calls += 1
                checks += 1
                passed += int(index + 1 < len(rows) and rows[index + 1][0] == "tool")
            elif isinstance(parsed, FinalAnswer):
                checks += 2
                passed += int(index == len(rows) - 1)
                passed += int(tool_calls > 0)
        elif role == "tool":
            checks += 1
            prior = _parse(rows[index - 1][1]) if index else None
            passed += int(isinstance(prior, ToolCall))
    return passed / checks if checks else 0.0


def tool_efficiency_reward(messages: Sequence[Any]) -> float:
    """Penalize exact repeated actions, loops, and trajectories without a final answer."""
    actions: list[str] = []
    repeated_errors = 0
    for role, content in _messages(messages):
        if role == "assistant":
            parsed = _parse(content)
            if isinstance(parsed, ToolCall):
                actions.append(
                    json.dumps(
                        {"name": parsed.name, "arguments": parsed.arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        elif role == "tool" and "repeated_action" in content:
            repeated_errors += 1
    if not actions:
        return 0.0
    duplicate_actions = len(actions) - len(set(actions))
    penalty = min(1.0, (duplicate_actions + repeated_errors) / len(actions))
    has_final = bool(final_answer_boundary_reward(messages))
    return max(0.0, (1.0 - penalty) * (1.0 if has_final else 0.5))


def answer_llm_judge_reward(
    judge_result: float | Mapping[str, Any] | None,
    *,
    fallback: float = 0.0,
) -> float:
    """Normalize a fixed external/local Judge result with deterministic failure fallback."""
    if judge_result is None:
        return fallback
    if isinstance(judge_result, Mapping):
        if judge_result.get("timed_out") or judge_result.get("failed"):
            return fallback
        judge_result = judge_result.get("score")
    try:
        score = float(judge_result)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(score):
        return fallback
    if score > 1.0:
        score /= 10.0
    return min(1.0, max(0.0, score))
