from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator


NATIVE_TOOL_CALL = re.compile(r"^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$", re.DOTALL)


def parse_native_agent_output(text: str) -> tuple[str, dict[str, Any] | str]:
    """Parse one Hermes/Qwen tool call, otherwise treat non-empty text as a final answer."""
    match = NATIVE_TOOL_CALL.fullmatch(text)
    if match:
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict):
            raise ValueError("tool-call payload must be an object")
        if not isinstance(payload.get("name"), str) or not payload["name"]:
            raise ValueError("tool-call name must be a non-empty string")
        if not isinstance(payload.get("arguments"), dict):
            raise ValueError("tool-call arguments must be an object")
        return "tool_call", payload
    answer = text.strip()
    if not answer:
        raise ValueError("empty assistant output")
    if "<tool_call>" in answer or "</tool_call>" in answer:
        raise ValueError("malformed or mixed tool-call output")
    return "answer", answer


def validate_tool_call(
    payload: dict[str, Any], tools: list[dict[str, Any]]
) -> tuple[bool, str | None]:
    by_name = {tool["function"]["name"]: tool["function"] for tool in tools}
    function = by_name.get(payload["name"])
    if function is None:
        return False, f"unknown tool: {payload['name']}"
    errors = sorted(
        Draft202012Validator(function["parameters"]).iter_errors(payload["arguments"]),
        key=lambda error: list(error.path),
    )
    if errors:
        return False, errors[0].message
    return True, None


def completion_score(record: dict[str, Any]) -> float:
    """Deterministic task score for the frozen train-coverage dev set."""
    expected = record.get("expected_tool")
    used = record.get("tool_sequence", [])
    final = record.get("final_answer") or ""
    components = [
        bool(final),
        expected in used,
        "simulat" in final.casefold(),
        record.get("invalid_tool_calls", 0) == 0,
    ]
    return sum(components) / len(components)
