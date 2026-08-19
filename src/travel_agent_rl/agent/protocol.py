from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ToolCall(BaseModel):
    kind: Literal["tool_call"] = "tool_call"
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    kind: Literal["answer"] = "answer"
    content: str = Field(min_length=1)


class AgentMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str


ParsedAgentOutput = ToolCall | FinalAnswer
_PARSED_OUTPUT = TypeAdapter(ParsedAgentOutput)
_TOOL_PATTERN = re.compile(r"^\s*<tool-call>\s*(.*?)\s*</tool-call[>}]\s*$", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"^\s*<answer>\s*(.*?)\s*</answer[>}]\s*$", re.DOTALL)


def parse_agent_output(text: str) -> ParsedAgentOutput:
    tool_match = _TOOL_PATTERN.fullmatch(text)
    if tool_match:
        payload = json.loads(tool_match.group(1))
        return _PARSED_OUTPUT.validate_python({"kind": "tool_call", **payload})

    answer_match = _ANSWER_PATTERN.fullmatch(text)
    if answer_match:
        return FinalAnswer(content=answer_match.group(1).strip())

    raise ValueError("assistant output must contain exactly one complete tool-call or answer block")
