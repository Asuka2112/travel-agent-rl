from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel

from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import ToolRequest, ToolResponse

from .protocol import FinalAnswer, ToolCall, parse_agent_output


class SessionStatus(str, Enum):
    RUNNING = "running"
    FORCE_ANSWER = "force_answer"
    FINISHED = "finished"


MAX_TOOL_OBSERVATION_CHARS = 2_000


def _compact_value(
    value: Any,
    *,
    list_limit: int = 3,
    string_limit: int = 500,
    max_depth: int = 6,
    depth: int = 0,
) -> Any:
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        return f"{value[:string_limit]}...[truncated {len(value) - string_limit} chars]"
    if isinstance(value, list):
        if depth >= max_depth:
            return {"truncated": True, "original_items": len(value)}
        compacted = [
            _compact_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                max_depth=max_depth,
                depth=depth + 1,
            )
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            compacted.append({"truncated_items": len(value) - list_limit})
        return compacted
    if isinstance(value, dict):
        if depth >= max_depth:
            return {"truncated": True, "original_keys": len(value)}
        return {
            str(key): _compact_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                max_depth=max_depth,
                depth=depth + 1,
            )
            for key, item in value.items()
        }
    return value


def _render_tool_observation(response: ToolResponse) -> str:
    payload = response.model_dump(mode="json")
    compacted = _compact_value(payload)
    encoded = json.dumps(compacted, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= MAX_TOOL_OBSERVATION_CHARS:
        return encoded

    compact_data = _compact_value(
        response.data,
        list_limit=1,
        string_limit=160,
        max_depth=3,
    )
    data_text = json.dumps(compact_data, ensure_ascii=False, separators=(",", ":"))
    fallback = {
        "request_id": response.request_id,
        "tool_name": response.tool_name,
        "status": response.status.value,
        "error": response.error.model_dump(mode="json") if response.error else None,
        "attempts": response.attempts,
        "cached": response.cached,
        "latency_ms": response.latency_ms,
        "data": {"preview": "", "truncated": True},
    }
    low, high = 0, len(data_text)
    rendered = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
    while low <= high:
        midpoint = (low + high) // 2
        fallback["data"]["preview"] = data_text[:midpoint]
        candidate = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
        if len(candidate) <= MAX_TOOL_OBSERVATION_CHARS:
            rendered = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return rendered


class SessionTransition(BaseModel):
    status: SessionStatus
    turns_used: int
    observation: str | None = None
    final_answer: str | None = None
    error_code: str | None = None


class AgentSession:
    PUBLIC_WEB_TOOLS = frozenset({"web_search", "web_scraping"})
    FORCE_ANSWER_MESSAGE = (
        "Maximum tool-action turns reached. Return one final response inside "
        "<answer>...</answer> and do not call another tool."
    )

    def __init__(
        self,
        environment: TravelEnvironment,
        max_turns: int = 13,
        max_public_web_actions: int | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if max_public_web_actions is not None and max_public_web_actions < 0:
            raise ValueError("max_public_web_actions cannot be negative")
        self.environment = environment
        self.max_turns = max_turns
        self.max_public_web_actions = max_public_web_actions
        self.public_web_actions_used = 0
        self.turns_used = 0
        self.status = SessionStatus.RUNNING
        self.action_counts: Counter[str] = Counter()

    def step(self, assistant_output: str) -> SessionTransition:
        if self.status is SessionStatus.FINISHED:
            raise RuntimeError("session has already finished")
        if self.status is SessionStatus.FORCE_ANSWER:
            try:
                parsed = parse_agent_output(assistant_output)
            except (ValueError, TypeError, json.JSONDecodeError):
                return self._force_transition(error_code="answer_required")
            if not isinstance(parsed, FinalAnswer):
                return self._force_transition(error_code="answer_required")
            return self._finish(parsed.content)

        try:
            parsed = parse_agent_output(assistant_output)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.turns_used += 1
            return self._tool_error("invalid_agent_output", str(exc))

        if isinstance(parsed, FinalAnswer):
            return self._finish(parsed.content)
        if not isinstance(parsed, ToolCall):
            raise AssertionError("parser returned an unsupported action")

        action_key = json.dumps(
            {"name": parsed.name, "arguments": parsed.arguments},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.action_counts[action_key] += 1
        self.turns_used += 1
        if self.action_counts[action_key] > 1:
            return self._tool_error("repeated_action", "identical tool action already executed")
        if parsed.name in self.PUBLIC_WEB_TOOLS and self.max_public_web_actions is not None:
            if self.public_web_actions_used >= self.max_public_web_actions:
                return self._tool_error(
                    "tool_budget_exceeded",
                    "public web action budget exhausted; use local evidence or answer",
                )
            self.public_web_actions_used += 1

        response = self.environment.execute(
            ToolRequest(tool_name=parsed.name, arguments=parsed.arguments)
        )
        response_for_model = response.model_copy(
            update={"data": {**response.data, "_agent_state": self._agent_state()}},
            deep=True,
        )
        observation = _render_tool_observation(response_for_model)
        if self.turns_used >= self.max_turns:
            self.status = SessionStatus.FORCE_ANSWER
            observation = f"{observation}\n{self.FORCE_ANSWER_MESSAGE}"
        return SessionTransition(
            status=self.status,
            turns_used=self.turns_used,
            observation=observation,
            error_code=response.error.code if response.error else None,
        )

    def _tool_error(self, code: str, message: str) -> SessionTransition:
        if self.turns_used >= self.max_turns:
            self.status = SessionStatus.FORCE_ANSWER
        observation = json.dumps(
            {
                "status": "error",
                "error": {"code": code, "message": message},
                "agent_state": self._agent_state(),
            },
            sort_keys=True,
        )
        if self.status is SessionStatus.FORCE_ANSWER:
            observation = f"{observation}\n{self.FORCE_ANSWER_MESSAGE}"
        return SessionTransition(
            status=self.status,
            turns_used=self.turns_used,
            observation=observation,
            error_code=code,
        )

    def _force_transition(self, error_code: str) -> SessionTransition:
        observation = json.dumps(
            {
                "status": "error",
                "error": {"code": error_code, "message": self.FORCE_ANSWER_MESSAGE},
                "agent_state": self._agent_state(),
            },
            sort_keys=True,
        )
        return SessionTransition(
            status=SessionStatus.FORCE_ANSWER,
            turns_used=self.turns_used,
            observation=f"{observation}\n{self.FORCE_ANSWER_MESSAGE}",
            error_code=error_code,
        )

    def _agent_state(self) -> dict[str, int | None]:
        remaining_web = None
        if self.max_public_web_actions is not None:
            remaining_web = max(0, self.max_public_web_actions - self.public_web_actions_used)
        return {
            "remaining_tool_turns": max(0, self.max_turns - self.turns_used),
            "remaining_public_web_actions": remaining_web,
        }

    def _finish(self, answer: str) -> SessionTransition:
        self.status = SessionStatus.FINISHED
        return SessionTransition(
            status=self.status,
            turns_used=self.turns_used,
            final_answer=answer,
        )
