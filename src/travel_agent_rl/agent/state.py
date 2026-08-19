from __future__ import annotations

import json
from collections import Counter
from enum import Enum

from pydantic import BaseModel

from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import ToolRequest

from .protocol import FinalAnswer, ToolCall, parse_agent_output


class SessionStatus(str, Enum):
    RUNNING = "running"
    FORCE_ANSWER = "force_answer"
    FINISHED = "finished"


class SessionTransition(BaseModel):
    status: SessionStatus
    turns_used: int
    observation: str | None = None
    final_answer: str | None = None
    error_code: str | None = None


class AgentSession:
    FORCE_ANSWER_MESSAGE = (
        "Maximum tool-action turns reached. Return one final response inside "
        "<answer>...</answer> and do not call another tool."
    )

    def __init__(self, environment: TravelEnvironment, max_turns: int = 13) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.environment = environment
        self.max_turns = max_turns
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

        response = self.environment.execute(
            ToolRequest(tool_name=parsed.name, arguments=parsed.arguments)
        )
        observation = response.model_dump_json()
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
            {"status": "error", "error": {"code": code, "message": message}},
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
        return SessionTransition(
            status=SessionStatus.FORCE_ANSWER,
            turns_used=self.turns_used,
            observation=self.FORCE_ANSWER_MESSAGE,
            error_code=error_code,
        )

    def _finish(self, answer: str) -> SessionTransition:
        self.status = SessionStatus.FINISHED
        return SessionTransition(
            status=self.status,
            turns_used=self.turns_used,
            final_answer=answer,
        )

