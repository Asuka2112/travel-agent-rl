from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from travel_agent_rl.agent.protocol import AgentMessage, ToolCall

from .client import TokenUsage


class TrajectoryMetadata(BaseModel):
    query_id: str
    teacher_model: str
    prompt_version: str
    started_at: datetime
    finished_at: datetime
    number_of_turns: int = Field(ge=0)
    tools_used: list[str] = Field(default_factory=list)
    success: bool
    exceptions: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = Field(ge=0)
    estimated_api_cost_usd: float | None = Field(default=None, ge=0)


class TrajectoryRecord(BaseModel):
    query_id: str
    query: str
    messages: list[AgentMessage]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_observations: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str | None = None
    metadata: TrajectoryMetadata

    @model_validator(mode="after")
    def validate_success_contract(self) -> TrajectoryRecord:
        if self.metadata.query_id != self.query_id:
            raise ValueError("metadata query_id must match trajectory query_id")
        if self.metadata.success and not self.final_answer:
            raise ValueError("successful trajectories require a final answer")
        return self
