from __future__ import annotations

import json
import re
from pydantic import BaseModel, Field

from travel_agent_rl.agent.protocol import AgentMessage
from travel_agent_rl.distill.client import ChatBackend
from travel_agent_rl.distill.models import TrajectoryRecord


class JudgeScores(BaseModel):
    task_relevance: float = Field(ge=1, le=5)
    completeness: float = Field(ge=1, le=5)
    constraint_satisfaction: float = Field(ge=1, le=5)
    factual_grounding: float = Field(ge=1, le=5)
    tool_use_correctness: float = Field(ge=1, le=5)
    tool_efficiency: float = Field(ge=1, le=5)
    final_answer_quality: float = Field(ge=1, le=5)

    @property
    def mean(self) -> float:
        values = self.model_dump().values()
        return sum(values) / len(type(self).model_fields)


class JudgeDecision(BaseModel):
    scores: JudgeScores
    rationale: str = Field(min_length=1)
    accepted: bool
    prompt_version: str


class StructuredJudge:
    def __init__(
        self,
        backend: ChatBackend,
        prompt: str,
        *,
        prompt_version: str = "judge-v1",
        mean_threshold: float = 4.0,
        minimum_dimension: float = 3.0,
    ) -> None:
        self.backend = backend
        self.prompt = prompt
        self.prompt_version = prompt_version
        self.mean_threshold = mean_threshold
        self.minimum_dimension = minimum_dimension

    def evaluate(self, trajectory: TrajectoryRecord) -> JudgeDecision:
        serialized = json.dumps(trajectory.model_dump(mode="json"), ensure_ascii=False)
        result = self.backend.complete(
            [AgentMessage(role="user", content=serialized)],
            self.prompt,
        )
        payload = self._extract_json(result.content)
        scores = JudgeScores.model_validate(payload["scores"])
        rationale = str(payload["rationale"]).strip()
        values = list(scores.model_dump().values())
        accepted = scores.mean >= self.mean_threshold and min(values) >= self.minimum_dimension
        return JudgeDecision(
            scores=scores,
            rationale=rationale,
            accepted=accepted,
            prompt_version=self.prompt_version,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, object]:
        stripped = text.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if fence:
            stripped = fence.group(1)
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("judge output must be a JSON object")
        return payload
