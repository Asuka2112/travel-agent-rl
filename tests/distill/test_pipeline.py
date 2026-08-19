from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from travel_agent_rl.agent import AgentSession
from travel_agent_rl.distill import (
    ChatResult,
    DistillationPipeline,
    TokenUsage,
    TrajectoryDistiller,
)
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import build_mock_registry


class ScriptedBackend:
    model = "scripted-teacher"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = deque(outputs)

    def complete(self, messages, system_prompt):
        return ChatResult(
            content=self.outputs.popleft(),
            model=self.model,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def session_factory() -> AgentSession:
    return AgentSession(TravelEnvironment(build_mock_registry()))


def test_generates_complete_multi_turn_trajectory() -> None:
    backend = ScriptedBackend(
        [
            '<tool-call>{"name":"poi_search","arguments":{"query":"museum","city":"Beijing"}}</tool-call>',
            "<answer>Visit the returned museum.</answer>",
        ]
    )
    record = TrajectoryDistiller(backend, session_factory, "prompt").generate("q1", "plan")

    assert record.metadata.success
    assert record.final_answer == "Visit the returned museum."
    assert record.metadata.tools_used == ["poi_search"]
    assert record.metadata.token_usage.total_tokens == 30
    assert [message.role for message in record.messages] == ["user", "assistant", "tool", "assistant"]


def test_pipeline_resumes_completed_ids(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"
    failed = tmp_path / "failed.jsonl"
    first_backend = ScriptedBackend(["<answer>done</answer>"])
    first = DistillationPipeline(TrajectoryDistiller(first_backend, session_factory, "prompt"))
    assert first.run([("q1", "one")], output, failed)["succeeded"] == 1

    second_backend = ScriptedBackend(["<answer>two</answer>"])
    second = DistillationPipeline(TrajectoryDistiller(second_backend, session_factory, "prompt"))
    stats = second.run([("q1", "one"), ("q2", "two")], output, failed)

    assert stats == {"submitted": 1, "resumed": 1, "succeeded": 1, "failed": 0}
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert {record["query_id"] for record in records} == {"q1", "q2"}


def test_failure_is_checkpointed(tmp_path: Path) -> None:
    backend = ScriptedBackend(["not valid"])
    distiller = TrajectoryDistiller(backend, session_factory, "prompt", max_model_calls=1)
    stats = DistillationPipeline(distiller).run(
        [("bad", "bad")], tmp_path / "out.jsonl", tmp_path / "failed.jsonl"
    )
    assert stats["failed"] == 1
    assert "bad" in (tmp_path / "failed.jsonl").read_text()
