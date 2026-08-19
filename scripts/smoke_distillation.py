#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from travel_agent_rl.agent import AgentSession
from travel_agent_rl.agent.protocol import AgentMessage
from travel_agent_rl.distill import (
    ChatResult,
    DistillationPipeline,
    TokenUsage,
    TrajectoryDistiller,
    TrajectoryRecord,
)
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.filtering import apply_rule_filter, canonicalize_trajectory
from travel_agent_rl.tools import build_mock_registry


class SmokeTeacher:
    model = "deterministic-smoke-teacher"

    def complete(self, messages: list[AgentMessage], system_prompt: str) -> ChatResult:
        if messages[-1].role == "user":
            content = (
                '<tool-call>{"name":"weather","arguments":'
                '{"city":"Hangzhou","date":"2026-08-20"}}</tool-call>'
            )
        else:
            content = "<answer>Use the clear-weather fixture to plan outdoor activities.</answer>"
        return ChatResult(content=content, model=self.model, usage=TokenUsage())


def main() -> None:
    registry = build_mock_registry()

    def session_factory() -> AgentSession:
        return AgentSession(TravelEnvironment(registry), max_turns=3)

    distiller = TrajectoryDistiller(
        SmokeTeacher(), session_factory, "smoke prompt", max_model_calls=4
    )
    queries = [(f"smoke-{index}", f"Plan mock trip {index}") for index in range(5)]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "trajectories.jsonl"
        stats = DistillationPipeline(distiller, max_workers=2).run(
            queries, output, root / "failed.jsonl"
        )
        records = [
            TrajectoryRecord.model_validate_json(line) for line in output.read_text().splitlines()
        ]
        accepted = [
            canonicalize_trajectory(record)
            for record in records
            if apply_rule_filter(record, set(registry.names), max_turns=3).accepted
        ]
    result = {**stats, "rule_and_format_accepted": len(accepted)}
    if result != {
        "submitted": 5,
        "resumed": 0,
        "succeeded": 5,
        "failed": 0,
        "rule_and_format_accepted": 5,
    }:
        raise SystemExit(json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
