#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from travel_agent_rl.agent import AgentSession
from travel_agent_rl.agent.protocol import AgentMessage
from travel_agent_rl.distill import (
    ChatResult,
    DistillationPipeline,
    OpenAICompatibleChatBackend,
    TokenUsage,
    TrajectoryDistiller,
)
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import build_live_registry, build_mock_registry


class MockTeacherBackend:
    """Offline checkpoint/resume smoke backend; never used for research data."""

    model = "deterministic-mock-teacher"

    def complete(self, messages: list[AgentMessage], system_prompt: str) -> ChatResult:
        if messages[-1].role == "user":
            content = (
                '<tool-call>{"name":"weather","arguments":'
                '{"city":"Hangzhou","date":"2026-08-20"}}</tool-call>'
            )
        else:
            content = "<answer>Mock-only trajectory; exclude from research datasets.</answer>"
        return ChatResult(content=content, model=self.model, usage=TokenUsage())


def read_queries(path: Path, limit: int | None) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                queries.append((str(row["query_id"]), str(row["query"])))
            except KeyError as exc:
                raise ValueError(f"missing query field on line {line_number}") from exc
            if limit is not None and len(queries) >= limit:
                break
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/distill/default.yaml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--mock", action="store_true", help="offline pipeline smoke only")
    args = parser.parse_args()

    root = args.root.resolve()
    config = yaml.safe_load((root / args.config).read_text())
    prompt = (root / config["prompt_path"]).read_text()
    if args.mock:
        backend = MockTeacherBackend()
        registry = build_mock_registry()
    else:
        provider = config["provider"]
        required = {
            name: os.getenv(provider[f"{name}_env"])
            for name in ("api_key", "base_url", "model")
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"missing teacher settings: {', '.join(missing)}")
        backend = OpenAICompatibleChatBackend(**required)
        registry = build_live_registry(root)

    def session_factory() -> AgentSession:
        environment = TravelEnvironment(
            registry,
            cache_dir=root / "data/intermediate/tool_cache",
        )
        return AgentSession(environment, max_turns=int(config["max_turns"]))

    distiller = TrajectoryDistiller(
        backend,
        session_factory,
        prompt,
        prompt_version=config["prompt_version"],
        max_model_calls=int(config["max_model_calls"]),
    )
    pipeline = DistillationPipeline(
        distiller,
        max_workers=args.workers or int(config["concurrency"]),
    )
    queries = read_queries(root / config["input_path"], args.limit)
    stats = pipeline.run(
        queries,
        root / config["output_path"],
        root / config["failed_path"],
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
