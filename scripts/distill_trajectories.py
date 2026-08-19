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
    RetryingChatBackend,
    TokenUsage,
    TrajectoryDistiller,
)
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import (
    build_distillation_registry,
    build_live_registry,
    build_mock_registry,
)


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


def read_queries(path: Path, limit: int | None) -> list[tuple[str, str, dict[str, object]]]:
    queries: list[tuple[str, str, dict[str, object]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                attributes = (
                    row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
                )
                difficulty = row.get("difficulty") or attributes.get("level")
                query_metadata: dict[str, object] = {
                    "difficulty": difficulty,
                    "attributes": attributes,
                    "source": row.get("source", {}),
                    "canonical_sha256": row.get("canonical_sha256"),
                    "constraint_sha256": row.get("constraint_sha256"),
                }
                queries.append((str(row["query_id"]), str(row["query"]), query_metadata))
            except KeyError as exc:
                raise ValueError(f"missing query field on line {line_number}") from exc
            if limit is not None and len(queries) >= limit:
                break
    return queries


def render_tool_catalog(registry) -> str:
    tools = []
    for name in registry.names:
        tool = registry.get(name)
        if tool is None:
            raise AssertionError(f"registry lost tool: {name}")
        tools.append(
            {"name": name, "description": tool.description, "arguments": tool.argument_schema}
        )
    return json.dumps(tools, ensure_ascii=False, sort_keys=True)


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
            name: os.getenv(provider[f"{name}_env"]) for name in ("api_key", "base_url", "model")
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"missing teacher settings: {', '.join(missing)}")
        backend = RetryingChatBackend(
            OpenAICompatibleChatBackend(
                **required,
                temperature=float(provider.get("temperature", 0.0)),
                seed=int(provider.get("seed", config["seed"])),
            ),
            max_attempts=int(provider.get("max_attempts", 3)),
        )
        if config.get("tool_registry") == "local_travel_database":
            registry = build_distillation_registry(
                root,
                database_path=root / config["database_path"],
            )
        else:
            registry = build_live_registry(root)

    prompt = f"{prompt.rstrip()}\n\nAvailable tools:\n{render_tool_catalog(registry)}\n"

    def session_factory() -> AgentSession:
        environment = TravelEnvironment(
            registry,
            cache_dir=root / "data/intermediate/tool_cache",
            cache_namespace=str(config.get("tool_cache_namespace", "default")),
        )
        return AgentSession(
            environment,
            max_turns=int(config["max_turns"]),
            max_public_web_actions=config.get("max_public_web_actions"),
        )

    distiller = TrajectoryDistiller(
        backend,
        session_factory,
        prompt,
        prompt_version=config["prompt_version"],
        teacher_revision=config.get("model", {}).get("revision"),
        snapshot_version=(
            int(os.environ["WEB_SNAPSHOT_VERSION"]) if os.getenv("WEB_SNAPSHOT_VERSION") else None
        ),
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
        raw_dir=root / config["raw_dir"] if config.get("raw_dir") else None,
        index_path=root / config["index_path"] if config.get("index_path") else None,
        checkpoint_path=(
            root / config["checkpoint_path"] if config.get("checkpoint_path") else None
        ),
        checkpoint_every=int(config.get("checkpoint_every", 50)),
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
