from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from travel_agent_rl.agent.protocol import FinalAnswer, ToolCall, parse_agent_output
from travel_agent_rl.tools.amap import AmapWeather
from travel_agent_rl.tools.firecrawl import FirecrawlScrape, FirecrawlSearch
from travel_agent_rl.tools.local_database import (
    LocalFlightSearch,
    LocalGroundRoutePlanning,
    LocalTravelNearbySearch,
    LocalTravelPoiSearch,
)
from travel_agent_rl.tools.simulator import TrainSearchSimulator


AGENT_TOOL_CLASSES = (
    LocalTravelPoiSearch,
    LocalTravelNearbySearch,
    AmapWeather,
    LocalGroundRoutePlanning,
    FirecrawlSearch,
    FirecrawlScrape,
    LocalFlightSearch,
    TrainSearchSimulator,
)


def build_agent_tool_schemas() -> list[dict[str, Any]]:
    """Return the eight production-visible tools in OpenAI/Hermes schema form."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool_class.name,
                "description": tool_class.description,
                "parameters": tool_class.argument_schema,
            },
        }
        for tool_class in AGENT_TOOL_CLASSES
    ]


def _convert_assistant(content: str) -> dict[str, str]:
    parsed = parse_agent_output(content)
    if isinstance(parsed, ToolCall):
        payload = {"name": parsed.name, "arguments": parsed.arguments}
        return {
            "role": "tool_call",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        }
    if isinstance(parsed, FinalAnswer):
        return {"role": "assistant", "content": parsed.content}
    raise AssertionError(f"unexpected parsed assistant output: {type(parsed)!r}")


def convert_frozen_trajectory(
    row: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one immutable trajectory to ms-swift's native agent data format."""
    messages: list[dict[str, str]] = []
    for message in row["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            messages.append(_convert_assistant(content))
        elif role == "tool":
            messages.append({"role": "tool_response", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        else:
            raise ValueError(f"unsupported frozen trajectory role: {role!r}")

    if not messages or messages[0]["role"] != "user":
        raise ValueError("trajectory must begin with a user message")
    if messages[-1]["role"] != "assistant":
        raise ValueError("trajectory must end with a final assistant answer")

    query_id = str(row["query_id"])
    difficulty = str(row.get("metadata", {}).get("difficulty", "unknown"))
    return {
        "query_id": query_id,
        "difficulty": difficulty,
        "tools": json.dumps(tools or build_agent_tool_schemas(), ensure_ascii=False),
        "messages": messages,
    }


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
