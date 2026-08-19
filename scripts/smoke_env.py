#!/usr/bin/env python3
from __future__ import annotations

import json

from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import ToolRequest, build_mock_registry
from travel_agent_rl.tools.mock import SMOKE_ARGUMENTS


def main() -> None:
    registry = build_mock_registry()
    environment = TravelEnvironment(registry)
    responses = [
        environment.execute(ToolRequest(tool_name=name, arguments=arguments)).model_dump(mode="json")
        for name, arguments in SMOKE_ARGUMENTS.items()
    ]
    if not all(response["status"] == "ok" for response in responses):
        raise SystemExit(1)
    print(json.dumps({"tools": len(responses), "success": True}, sort_keys=True))


if __name__ == "__main__":
    main()
