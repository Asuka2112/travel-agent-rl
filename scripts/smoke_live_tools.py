#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import ToolRequest, build_live_registry


def main() -> None:
    root = Path(".")
    environment = TravelEnvironment(
        build_live_registry(root),
        max_attempts=3,
        cache_dir=root / "data/intermediate/tool_cache",
    )
    calls = (
        ("poi_search", {"city": "杭州", "query": "博物馆", "limit": 2}),
        (
            "nearby_search",
            {"latitude": 30.2500, "longitude": 120.1600, "query": "咖啡", "limit": 2},
        ),
        ("weather", {"city": "330100", "date": "2026-08-20"}),
        (
            "route_planning",
            {
                "origin": "120.155153,30.274085",
                "destination": "120.212199,30.290691",
                "mode": "walk",
            },
        ),
        ("web_search", {"query": "Hangzhou museum official", "limit": 2}),
        ("web_scraping", {"url": "https://example.com"}),
        (
            "flight_search",
            {"origin": "HGH", "destination": "PEK", "date": "2026-08-20"},
        ),
        (
            "train_search",
            {"origin": "Hangzhou", "destination": "Shanghai", "date": "2026-08-20"},
        ),
    )
    responses = [
        environment.execute(ToolRequest(tool_name=name, arguments=arguments)).model_dump(
            mode="json"
        )
        for name, arguments in calls
    ]
    print(json.dumps(responses, indent=2, ensure_ascii=False))
    if not all(response["status"] == "ok" for response in responses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
