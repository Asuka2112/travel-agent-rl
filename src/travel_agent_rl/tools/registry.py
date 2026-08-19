from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

from .base import BaseTool
from .mock import MOCK_TOOL_DEFINITIONS, DeterministicMockTool


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def build_mock_registry() -> ToolRegistry:
    return ToolRegistry(
        DeterministicMockTool(
            name=item["name"],
            description=item["description"],
            argument_schema=item["schema"],
            fixture=item["fixture"],
        )
        for item in MOCK_TOOL_DEFINITIONS
    )


def build_live_registry(
    root: Path,
    *,
    amap_api_key: str | None = None,
    firecrawl_api_key: str | None = None,
) -> ToolRegistry:
    from .amap import AmapNearbySearch, AmapPoiSearch, AmapRoutePlanning, AmapWeather
    from .firecrawl import FirecrawlScrape, FirecrawlSearch
    from .simulator import FlightSearchSimulator, TrainSearchSimulator

    amap_key = amap_api_key or os.getenv("AMAP_API_KEY")
    firecrawl_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
    missing = [
        name
        for name, value in (
            ("AMAP_API_KEY", amap_key),
            ("FIRECRAWL_API_KEY", firecrawl_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"missing live-tool credentials: {', '.join(missing)}")
    return ToolRegistry(
        (
            AmapPoiSearch(amap_key),
            AmapNearbySearch(amap_key),
            AmapWeather(amap_key),
            AmapRoutePlanning(amap_key),
            FirecrawlSearch(firecrawl_key),
            FirecrawlScrape(firecrawl_key),
            FlightSearchSimulator(root / "data/fixtures/flight_inventory.json"),
            TrainSearchSimulator(root / "data/fixtures/train_inventory.json"),
        )
    )


def build_distillation_registry(
    root: Path,
    *,
    database_path: Path | None = None,
    amap_api_key: str | None = None,
    firecrawl_api_key: str | None = None,
) -> ToolRegistry:
    from .amap import AmapWeather
    from .firecrawl import FirecrawlScrape, FirecrawlSearch
    from .local_database import (
        LocalFlightSearch,
        LocalGroundRoutePlanning,
        LocalTravelNearbySearch,
        LocalTravelPoiSearch,
    )
    from .simulator import TrainSearchSimulator

    amap_key = amap_api_key or os.getenv("AMAP_API_KEY")
    firecrawl_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
    missing = [
        name
        for name, value in (
            ("AMAP_API_KEY", amap_key),
            ("FIRECRAWL_API_KEY", firecrawl_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"missing distillation-tool credentials: {', '.join(missing)}")
    resolved_database = database_path or root / "data/processed/agent_star_travel.sqlite3"
    return ToolRegistry(
        (
            LocalTravelPoiSearch(resolved_database),
            LocalTravelNearbySearch(resolved_database),
            AmapWeather(amap_key),
            LocalGroundRoutePlanning(resolved_database),
            FirecrawlSearch(firecrawl_key),
            FirecrawlScrape(firecrawl_key),
            LocalFlightSearch(resolved_database),
            TrainSearchSimulator(root / "data/fixtures/train_inventory.json"),
        )
    )
