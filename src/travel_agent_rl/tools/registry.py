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
    web_mode: str | None = None,
    web_backend: str | None = None,
    firecrawl_base_url: str | None = None,
    snapshot_path: Path | None = None,
    web_run_id: str | None = None,
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
    if not amap_key:
        raise ValueError("missing distillation-tool credentials: AMAP_API_KEY")
    resolved_mode = web_mode or os.getenv("WEB_MODE", "replay_only")
    resolved_backend = web_backend or os.getenv("WEB_BACKEND", "selfhosted")
    if resolved_mode not in {"online_capture", "replay_only", "mock"}:
        raise ValueError(f"unsupported WEB_MODE: {resolved_mode}")
    if resolved_backend not in {"selfhosted", "cloud"}:
        raise ValueError(f"unsupported WEB_BACKEND: {resolved_backend}")
    resolved_database = database_path or root / "data/processed/agent_star_travel.sqlite3"
    if resolved_mode == "mock":
        mock_registry = build_mock_registry()
        web_tools = [mock_registry.get("web_search"), mock_registry.get("web_scraping")]
    else:
        from .web_snapshot import SnapshotWebTool, WebSnapshotStore

        firecrawl_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        if resolved_backend == "cloud":
            if resolved_mode == "online_capture":
                raise ValueError("Firecrawl Cloud capture is disabled for formal distillation")
            base_url = firecrawl_base_url or "https://api.firecrawl.dev/v2"
            backend_name = "firecrawl_cloud"
        else:
            base_url = firecrawl_base_url or os.getenv(
                "FIRECRAWL_BASE_URL", "http://127.0.0.1:3002/v1"
            )
            backend_name = "firecrawl_selfhosted"
            firecrawl_key = None
        store = WebSnapshotStore(
            snapshot_path or root / "data/intermediate/web_snapshot",
            run_id=web_run_id or os.getenv("WEB_RUN_ID", "unspecified"),
        )
        web_tools = [
            SnapshotWebTool(
                FirecrawlSearch(
                    firecrawl_key, base_url=base_url, backend=backend_name
                ),
                store,
                mode=resolved_mode,
            ),
            SnapshotWebTool(
                FirecrawlScrape(
                    firecrawl_key, base_url=base_url, backend=backend_name
                ),
                store,
                mode=resolved_mode,
            ),
        ]
    if any(tool is None for tool in web_tools):
        raise AssertionError("web tool registry is incomplete")
    return ToolRegistry(
        (
            LocalTravelPoiSearch(resolved_database),
            LocalTravelNearbySearch(resolved_database),
            AmapWeather(amap_key),
            LocalGroundRoutePlanning(resolved_database),
            *web_tools,
            LocalFlightSearch(resolved_database),
            TrainSearchSimulator(root / "data/fixtures/train_inventory.json"),
        )
    )
