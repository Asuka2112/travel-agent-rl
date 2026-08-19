import json
from pathlib import Path

import httpx

from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import ToolRequest, ToolStatus
from travel_agent_rl.tools.amap import AmapPoiSearch
from travel_agent_rl.tools.firecrawl import FirecrawlSearch
from travel_agent_rl.tools.registry import ToolRegistry, build_live_registry
from travel_agent_rl.tools.simulator import FlightSearchSimulator


def test_amap_adapter_maps_provider_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "test-key"
        assert request.url.params["keywords"] == "museum"
        return httpx.Response(
            200,
            json={"status": "1", "count": "1", "pois": [{"name": "Museum"}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    environment = TravelEnvironment(ToolRegistry([AmapPoiSearch("test-key", client)]))
    response = environment.execute(
        ToolRequest(tool_name="poi_search", arguments={"city": "Hangzhou", "query": "museum"})
    )
    assert response.status is ToolStatus.OK
    assert response.data["items"] == [{"name": "Museum"}]
    assert "test-key" not in response.model_dump_json()


def test_firecrawl_rate_limit_is_retried_and_normalized() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"success": False})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    environment = TravelEnvironment(
        ToolRegistry([FirecrawlSearch("test-key", client)]),
        max_attempts=2,
    )
    response = environment.execute(
        ToolRequest(tool_name="web_search", arguments={"query": "travel"})
    )
    assert calls == 2
    assert response.status is ToolStatus.ERROR
    assert response.error is not None
    assert response.error.code == "rate_limited"
    assert response.error.provider == "firecrawl"


def test_deterministic_flight_inventory_orders_by_price(tmp_path: Path) -> None:
    inventory = [
        {
            "number": "A1",
            "origin": "HGH",
            "destination": "PEK",
            "date": "2026-08-20",
            "departure": "08:00",
            "price_cny": 700,
        },
        {
            "number": "A2",
            "origin": "HGH",
            "destination": "PEK",
            "date": "2026-08-20",
            "departure": "09:00",
            "price_cny": 600,
        },
    ]
    path = tmp_path / "flights.json"
    path.write_text(json.dumps(inventory))
    environment = TravelEnvironment(ToolRegistry([FlightSearchSimulator(path)]))
    response = environment.execute(
        ToolRequest(
            tool_name="flight_search",
            arguments={"origin": "hgh", "destination": "pek", "date": "2026-08-20"},
        )
    )
    assert [item["number"] for item in response.data["options"]] == ["A2", "A1"]


def test_live_registry_requires_both_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AMAP_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    try:
        build_live_registry(tmp_path, amap_api_key="", firecrawl_api_key="")
    except ValueError as exc:
        assert "AMAP_API_KEY" in str(exc)
        assert "FIRECRAWL_API_KEY" in str(exc)
    else:
        raise AssertionError("missing credentials must be rejected")
