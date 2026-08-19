from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base import BaseTool


class DeterministicMockTool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str,
        argument_schema: dict[str, Any],
        fixture: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.argument_schema = argument_schema
        self._fixture = fixture

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(self._fixture)
        result["mock"] = True
        result["echo"] = deepcopy(arguments)
        return result


def _object_schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


MOCK_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "poi_search",
        "description": "Search points of interest in a city.",
        "schema": _object_schema(
            ["city", "query"],
            {"city": {"type": "string"}, "query": {"type": "string"}},
        ),
        "fixture": {"items": [{"name": "West Lake", "city": "Hangzhou"}]},
    },
    {
        "name": "nearby_search",
        "description": "Search amenities near a coordinate.",
        "schema": _object_schema(
            ["latitude", "longitude", "query"],
            {
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                "query": {"type": "string"},
            },
        ),
        "fixture": {"items": [{"name": "Mock Cafe", "distance_m": 240}]},
    },
    {
        "name": "weather",
        "description": "Return a deterministic weather forecast.",
        "schema": _object_schema(
            ["city", "date"],
            {"city": {"type": "string"}, "date": {"type": "string", "format": "date"}},
        ),
        "fixture": {"condition": "clear", "temperature_c": [18, 27]},
    },
    {
        "name": "route_planning",
        "description": "Plan a route between two locations.",
        "schema": _object_schema(
            ["origin", "destination", "mode"],
            {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "mode": {"enum": ["walk", "drive", "transit"]},
            },
        ),
        "fixture": {"distance_km": 6.2, "duration_min": 32},
    },
    {
        "name": "web_search",
        "description": "Search public web pages.",
        "schema": _object_schema(["query"], {"query": {"type": "string"}}),
        "fixture": {"results": [{"title": "Mock result", "url": "https://example.test"}]},
    },
    {
        "name": "web_scraping",
        "description": "Extract text from a public web page.",
        "schema": _object_schema(["url"], {"url": {"type": "string", "pattern": "^https?://"}}),
        "fixture": {"title": "Mock page", "text": "Offline fixture content."},
    },
    {
        "name": "flight_search",
        "description": "Search a deterministic flight inventory.",
        "schema": _object_schema(
            ["origin", "destination", "date"],
            {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "format": "date"},
            },
        ),
        "fixture": {"options": [{"number": "TA100", "price_cny": 680}]},
    },
    {
        "name": "train_search",
        "description": "Search a deterministic train inventory.",
        "schema": _object_schema(
            ["origin", "destination", "date"],
            {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "format": "date"},
            },
        ),
        "fixture": {"options": [{"number": "G100", "price_cny": 117.5}]},
    },
)

SMOKE_ARGUMENTS: dict[str, dict[str, Any]] = {
    "poi_search": {"city": "Hangzhou", "query": "museum"},
    "nearby_search": {"latitude": 30.25, "longitude": 120.16, "query": "cafe"},
    "weather": {"city": "Hangzhou", "date": "2026-08-20"},
    "route_planning": {
        "origin": "West Lake",
        "destination": "East Station",
        "mode": "transit",
    },
    "web_search": {"query": "Hangzhou museum opening hours"},
    "web_scraping": {"url": "https://example.test/travel"},
    "flight_search": {"origin": "HGH", "destination": "PEK", "date": "2026-08-20"},
    "train_search": {
        "origin": "Hangzhou",
        "destination": "Shanghai",
        "date": "2026-08-20",
    },
}
