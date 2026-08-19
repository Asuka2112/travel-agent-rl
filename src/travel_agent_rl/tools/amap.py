from __future__ import annotations

from typing import Any

import httpx

from .base import BaseTool, ToolExecutionError, TransientToolError
from .http import JsonHttpClient
from .mock import _object_schema

AMAP_BASE_URL = "https://restapi.amap.com"


class AmapTool(BaseTool):
    endpoint: str

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ValueError("AMAP_API_KEY is required")
        self.api_key = api_key
        self._owned_client = client is None
        self.client = client or httpx.Client(timeout=httpx.Timeout(10.0))
        self.http = JsonHttpClient(self.client, "amap")

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        payload = self.http.request_json(
            "GET",
            f"{AMAP_BASE_URL}{self.endpoint}",
            params={"key": self.api_key, "output": "JSON", **params},
        )
        if str(payload.get("status")) != "1":
            code = str(payload.get("infocode", "unknown"))
            message = str(payload.get("info", "AMap request failed"))
            if code in {"10003", "10004", "10019", "10020", "10021"}:
                raise TransientToolError(
                    message,
                    code=f"amap_{code}",
                    provider="amap",
                )
            raise ToolExecutionError(message, code=f"amap_{code}", provider="amap")
        return payload


class AmapPoiSearch(AmapTool):
    name = "poi_search"
    description = "Search AMap points of interest by city and keyword."
    endpoint = "/v3/place/text"
    argument_schema = _object_schema(
        ["city", "query"],
        {
            "city": {"type": "string", "minLength": 1},
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self._get(
            {
                "city": arguments["city"],
                "keywords": arguments["query"],
                "offset": arguments.get("limit", 10),
                "extensions": "base",
            }
        )
        return {"items": payload.get("pois", []), "count": payload.get("count", "0")}


class AmapNearbySearch(AmapTool):
    name = "nearby_search"
    description = "Search AMap points of interest near a coordinate."
    endpoint = "/v3/place/around"
    argument_schema = _object_schema(
        ["latitude", "longitude", "query"],
        {
            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
            "longitude": {"type": "number", "minimum": -180, "maximum": 180},
            "query": {"type": "string", "minLength": 1},
            "radius_m": {"type": "integer", "minimum": 1, "maximum": 50000},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        location = f"{arguments['longitude']:.6f},{arguments['latitude']:.6f}"
        payload = self._get(
            {
                "location": location,
                "keywords": arguments["query"],
                "radius": arguments.get("radius_m", 3000),
                "offset": arguments.get("limit", 10),
                "extensions": "base",
            }
        )
        return {"items": payload.get("pois", []), "count": payload.get("count", "0")}


class AmapWeather(AmapTool):
    name = "weather"
    description = "Return AMap weather forecast for an administrative code."
    endpoint = "/v3/weather/weatherInfo"
    argument_schema = _object_schema(
        ["city", "date"],
        {
            "city": {"type": "string", "minLength": 1},
            "date": {"type": "string"},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self._get({"city": arguments["city"], "extensions": "all"})
        forecasts = payload.get("forecasts", [])
        casts = forecasts[0].get("casts", []) if forecasts else []
        requested = [item for item in casts if item.get("date") == arguments["date"]]
        return {"forecasts": requested or casts, "requested_date": arguments["date"]}


class AmapRoutePlanning(AmapTool):
    name = "route_planning"
    description = "Plan an AMap walk, drive, or transit route between coordinates."
    argument_schema = _object_schema(
        ["origin", "destination", "mode"],
        {
            "origin": {"type": "string", "pattern": r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$"},
            "destination": {
                "type": "string",
                "pattern": r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$",
            },
            "mode": {"enum": ["walk", "drive", "transit"]},
            "city": {"type": "string"},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        mode = arguments["mode"]
        self.endpoint = {
            "walk": "/v3/direction/walking",
            "drive": "/v3/direction/driving",
            "transit": "/v3/direction/transit/integrated",
        }[mode]
        params = {"origin": arguments["origin"], "destination": arguments["destination"]}
        if mode == "transit":
            if not arguments.get("city"):
                raise ToolExecutionError(
                    "city is required for transit routes",
                    code="missing_transit_city",
                    provider="amap",
                )
            params["city"] = arguments["city"]
        payload = self._get(params)
        return {"mode": mode, "route": payload.get("route", {})}
