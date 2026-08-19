from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolExecutionError
from .mock import _object_schema


class InventorySearchTool(BaseTool):
    def __init__(self, inventory_path: Path) -> None:
        self.inventory_path = inventory_path
        payload = json.loads(inventory_path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"inventory must be a list: {inventory_path}")
        self.inventory: list[dict[str, Any]] = payload

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        matches = [
            item
            for item in self.inventory
            if str(item["origin"]).casefold() == str(arguments["origin"]).casefold()
            and str(item["destination"]).casefold() == str(arguments["destination"]).casefold()
            and item["date"] == arguments["date"]
        ]
        matches.sort(key=lambda item: (item["price_cny"], item["departure"], item["number"]))
        return {"options": matches, "count": len(matches), "simulator": True}


class FlightSearchSimulator(InventorySearchTool):
    name = "flight_search"
    description = "Search the deterministic local flight inventory."
    argument_schema = _object_schema(
        ["origin", "destination", "date"],
        {
            "origin": {"type": "string", "minLength": 1},
            "destination": {"type": "string", "minLength": 1},
            "date": {"type": "string"},
        },
    )


class TrainSearchSimulator(InventorySearchTool):
    name = "train_search"
    description = (
        "Search the versioned deterministic US intercity train simulator. Use exact city "
        "names and a YYYY-MM-DD travel date. Results are simulated recurring schedules, "
        "not live fares; prefer this tool when rail is requested or is plausible for a "
        "short/medium-distance city pair."
    )
    argument_schema = _object_schema(
        ["origin", "destination", "date"],
        {
            "origin": {"type": "string", "minLength": 1},
            "destination": {"type": "string", "minLength": 1},
            "date": {"type": "string", "format": "date"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        origin = str(arguments["origin"]).strip()
        destination = str(arguments["destination"]).strip()
        travel_date = str(arguments["date"]).strip()
        limit = int(arguments.get("limit", 10))
        try:
            date.fromisoformat(travel_date)
        except ValueError as exc:
            raise ToolExecutionError(
                "date must be a valid YYYY-MM-DD calendar date",
                code="invalid_arguments",
            ) from exc

        matches = [
            {
                **item,
                "origin": origin,
                "destination": destination,
                "date": travel_date,
                "simulated": True,
            }
            for item in self.inventory
            if str(item["origin"]).casefold() == origin.casefold()
            and str(item["destination"]).casefold() == destination.casefold()
        ]
        matches.sort(key=lambda item: (item["price_usd"], item["departure"], item["number"]))
        options = matches[:limit]
        return {
            "options": options,
            "count": len(options),
            "source": "travel-agent-rl/train-simulator-us@v1",
            "simulator": True,
            "date_semantics": "versioned recurring schedule applied to the requested date",
            "guidance": (
                None
                if options
                else "No simulated direct train route is available for this exact city pair."
            ),
        }
