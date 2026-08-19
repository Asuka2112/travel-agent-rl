from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseTool
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
    description = "Search the deterministic local train inventory."
    argument_schema = FlightSearchSimulator.argument_schema

