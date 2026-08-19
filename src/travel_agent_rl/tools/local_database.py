from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from .base import BaseTool
from .mock import _object_schema


def _like(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _terms(query: str) -> list[str]:
    stopwords = {
        "accommodation",
        "accommodations",
        "attraction",
        "attractions",
        "dining",
        "food",
        "hotel",
        "hotels",
        "place",
        "places",
        "restaurant",
        "restaurants",
        "stay",
        "tourist",
    }
    return [term for term in query.casefold().replace(",", " ").split() if term not in stopwords]


class SQLiteTravelTool(BaseTool):
    def __init__(self, database_path: Path) -> None:
        if not database_path.is_file():
            raise FileNotFoundError(database_path)
        self.database_path = database_path.resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro",
            uri=True,
            timeout=10.0,
        )
        connection.row_factory = sqlite3.Row
        return connection


class LocalTravelPoiSearch(SQLiteTravelTool):
    name = "poi_search"
    description = (
        "Search the versioned 2022 US travel database for attractions, restaurants, "
        "and accommodations. Use an exact city name plus a category, cuisine, room, "
        "house-rule, or place keyword."
    )
    argument_schema = _object_schema(
        ["city", "query"],
        {
            "city": {"type": "string", "minLength": 1},
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        city = str(arguments["city"]).strip()
        query = str(arguments["query"]).strip()
        limit = int(arguments.get("limit", 10))
        folded = query.casefold()
        terms = _terms(query)
        restaurant_cues = {"restaurant", "restaurants", "dining", "food", "cuisine"}
        accommodation_cues = {
            "accommodation",
            "accommodations",
            "hotel",
            "hotels",
            "home",
            "lodging",
            "rental",
            "room",
            "stay",
        }
        attraction_cues = {"attraction", "attractions", "museum", "park", "tourist"}
        words = set(folded.replace(",", " ").split())
        selected: set[str] = set()
        if words & restaurant_cues:
            selected.add("restaurant")
        if words & accommodation_cues:
            selected.add("accommodation")
        if words & attraction_cues:
            selected.add("attraction")
        if not selected:
            selected = {"restaurant", "accommodation", "attraction"}

        accommodation_filters = self._accommodation_filters(folded)
        if "accommodation" in selected:
            ignored = {
                "allow",
                "allowed",
                "children",
                "capacity",
                "entire",
                "for",
                "guest",
                "guests",
                "group",
                "home",
                "maximum",
                "night",
                "nights",
                "no",
                "non-shared",
                "not",
                "of",
                "people",
                "permit",
                "permitted",
                "person",
                "pets",
                "private",
                "restriction",
                "restrictions",
                "room",
                "shared",
                "smoking",
                "to",
                "under",
                "up",
                "visitor",
                "visitors",
                "without",
            }
            terms = [term for term in terms if term not in ignored and not term.isdigit()]

        items: list[dict[str, Any]] = []
        with self._connect() as connection:
            if "restaurant" in selected:
                items.extend(self._restaurants(connection, city, terms, limit))
            if "accommodation" in selected:
                items.extend(
                    self._accommodations(
                        connection,
                        city,
                        terms,
                        limit,
                        **accommodation_filters,
                    )
                )
            if "attraction" in selected:
                items.extend(self._attractions(connection, city, terms, limit))
        items = items[:limit]
        return {
            "items": items,
            "count": len(items),
            "city": city,
            "query": query,
            "source": "agent-star-travel-database@985c8c7",
            "applied_filters": accommodation_filters if "accommodation" in selected else {},
            "guidance": (
                "Do not search for airport names or codes. flight_search accepts exact "
                "city names directly; if no flight exists, try route_planning once."
                if "airport" in folded
                else None
            ),
        }

    @staticmethod
    def _accommodation_filters(query: str) -> dict[str, Any]:
        room_type = None
        if "entire" in query:
            room_type = "Entire home/apt"
        elif "private" in query:
            room_type = "Private room"
        elif "shared" in query and "not shared" not in query and "non-shared" not in query:
            room_type = "Shared room"

        allowed_rules: list[str] = []
        allowed_cues = ("allow", "permitted", "without restriction")
        if any(cue in query for cue in allowed_cues):
            for phrase, prohibition in (
                ("smoking", "No smoking"),
                ("visitor", "No visitors"),
                ("pet", "No pets"),
                ("children under 10", "No children under 10"),
            ):
                if phrase in query:
                    allowed_rules.append(prohibition)

        capacity_match = re.search(
            r"(?:capacity(?: of)?|for|group of)\s+(\d+)\s+(?:people|persons?|guests?)",
            query,
        )
        nights_match = re.search(r"(?:for|maximum|up to)\s+(\d+)\s+nights?", query)
        return {
            "room_type": room_type,
            "exclude_shared": "not shared" in query or "non-shared" in query,
            "allowed_rules": allowed_rules,
            "capacity": int(capacity_match.group(1)) if capacity_match else None,
            "maximum_minimum_nights": int(nights_match.group(1)) if nights_match else None,
        }

    @staticmethod
    def _where_terms(fields: tuple[str, ...], terms: list[str]) -> tuple[str, list[str]]:
        if not terms:
            return "", []
        clauses: list[str] = []
        params: list[str] = []
        for term in terms:
            clauses.append(
                "(" + " OR ".join(f"{field} LIKE ? ESCAPE '\\'" for field in fields) + ")"
            )
            params.extend([_like(term)] * len(fields))
        return " AND " + " AND ".join(clauses), params

    def _restaurants(
        self,
        connection: sqlite3.Connection,
        city: str,
        terms: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        where, params = self._where_terms(("name", "cuisines"), terms)
        rows = connection.execute(
            "SELECT name,city,cuisines,average_cost,aggregate_rating FROM restaurants "
            f"WHERE city = ? COLLATE NOCASE{where} "
            "ORDER BY aggregate_rating DESC, average_cost ASC, name COLLATE NOCASE LIMIT ?",
            [city, *params, limit],
        )
        return [
            {
                "category": "restaurant",
                "name": row["name"],
                "city": row["city"],
                "cuisines": row["cuisines"],
                "average_cost_usd": row["average_cost"],
                "rating": row["aggregate_rating"],
            }
            for row in rows
        ]

    def _accommodations(
        self,
        connection: sqlite3.Connection,
        city: str,
        terms: list[str],
        limit: int,
        *,
        room_type: str | None,
        exclude_shared: bool,
        allowed_rules: list[str],
        capacity: int | None,
        maximum_minimum_nights: int | None,
    ) -> list[dict[str, Any]]:
        where, params = self._where_terms(("name", "room_type", "house_rules"), terms)
        if room_type is not None:
            where += " AND room_type = ? COLLATE NOCASE"
            params.append(room_type)
        if exclude_shared:
            where += " AND room_type != 'Shared room' COLLATE NOCASE"
        for prohibition in allowed_rules:
            where += " AND COALESCE(house_rules, '') NOT LIKE ? COLLATE NOCASE"
            params.append(_like(prohibition))
        if capacity is not None:
            where += " AND maximum_occupancy >= ?"
            params.append(str(capacity))
        if maximum_minimum_nights is not None:
            where += " AND minimum_nights <= ?"
            params.append(str(maximum_minimum_nights))
        rows = connection.execute(
            "SELECT name,room_type,price,minimum_nights,review_rate,house_rules,"
            "maximum_occupancy,city FROM accommodations "
            f"WHERE city = ? COLLATE NOCASE{where} "
            "ORDER BY review_rate DESC, price ASC, name COLLATE NOCASE LIMIT ?",
            [city, *params, limit],
        )
        return [
            {
                "category": "accommodation",
                "name": row["name"],
                "city": row["city"],
                "room_type": row["room_type"],
                "price_per_night_usd": row["price"],
                "minimum_nights": row["minimum_nights"],
                "review_rate": row["review_rate"],
                "house_rules": row["house_rules"],
                "maximum_occupancy": row["maximum_occupancy"],
            }
            for row in rows
        ]

    def _attractions(
        self,
        connection: sqlite3.Connection,
        city: str,
        terms: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        where, params = self._where_terms(("name", "address"), terms)
        rows = connection.execute(
            "SELECT name,latitude,longitude,address,phone,website,city FROM attractions "
            f"WHERE city = ? COLLATE NOCASE{where} "
            "ORDER BY name COLLATE NOCASE LIMIT ?",
            [city, *params, limit],
        )
        return [
            {
                "category": "attraction",
                "name": row["name"],
                "city": row["city"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "address": row["address"],
                "phone": row["phone"],
                "website": row["website"],
            }
            for row in rows
        ]


class LocalTravelNearbySearch(SQLiteTravelTool):
    name = "nearby_search"
    description = "Search nearby US attractions in the versioned 2022 travel database."
    argument_schema = _object_schema(
        ["latitude", "longitude", "query"],
        {
            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
            "longitude": {"type": "number", "minimum": -180, "maximum": 180},
            "query": {"type": "string", "minLength": 1},
            "radius_m": {"type": "integer", "minimum": 1, "maximum": 100000},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        latitude = float(arguments["latitude"])
        longitude = float(arguments["longitude"])
        query = str(arguments["query"]).casefold().strip()
        radius_m = int(arguments.get("radius_m", 50_000))
        limit = int(arguments.get("limit", 10))
        matches: list[tuple[float, sqlite3.Row]] = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name,latitude,longitude,address,city FROM attractions "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
            )
            for row in rows:
                if (
                    query not in {"attraction", "attractions", "place", "places"}
                    and query not in str(row["name"]).casefold()
                ):
                    continue
                distance = self._haversine_m(latitude, longitude, row["latitude"], row["longitude"])
                if distance <= radius_m:
                    matches.append((distance, row))
        matches.sort(key=lambda item: (item[0], str(item[1]["name"]).casefold()))
        items = [
            {
                "category": "attraction",
                "name": row["name"],
                "city": row["city"],
                "address": row["address"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "distance_m": round(distance, 1),
            }
            for distance, row in matches[:limit]
        ]
        return {
            "items": items,
            "count": len(items),
            "source": "agent-star-travel-database@985c8c7",
        }

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class LocalFlightSearch(SQLiteTravelTool):
    name = "flight_search"
    description = (
        "Search the versioned 2022 US flight database. Origin and destination must be "
        "exact city names and date must be YYYY-MM-DD."
    )
    argument_schema = _object_schema(
        ["origin", "destination", "date"],
        {
            "origin": {"type": "string", "minLength": 1},
            "destination": {"type": "string", "minLength": 1},
            "date": {"type": "string", "format": "date"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        origin = str(arguments["origin"]).strip()
        destination = str(arguments["destination"]).strip()
        date = str(arguments["date"]).strip()
        limit = int(arguments.get("limit", 10))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT flight_number,price,departure_time,arrival_time,elapsed_time,"
                "flight_date,origin,destination,distance_miles FROM flights "
                "WHERE origin = ? COLLATE NOCASE AND destination = ? COLLATE NOCASE "
                "AND flight_date = ? ORDER BY price ASC, departure_time ASC, flight_number ASC "
                "LIMIT ?",
                (origin, destination, date, limit),
            )
            options = [
                {
                    "number": row["flight_number"],
                    "price_usd": row["price"],
                    "departure": row["departure_time"],
                    "arrival": row["arrival_time"],
                    "duration": row["elapsed_time"],
                    "date": row["flight_date"],
                    "origin": row["origin"],
                    "destination": row["destination"],
                    "distance_miles": row["distance_miles"],
                }
                for row in rows
            ]
        return {
            "options": options,
            "count": len(options),
            "source": "agent-star-travel-database@985c8c7",
            "guidance": (
                None
                if options
                else "No exact-city flight found; try route_planning once, not an airport search."
            ),
        }


class LocalGroundRoutePlanning(SQLiteTravelTool):
    name = "route_planning"
    description = (
        "Look up versioned intercity driving cost, duration, and distance between exact "
        "US city names."
    )
    argument_schema = _object_schema(
        ["origin", "destination", "mode"],
        {
            "origin": {"type": "string", "minLength": 1},
            "destination": {"type": "string", "minLength": 1},
            "mode": {"enum": ["walk", "drive", "transit"]},
            "city": {"type": "string"},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        origin = str(arguments["origin"]).strip()
        destination = str(arguments["destination"]).strip()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT origin,destination,cost,duration,distance FROM distances "
                "WHERE origin = ? COLLATE NOCASE AND destination = ? COLLATE NOCASE LIMIT 1",
                (origin, destination),
            ).fetchone()
        route = (
            {
                "origin": row["origin"],
                "destination": row["destination"],
                "estimated_cost_usd": row["cost"],
                "duration": row["duration"],
                "distance": row["distance"],
            }
            if row is not None
            else None
        )
        return {
            "mode": arguments["mode"],
            "route": route,
            "count": int(route is not None),
            "source": "agent-star-travel-database@985c8c7",
        }
