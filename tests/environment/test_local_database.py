from __future__ import annotations

import sqlite3
from pathlib import Path

from travel_agent_rl.tools.local_database import (
    LocalFlightSearch,
    LocalGroundRoutePlanning,
    LocalTravelNearbySearch,
    LocalTravelPoiSearch,
)


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "travel.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE flights (
                flight_number TEXT, price REAL, departure_time TEXT, arrival_time TEXT,
                elapsed_time TEXT, flight_date TEXT, origin TEXT, destination TEXT,
                distance_miles REAL
            );
            CREATE TABLE attractions (
                name TEXT, latitude REAL, longitude REAL, address TEXT, phone TEXT,
                website TEXT, city TEXT
            );
            CREATE TABLE accommodations (
                name TEXT, room_type TEXT, price REAL, minimum_nights REAL,
                review_rate REAL, house_rules TEXT, maximum_occupancy REAL, city TEXT
            );
            CREATE TABLE restaurants (
                name TEXT, city TEXT, cuisines TEXT, average_cost REAL,
                aggregate_rating REAL
            );
            CREATE TABLE distances (
                origin TEXT, destination TEXT, cost REAL, duration TEXT, distance TEXT
            );
            INSERT INTO flights VALUES
                ('F2', 250, '09:00', '11:00', '2 hours', '2022-06-07',
                 'Newark', 'Seattle', 2402),
                ('F1', 200, '10:00', '12:00', '2 hours', '2022-06-07',
                 'Newark', 'Seattle', 2402);
            INSERT INTO attractions VALUES
                ('City Museum', 47.6000, -122.3300, '1 Main St', '555',
                 'https://museum.test', 'Seattle');
            INSERT INTO accommodations VALUES
                ('Quiet Private Room', 'Private room', 90, 2, 4.8,
                 'No smoking', 2, 'Seattle'),
                ('Visitor Friendly Home', 'Entire home/apt', 140, 2, 4.7,
                 'No pets', 8, 'Seattle');
            INSERT INTO restaurants VALUES
                ('BBQ Place', 'Seattle', 'BBQ, American', 30, 4.7);
            INSERT INTO distances VALUES
                ('Seattle', 'Portland', 45, '3 hours', '174 miles');
            """
        )
    return path


def test_local_flight_search_orders_by_usd_price(tmp_path: Path) -> None:
    result = LocalFlightSearch(_database(tmp_path)).run(
        {
            "origin": "newark",
            "destination": "SEATTLE",
            "date": "2022-06-07",
        }
    )
    assert [option["number"] for option in result["options"]] == ["F1", "F2"]
    assert result["options"][0]["price_usd"] == 200


def test_local_poi_search_covers_travel_categories(tmp_path: Path) -> None:
    tool = LocalTravelPoiSearch(_database(tmp_path))
    restaurant = tool.run({"city": "Seattle", "query": "BBQ restaurants"})
    accommodation = tool.run({"city": "Seattle", "query": "private room"})
    attraction = tool.run({"city": "Seattle", "query": "museum"})

    assert restaurant["items"][0]["category"] == "restaurant"
    assert accommodation["items"][0]["category"] == "accommodation"
    assert attraction["items"][0]["category"] == "attraction"


def test_local_accommodation_search_interprets_allowed_rules_and_capacity(
    tmp_path: Path,
) -> None:
    result = LocalTravelPoiSearch(_database(tmp_path)).run(
        {
            "city": "Seattle",
            "query": "entire home visitors allowed for 7 people for 3 nights",
        }
    )

    assert [item["name"] for item in result["items"]] == ["Visitor Friendly Home"]
    assert result["applied_filters"]["allowed_rules"] == ["No visitors"]
    assert result["applied_filters"]["capacity"] == 7


def test_local_nearby_and_ground_route_are_deterministic(tmp_path: Path) -> None:
    database = _database(tmp_path)
    nearby = LocalTravelNearbySearch(database).run(
        {
            "latitude": 47.6001,
            "longitude": -122.3301,
            "query": "museum",
            "radius_m": 1_000,
        }
    )
    route = LocalGroundRoutePlanning(database).run(
        {"origin": "seattle", "destination": "PORTLAND", "mode": "drive"}
    )

    assert nearby["count"] == 1
    assert nearby["items"][0]["distance_m"] < 20
    assert route["route"]["estimated_cost_usd"] == 45
