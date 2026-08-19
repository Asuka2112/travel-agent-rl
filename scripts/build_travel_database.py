#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SOURCE_REVISION = "985c8c7ba3e6beb986b3b0a55a2b064d18f2e68e"
EXPECTED_SHA256 = {
    "attractions.csv": "8007fa851ba5b3adb205fbd298169c95116b297cef7ccd21dbea26ecce2605d8",
    "clean_Flights_2022.csv": "8dafdb0e3f8b79ce599a1e612a772865295bc226b46e5fb278368f7255b11cee",
    "clean_accommodations_2022.csv": "7e70cc8cb573e824267be47153a3cf7934c839f01c955fb523d18be405921d86",
    "clean_restaurant_2022.csv": "a38ef00a3398b1ef78306a9b1851c491723d27899c6a1d4a3d6fe042929c0f26",
    "distance.csv": "7dc1a26dc7877ae96ce63e74be0d9660bbdef96ffccaffce441128e3a4864a27",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float | None:
    text = _text(value)
    if not text or text.casefold() in {"nan", "none", "unknown", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _rows(path: Path, fields: tuple[str, ...]) -> Iterator[tuple[Any, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield tuple(row[field] for field in fields)


def _batched(rows: Iterable[tuple[Any, ...]], size: int) -> Iterator[list[tuple[Any, ...]]]:
    batch: list[tuple[Any, ...]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _insert_batches(
    connection: sqlite3.Connection,
    statement: str,
    rows: Iterable[tuple[Any, ...]],
    batch_size: int,
) -> int:
    count = 0
    for batch in _batched(rows, batch_size):
        connection.executemany(statement, batch)
        count += len(batch)
    return count


def build_database(source_dir: Path, output_path: Path, batch_size: int) -> dict[str, int]:
    for filename, expected in EXPECTED_SHA256.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {filename}: {actual}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-200000;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE flights (
                flight_number TEXT NOT NULL,
                price REAL,
                departure_time TEXT,
                arrival_time TEXT,
                elapsed_time TEXT,
                flight_date TEXT NOT NULL,
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                distance_miles REAL
            );
            CREATE TABLE attractions (
                name TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                address TEXT,
                phone TEXT,
                website TEXT,
                city TEXT NOT NULL
            );
            CREATE TABLE accommodations (
                name TEXT NOT NULL,
                room_type TEXT,
                price REAL,
                minimum_nights REAL,
                review_rate REAL,
                house_rules TEXT,
                maximum_occupancy REAL,
                city TEXT NOT NULL
            );
            CREATE TABLE restaurants (
                name TEXT NOT NULL,
                city TEXT NOT NULL,
                cuisines TEXT,
                average_cost REAL,
                aggregate_rating REAL
            );
            CREATE TABLE distances (
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                cost REAL,
                duration TEXT,
                distance TEXT
            );
            """
        )
        stats: dict[str, int] = {}
        flight_rows = (
            (
                _text(row[0]),
                _number(row[1]),
                _text(row[2]),
                _text(row[3]),
                _text(row[4]),
                _text(row[5]),
                _text(row[6]),
                _text(row[7]),
                _number(row[8]),
            )
            for row in _rows(
                source_dir / "clean_Flights_2022.csv",
                (
                    "Flight Number",
                    "Price",
                    "DepTime",
                    "ArrTime",
                    "ActualElapsedTime",
                    "FlightDate",
                    "OriginCityName",
                    "DestCityName",
                    "Distance",
                ),
            )
        )
        stats["flights"] = _insert_batches(
            connection,
            "INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?)",
            flight_rows,
            batch_size,
        )
        attraction_rows = (
            (
                _text(row[0]),
                _number(row[1]),
                _number(row[2]),
                _text(row[3]),
                _text(row[4]),
                _text(row[5]),
                _text(row[6]),
            )
            for row in _rows(
                source_dir / "attractions.csv",
                ("Name", "Latitude", "Longitude", "Address", "Phone", "Website", "City"),
            )
        )
        stats["attractions"] = _insert_batches(
            connection,
            "INSERT INTO attractions VALUES (?,?,?,?,?,?,?)",
            attraction_rows,
            batch_size,
        )
        accommodation_rows = (
            (
                _text(row[0]),
                _text(row[1]),
                _number(row[2]),
                _number(row[3]),
                _number(row[4]),
                _text(row[5]),
                _number(row[6]),
                _text(row[7]),
            )
            for row in _rows(
                source_dir / "clean_accommodations_2022.csv",
                (
                    "NAME",
                    "room type",
                    "price",
                    "minimum nights",
                    "review rate number",
                    "house_rules",
                    "maximum occupancy",
                    "city",
                ),
            )
        )
        stats["accommodations"] = _insert_batches(
            connection,
            "INSERT INTO accommodations VALUES (?,?,?,?,?,?,?,?)",
            accommodation_rows,
            batch_size,
        )
        restaurant_rows = (
            (
                _text(row[0]),
                _text(row[1]),
                _text(row[2]),
                _number(row[3]),
                _number(row[4]),
            )
            for row in _rows(
                source_dir / "clean_restaurant_2022.csv",
                ("Name", "City", "Cuisines", "Average Cost", "Aggregate Rating"),
            )
        )
        stats["restaurants"] = _insert_batches(
            connection,
            "INSERT INTO restaurants VALUES (?,?,?,?,?)",
            restaurant_rows,
            batch_size,
        )
        distance_rows = (
            (_text(row[0]), _text(row[1]), _number(row[2]), _text(row[3]), _text(row[4]))
            for row in _rows(
                source_dir / "distance.csv",
                ("origin", "destination", "cost", "duration", "distance"),
            )
        )
        stats["distances"] = _insert_batches(
            connection,
            "INSERT INTO distances VALUES (?,?,?,?,?)",
            distance_rows,
            batch_size,
        )
        connection.executescript(
            """
            CREATE INDEX flights_route_date_idx
                ON flights(origin COLLATE NOCASE, destination COLLATE NOCASE, flight_date, price);
            CREATE INDEX attractions_city_idx ON attractions(city COLLATE NOCASE, name COLLATE NOCASE);
            CREATE INDEX accommodations_city_idx
                ON accommodations(city COLLATE NOCASE, price, review_rate);
            CREATE INDEX restaurants_city_idx
                ON restaurants(city COLLATE NOCASE, aggregate_rating, name COLLATE NOCASE);
            CREATE INDEX distances_route_idx
                ON distances(origin COLLATE NOCASE, destination COLLATE NOCASE);
            ANALYZE;
            """
        )
        metadata = {
            "source_repository": "xxwu/Agent-STAR-TravelDatabase",
            "source_revision": SOURCE_REVISION,
            "declared_license": "mit",
            **{f"rows_{key}": str(value) for key, value in stats.items()},
            **{f"sha256_{key}": value for key, value in EXPECTED_SHA256.items()},
        }
        connection.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    os.replace(temporary, output_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/agent_star_travel_database"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/agent_star_travel.sqlite3"),
    )
    parser.add_argument("--batch-size", type=int, default=20_000)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    stats = build_database(args.source_dir, args.output, args.batch_size)
    for name, count in sorted(stats.items()):
        print(f"{name}={count}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
