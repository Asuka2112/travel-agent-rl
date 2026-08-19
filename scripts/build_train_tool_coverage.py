#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROUTES = (
    ("Boston", "New York City"),
    ("New York City", "Boston"),
    ("New York City", "Philadelphia"),
    ("Philadelphia", "New York City"),
    ("Philadelphia", "Washington"),
    ("Washington", "Philadelphia"),
    ("Washington", "Richmond"),
    ("Richmond", "Washington"),
    ("Portland", "Seattle"),
    ("Seattle", "Portland"),
    ("Los Angeles", "San Diego"),
    ("San Diego", "Los Angeles"),
    ("Chicago", "Milwaukee"),
    ("Milwaukee", "Chicago"),
    ("Chicago", "St. Louis"),
    ("St. Louis", "Chicago"),
    ("Sacramento", "Oakland"),
    ("Oakland", "Sacramento"),
    ("Raleigh", "Charlotte"),
    ("Charlotte", "Raleigh"),
    ("New Haven", "New York City"),
    ("New York City", "New Haven"),
    ("Boston", "New York City"),
    ("Portland", "Seattle"),
)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/train_tool_coverage_24.jsonl")
    )
    parser.add_argument(
        "--frozen", type=Path, default=Path("data/manifests/train_queries_1835.jsonl")
    )
    args = parser.parse_args()

    frozen_rows = [
        json.loads(line)
        for line in args.frozen.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frozen_ids = {row["query_id"] for row in frozen_rows}
    frozen_queries = {row["query"].casefold().strip() for row in frozen_rows}
    rows = []
    for index, (origin, destination) in enumerate(ROUTES, start=1):
        date = f"2022-{((index - 1) % 7) + 1:02d}-{((index * 3 - 1) % 25) + 1:02d}"
        query = (
            f"Plan a practical two-day trip for one traveler from {origin} to "
            f"{destination}, departing on {date}. I explicitly prefer intercity rail "
            "over flying or driving. Use an available train option, state that schedule "
            "evidence is simulated, and continue the plan using the train observation. "
            "Keep the transport cost under $100 if the simulator offers such an option."
        )
        query_id = f"coverage:train:v1:{index:02d}"
        if query_id in frozen_ids or query.casefold().strip() in frozen_queries:
            raise SystemExit(f"coverage row contaminates frozen data: {query_id}")
        row = {
            "query_id": query_id,
            "query": query,
            "coverage_only": True,
            "excluded_from_training_and_evaluation": True,
            "origin": origin,
            "destination": destination,
            "date": date,
            "expected_tool": "train_search",
            "fixture_source": "travel-agent-rl/train-simulator-us@v1",
        }
        row["row_sha256"] = hashlib.sha256(canonical(row)).hexdigest()
        rows.append(row)

    payload = b"".join(canonical(row) + b"\n" for row in rows)
    if args.output.exists():
        if args.output.read_bytes() != payload:
            raise SystemExit(f"refusing to overwrite different coverage manifest: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(args.output)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "frozen_id_overlap": 0,
                "frozen_exact_query_overlap": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
