from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


LENGTH_BUCKETS = (4096, 8192, 16384, 24576, 32768)


def nearest_rank(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def length_bucket(length: int) -> str:
    if length <= 4096:
        return "<=4K"
    if length <= 8192:
        return "4K-8K"
    if length <= 16384:
        return "8K-16K"
    if length <= 24576:
        return "16K-24K"
    if length <= 32768:
        return "24K-32K"
    return ">32K"


def select_label_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 15 distinct, deterministic rows across required audit categories."""
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_matching(predicate, count: int) -> None:
        matches = [
            row
            for row in rows
            if row["query_id"] not in selected_ids and predicate(row)
        ]
        matches.sort(key=lambda row: row["query_id"])
        if len(matches) < count:
            raise ValueError(f"only {len(matches)} rows available for an audit category")
        for row in matches[:count]:
            selected.append(row)
            selected_ids.add(row["query_id"])

    for difficulty in ("easy", "medium", "hard"):
        add_matching(lambda row, difficulty=difficulty: row["difficulty"] == difficulty, 3)
    add_matching(lambda row: _uses_tool(row, "train_search"), 3)
    add_matching(
        lambda row: _uses_tool(row, "web_search") or _uses_tool(row, "web_scraping"),
        3,
    )
    return selected


def _uses_tool(row: dict[str, Any], tool_name: str) -> bool:
    needle = f'"name":"{tool_name}"'
    return any(
        message["role"] == "tool_call" and needle in message["content"]
        for message in row["messages"]
    )
