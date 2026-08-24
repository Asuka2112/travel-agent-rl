#!/usr/bin/env python3
"""Summarize immutable Base/SFT/RL evaluation failure modes by query."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent_rl.sft.data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("base", "sft", "rl"):
        parser.add_argument(f"--{label}-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_records(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("[0-9][0-9].json"))]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    error_types: Counter[str] = Counter()
    failures = []
    for row in records:
        for error in row.get("errors", []):
            error_types[str(error).split(":", 1)[0]] += 1
        reasons = []
        if not row.get("task_complete"):
            reasons.append("task_incomplete")
        if not row.get("final_answer_present"):
            reasons.append("missing_final_answer")
        if row.get("invalid_tool_calls"):
            reasons.append("invalid_tool_call")
        if row.get("loop"):
            reasons.append("loop")
        if row.get("replay_misses"):
            reasons.append("snapshot_miss_observed")
        if row.get("unknown_exception"):
            reasons.append("unknown_exception")
        if reasons:
            failures.append(
                {
                    "query_id": row["query_id"],
                    "reasons": reasons,
                    "errors": row.get("errors", []),
                    "tool_sequence": row.get("tool_sequence", []),
                    "task_score": row.get("task_score", 0.0),
                }
            )
    return {
        "queries": len(records),
        "failure_queries": len(failures),
        "error_types": dict(error_types),
        "snapshot_misses": sum(int(row.get("replay_misses", 0)) for row in records),
        "loops": sum(bool(row.get("loop")) for row in records),
        "invalid_tool_calls": sum(int(row.get("invalid_tool_calls", 0)) for row in records),
        "unknown_exceptions": sum(bool(row.get("unknown_exception")) for row in records),
        "records": failures,
    }


def main() -> None:
    args = parse_args()
    directories = {label: getattr(args, f"{label}_records") for label in ("base", "sft", "rl")}
    records = {label: load_records(path) for label, path in directories.items()}
    query_sets = [{row["query_id"] for row in rows} for rows in records.values()]
    if not query_sets or len({frozenset(items) for items in query_sets}) != 1:
        raise RuntimeError("failure analysis inputs do not contain identical frozen query ids")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_ids_identical": True,
        "inputs": {
            label: {
                "directory": str(path),
                "summary_sha256": sha256_file(path / "summary.json"),
            }
            for label, path in directories.items()
        },
        "models": {label: summarize(rows) for label, rows in records.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({label: {k: v for k, v in row.items() if k != "records"} for label, row in payload["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
