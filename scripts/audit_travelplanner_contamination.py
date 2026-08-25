#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from travel_agent_rl.data.prepare import (
    CONSTRAINT_FIELDS,
    canonicalize_query,
    constraint_fingerprint,
    sha256_bytes,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen TravelPlanner contamination audit")
    parser.add_argument("--validation", type=Path, default=Path("data/raw/travelplanner/validation.csv"))
    parser.add_argument(
        "--candidate",
        action="append",
        type=Path,
        default=None,
        help="JSONL training-derived artifact; may be repeated",
    )
    parser.add_argument("--threshold", type=float, default=99.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/final/travelplanner_validation/contamination_audit.json"),
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _query(row: dict[str, Any]) -> str:
    if row.get("query"):
        return str(row["query"])
    messages = row.get("messages") or []
    for message in messages:
        if message.get("role") == "user":
            return str(message["content"])
    raise ValueError("candidate row has no query or user message")


def _attributes(row: dict[str, Any]) -> dict[str, Any]:
    attributes = dict(row.get("attributes") or {})
    for field in CONSTRAINT_FIELDS:
        if field in row and field not in attributes:
            attributes[field] = row[field]
    return attributes


def main() -> None:
    args = parse_args()
    candidates = args.candidate or [
        Path("data/manifests/train_queries_1835.jsonl"),
        Path("data/processed/sft.jsonl"),
        Path("data/processed/rl_queries.jsonl"),
        Path("data/processed/dev.jsonl"),
    ]
    with args.validation.open(encoding="utf-8", newline="") as handle:
        validation = list(csv.DictReader(handle))

    validation_canonical = [canonicalize_query(str(row["query"])) for row in validation]
    validation_hashes = {sha256_bytes(text.encode()) for text in validation_canonical}
    validation_constraints = {
        constraint_fingerprint(row, CONSTRAINT_FIELDS): index
        for index, row in enumerate(validation)
    }
    artifact_results = []
    all_exact: list[dict[str, Any]] = []
    all_constraints: list[dict[str, Any]] = []
    all_fuzzy: list[dict[str, Any]] = []

    for path in candidates:
        rows = _read_jsonl(path)
        exact = []
        constraints = []
        fuzzy = []
        for index, row in enumerate(rows):
            canonical = canonicalize_query(_query(row))
            query_hash = sha256_bytes(canonical.encode())
            if query_hash in validation_hashes:
                exact.append({"candidate_row": index, "query_sha256": query_hash})
            attributes = _attributes(row)
            if all(field in attributes for field in CONSTRAINT_FIELDS):
                fingerprint = constraint_fingerprint(attributes, CONSTRAINT_FIELDS)
                if fingerprint in validation_constraints:
                    constraints.append(
                        {
                            "candidate_row": index,
                            "validation_row": validation_constraints[fingerprint],
                            "constraint_sha256": fingerprint,
                        }
                    )
            match = process.extractOne(
                canonical,
                validation_canonical,
                scorer=fuzz.ratio,
                score_cutoff=args.threshold,
            )
            if match is not None and query_hash not in validation_hashes:
                fuzzy.append(
                    {
                        "candidate_row": index,
                        "validation_row": match[2],
                        "score": match[1],
                    }
                )
        entry = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "exact_or_canonical_overlaps": exact,
            "constraint_fingerprint_overlaps": constraints,
            "fuzzy_overlaps_at_or_above_threshold": fuzzy,
        }
        artifact_results.append(entry)
        all_exact.extend({"path": str(path), **item} for item in exact)
        all_constraints.extend({"path": str(path), **item} for item in constraints)
        all_fuzzy.extend({"path": str(path), **item} for item in fuzzy)

    passed = not (all_exact or all_constraints or all_fuzzy)
    result = {
        "audit_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "protected_split": "official TravelPlanner validation",
            "fuzzy_metric": "rapidfuzz.fuzz.ratio on canonical query",
            "fuzzy_threshold": args.threshold,
            "constraint_fields": list(CONSTRAINT_FIELDS),
        },
        "validation": {
            "path": str(args.validation),
            "sha256": sha256_file(args.validation),
            "rows": len(validation),
        },
        "candidate_artifacts": artifact_results,
        "totals": {
            "candidate_rows_scanned": sum(item["rows"] for item in artifact_results),
            "exact_or_canonical_overlaps": len(all_exact),
            "constraint_fingerprint_overlaps": len(all_constraints),
            "fuzzy_overlaps_at_or_above_threshold": len(all_fuzzy),
        },
        "gate": "passed" if passed else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["totals"], sort_keys=True))
    if not passed:
        raise SystemExit("contamination audit failed")


if __name__ == "__main__":
    main()
