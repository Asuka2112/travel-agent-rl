#!/usr/bin/env python3
"""Build an immutable, contamination-checked SFT offline-validation subset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent_rl.sft.data import (
    build_agent_tool_schemas,
    convert_frozen_trajectory,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def rank(seed: int, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:sft-heldout-v1:{query_id}".encode()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n" for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/intermediate/sft/heldout_validation_64.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/sft_heldout_validation_64_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    paths = {
        "accepted": root / "data/intermediate/cleaning/judge_accepted.jsonl",
        "sft": root / "data/processed/sft.jsonl",
        "rl": root / "data/processed/rl_queries.jsonl",
        "dev": root / "data/processed/dev.jsonl",
        "frozen_test": root / "data/manifests/frozen_test_ids.json",
        "split_manifest": root / "data/manifests/processed_split_manifest.json",
    }
    source_hashes_before = {
        name: sha256_file(path) for name, path in paths.items()
    }
    accepted = read_jsonl(paths["accepted"])
    excluded: set[str] = set()
    for name in ("sft", "rl", "dev"):
        excluded.update(str(row["query_id"]) for row in read_jsonl(paths[name]))
    frozen_test = json.loads(paths["frozen_test"].read_text(encoding="utf-8"))
    excluded.update(str(value) for value in frozen_test["ids"])

    candidates = [row for row in accepted if str(row["query_id"]) not in excluded]
    if len(candidates) < args.size:
        raise RuntimeError(f"only {len(candidates)} eligible held-out trajectories")
    selected = sorted(
        candidates,
        key=lambda row: (rank(args.seed, str(row["query_id"])), str(row["query_id"])),
    )[: args.size]
    selected.sort(key=lambda row: str(row["query_id"]))
    selected_ids = {str(row["query_id"]) for row in selected}
    if selected_ids & excluded:
        raise AssertionError("held-out validation overlaps a prohibited split")

    tools = build_agent_tool_schemas()
    converted = [convert_frozen_trajectory(row, tools=tools) for row in selected]
    payload = jsonl_bytes(converted)
    output = root / args.output
    atomic_write(output, payload)
    source_hashes_after = {
        name: sha256_file(path) for name, path in paths.items()
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("a frozen source changed during held-out selection")

    difficulty = Counter(str(row["difficulty"]).title() for row in converted)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "fixed-sha256-rank-v1",
        "seed": args.seed,
        "row_count": len(converted),
        "candidate_count_after_exclusions": len(candidates),
        "difficulty": dict(sorted(difficulty.items())),
        "query_ids": sorted(selected_ids),
        "exclusions": {
            "sft": 881,
            "rl": 200,
            "synthetic_dev": 16,
            "frozen_test": len(frozen_test["ids"]),
            "overlap": 0,
        },
        "parent_contamination": json.loads(
            paths["split_manifest"].read_text(encoding="utf-8")
        )["parent_contamination"],
        "source_sha256": source_hashes_before,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "format": "ms-swift-native-agent-v1",
        "template": "qwen3_nothinking/hermes",
    }
    atomic_write(
        root / args.manifest,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
