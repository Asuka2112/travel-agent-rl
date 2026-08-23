#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from travel_agent_rl.sft.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/intermediate/sft/ms_swift_agent_881.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/intermediate/sft/smoke_32.jsonl")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/sft_smoke_32_manifest.json")
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--easy", type=int, default=12)
    parser.add_argument("--medium", type=int, default=12)
    parser.add_argument("--hard", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open() if line.strip()]
    by_difficulty = {
        difficulty: [row for row in rows if row["difficulty"] == difficulty]
        for difficulty in ("easy", "medium", "hard")
    }
    rng = random.Random(args.seed)
    selected = []
    counts = {"easy": args.easy, "medium": args.medium, "hard": args.hard}
    for difficulty, count in counts.items():
        choices = sorted(by_difficulty[difficulty], key=lambda row: row["query_id"])
        selected.extend(rng.sample(choices, count))
    selected.sort(key=lambda row: hashlib.sha256(row["query_id"].encode()).hexdigest())
    expected_count = sum(counts.values())
    if len({row["query_id"] for row in selected}) != expected_count:
        raise RuntimeError("smoke subset is not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "row_count": len(selected),
        "query_ids": [row["query_id"] for row in selected],
        "difficulty": {
            difficulty: sum(row["difficulty"] == difficulty for row in selected)
            for difficulty in ("easy", "medium", "hard")
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
