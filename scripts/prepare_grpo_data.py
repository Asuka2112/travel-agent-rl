#!/usr/bin/env python3
"""Create immutable fixed-hash smoke/pilot views of the frozen 200 RL queries."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent_rl.sft.data import build_agent_tool_schemas, sha256_file

SOURCE = Path("data/processed/rl_queries.jsonl")
OUTPUTS = {
    "smoke": (12, Path("data/intermediate/grpo/smoke_12.jsonl")),
    "pilot": (48, Path("data/intermediate/grpo/pilot_48.jsonl")),
    "formal": (200, Path("data/intermediate/grpo/formal_200.jsonl")),
}
SEED = 20260819


def rank(query_id: str) -> str:
    return hashlib.sha256(f"{SEED}:grpo-subsets-v1:{query_id}".encode()).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    source_hash_before = sha256_file(SOURCE)
    source_rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
    if len(source_rows) != 200 or len({row["query_id"] for row in source_rows}) != 200:
        raise RuntimeError("frozen RL source is not the expected unique 200-query set")
    ordered = sorted(source_rows, key=lambda row: (rank(row["query_id"]), row["query_id"]))
    tools = json.dumps(build_agent_tool_schemas(), ensure_ascii=False, separators=(",", ":"))
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "fixed-sha256-prefix-v1",
        "seed": SEED,
        "source": str(SOURCE),
        "source_rows": len(source_rows),
        "source_sha256": source_hash_before,
        "subsets": {},
    }
    for name, (size, path) in OUTPUTS.items():
        selected = ordered[:size]
        converted = [
            {
                "query_id": row["query_id"],
                "difficulty": str(row.get("attributes", {}).get("level", "unknown")),
                "messages": [{"role": "user", "content": row["query"]}],
                "tools": tools,
            }
            for row in selected
        ]
        payload = b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n"
            for row in converted
        )
        atomic_write(path, payload)
        manifest["subsets"][name] = {
            "path": str(path),
            "rows": size,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "query_ids": [row["query_id"] for row in converted],
            "difficulty": dict(Counter(row["difficulty"] for row in converted)),
        }
    if sha256_file(SOURCE) != source_hash_before:
        raise RuntimeError("frozen RL source changed during subset generation")
    atomic_write(
        Path("data/manifests/grpo_subsets_manifest.json"),
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
