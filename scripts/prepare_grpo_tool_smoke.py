#!/usr/bin/env python3
"""Freeze a tool-demanding GRPO smoke view without changing the RL split.

The original hash-prefix smoke is retained as an unbiased resource probe.  This
second engineering fixture uses only hard queries from the already-frozen RL
set, where the prompt carries transport and accommodation constraints that
cannot be checked responsibly without tools.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from travel_agent_rl.sft.data import build_agent_tool_schemas, sha256_file

SOURCE = Path("data/processed/rl_queries.jsonl")
OUTPUT = Path("data/intermediate/grpo/smoke_tool_12.jsonl")
MANIFEST = Path("data/manifests/grpo_smoke_tool_coverage_manifest.json")
SEED = 20260819
SIZE = 12


def _rank(query_id: str) -> str:
    return hashlib.sha256(
        f"{SEED}:grpo-tool-smoke-v1:{query_id}".encode()
    ).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    source_hash = sha256_file(SOURCE)
    rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
    if len(rows) != 200 or len({row["query_id"] for row in rows}) != 200:
        raise RuntimeError("frozen RL source is not the expected unique 200-query set")
    hard = [
        row
        for row in rows
        if str(row.get("attributes", {}).get("level", "")).lower() == "hard"
    ]
    selected = sorted(hard, key=lambda row: (_rank(row["query_id"]), row["query_id"]))[
        :SIZE
    ]
    if len(selected) != SIZE:
        raise RuntimeError(f"expected at least {SIZE} frozen hard RL queries")
    tools = json.dumps(build_agent_tool_schemas(), ensure_ascii=False, separators=(",", ":"))
    converted = [
        {
            "query_id": row["query_id"],
            "difficulty": "hard",
            "messages": [{"role": "user", "content": row["query"]}],
            "tools": tools,
        }
        for row in selected
    ]
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n"
        for row in converted
    )
    _atomic_write(OUTPUT, payload)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "engineering tool-chain coverage only; not a new train/eval split",
        "algorithm": "hard-only-fixed-sha256-prefix-v1",
        "seed": SEED,
        "source": str(SOURCE),
        "source_rows": len(rows),
        "source_sha256": source_hash,
        "output": str(OUTPUT),
        "rows": len(converted),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "query_ids": [row["query_id"] for row in converted],
    }
    _atomic_write(
        MANIFEST,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    if sha256_file(SOURCE) != source_hash:
        raise RuntimeError("frozen RL source changed while creating smoke fixture")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
