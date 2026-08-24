#!/usr/bin/env python3
"""Build a synthetic, non-training regression set for replay-policy v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MISS_QUERY = "flights from Fort Lauderdale to North Carolina cities March 2022"


def canonical_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/intermediate/grpo/smoke_tool_12.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/intermediate/grpo/replay_v2_regression_3.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/replay_v2_regression_manifest.json"),
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in args.source.open() if line.strip()]
    source = next(record for record in records if record["query_id"] == "agent_star:hard_trsd60")
    original = source["messages"][0]["content"]
    instruction = (
        "Replay-policy regression test. First call web_search exactly once with the exact "
        f'query string "{MISS_QUERY}". The frozen snapshot intentionally has no entry. '
        "Treat the structured snapshot_miss as a valid observation, then call flight_search "
        "for travel evidence and finish with a useful answer. Do not repeat web_search.\n\n"
    )
    derived = []
    for index in range(3):
        record = dict(source)
        record["query_id"] = f"replay_v2_regression:{index}"
        record["difficulty"] = "regression"
        record["messages"] = [{"role": "user", "content": instruction + original}]
        derived.append(record)

    payload = "".join(canonical_line(record) + "\n" for record in derived).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "synthetic replay-policy regression only; excluded from frozen train/eval",
        "source": str(args.source),
        "source_sha256": source_sha,
        "source_query_id": source["query_id"],
        "required_snapshot_miss_query": MISS_QUERY,
        "record_count": len(derived),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_version": 4,
        "web_mode": "replay_only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
