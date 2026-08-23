#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from travel_agent_rl.sft.data import (
    build_agent_tool_schemas,
    convert_frozen_trajectory,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare immutable SFT data for ms-swift")
    parser.add_argument("--input", type=Path, default=Path("data/processed/sft.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/intermediate/sft/ms_swift_agent_881.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/sft_agent_881_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_hash_before = sha256_file(args.input)
    tools = build_agent_tool_schemas()
    rows = [json.loads(line) for line in args.input.open() if line.strip()]
    converted = [convert_frozen_trajectory(row, tools=tools) for row in rows]
    if len(converted) != 881:
        raise ValueError(f"expected 881 frozen SFT trajectories, got {len(converted)}")
    if len({row["query_id"] for row in converted}) != len(converted):
        raise ValueError("duplicate query_id after conversion")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    input_hash_after = sha256_file(args.input)
    if input_hash_before != input_hash_after:
        raise RuntimeError("frozen source data changed during conversion")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "ms-swift-native-agent-v1",
        "agent_template": "hermes",
        "template": "qwen3_nothinking",
        "loss_scale": "default",
        "row_count": len(converted),
        "tool_count": len(tools),
        "tool_names": [tool["function"]["name"] for tool in tools],
        "source": str(args.input),
        "source_sha256": input_hash_before,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
