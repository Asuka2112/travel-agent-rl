#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import yaml

from travel_agent_rl.distill import OpenAICompatibleChatBackend, TrajectoryRecord
from travel_agent_rl.filtering import StructuredJudge, apply_rule_filter, canonicalize_trajectory
from travel_agent_rl.tools import build_mock_registry


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/filter/default.yaml"))
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    config = yaml.safe_load((root / args.config).read_text())
    records = [
        TrajectoryRecord.model_validate_json(line)
        for line in (root / config["input_path"]).read_text().splitlines()
        if line.strip()
    ]
    allowed_tools = set(build_mock_registry().names)
    rejected_reasons: Counter[str] = Counter()
    accepted: list[TrajectoryRecord] = []
    for record in records:
        result = apply_rule_filter(record, allowed_tools, max_turns=int(config["max_turns"]))
        if result.accepted:
            accepted.append(record)
        else:
            rejected_reasons.update(result.reasons)

    write_jsonl(
        root / config["rule_accepted_path"],
        [record.model_dump(mode="json") for record in accepted],
    )
    canonical = [canonicalize_trajectory(record) for record in accepted]
    write_jsonl(
        root / config["canonical_path"],
        [record.model_dump(mode="json") for record in canonical],
    )

    judged: list[TrajectoryRecord] = []
    if not args.skip_judge:
        env = {
            "api_key": os.getenv("JUDGE_API_KEY"),
            "base_url": os.getenv("JUDGE_BASE_URL"),
            "model": os.getenv("JUDGE_MODEL"),
        }
        missing = [name for name, value in env.items() if not value]
        if missing:
            raise SystemExit(f"missing judge settings: {', '.join(missing)}")
        backend = OpenAICompatibleChatBackend(**env)
        judge = StructuredJudge(
            backend,
            (root / config["judge_prompt_path"]).read_text(),
            prompt_version=config["judge_prompt_version"],
            mean_threshold=float(config["judge_mean_threshold"]),
            minimum_dimension=float(config["judge_minimum_dimension"]),
        )
        for record in canonical:
            if judge.evaluate(record).accepted:
                judged.append(record)
        write_jsonl(
            root / config["judge_accepted_path"],
            [record.model_dump(mode="json") for record in judged],
        )

    stats = {
        "input": len(records),
        "rule_accepted": len(accepted),
        "rule_rejected": len(records) - len(accepted),
        "rejection_reasons": dict(sorted(rejected_reasons.items())),
        "judge_accepted": None if args.skip_judge else len(judged),
    }
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
