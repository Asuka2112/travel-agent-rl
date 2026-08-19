#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from travel_agent_rl.distill import (
    OpenAICompatibleChatBackend,
    RetryingChatBackend,
    TrajectoryRecord,
)
from travel_agent_rl.filtering import StructuredJudge, apply_rule_filter, canonicalize_trajectory
from travel_agent_rl.tools import build_mock_registry


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def judge_checkpoint_path(directory: Path, query_id: str) -> Path:
    readable = "".join(character if character.isalnum() else "_" for character in query_id)
    digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:12]
    return directory / f"{readable[:80]}__{digest}.json"


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
    audit: list[dict[str, object]] = []
    for record in records:
        result = apply_rule_filter(record, allowed_tools, max_turns=int(config["max_turns"]))
        audit.append(
            {
                "accepted": result.accepted,
                "drop_reason": [] if result.accepted else result.reasons,
                "query_id": record.query_id,
                "reasons": result.reasons,
                "stage": "rule",
            }
        )
        if result.accepted:
            accepted.append(record)
        else:
            rejected_reasons.update(result.reasons)

    write_jsonl(
        root / config["rule_accepted_path"],
        [record.model_dump(mode="json") for record in accepted],
    )
    canonical: list[TrajectoryRecord] = []
    format_rejected_reasons: Counter[str] = Counter()
    for record in accepted:
        try:
            formatted = canonicalize_trajectory(record)
        except Exception as exc:
            reason = f"format_exception:{type(exc).__name__}"
            format_rejected_reasons[reason] += 1
            audit.append(
                {
                    "accepted": False,
                    "drop_reason": [reason],
                    "query_id": record.query_id,
                    "stage": "format",
                }
            )
        else:
            canonical.append(formatted)
            audit.append(
                {
                    "accepted": True,
                    "drop_reason": [],
                    "query_id": record.query_id,
                    "stage": "format",
                }
            )
    write_jsonl(
        root / config["canonical_path"],
        [record.model_dump(mode="json") for record in canonical],
    )

    judged: list[TrajectoryRecord] = []
    judge_resumed = 0
    if not args.skip_judge:
        env = {
            "api_key": os.getenv("JUDGE_API_KEY"),
            "base_url": os.getenv("JUDGE_BASE_URL"),
            "model": os.getenv("JUDGE_MODEL"),
        }
        missing = [name for name, value in env.items() if not value]
        if missing:
            raise SystemExit(f"missing judge settings: {', '.join(missing)}")
        backend = RetryingChatBackend(
            OpenAICompatibleChatBackend(**env, temperature=0.0, seed=20260819),
            max_attempts=3,
        )
        judge = StructuredJudge(
            backend,
            (root / config["judge_prompt_path"]).read_text(),
            prompt_version=config["judge_prompt_version"],
            mean_threshold=float(config["judge_mean_threshold"]),
            minimum_dimension=float(config["judge_minimum_dimension"]),
        )
        judge_dir = root / config.get(
            "judge_raw_dir",
            str(Path(config["audit_path"]).parent / "judge_raw"),
        )
        judge_dir.mkdir(parents=True, exist_ok=True)

        def evaluate(record: TrajectoryRecord) -> dict[str, object]:
            try:
                decision = judge.evaluate(record)
            except Exception as exc:
                reason = f"judge_exception:{type(exc).__name__}"
                return {
                    "accepted": False,
                    "drop_reason": [reason],
                    "query_id": record.query_id,
                    "stage": "judge",
                }
            return {
                "accepted": decision.accepted,
                "drop_reason": [] if decision.accepted else ["judge_below_threshold"],
                "judge_dimensions": decision.scores.model_dump(mode="json"),
                "judge_score": decision.scores.mean,
                "prompt_version": decision.prompt_version,
                "query_id": record.query_id,
                "rationale": decision.rationale,
                "score_mean": decision.scores.mean,
                "scores": decision.scores.model_dump(mode="json"),
                "stage": "judge",
            }

        judge_rows: dict[str, dict[str, object]] = {}
        pending: list[TrajectoryRecord] = []
        for record in canonical:
            checkpoint = judge_checkpoint_path(judge_dir, record.query_id)
            if checkpoint.is_file():
                judge_rows[record.query_id] = json.loads(checkpoint.read_text(encoding="utf-8"))
                judge_resumed += 1
            else:
                pending.append(record)
        with ThreadPoolExecutor(max_workers=int(config.get("judge_concurrency", 1))) as executor:
            futures = {executor.submit(evaluate, record): record for record in pending}
            for future in as_completed(futures):
                record = futures[future]
                row = future.result()
                write_json(judge_checkpoint_path(judge_dir, record.query_id), row)
                judge_rows[record.query_id] = row

        for record in canonical:
            row = judge_rows[record.query_id]
            audit.append(row)
            if row["accepted"]:
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
        "format_accepted": len(canonical),
        "format_rejected": len(accepted) - len(canonical),
        "format_rejection_reasons": dict(sorted(format_rejected_reasons.items())),
        "judge_accepted": None if args.skip_judge else len(judged),
        "judge_resumed": judge_resumed,
        "stages": {
            "rule": {"before_count": len(records), "after_count": len(accepted)},
            "format": {"before_count": len(accepted), "after_count": len(canonical)},
            "judge": {
                "before_count": len(canonical),
                "after_count": None if args.skip_judge else len(judged),
            },
        },
    }
    write_jsonl(root / config["audit_path"], audit)
    write_json(root / config["stats_path"], stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
