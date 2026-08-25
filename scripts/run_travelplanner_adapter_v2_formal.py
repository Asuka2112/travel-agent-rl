#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent_rl.evaluation.travelplanner_adapter_v2 import (
    adapt_final_answer_v2,
    validate_official_schema,
)


LABELS = ("base", "sft", "rl")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_answer(answer: str | None) -> str:
    return sha256_bytes((answer or "").encode("utf-8"))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot frozen-output Adapter v2 postprocess")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = repo / "outputs/final/travelplanner_adapter_v2"
    raw_root = repo / "outputs/final/travelplanner_validation"
    freeze_path = output / "adapter_freeze_manifest.json"
    lock_path = output / "formal_run_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"formal adapter run is immutable and already started: {lock_path}")
    freeze = load(freeze_path)
    adapter_path = repo / "src/travel_agent_rl/evaluation/travelplanner_adapter_v2.py"
    if sha256_file(adapter_path) != freeze["adapter_sha256"]:
        raise RuntimeError("adapter source differs from frozen hash")
    raw_manifest = load(output / "raw_answer_manifest.pre_formal.json")

    started_at = datetime.now(timezone.utc).isoformat()
    lock_path.write_text(
        json.dumps(
            {
                "status": "started",
                "started_at": started_at,
                "adapter_sha256": freeze["adapter_sha256"],
                "formal_generation_attempt": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries: dict[str, Any] = {}
    for label in LABELS:
        records: list[dict[str, Any]] = []
        submissions: list[dict[str, Any]] = []
        expected = raw_manifest["labels"][label]["items"]
        for task_id in range(1, 181):
            source = load(raw_root / label / f"{task_id:03d}.json")
            answer = source.get("final_answer")
            answer_hash = sha256_answer(answer)
            if answer_hash != expected[task_id - 1]["final_answer_sha256"]:
                raise RuntimeError(f"frozen raw answer hash mismatch: {label}/{task_id:03d}")
            plan = adapt_final_answer_v2(answer)
            schema_valid, schema_errors = validate_official_schema(plan)
            records.append(
                {
                    "idx": task_id - 1,
                    "task_id": task_id,
                    "query_id": source.get("query_id"),
                    "raw_final_answer_sha256": answer_hash,
                    "parser_valid": plan is not None,
                    "schema_valid": schema_valid,
                    "schema_errors": schema_errors,
                    "plan": plan,
                }
            )
            submissions.append(
                {"idx": task_id - 1, "query": source.get("query"), "plan": plan}
            )
        label_dir = output / label
        label_dir.mkdir(parents=True, exist_ok=False)
        records_path = label_dir / "parsed_records.jsonl"
        submission_path = label_dir / "official_submission.jsonl"
        write_jsonl(records_path, records)
        write_jsonl(submission_path, submissions)
        summaries[label] = {
            "n": len(records),
            "parser_valid": sum(row["parser_valid"] for row in records),
            "schema_valid": sum(row["schema_valid"] for row in records),
            "parser_failure": sum(not row["parser_valid"] for row in records),
            "parsed_records_sha256": sha256_file(records_path),
            "official_submission_sha256": sha256_file(submission_path),
        }

    completed_at = datetime.now(timezone.utc).isoformat()
    lock_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "adapter_sha256": freeze["adapter_sha256"],
                "formal_generation_attempt": 1,
                "model_inference_runs": 0,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
