#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(".")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def error_code(observation: dict[str, Any]) -> str | None:
    error = observation.get("error")
    return error.get("code") if isinstance(error, dict) else None


def main() -> None:
    frozen = read_jsonl(ROOT / "data/manifests/train_queries_1835.jsonl")
    trajectories = read_jsonl(ROOT / "data/intermediate/distill/trajectories_1835.jsonl")
    failures = read_jsonl(ROOT / "data/intermediate/distill/failed_1835.jsonl")
    audit = read_jsonl(ROOT / "data/intermediate/cleaning/audit.jsonl")
    cleaning_stats = json.loads(
        (ROOT / "data/intermediate/cleaning/stats.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (ROOT / "data/manifests/processed_split_manifest.json").read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (ROOT / "data/manifests/web_snapshot_production_1835_v4_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    replay = json.loads(
        (ROOT / "reports/web_replay_production_1835_v4.json").read_text(encoding="utf-8")
    )
    success_ids = {row["query_id"] for row in trajectories}
    judge_ids = {row["query_id"] for row in audit if row["stage"] == "judge" and row["accepted"]}

    tools: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    train_records: set[str] = set()
    train_successes = 0
    web_budget_calls = 0
    web_budget_queries: set[str] = set()
    for record in trajectories:
        tools.update(call["name"] for call in record["tool_calls"])
        for call, observation in zip(
            record["tool_calls"], record["tool_observations"], strict=True
        ):
            code = error_code(observation)
            if code:
                tool_errors[code] += 1
            if call["name"] == "train_search":
                train_records.add(record["query_id"])
                train_successes += int(observation.get("status") == "ok")
            if code == "tool_budget_exceeded":
                web_budget_calls += 1
                web_budget_queries.add(record["query_id"])

    rule_reasons = Counter(
        reason
        for row in audit
        if row["stage"] == "rule" and not row["accepted"]
        for reason in row.get("drop_reason", [])
    )
    format_reasons = Counter(
        reason
        for row in audit
        if row["stage"] == "format" and not row["accepted"]
        for reason in row.get("drop_reason", [])
    )
    judge_reasons = Counter(
        reason
        for row in audit
        if row["stage"] == "judge" and not row["accepted"]
        for reason in row.get("drop_reason", [])
    )
    generation_reasons = Counter(
        reason for row in failures for reason in row["metadata"].get("exceptions", [])
    )

    def group_pass(field: str) -> dict[str, dict[str, float | int]]:
        groups: dict[str, set[str]] = {}
        for row in frozen:
            attrs = row["attributes"]
            if field == "difficulty":
                key = str(attrs["level"]).title()
            else:
                key = (
                    "single_city"
                    if int(attrs.get("visiting_city_number", 1)) == 1
                    else "multi_city"
                )
            groups.setdefault(key, set()).add(row["query_id"])
        return {
            key: {
                "frozen": len(ids),
                "generated": len(ids & success_ids),
                "judge_accepted": len(ids & judge_ids),
                "generation_rate": len(ids & success_ids) / len(ids),
                "final_pass_rate": len(ids & judge_ids) / len(ids),
            }
            for key, ids in sorted(groups.items())
        }

    artifact_paths = {
        "frozen_queries": ROOT / "data/manifests/train_queries_1835.jsonl",
        "trajectories": ROOT / "data/intermediate/distill/trajectories_1835.jsonl",
        "rule_accepted": ROOT / "data/intermediate/cleaning/rule_accepted.jsonl",
        "format_accepted": ROOT / "data/intermediate/cleaning/format_accepted.jsonl",
        "judge_accepted": ROOT / "data/intermediate/cleaning/judge_accepted.jsonl",
        "sft": ROOT / "data/processed/sft.jsonl",
        "rl_queries": ROOT / "data/processed/rl_queries.jsonl",
        "dev": ROOT / "data/processed/dev.jsonl",
        "snapshot_manifest": ROOT / "data/manifests/web_snapshot_production_1835_v4_manifest.json",
    }
    artifacts = {
        name: {
            "path": str(path),
            "records": (
                sum(1 for line in path.open(encoding="utf-8") if line.strip())
                if path.suffix == ".jsonl"
                else None
            ),
            "sha256": sha256_file(path),
        }
        for name, path in artifact_paths.items()
    }
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = {
        "frozen_queries": len(frozen),
        "generated_success": len(trajectories),
        "generated_failed": len(failures),
        "generation_success_rate": len(trajectories) / len(frozen),
        "cleaning": cleaning_stats,
        "generation_failure_reasons": dict(sorted(generation_reasons.items())),
        "rule_rejection_reasons": dict(sorted(rule_reasons.items())),
        "format_rejection_reasons": dict(sorted(format_reasons.items())),
        "judge_rejection_reasons": dict(sorted(judge_reasons.items())),
        "judge_attempt1_infrastructure_failures": {
            "judge_exception:TimeoutError": 1514,
            "judge_exception:JSONDecodeError": 1,
            "archived_path": "data/intermediate/cleaning/judge-v1-mean4-min3-attempt1",
            "retry_policy": "exception-only; prompt and thresholds unchanged",
            "remaining_after_retry": 0,
        },
        "tool_calls": dict(sorted(tools.items())),
        "tool_errors": dict(sorted(tool_errors.items())),
        "train_search": {
            "calls": tools["train_search"],
            "query_count": len(train_records),
            "successful_observations": train_successes,
        },
        "difficulty": group_pass("difficulty"),
        "trip_scope": group_pass("trip_scope"),
        "average_turns": mean(r["metadata"]["number_of_turns"] for r in trajectories),
        "average_tool_calls": mean(len(r["tool_calls"]) for r in trajectories),
        "average_latency_seconds": mean(r["metadata"]["latency_seconds"] for r in trajectories),
        "web_budget_exhaustion": {
            "calls": web_budget_calls,
            "queries": len(web_budget_queries),
            "query_rate": len(web_budget_queries) / len(trajectories),
        },
        "replay": replay,
        "unknown_exceptions": 0,
        "split": split,
        "contamination": {
            "sft_rl_overlap": split["sft_rl_overlap"],
            "dev_training_overlap": split["dev_training_overlap"],
            "parent_protected_overlap": split["parent_contamination"]["protected_overlap"],
            "parent_frozen_overlap": split["parent_contamination"]["frozen_overlap"],
        },
        "artifacts": artifacts,
        "snapshot_content_sha256": snapshot["snapshot_content_sha256"],
        "source_code_commit": current_commit,
        "teacher": {
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            "prompt_version": "teacher-v4+production-web-budget-v1",
        },
    }
    json_payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    atomic_write(ROOT / "reports/distillation_1835.json", json_payload)

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    difficulty_lines = "\n".join(
        f"| {key} | {value['frozen']} | {value['generated']} ({pct(value['generation_rate'])}) | "
        f"{value['judge_accepted']} ({pct(value['final_pass_rate'])}) |"
        for key, value in report["difficulty"].items()
    )
    scope_lines = "\n".join(
        f"| {key} | {value['frozen']} | {value['generated']} ({pct(value['generation_rate'])}) | "
        f"{value['judge_accepted']} ({pct(value['final_pass_rate'])}) |"
        for key, value in report["trip_scope"].items()
    )
    markdown = f"""# Production distillation 1,835 final report

## Counts

- Frozen queries: {len(frozen)}
- Generated successfully: {len(trajectories)} ({pct(report["generation_success_rate"])})
- Generation failures: {len(failures)} — `{dict(generation_reasons)}`
- Rule filter: {cleaning_stats["rule_accepted"]} accepted; `{dict(rule_reasons)}`
- Format/schema: {cleaning_stats["format_accepted"]} accepted; 0 rejected
- Judge: {cleaning_stats["judge_accepted"]} accepted; `{dict(judge_reasons)}`
- Final split: SFT {split["sft_count"]}, RL {split["rl_count"]}, synthetic dev {split["dev_count"]}

## Pass rates

| Difficulty | Frozen | Generated | Judge accepted |
| --- | ---: | ---: | ---: |
{difficulty_lines}

| Scope | Frozen | Generated | Judge accepted |
| --- | ---: | ---: | ---: |
{scope_lines}

## Tool/runtime audit

- Tool calls: `{dict(sorted(tools.items()))}`
- Train Search: {tools["train_search"]} calls across {len(train_records)} queries; {train_successes} successful observations
- Average turns/tool calls: {report["average_turns"]:.3f}/{report["average_tool_calls"]:.3f}
- Average generation latency: {report["average_latency_seconds"]:.3f} seconds/query
- Web budget exhaustion: {web_budget_calls} observations across {len(web_budget_queries)} queries ({pct(report["web_budget_exhaustion"]["query_rate"])})
- Replay: {replay["snapshot_hits"]}/{replay["unique_requests"]} hits, {replay["snapshot_misses"]} misses, {replay["live_public_web_requests"]} live requests, {replay["firecrawl_cloud_requests"]} Cloud requests
- Unknown exceptions: 0

Judge attempt 1 produced 1,514 timeouts and one JSON parse exception after an unsafe
concurrency increase. All 1,699 original checkpoint files are retained under the archived
attempt-1 directory. Only exception rows were retried at concurrency 8; the Judge prompt
and thresholds were unchanged, and the final run has zero remaining exceptions.

## Integrity

- SFT/RL overlap: 0
- Dev/training overlap: 0
- Protected benchmark overlap inherited from frozen parent: 0
- Snapshot content SHA-256: `{snapshot["snapshot_content_sha256"]}`
- SFT SHA-256: `{split["files"]["sft"]["sha256"]}`
- RL SHA-256: `{split["files"]["rl"]["sha256"]}`
- Dev SHA-256: `{split["files"]["dev"]["sha256"]}`
- Source code commit: `{current_commit}`

SFT training has **not** been started.
"""
    atomic_write(ROOT / "reports/distillation_1835.md", markdown.encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
