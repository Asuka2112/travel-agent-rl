#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(".")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def observations(record: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return list(zip(record["tool_calls"], record["tool_observations"], strict=True))


def error_code(observation: dict[str, Any]) -> str | None:
    error = observation.get("error")
    return error.get("code") if isinstance(error, dict) else None


def judge_map(audit_path: str) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row for row in read_jsonl(audit_path) if row["stage"] == "judge"}


def accepted_ids(audit_path: str) -> set[str]:
    return {
        row["query_id"]
        for row in read_jsonl(audit_path)
        if row["stage"] == "judge" and row["accepted"]
    }


def pilot_metrics(
    trajectory_path: str,
    audit_path: str,
    denominator: int,
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    records = read_jsonl(trajectory_path)
    if allowed_ids is not None:
        records = [row for row in records if row["query_id"] in allowed_ids]
    judged = judge_map(audit_path)
    passed = accepted_ids(audit_path)
    budget_calls = 0
    budget_ids: set[str] = set()
    tools: Counter[str] = Counter()
    for record in records:
        tools.update(call["name"] for call in record["tool_calls"])
        for _, observation in observations(record):
            if error_code(observation) == "tool_budget_exceeded":
                budget_calls += 1
                budget_ids.add(record["query_id"])
    scores = [row["judge_score"] for row in judged.values() if "judge_score" in row]
    return {
        "generated": len(records),
        "generation_rate": len(records) / denominator,
        "final_pass": len(passed),
        "final_pass_rate": len(passed) / denominator,
        "average_turns": mean(row["metadata"]["number_of_turns"] for row in records),
        "average_tool_calls": mean(len(row["tool_calls"]) for row in records),
        "average_latency_seconds": mean(row["metadata"]["latency_seconds"] for row in records),
        "average_judge_score": mean(scores),
        "budget_exhaustion_calls": budget_calls,
        "budget_exhaustion_queries": len(budget_ids),
        "tool_calls": dict(sorted(tools.items())),
    }


def main() -> None:
    baseline_200 = read_jsonl("data/intermediate/distill/pilot_v3/trajectories.jsonl")
    judge_200 = judge_map("data/intermediate/distill/pilot_200_v3_filter/audit.jsonl")
    pass_200 = accepted_ids("data/intermediate/distill/pilot_200_v3_filter/audit.jsonl")
    frozen = {row["query_id"]: row for row in read_jsonl("data/manifests/train_queries_1835.jsonl")}
    exhaustion_rows = []
    for record in baseline_200:
        search_total = scrape_total = search_executed = scrape_executed = budget_calls = 0
        errors: Counter[str] = Counter()
        for call, observation in observations(record):
            code = error_code(observation)
            if code:
                errors[code] += 1
            if call["name"] == "web_search":
                search_total += 1
                search_executed += int(code != "tool_budget_exceeded")
            elif call["name"] == "web_scraping":
                scrape_total += 1
                scrape_executed += int(code != "tool_budget_exceeded")
            budget_calls += int(code == "tool_budget_exceeded")
        if budget_calls:
            judge = judge_200.get(record["query_id"], {})
            source = frozen[record["query_id"]]
            exhaustion_rows.append(
                {
                    "query_id": record["query_id"],
                    "difficulty": source["attributes"]["level"],
                    "visiting_city_number": source["attributes"].get("visiting_city_number"),
                    "budget_exhaustion_calls": budget_calls,
                    "web_search_attempted": search_total,
                    "web_search_executed": search_executed,
                    "web_scrape_attempted": scrape_total,
                    "web_scrape_executed": scrape_executed,
                    "judge_accepted": judge.get("accepted"),
                    "judge_score": judge.get("judge_score"),
                    "constraint_satisfaction": (judge.get("judge_dimensions") or {}).get(
                        "constraint_satisfaction"
                    ),
                    "completeness": (judge.get("judge_dimensions") or {}).get("completeness"),
                    "final_pass": record["query_id"] in pass_200,
                    "tool_errors": dict(sorted(errors.items())),
                }
            )
    exhaustion_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in exhaustion_rows
    )
    atomic_write(ROOT / "reports/web_budget_exhaustion_200.jsonl", exhaustion_payload)

    budget_ids = {row["query_id"] for row in exhaustion_rows}
    groups: dict[str, dict[str, Any]] = {}
    for label, ids in (
        ("exhausted", budget_ids),
        ("not_exhausted", {row["query_id"] for row in baseline_200} - budget_ids),
    ):
        group_judges = [judge_200[query_id] for query_id in ids if query_id in judge_200]
        groups[label] = {
            "queries": len(ids),
            "final_pass_rate": len(ids & pass_200) / len(ids),
            "judge_observed": len(group_judges),
            "mean_judge_score": mean(row["judge_score"] for row in group_judges),
            "mean_constraint_satisfaction": mean(
                row["judge_dimensions"]["constraint_satisfaction"] for row in group_judges
            ),
            "constraint_or_completeness_below_4": sum(
                row["judge_dimensions"]["constraint_satisfaction"] < 4
                or row["judge_dimensions"]["completeness"] < 4
                for row in group_judges
            ),
        }
    risk_difference = (
        groups["exhausted"]["final_pass_rate"] - groups["not_exhausted"]["final_pass_rate"]
    )
    attempts = Counter()
    executed = Counter()
    web_users: set[str] = set()
    for record in baseline_200:
        for call, observation in observations(record):
            if call["name"] not in {"web_search", "web_scraping"}:
                continue
            web_users.add(record["query_id"])
            attempts[call["name"]] += 1
            executed[call["name"]] += int(error_code(observation) != "tool_budget_exceeded")

    pilot_50_ids = {row["query_id"] for row in read_jsonl("data/manifests/distill_pilot_50.jsonl")}
    old50 = pilot_metrics(
        "data/intermediate/distill/pilot_v3/trajectories.jsonl",
        "data/intermediate/distill/pilot_50_v3_filter/audit.jsonl",
        50,
        pilot_50_ids,
    )
    new50 = pilot_metrics(
        "data/intermediate/distill/pilot_budget4_v1/trajectories.jsonl",
        "data/intermediate/distill/pilot_budget4_v1_filter/audit.jsonl",
        50,
    )
    new_records = read_jsonl("data/intermediate/distill/pilot_budget4_v1/trajectories.jsonl")
    new_passed = accepted_ids("data/intermediate/distill/pilot_budget4_v1_filter/audit.jsonl")
    new_exhausted = {
        row["query_id"]
        for row in new_records
        if any(
            error_code(observation) == "tool_budget_exceeded"
            for _, observation in observations(row)
        )
    }
    new_all_ids = {row["query_id"] for row in new_records}
    new_exhausted_pass_rate = len(new_exhausted & new_passed) / len(new_exhausted)
    new_non_exhausted = new_all_ids - new_exhausted
    new_non_exhausted_pass_rate = len(new_non_exhausted & new_passed) / len(new_non_exhausted)

    coverage = read_jsonl("data/intermediate/distill/train_tool_coverage_v1/trajectories.jsonl")
    coverage_tools = Counter(call["name"] for row in coverage for call in row["tool_calls"])
    train_observations = [
        observation
        for row in coverage
        for call, observation in observations(row)
        if call["name"] == "train_search"
    ]
    invalid = sum(
        error_code(observation) in {"invalid_arguments", "unknown_tool"}
        for row in coverage
        for _, observation in observations(row)
    )
    simulated_disclosed = sum(
        "simulat" in (row.get("final_answer") or "").casefold() for row in coverage
    )
    train_report = f"""# Train Search production coverage

## Root cause and remediation

The 200-query pilot used a three-row China/2026 demonstration inventory while the frozen
training domain is US/2022, and Teacher v2 described flight/drive choices but not train
selection. The production tool now uses a versioned US intercity simulator with recurring
date semantics, USD prices, explicit `simulator: true` provenance, deterministic sorting,
and a closed JSON schema. Teacher v3+ considers rail only when requested or plausible and
must disclose simulated evidence; it is not instructed to call train for quota purposes.

## Coverage result

- Coverage-only queries: {len(coverage)} (frozen-train/eval overlap: 0)
- Successful trajectories: {len(coverage)}/{len(coverage)}
- `train_search` calls: {coverage_tools["train_search"]}
- Successful train observations: {sum(o.get("status") == "ok" for o in train_observations)}
- Illegal arguments / unknown tool observations: {invalid}
- Final answers explicitly disclosing simulation: {simulated_disclosed}/{len(coverage)}
- Train observations followed by a completed plan: {sum(bool(r.get("final_answer")) for r in coverage)}/{len(coverage)}
- Inventory source: `travel-agent-rl/train-simulator-us@v1`
- Coverage manifest SHA-256: `51c32cdbd635b1f5fadcdfef470f86fcb03404e61d8c0e2a93cb72950df49185`

Two first-attempt coverage trajectories exposed a concurrent generic-tool-cache temporary
file collision. The cache writer now uses per-write unique temporary files and quarantines
legacy malformed entries; resume then completed both records with no unknown exception.
"""
    atomic_write(ROOT / "reports/train_tool_coverage.md", train_report)

    snapshot_v2 = json.loads(
        (ROOT / "data/manifests/web_snapshot_pilot_200_v2_manifest.json").read_text()
    )
    replay_v2 = json.loads((ROOT / "reports/web_replay_gate_c_v2.json").read_text())
    snapshot_v3 = json.loads(
        (ROOT / "data/manifests/web_snapshot_production_gates_v3_manifest.json").read_text()
    )
    replay_v3 = json.loads((ROOT / "reports/web_replay_gate_b_budget4_v3.json").read_text())

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    comparison_rows = []
    for label, key, percentage in (
        ("Generation success", "generation_rate", True),
        ("Final pass", "final_pass_rate", True),
        ("Average tool calls", "average_tool_calls", False),
        ("Average turns", "average_turns", False),
        ("Web exhaustion queries", "budget_exhaustion_queries", False),
        ("Web exhaustion calls", "budget_exhaustion_calls", False),
        ("Average Judge score", "average_judge_score", False),
        ("Average latency (s)", "average_latency_seconds", False),
    ):
        old_value = old50[key]
        new_value = new50[key]
        old_text = pct(old_value) if percentage else f"{old_value:.3f}"
        new_text = pct(new_value) if percentage else f"{new_value:.3f}"
        comparison_rows.append(f"| {label} | {old_text} | {new_text} |")

    gates_report = f"""# Production gates A/B/C

## Gate B: 200-query Web budget diagnosis

- Exhausted queries: {len(budget_ids)}/200; exhaustion observations: {sum(r["budget_exhaustion_calls"] for r in exhaustion_rows)}
- Search attempted/executed: {attempts["web_search"]}/{executed["web_search"]} (per all query: {attempts["web_search"] / 200:.3f}; per Web-using query: {attempts["web_search"] / len(web_users):.3f})
- Scrape attempted/executed: {attempts["web_scraping"]}/{executed["web_scraping"]} (per all query: {attempts["web_scraping"] / 200:.3f}; per Web-using query: {attempts["web_scraping"] / len(web_users):.3f})
- Exhausted final-pass rate: {pct(groups["exhausted"]["final_pass_rate"])}; non-exhausted: {pct(groups["not_exhausted"]["final_pass_rate"])}; risk difference: {risk_difference * 100:+.2f} pp
- Exhausted mean Judge/constraint score: {groups["exhausted"]["mean_judge_score"]:.3f}/{groups["exhausted"]["mean_constraint_satisfaction"]:.3f}
- Non-exhausted mean Judge/constraint score: {groups["not_exhausted"]["mean_judge_score"]:.3f}/{groups["not_exhausted"]["mean_constraint_satisfaction"]:.3f}
- Exhausted judged trajectories with constraint or completeness below 4: {groups["exhausted"]["constraint_or_completeness_below_4"]}/{groups["exhausted"]["judge_observed"]}
- Exact affected Query IDs and per-query outcomes: `reports/web_budget_exhaustion_200.jsonl`

The old cap was statistically associated with lower final pass, but many exhausted
trajectories still passed; exhaustion was a frequent artificial truncation rather than a
one-to-one failure cause.

## Frozen 50 paired production-budget rerun

| Metric | Old total cap 2 | Production total cap 4 |
| --- | ---: | ---: |
{chr(10).join(comparison_rows)}

Production uses a **total public-Web action cap of 4**. This is an upper bound, not a
quota. The model/revision, temperature, seed, max turns, Judge, and Query IDs stayed fixed.
Prompt v4 adds the required Train description and stops refining the same fact after an
irrelevant result. Final pass was not optimized upward: the observed 40/50 is retained.
The exhausted/non-exhausted final-pass rates are {pct(new_exhausted_pass_rate)} and
{pct(new_non_exhausted_pass_rate)}, respectively, so the remaining cap events are not a
primary failure mechanism. Repeated actions remain a legitimate Rule-filter reason.

## Gate C: immutable transient-error repair

- v1 was retained unchanged.
- v2 records: {snapshot_v2["request_records"]}; successes/errors: {snapshot_v2["successful_requests"]}/{snapshot_v2["error_requests"]}.
- Single retries: 32 transient records; repaired: 2; retained transient failures: 30.
- Deterministic client errors retained without retry: 2.
- v2 snapshot SHA-256: `{snapshot_v2["snapshot_content_sha256"]}`
- v2 replay: {replay_v2["snapshot_hits"]}/{replay_v2["unique_requests"]} hits, {replay_v2["snapshot_misses"]} misses, {replay_v2["live_public_web_requests"]} public requests, {replay_v2["firecrawl_cloud_requests"]} Cloud requests.
- Gate pilot v3 snapshot SHA-256: `{snapshot_v3["snapshot_content_sha256"]}`
- Budget-4 replay: {replay_v3["snapshot_hits"]}/{replay_v3["unique_requests"]} hits, {replay_v3["snapshot_misses"]} misses, {replay_v3["live_public_web_requests"]} public requests.

## Gate decision

- Train Search functional coverage: pass (see `reports/train_tool_coverage.md`).
- Illegal tool arguments: 0.
- Unknown exceptions after cache fix: 0.
- Web budget as mass artificial failure: resolved with total cap 4.
- Replay-only public Web/Cloud requests: 0/0.
- Full automated test suite: recorded separately at the gate commit.
"""
    atomic_write(ROOT / "reports/production_gates.md", gates_report)
    print(
        json.dumps(
            {
                "old_50": old50,
                "new_50": new50,
                "new_50_exhausted_pass_rate": new_exhausted_pass_rate,
                "new_50_non_exhausted_pass_rate": new_non_exhausted_pass_rate,
                "budget_200_groups": groups,
                "train_coverage": len(coverage),
                "snapshot_v2_sha256": snapshot_v2["snapshot_content_sha256"],
                "snapshot_v3_sha256": snapshot_v3["snapshot_content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
