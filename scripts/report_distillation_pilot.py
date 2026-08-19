#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ALLOWED_TOOLS = {
    "poi_search",
    "nearby_search",
    "weather",
    "route_planning",
    "web_search",
    "web_scraping",
    "flight_search",
    "train_search",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--filter-audit", type=Path, required=True)
    parser.add_argument("--web-events", type=Path, required=True)
    parser.add_argument("--capture-run-id", action="append", default=[])
    parser.add_argument("--replay-summary", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    query_ids = {row["query_id"] for row in manifest}
    trajectories = [row for row in read_jsonl(args.trajectories) if row["query_id"] in query_ids]
    trajectory_ids = {row["query_id"] for row in trajectories}
    audit = [row for row in read_jsonl(args.filter_audit) if row["query_id"] in query_ids]
    web_events = [
        row for row in read_jsonl(args.web_events) if row.get("run_id") in set(args.capture_run_id)
    ]

    rule_reasons: Counter[str] = Counter()
    judge_reasons: Counter[str] = Counter()
    final_ids: set[str] = set()
    for row in audit:
        if row["stage"] == "rule" and not row["accepted"]:
            rule_reasons.update(row.get("drop_reason", []))
        if row["stage"] == "judge":
            if row["accepted"]:
                final_ids.add(row["query_id"])
            else:
                judge_reasons.update(row.get("drop_reason", []))

    tool_usage: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    repeated_records = 0
    invalid_tool_calls = 0
    protocol_error_records = 0
    protocol_error_count = 0
    exception_records = 0
    exceptions: Counter[str] = Counter()
    for record in trajectories:
        action_counts: Counter[str] = Counter()
        for call in record.get("tool_calls", []):
            tool_usage[call["name"]] += 1
            if call["name"] not in ALLOWED_TOOLS:
                invalid_tool_calls += 1
            action_counts[json.dumps(call, sort_keys=True)] += 1
        if any(count > 1 for count in action_counts.values()):
            repeated_records += 1
        for observation in record.get("tool_observations", []):
            error = observation.get("error")
            if isinstance(error, dict) and error.get("code"):
                tool_errors[str(error["code"])] += 1
        metadata = record["metadata"]
        protocol_errors = metadata.get("protocol_errors", [])
        if protocol_errors:
            protocol_error_records += 1
            protocol_error_count += len(protocol_errors)
        if metadata.get("exceptions"):
            exception_records += 1
            exceptions.update(str(value).split(":", 1)[0] for value in metadata["exceptions"])
    for name in ALLOWED_TOOLS:
        tool_usage.setdefault(name, 0)

    total = len(manifest)
    succeeded = len(trajectory_ids)
    web_requests = len(web_events)
    web_hits = sum(bool(row.get("snapshot_hit")) for row in web_events)
    web_live = sum(bool(row.get("live_request")) for row in web_events)
    cloud_calls = sum(row.get("backend") == "firecrawl_cloud" for row in web_events)
    replay = (
        json.loads(args.replay_summary.read_text(encoding="utf-8"))
        if args.replay_summary and args.replay_summary.is_file()
        else None
    )
    latencies = [float(row["metadata"]["latency_seconds"]) for row in trajectories]
    turns = [int(row["metadata"]["number_of_turns"]) for row in trajectories]
    model_calls = [int(row["metadata"].get("number_of_model_calls", 0)) for row in trajectories]
    generation_hours_1835 = (mean(latencies) * 1835 / args.concurrency / 3600) if latencies else 0

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "manifest_records": total,
        "generation_succeeded": succeeded,
        "generation_success_rate": percent(succeeded, total),
        "final_accepted": len(final_ids),
        "final_pass_rate": percent(len(final_ids), total),
        "average_turns": mean(turns) if turns else 0,
        "average_model_calls": mean(model_calls) if model_calls else 0,
        "average_generation_latency_seconds": mean(latencies) if latencies else 0,
        "tool_usage": dict(sorted(tool_usage.items())),
        "tool_error_distribution": dict(sorted(tool_errors.items())),
        "repeated_action_records": repeated_records,
        "repeated_action_rate": percent(repeated_records, total),
        "invalid_tool_calls": invalid_tool_calls,
        "invalid_tool_rate": percent(invalid_tool_calls, sum(tool_usage.values())),
        "protocol_error_records": protocol_error_records,
        "protocol_error_count": protocol_error_count,
        "exception_records": exception_records,
        "exception_rate": percent(exception_records, total),
        "exception_distribution": dict(sorted(exceptions.items())),
        "rule_drop_reasons": dict(sorted(rule_reasons.items())),
        "judge_drop_reasons": dict(sorted(judge_reasons.items())),
        "web_snapshot_requests": web_requests,
        "web_snapshot_hits": web_hits,
        "web_snapshot_misses": web_requests - web_hits,
        "web_cache_hit_rate": percent(web_hits, web_requests),
        "live_public_web_requests": web_live,
        "firecrawl_cloud_requests": cloud_calls,
        "firecrawl_cloud_credit_usage": 0 if cloud_calls == 0 else None,
        "replay_verification": replay,
        "estimated_generation_hours_for_1835": generation_hours_1835,
        "acceptance": {
            "generation_at_least_95_percent": percent(succeeded, total) >= 95,
            "final_pass_at_least_85_percent": percent(len(final_ids), total) >= 85,
            "unknown_exceptions_zero": exception_records == 0,
            "firecrawl_cloud_zero": cloud_calls == 0,
            "replay_live_web_zero": bool(replay) and replay["live_public_web_requests"] == 0,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Distillation pilot: {total} frozen queries",
        "",
        "## Acceptance summary",
        "",
        f"- Generation: {succeeded}/{total} ({summary['generation_success_rate']:.2f}%)",
        f"- Final pass: {len(final_ids)}/{total} ({summary['final_pass_rate']:.2f}%)",
        f"- Average turns / model calls: {summary['average_turns']:.2f} / {summary['average_model_calls']:.2f}",
        f"- Average generation latency: {summary['average_generation_latency_seconds']:.2f} seconds/query",
        f"- Repeated-action records: {repeated_records} ({summary['repeated_action_rate']:.2f}%)",
        f"- Invalid tool calls: {invalid_tool_calls}",
        f"- Audited protocol corrections: {protocol_error_count} across {protocol_error_records} records",
        f"- Exception records: {exception_records}",
        f"- Firecrawl Cloud calls/credits: {cloud_calls} / {summary['firecrawl_cloud_credit_usage']}",
        "",
        "## Tool calls",
        "",
        "| Tool | Calls |",
        "| --- | ---: |",
        *[f"| `{name}` | {count} |" for name, count in sorted(tool_usage.items())],
        "",
        "## Filtering failures",
        "",
        f"- Rule: `{json.dumps(dict(sorted(rule_reasons.items())), sort_keys=True)}`",
        f"- Judge: `{json.dumps(dict(sorted(judge_reasons.items())), sort_keys=True)}`",
        f"- Tool errors: `{json.dumps(dict(sorted(tool_errors.items())), sort_keys=True)}`",
        "",
        "## Web snapshot",
        "",
        f"- Capture events: {web_requests}; hits {web_hits}; misses {web_requests - web_hits}; hit rate {summary['web_cache_hit_rate']:.2f}%",
        f"- Live public-web requests during capture: {web_live}",
        f"- Replay verification: `{json.dumps(replay, sort_keys=True)}`",
        "",
        "## Projection",
        "",
        f"At concurrency {args.concurrency}, the measured mean generation latency projects to approximately {generation_hours_1835:.2f} hours for 1,835 queries. This excludes the separate Judge pass and operational buffer.",
        "",
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
