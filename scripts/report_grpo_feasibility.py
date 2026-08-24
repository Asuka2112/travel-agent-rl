#!/usr/bin/env python3
"""Aggregate immutable smoke/pilot GRPO artifacts into an auditable report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def finite_numbers(payload: Any) -> bool:
    if isinstance(payload, float):
        return math.isfinite(payload)
    if isinstance(payload, dict):
        return all(finite_numbers(value) for value in payload.values())
    if isinstance(payload, list):
        return all(finite_numbers(value) for value in payload)
    return True


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate_resources(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    gpu_peak: dict[int, int] = {}
    gpu_util_peak: dict[int, int] = {}
    host_used_peak = 0
    project_rss_peak = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpu = int(row["gpu_index"])
            gpu_peak[gpu] = max(gpu_peak.get(gpu, 0), int(row["gpu_used_mib"]))
            gpu_util_peak[gpu] = max(gpu_util_peak.get(gpu, 0), int(row["gpu_utilization_percent"]))
            host_used_peak = max(
                host_used_peak,
                int(row["host_ram_total_bytes"]) - int(row["host_ram_available_bytes"]),
            )
            project_rss_peak = max(project_rss_peak, int(row["project_rss_bytes"]))
    return {
        "gpu_peak_used_mib": {str(key): value for key, value in sorted(gpu_peak.items())},
        "gpu_peak_utilization_percent": {
            str(key): value for key, value in sorted(gpu_util_peak.items())
        },
        "host_ram_used_peak_gib": host_used_peak / 2**30,
        "project_rss_peak_gib": project_rss_peak / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--parameter-change", type=Path)
    parser.add_argument("--minimum-steps", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.output_root.glob("v*") if path.is_dir())
    logs = [item for run in run_dirs for item in read_jsonl(run / "logging.jsonl")]
    step_logs = [item for item in logs if "loss" in item and "global_step/max_steps" in item]
    completions = [item for run in run_dirs for item in read_jsonl(run / "completions.jsonl")]
    telemetry = read_jsonl(args.telemetry)
    steps = [int(item["global_step/max_steps"].split("/")[0]) for item in step_logs]
    rewards = [float(item["reward"]) for item in step_logs]
    reward_stds = [float(item["reward_std"]) for item in step_logs]
    advantages = [float(value) for batch in completions for value in batch.get("advantages", [])]
    component_names = sorted(
        {
            key.removeprefix("rewards/").removesuffix("/mean")
            for item in step_logs
            for key in item
            if key.startswith("rewards/") and key.endswith("/mean")
        }
    )
    component_stats = {
        name: {
            "mean": mean([float(item[f"rewards/{name}/mean"]) for item in step_logs]),
            "mean_logged_std": mean([float(item[f"rewards/{name}/std"]) for item in step_logs]),
            "nonconstant": any(float(item[f"rewards/{name}/std"]) > 0 for item in step_logs),
        }
        for name in component_names
    }
    tool_usage = Counter(tool for item in telemetry for tool in item.get("tool_sequence", []))
    replay_miss_details = [
        detail for item in telemetry for detail in item.get("replay_miss_details", [])
    ]
    all_errors = Counter(error for item in telemetry for error in item.get("errors", []))
    checkpoints = sorted(
        str(path) for run in run_dirs for path in run.glob("checkpoint-*") if path.is_dir()
    )
    parameter_change = (
        json.loads(args.parameter_change.read_text())
        if args.parameter_change and args.parameter_change.exists()
        else None
    )
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "output_root": str(args.output_root),
        "run_directories": [str(path) for path in run_dirs],
        "optimizer_steps": max(steps, default=0),
        "minimum_steps": args.minimum_steps,
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "resume_observed": len(run_dirs) >= 2 and max(steps, default=0) >= args.minimum_steps,
        "full_parameter": all(
            "100.0000%" in str(item.get("model_parameter_info", ""))
            for item in logs
            if "model_parameter_info" in item
        ),
        "reward_mean": mean(rewards),
        "reward_std_mean": mean(reward_stds),
        "reward_variance_observed": any(value > 0 for value in reward_stds),
        "reward_components": component_stats,
        "advantage_nonzero_ratio": (
            sum(abs(value) > 1e-8 for value in advantages) / len(advantages) if advantages else None
        ),
        "losses": [float(item["loss"]) for item in step_logs],
        "grad_norms": [float(item["grad_norm"]) for item in step_logs],
        "kl_mean": mean([float(item["kl"]) for item in step_logs]),
        "completion_length_mean": mean(
            [float(item["completions/mean_length"]) for item in step_logs]
        ),
        "logged_average_turns": mean([float(item["num_turns"]) for item in step_logs]),
        "average_turns": mean([float(item.get("num_turns", 0)) for item in telemetry]),
        "rollouts": len(telemetry),
        "average_rollout_latency_seconds": mean(
            [float(item["rollout_latency_seconds"]) for item in telemetry]
        ),
        "average_tool_calls": mean([float(item.get("tool_calls", 0)) for item in telemetry]),
        "tool_usage": dict(sorted(tool_usage.items())),
        "final_answer_rate": (
            sum(bool(item.get("final_answer_present")) for item in telemetry) / len(telemetry)
            if telemetry and all("final_answer_present" in item for item in telemetry)
            else None
        ),
        "forced_terminal_rate": (
            sum(bool(item.get("forced_terminal")) for item in telemetry) / len(telemetry)
            if telemetry
            else None
        ),
        "repeated_action_rate": (
            sum(bool(item.get("repeated_action")) for item in telemetry) / len(telemetry)
            if telemetry and all("repeated_action" in item for item in telemetry)
            else None
        ),
        "error_distribution": dict(sorted(all_errors.items())),
        "unknown_exceptions": sum(
            count for error, count in all_errors.items() if error.startswith("unhandled_tool_error")
        ),
        "invalid_tool_errors": sum(
            count
            for error, count in all_errors.items()
            if error.startswith(("invalid_tool", "unknown_tool", "invalid_arguments"))
        ),
        "replay_misses": sum(int(item.get("replay_misses", 0)) for item in telemetry),
        "replay_miss_unique": len(
            {
                json.dumps(
                    {
                        "tool": item.get("tool"),
                        "arguments": item.get("arguments", {}),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for item in replay_miss_details
            }
        ),
        "replay_miss_affected_queries": sorted(
            {
                str(item.get("query_id"))
                for item in telemetry
                if item.get("query_id") is not None and item.get("replay_misses", 0)
            }
        ),
        "replay_miss_recovered": sum(
            int(item.get("replay_miss_recovered", 0)) for item in telemetry
        ),
        "replay_miss_final_failure": sum(
            int(item.get("replay_miss_final_failure", 0)) for item in telemetry
        ),
        "public_network_requests": sum(
            int(item.get("public_network_requests", 0)) for item in telemetry
        ),
        "all_rollouts_saw_eight_tools": bool(telemetry)
        and all(int(item.get("tool_schema_count", 0)) == 8 for item in telemetry),
        "observation_backfill_consistent": bool(telemetry)
        and all(
            int(item.get("observation_count", 0)) == int(item.get("tool_calls", 0))
            for item in telemetry
        ),
        "rollout_alignment_verified": bool(telemetry)
        and all(bool(item.get("alignment_verified")) for item in telemetry),
        "tool_observations_excluded_from_policy_loss": bool(telemetry)
        and all(
            bool(item.get("tool_observations_excluded_from_policy_loss")) for item in telemetry
        ),
        "no_nan_or_inf": finite_numbers(logs)
        and finite_numbers(completions)
        and finite_numbers(telemetry),
        "parameter_change": parameter_change,
        "resources": aggregate_resources(args.resources),
    }
    metrics["replay_misses_all_logged"] = len(replay_miss_details) == metrics["replay_misses"]
    metrics["replay_misses_all_deterministic"] = all(
        detail.get("structured") is True
        and detail.get("error_code") == "snapshot_miss"
        and detail.get("retryable") is False
        and detail.get("public_network_used") is False
        for detail in replay_miss_details
    )
    metrics["replay_misses_nonfabricated"] = all(
        detail.get("results_empty") is True and detail.get("status") == "error"
        for detail in replay_miss_details
    )
    metrics["rollout_continued_after_replay_miss"] = bool(
        metrics["replay_misses"] == 0 or metrics["replay_miss_recovered"] > 0
    )
    metrics["pipeline_feasibility_pass"] = bool(
        metrics["optimizer_steps"] >= args.minimum_steps
        and metrics["reward_variance_observed"]
        and metrics["advantage_nonzero_ratio"] not in {None, 0.0}
        and metrics["all_rollouts_saw_eight_tools"]
        and metrics["observation_backfill_consistent"]
        and metrics["rollout_alignment_verified"]
        and metrics["tool_observations_excluded_from_policy_loss"]
        and metrics["public_network_requests"] == 0
        and metrics["unknown_exceptions"] == 0
        and metrics["invalid_tool_errors"] == 0
        and metrics["no_nan_or_inf"]
        and (parameter_change is None or parameter_change["parameters_changed"])
    )
    metrics["replay_policy_version"] = 2
    metrics["replay_policy_v2_gate_pass"] = bool(
        metrics["public_network_requests"] == 0
        and metrics["replay_misses_all_logged"]
        and metrics["replay_misses_all_deterministic"]
        and metrics["replay_misses_nonfabricated"]
        and metrics["rollout_continued_after_replay_miss"]
    )
    metrics["strict_replay_gate_pass"] = metrics["replay_policy_v2_gate_pass"]
    metrics["gate_pass"] = bool(
        metrics["pipeline_feasibility_pass"] and metrics["replay_policy_v2_gate_pass"]
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# {args.stage.replace('_', ' ').title()} GRPO Feasibility",
        "",
        f"Generated: {metrics['generated_at']}",
        "",
        f"- Gate pass: **{metrics['gate_pass']}**",
        f"- Full-parameter pipeline feasibility: {metrics['pipeline_feasibility_pass']}",
        f"- Replay policy v2 gate: {metrics['replay_policy_v2_gate_pass']}",
        f"- Optimizer steps: {metrics['optimizer_steps']} (minimum {args.minimum_steps})",
        f"- Full-parameter trainer: {metrics['full_parameter']}",
        f"- Checkpoints: {metrics['checkpoint_count']}",
        f"- Checkpoint resume observed: {metrics['resume_observed']}",
        f"- Rollouts: {metrics['rollouts']}",
        f"- Mean rollout latency: {metrics['average_rollout_latency_seconds']}",
        f"- Mean turns: {metrics['average_turns']}",
        f"- Mean tool calls: {metrics['average_tool_calls']}",
        f"- Reward mean / mean group std: {metrics['reward_mean']} / {metrics['reward_std_mean']}",
        f"- Advantage non-zero ratio: {metrics['advantage_nonzero_ratio']}",
        f"- Mean KL: {metrics['kl_mean']}",
        f"- Eight schemas visible in every rollout: {metrics['all_rollouts_saw_eight_tools']}",
        f"- Tool/observation counts aligned: {metrics['observation_backfill_consistent']}",
        f"- Replay misses / public requests: {metrics['replay_misses']} / {metrics['public_network_requests']}",
        f"- Replay miss unique / affected queries: {metrics['replay_miss_unique']} / {len(metrics['replay_miss_affected_queries'])}",
        f"- Miss recovered / final failure: {metrics['replay_miss_recovered']} / {metrics['replay_miss_final_failure']}",
        f"- Misses deterministic / logged / non-fabricated: {metrics['replay_misses_all_deterministic']} / {metrics['replay_misses_all_logged']} / {metrics['replay_misses_nonfabricated']}",
        f"- No NaN/Inf: {metrics['no_nan_or_inf']}",
        "",
        "## Tool usage",
        "",
        "| Tool | Calls |",
        "|---|---:|",
        *[f"| `{name}` | {count} |" for name, count in metrics["tool_usage"].items()],
        "",
        "## Reward components",
        "",
        "| Reward | Mean | Mean logged std | Nonconstant |",
        "|---|---:|---:|---:|",
        *[
            f"| `{name}` | {values['mean']} | {values['mean_logged_std']} | {values['nonconstant']} |"
            for name, values in component_stats.items()
        ],
        "",
        "The machine-readable JSON is the source of truth for all values in this report.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
