#!/usr/bin/env python3
"""Merge parity-checked Base/SFT/RL evaluation and fixed-Judge metrics."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent_rl.sft.data import sha256_file

LABELS = ("base", "sft", "rl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in LABELS:
        parser.add_argument(f"--{label}-summary", type=Path, required=True)
        parser.add_argument(f"--{label}-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    summaries = {label: load(getattr(args, f"{label}_summary")) for label in LABELS}
    judges = {label: load(getattr(args, f"{label}_judge")) for label in LABELS}
    parity = {
        "dev_sha256": len({row["dev_sha256"] for row in summaries.values()}) == 1,
        "tool_backend": len({row["tool_backend"] for row in summaries.values()}) == 1,
        "query_count": len({row["queries"] for row in summaries.values()}) == 1,
        "judge_model": len({row["judge_model"] for row in judges.values()}) == 1,
        "judge_prompt_sha256": len({row["judge_prompt_sha256"] for row in judges.values()}) == 1,
    }
    if not all(parity.values()):
        raise RuntimeError(f"evaluation parity failed: {parity}")
    metrics = {}
    for label in LABELS:
        summary = summaries[label]
        judge = judges[label]
        metrics[label] = {
            "task_success": summary["task_completion_rate"],
            "constraint_satisfaction": summary["constraint_satisfaction_rate"],
            "valid_tool_call_rate": summary["valid_tool_call_rate"],
            "invalid_tool_call_rate": summary["invalid_tool_call_rate"],
            "final_answer_rate": summary["final_answer_rate"],
            "average_turns": summary["average_turns"],
            "average_tool_calls": summary["average_tool_calls"],
            "loop_rate": summary["loop_rate"],
            "tool_efficiency": summary["tool_efficiency"],
            "judge_score_1_to_5": judge["mean_score_1_to_5"],
            "judge_score_normalized": judge["mean_normalized_score"],
            "replay_misses": summary["replay_misses"],
            "public_network_requests": summary["public_network_requests"],
            "unknown_exceptions": summary["unknown_exceptions"],
        }
    inputs = {}
    for label in LABELS:
        summary_path = getattr(args, f"{label}_summary")
        judge_path = getattr(args, f"{label}_judge")
        inputs[label] = {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "judge": str(judge_path),
            "judge_sha256": sha256_file(judge_path),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parity": parity,
        "dev_sha256": next(iter(summaries.values()))["dev_sha256"],
        "tool_backend": next(iter(summaries.values()))["tool_backend"],
        "judge_model": next(iter(judges.values()))["judge_model"],
        "judge_prompt_sha256": next(iter(judges.values()))["judge_prompt_sha256"],
        "inputs": inputs,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
