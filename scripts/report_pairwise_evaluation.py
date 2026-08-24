#!/usr/bin/env python3
"""Build position-audited Base/SFT/RL pairwise metrics from fixed-Judge scores."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from travel_agent_rl.sft.data import sha256_file

PATH_DIMENSIONS = ("factual_grounding", "tool_use_correctness", "tool_efficiency")
FINAL_DIMENSIONS = (
    "task_relevance",
    "completeness",
    "constraint_satisfaction",
    "final_answer_quality",
)
PATH_WEIGHT = 0.30
FINAL_WEIGHT = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("base", "sft", "rl"):
        parser.add_argument(f"--{label}-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["query_id"]: row for row in payload["results"]}


def weighted_score(scores: dict[str, float]) -> dict[str, float]:
    path = sum(float(scores[name]) for name in PATH_DIMENSIONS) / len(PATH_DIMENSIONS) / 5.0
    final = sum(float(scores[name]) for name in FINAL_DIMENSIONS) / len(FINAL_DIMENSIONS) / 5.0
    return {
        "path": path,
        "final_answer": final,
        "weighted": PATH_WEIGHT * path + FINAL_WEIGHT * final,
    }


def decide(left: float, right: float, tolerance: float = 1e-12) -> str:
    if abs(left - right) <= tolerance:
        return "tie"
    return "left" if left > right else "right"


def logical_winner(decision: str, left: str, right: str) -> str:
    if decision == "tie":
        return "tie"
    return left if decision == "left" else right


def evaluate_pair(left: str, right: str, candidates: dict[str, dict[str, dict[str, Any]]]) -> dict:
    query_ids = sorted(set(candidates[left]) & set(candidates[right]))
    rows = []
    wins = ties = losses = disagreements = 0
    for query_id in query_ids:
        left_score = weighted_score(candidates[left][query_id]["scores"])
        right_score = weighted_score(candidates[right][query_id]["scores"])
        ab = decide(left_score["weighted"], right_score["weighted"])
        ba = decide(right_score["weighted"], left_score["weighted"])
        ab_winner = logical_winner(ab, left, right)
        ba_winner = logical_winner(ba, right, left)
        disagreement = ab_winner != ba_winner
        disagreements += int(disagreement)
        consensus = "tie" if disagreement else ab_winner
        wins += int(consensus == left)
        losses += int(consensus == right)
        ties += int(consensus == "tie")
        rows.append(
            {
                "query_id": query_id,
                "scores": {left: left_score, right: right_score},
                "ab": {"left": left, "right": right, "winner": ab_winner},
                "ba": {"left": right, "right": left, "winner": ba_winner},
                "position_disagreement": disagreement,
                "consensus_winner": consensus,
            }
        )
    total = len(rows)
    return {
        "left": left,
        "right": right,
        "queries": total,
        "win": wins,
        "tie": ties,
        "loss": losses,
        "position_disagreement": disagreements,
        "pairwise_score": (wins + 0.5 * ties) / total if total else 0.0,
        "records": rows,
    }


def main() -> None:
    args = parse_args()
    paths = {label: getattr(args, f"{label}_judge") for label in ("base", "sft", "rl")}
    candidates = {label: load_scores(path) for label, path in paths.items()}
    pairs = [evaluate_pair(left, right, candidates) for left, right in combinations(candidates, 2)]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "position-invariant fixed-Judge dimensional scoring with explicit A/B and B/A audit",
        "path_weight": PATH_WEIGHT,
        "final_answer_weight": FINAL_WEIGHT,
        "path_dimensions": list(PATH_DIMENSIONS),
        "final_answer_dimensions": list(FINAL_DIMENSIONS),
        "judge_inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in paths.items()
        },
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pairs": [{k: v for k, v in row.items() if k != "records"} for row in pairs]}, indent=2))


if __name__ == "__main__":
    main()
