#!/usr/bin/env python3
"""Run true A/B and B/A fixed-Base pairwise judging for Base/SFT/RL."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from swift.llm import get_model_tokenizer

from travel_agent_rl.filtering.judge import JudgeScores
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
LABELS = ("base", "sft", "rl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    for label in LABELS:
        parser.add_argument(f"--{label}-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def load_records(directory: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("[0-9][0-9].json"))
    ]
    return {str(row["query_id"]): row for row in rows}


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": item.get("role"), "content": str(item.get("content", ""))[:2000]}
        for item in record.get("messages", [])
    ]
    return {
        "messages": messages,
        "final_answer": record.get("final_answer"),
        "tool_sequence": record.get("tool_sequence", []),
        "invalid_tool_calls": record.get("invalid_tool_calls", 0),
        "loop": record.get("loop", False),
        "replay_misses": record.get("replay_misses", 0),
    }


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    first, last = stripped.find("{"), stripped.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("pairwise Judge output has no JSON object")
    payload = json.loads(stripped[first : last + 1])
    if not isinstance(payload, dict):
        raise TypeError("pairwise Judge output is not an object")
    return payload


def weighted_score(scores: dict[str, float]) -> dict[str, float]:
    path = sum(float(scores[name]) for name in PATH_DIMENSIONS) / len(PATH_DIMENSIONS) / 5.0
    final = sum(float(scores[name]) for name in FINAL_DIMENSIONS) / len(FINAL_DIMENSIONS) / 5.0
    return {"path": path, "final_answer": final, "weighted": PATH_WEIGHT * path + FINAL_WEIGHT * final}


def winner(left: str, right: str, left_score: float, right_score: float) -> str:
    if abs(left_score - right_score) <= 1e-12:
        return "tie"
    return left if left_score > right_score else right


def aggregate_pair(left: str, right: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    records = []
    wins = ties = losses = disagreements = 0
    for query_id, orientations in sorted(by_query.items()):
        logical_winners = [row["winner"] for row in orientations]
        disagreement = len(set(logical_winners)) != 1
        consensus = "tie" if disagreement else logical_winners[0]
        wins += int(consensus == left)
        losses += int(consensus == right)
        ties += int(consensus == "tie")
        disagreements += int(disagreement)
        records.append(
            {
                "query_id": query_id,
                "orientations": orientations,
                "position_disagreement": disagreement,
                "consensus_winner": consensus,
            }
        )
    total = len(records)
    return {
        "left": left,
        "right": right,
        "queries": total,
        "win": wins,
        "tie": ties,
        "loss": losses,
        "position_disagreement": disagreements,
        "pairwise_score": (wins + 0.5 * ties) / total if total else 0.0,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = get_model_tokenizer(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_impl="flash_attention_2",
        max_model_len=32768,
    )
    model.eval()
    prompt_path = Path("prompts/judge/pairwise_v1.txt")
    prompt = prompt_path.read_text(encoding="utf-8")
    directories = {label: getattr(args, f"{label}_dir") for label in LABELS}
    candidates = {label: load_records(path) for label, path in directories.items()}
    query_sets = {frozenset(rows) for rows in candidates.values()}
    if len(query_sets) != 1:
        raise RuntimeError("pairwise inputs do not contain identical frozen query IDs")
    fallback = JudgeScores(**{name: 1 for name in JudgeScores.model_fields})
    pair_outputs = []
    total_call = 0
    for left, right in combinations(LABELS, 2):
        orientation_rows = []
        for query_id in sorted(candidates[left]):
            query = candidates[left][query_id]["query"]
            for candidate_a, candidate_b in ((left, right), (right, left)):
                total_call += 1
                user_payload = {
                    "query_id": query_id,
                    "query": query,
                    "candidate_a": compact_record(candidates[candidate_a][query_id]),
                    "candidate_b": compact_record(candidates[candidate_b][query_id]),
                }
                conversation = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ]
                encoded = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).to(args.device)
                error = None
                raw = ""
                try:
                    with torch.inference_mode():
                        output = model.generate(
                            encoded,
                            do_sample=False,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    raw = tokenizer.decode(output[0, encoded.shape[1] :], skip_special_tokens=True)
                    parsed = extract_json(raw)
                    score_a = JudgeScores.model_validate(parsed["candidate_a"]["scores"])
                    score_b = JudgeScores.model_validate(parsed["candidate_b"]["scores"])
                    rationale = str(parsed.get("rationale", "fixed pairwise Judge"))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    score_a = score_b = fallback
                    rationale = "deterministic pairwise Judge failure fallback"
                weighted_a = weighted_score(score_a.model_dump())
                weighted_b = weighted_score(score_b.model_dump())
                orientation_rows.append(
                    {
                        "query_id": query_id,
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "scores": {candidate_a: weighted_a, candidate_b: weighted_b},
                        "dimension_scores": {
                            candidate_a: score_a.model_dump(),
                            candidate_b: score_b.model_dump(),
                        },
                        "winner": winner(
                            candidate_a, candidate_b, weighted_a["weighted"], weighted_b["weighted"]
                        ),
                        "rationale": rationale,
                        "error": error,
                        "raw_output": raw,
                    }
                )
                print(f"judged {total_call}/96 {left}_vs_{right} {query_id} {candidate_a}/{candidate_b}", flush=True)
        pair_outputs.append(aggregate_pair(left, right, orientation_rows))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "true fixed-Base A/B and B/A pairwise judging with independent dimensional scores",
        "judge_model": str(args.model),
        "judge_prompt": str(prompt_path),
        "judge_prompt_sha256": sha256_file(prompt_path),
        "seed": args.seed,
        "path_weight": PATH_WEIGHT,
        "final_answer_weight": FINAL_WEIGHT,
        "path_dimensions": list(PATH_DIMENSIONS),
        "final_answer_dimensions": list(FINAL_DIMENSIONS),
        "input_summaries": {
            label: {"directory": str(path), "summary_sha256": sha256_file(path / "summary.json")}
            for label, path in directories.items()
        },
        "judge_calls": total_call,
        "judge_failures": sum(
            row["error"] is not None
            for pair in pair_outputs
            for record in pair["records"]
            for row in record["orientations"]
        ),
        "pairs": pair_outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"judge_calls": total_call, "judge_failures": payload["judge_failures"], "pairs": [{k: v for k, v in pair.items() if k != "records"} for pair in pair_outputs]}, indent=2))


if __name__ == "__main__":
    main()
