#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from swift.llm import get_model_tokenizer, get_template

from evaluate_agent_model import _generate, _observation, build_full_replay_environment
from travel_agent_rl.evaluation.agent_eval import parse_native_agent_output, validate_tool_call
from travel_agent_rl.evaluation.travelplanner import (
    adapt_final_answer,
    is_structurally_complete,
    wilson_interval,
)
from travel_agent_rl.sft.data import build_agent_tool_schemas, sha256_file
from travel_agent_rl.tools.simulator import TrainSearchSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen TravelPlanner validation generation")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=Path("data/raw/travelplanner/validation.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", choices=("base", "sft", "rl"), required=True)
    parser.add_argument("--max-turns", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def load_validation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        row["idx"] = index
        for key in ("days", "visiting_city_number", "people_number", "budget"):
            row[key] = int(row[key])
        for key in ("date", "local_constraint", "reference_information"):
            if isinstance(row.get(key), str):
                try:
                    row[key] = ast.literal_eval(row[key])
                except (SyntaxError, ValueError):
                    pass
    return rows


def evaluate_one(row, model, tokenizer, template, tools, environment, train_tool, args):
    started = perf_counter()
    # This is intentionally the same model-native prompt used for internal frozen eval:
    # no validation metadata, reference information, evaluator rules, or system cue is added.
    messages = [{"role": "user", "content": row["query"]}]
    tool_sequence: list[str] = []
    raw_outputs: list[str] = []
    errors: list[str] = []
    seen_signatures: set[str] = set()
    repeated = False
    invalid = 0
    replay_misses = 0
    final_answer = None
    for _turn in range(1, args.max_turns + 1):
        output = _generate(model, tokenizer, template, messages, tools, args.max_new_tokens)
        raw_outputs.append(output)
        try:
            kind, parsed = parse_native_agent_output(output)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid += 1
            errors.append(f"{type(exc).__name__}: {exc}")
            break
        if kind == "answer":
            final_answer = str(parsed)
            messages.append({"role": "assistant", "content": final_answer})
            break
        payload = parsed
        assert isinstance(payload, dict)
        valid, reason = validate_tool_call(payload, tools)
        if not valid:
            invalid += 1
            errors.append(str(reason))
            break
        signature = json.dumps(payload, sort_keys=True)
        repeated = repeated or signature in seen_signatures
        seen_signatures.add(signature)
        tool_sequence.append(payload["name"])
        messages.append({"role": "tool_call", "content": json.dumps(payload, ensure_ascii=False)})
        observation = _observation(payload, train_tool, environment)
        if (observation.get("error") or {}).get("code") == "snapshot_miss":
            replay_misses += 1
        messages.append(
            {
                "role": "tool_response",
                "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
            }
        )
    official_plan = adapt_final_answer(final_answer)
    task_complete = is_structurally_complete(official_plan, row["days"])
    return {
        "idx": row["idx"],
        "query": row["query"],
        "expected_days": row["days"],
        "tool_sequence": tool_sequence,
        "valid_tool_calls": len(tool_sequence),
        "invalid_tool_calls": invalid,
        "final_answer": final_answer,
        "official_plan": official_plan,
        "final_answer_present": bool(final_answer),
        "task_complete": task_complete,
        "turns": len(raw_outputs),
        "tool_calls": len(tool_sequence),
        "loop": repeated,
        "raw_outputs": raw_outputs,
        "messages": messages,
        "errors": errors,
        "replay_misses": replay_misses,
        "latency_seconds": perf_counter() - started,
        "unknown_exception": False,
    }


def failure_record(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "idx": row["idx"],
        "query": row["query"],
        "expected_days": row["days"],
        "tool_sequence": [],
        "valid_tool_calls": 0,
        "invalid_tool_calls": 0,
        "final_answer": None,
        "official_plan": None,
        "final_answer_present": False,
        "task_complete": False,
        "turns": 0,
        "tool_calls": 0,
        "loop": False,
        "raw_outputs": [],
        "messages": [],
        "errors": [f"{type(exc).__name__}: {exc}"],
        "replay_misses": 0,
        "latency_seconds": 0.0,
        "unknown_exception": True,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    valid = sum(row["valid_tool_calls"] for row in records)
    invalid = sum(row["invalid_tool_calls"] for row in records)
    final_answers = sum(row["final_answer_present"] for row in records)
    task_successes = sum(row["task_complete"] for row in records)
    return {
        "n": n,
        "task_success": {
            "numerator": task_successes,
            "denominator": n,
            "rate": task_successes / n,
            "ci95_wilson": wilson_interval(task_successes, n),
            "definition": "final answer adapted to exactly the requested number of official-schema days",
        },
        "valid_tool_rate": valid / (valid + invalid) if valid + invalid else 0.0,
        "valid_tool_calls": valid,
        "invalid_tool_calls": invalid,
        "final_answer_rate": final_answers / n,
        "final_answer_numerator": final_answers,
        "average_turns": statistics.fmean(row["turns"] for row in records),
        "average_tool_calls": statistics.fmean(row["tool_calls"] for row in records),
        "loop_rate": sum(row["loop"] for row in records) / n,
        "tool_usage": dict(Counter(tool for row in records for tool in row["tool_sequence"])),
        "replay_misses": sum(row["replay_misses"] for row in records),
        "unknown_exceptions": sum(row["unknown_exception"] for row in records),
        "public_network_requests": 0,
        "average_latency_seconds": statistics.fmean(row["latency_seconds"] for row in records),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluation: {args.output_dir}")
    if args.max_turns != 13 or args.max_new_tokens != 768 or args.seed != 20260819:
        raise ValueError("external validation protocol is frozen at turns=13, tokens=768, seed=20260819")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model, tokenizer = get_model_tokenizer(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_impl=args.attn_impl,
        max_model_len=32768,
    )
    model.eval()
    template = get_template(
        tokenizer.model_meta.template,
        tokenizer,
        max_length=32768,
        truncation_strategy="raise",
        agent_template="hermes",
        loss_scale="default",
    )
    template.set_mode("vllm")
    tools = build_agent_tool_schemas()
    environment = build_full_replay_environment(Path.cwd())
    train_tool = TrainSearchSimulator(Path("data/fixtures/train_inventory.json"))
    rows = load_validation(args.validation)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]

    args.output_dir.mkdir(parents=True)
    records = []
    submission_path = args.output_dir / "official_submission.jsonl"
    with submission_path.open("w", encoding="utf-8") as submission:
        for index, row in enumerate(rows, 1):
            try:
                record = evaluate_one(
                    row, model, tokenizer, template, tools, environment, train_tool, args
                )
            except Exception as exc:
                record = failure_record(row, exc)
            records.append(record)
            (args.output_dir / f"{index:03d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            submission.write(
                json.dumps(
                    {"idx": row["idx"], "query": row["query"], "plan": record["official_plan"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"evaluated {index}/{len(rows)} validation:{row['idx']}", flush=True)
    summary = summarize(records)
    summary.update(
        {
            "label": args.label,
            "model": str(args.model),
            "validation_sha256": sha256_file(args.validation),
            "submission_sha256": sha256_file(submission_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "prompt": "raw query only; no system or benchmark-specific cue",
                "template": "qwen3_nothinking + native hermes agent format",
                "decoding": "greedy (do_sample=false)",
                "max_turns": args.max_turns,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "tool_backend": "full_replay_snapshot_v4",
                "adapter": "deterministic final-answer-only interface adapter",
            },
        }
    )
    (args.output_dir / "project_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
