#!/usr/bin/env python3
"""Score immutable agent-evaluation records with the fixed local Base judge."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from swift.llm import get_model_tokenizer

from travel_agent_rl.filtering.judge import JudgeScores
from travel_agent_rl.sft.data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    first, last = stripped.find("{"), stripped.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("judge output has no JSON object")
    payload = json.loads(stripped[first : last + 1])
    if not isinstance(payload, dict):
        raise TypeError("judge output is not an object")
    return payload


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for item in record.get("messages", []):
        content = str(item.get("content", ""))
        messages.append({"role": item.get("role"), "content": content[:2000]})
    return {
        "query_id": record["query_id"],
        "query": record["query"],
        "messages": messages,
        "final_answer": record.get("final_answer"),
        "tool_sequence": record.get("tool_sequence", []),
        "invalid_tool_calls": record.get("invalid_tool_calls", 0),
        "loop": record.get("loop", False),
        "replay_misses": record.get("replay_misses", 0),
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
    prompt = Path("prompts/judge/v1.txt").read_text(encoding="utf-8")
    record_paths = sorted(args.input_dir.glob("[0-9][0-9].json"))
    results = []
    for index, path in enumerate(record_paths, 1):
        record = json.loads(path.read_text(encoding="utf-8"))
        conversation = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(compact_record(record), ensure_ascii=False),
            },
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
            payload = extract_json(raw)
            scores = JudgeScores.model_validate(payload["scores"])
            rationale = str(payload.get("rationale", "fixed local judge"))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            scores = JudgeScores(**{name: 1 for name in JudgeScores.model_fields})
            rationale = "deterministic Judge failure fallback"
        results.append(
            {
                "query_id": record["query_id"],
                "scores": scores.model_dump(),
                "mean": scores.mean,
                "normalized_score": scores.mean / 5.0,
                "rationale": rationale,
                "error": error,
                "raw_output": raw,
            }
        )
        print(f"judged {index}/{len(record_paths)} {record['query_id']}", flush=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": str(args.model),
        "judge_role": "fixed local Base model; never updated by GRPO",
        "judge_prompt": "prompts/judge/v1.txt",
        "judge_prompt_sha256": sha256_file(Path("prompts/judge/v1.txt")),
        "input_dir": str(args.input_dir),
        "records": len(results),
        "mean_score_1_to_5": fmean(row["mean"] for row in results),
        "mean_normalized_score": fmean(row["normalized_score"] for row in results),
        "failures": sum(row["error"] is not None for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
