#!/usr/bin/env python3
"""Evaluate SFT agent behavior at three points on immutable teacher trajectories."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from peft import PeftModel
from swift.llm import get_model_tokenizer, get_template

from travel_agent_rl.evaluation.agent_eval import parse_native_agent_output, validate_tool_call
from travel_agent_rl.sft.data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="flash_attention_2")
    return parser.parse_args()


def generate(model, tokenizer, template, messages, tools, max_new_tokens: int) -> str:
    item = template.encode({"messages": messages, "tools": tools})
    input_ids = torch.tensor([item["input_ids"]], device=model.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def classify(output: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        kind, parsed = parse_native_agent_output(output)
        if kind == "answer":
            return {"kind": kind, "valid": True, "name": None, "error": None}
        valid, reason = validate_tool_call(parsed, tools)
        return {
            "kind": kind,
            "valid": valid,
            "name": parsed["name"],
            "arguments": parsed["arguments"],
            "error": reason,
        }
    except Exception as exc:
        return {
            "kind": "invalid",
            "valid": False,
            "name": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def reference(message: dict[str, Any]) -> dict[str, Any]:
    if message["role"] == "assistant":
        return {"kind": "answer", "name": None}
    payload = json.loads(message["content"])
    return {"kind": "tool_call", "name": payload["name"]}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluation: {args.output_dir}")
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
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
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
    rows = [json.loads(line) for line in args.data.open() if line.strip()][: args.max_cases]
    args.output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    started = perf_counter()
    for index, row in enumerate(rows, 1):
        messages = row["messages"]
        tools = json.loads(row["tools"])
        checkpoints = [
            ("initial", messages[:1], messages[1], 256),
            ("post_first_observation", messages[:3], messages[3], 256),
            ("pre_final", messages[:-1], messages[-1], args.max_new_tokens),
        ]
        case = {"query_id": row["query_id"], "difficulty": row["difficulty"], "checks": []}
        for label, context, expected_message, max_tokens in checkpoints:
            output = generate(model, tokenizer, template, context, row["tools"], max_tokens)
            parsed = classify(output, tools)
            expected = reference(expected_message)
            parsed["label"] = label
            parsed["expected_kind"] = expected["kind"]
            parsed["expected_name"] = expected["name"]
            parsed["protocol_pass"] = bool(parsed["valid"] and parsed["kind"] == expected["kind"])
            parsed["reference_match"] = bool(
                parsed["protocol_pass"]
                and (expected["name"] is None or parsed["name"] == expected["name"])
            )
            parsed["output"] = output
            case["checks"].append(parsed)
        records.append(case)
        print(f"evaluated {index}/{len(rows)} {row['query_id']}", flush=True)
    all_checks = [check for record in records for check in record["checks"]]
    by_label = {}
    for label in ("initial", "post_first_observation", "pre_final"):
        selected = [check for check in all_checks if check["label"] == label]
        by_label[label] = {
            "count": len(selected),
            "protocol_pass_rate": sum(check["protocol_pass"] for check in selected) / len(selected),
            "reference_match_rate": sum(check["reference_match"] for check in selected) / len(selected),
            "invalid_count": sum(not check["valid"] for check in selected),
        }
    summary = {
        "queries": len(records),
        "checks": len(all_checks),
        "protocol_pass_rate": sum(check["protocol_pass"] for check in all_checks) / len(all_checks),
        "reference_match_rate": sum(check["reference_match"] for check in all_checks) / len(all_checks),
        "invalid_count": sum(not check["valid"] for check in all_checks),
        "by_checkpoint": by_label,
        "latency_seconds": perf_counter() - started,
        "data_sha256": sha256_file(args.data),
        "adapter": str(args.adapter),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SFT Trajectory-context Smoke Evaluation",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "This diagnostic uses immutable teacher observations as context; it does not replace the "
        "free-running frozen Agent evaluation. It isolates three behaviors: initial legal tool "
        "selection, legal continuation after an observation, and final-answer generation after "
        "the complete evidence trajectory.",
        "",
        f"- Data: `{args.data}`",
        f"- Data SHA-256: `{summary['data_sha256']}`",
        f"- Adapter: `{args.adapter}`",
        f"- Queries: {summary['queries']}",
        f"- Total checks: {summary['checks']}",
        f"- Protocol pass rate: {summary['protocol_pass_rate']:.2%}",
        f"- Reference action match rate: {summary['reference_match_rate']:.2%}",
        f"- Invalid outputs: {summary['invalid_count']}",
        "",
        "| Checkpoint | Count | Protocol pass | Reference match | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in by_label.items():
        lines.append(
            f"| {label} | {values['count']} | {values['protocol_pass_rate']:.2%} | "
            f"{values['reference_match_rate']:.2%} | {values['invalid_count']} |"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
