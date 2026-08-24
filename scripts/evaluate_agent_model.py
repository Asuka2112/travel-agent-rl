#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from peft import PeftModel
from swift.llm import get_model_tokenizer, get_template

from travel_agent_rl.evaluation.agent_eval import (
    completion_score,
    parse_native_agent_output,
    validate_tool_call,
)
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.sft.data import build_agent_tool_schemas, sha256_file
from travel_agent_rl.tools.base import BaseTool, ToolExecutionError, ToolRequest
from travel_agent_rl.tools.registry import (
    ToolRegistry,
    build_distillation_registry,
    build_mock_registry,
)
from travel_agent_rl.tools.simulator import TrainSearchSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen interactive Agent evaluation")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dev", type=Path, default=Path("data/processed/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--tool-backend",
        choices=("frozen_train", "mock_all", "full_replay"),
        default="frozen_train",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _generate(model, tokenizer, template, messages, tools, max_new_tokens: int) -> str:
    item = template.encode({"messages": messages, "tools": json.dumps(tools, ensure_ascii=False)})
    input_ids = torch.tensor([item["input_ids"]], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def _observation(
    payload: dict[str, Any],
    train_tool: TrainSearchSimulator,
    mock_environment: TravelEnvironment | None,
) -> dict[str, Any]:
    if mock_environment is not None:
        return mock_environment.execute(
            ToolRequest(tool_name=payload["name"], arguments=payload["arguments"])
        ).model_dump(mode="json")
    if payload["name"] == "train_search":
        response = train_tool.invoke(
            ToolRequest(tool_name=payload["name"], arguments=payload["arguments"])
        )
        return response.model_dump(mode="json")
    return {
        "status": "error",
        "tool_name": payload["name"],
        "error": {
            "code": "frozen_dev_tool_unavailable",
            "message": (
                "This frozen synthetic dev set contains only train coverage fixtures; "
                "choose train_search when it satisfies the explicit rail request."
            ),
        },
    }


class OfflineWeatherTool(BaseTool):
    """Expose the frozen weather protocol without making a public-network request."""

    name = "weather"
    description = "Offline evaluation weather boundary; live public requests are disabled."
    argument_schema = next(
        tool["function"]["parameters"]
        for tool in build_agent_tool_schemas()
        if tool["function"]["name"] == "weather"
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ToolExecutionError(
            "live weather is disabled in frozen replay evaluation",
            code="offline_weather_unavailable",
            provider="frozen_evaluation",
        )


def build_full_replay_environment(root: Path) -> TravelEnvironment:
    # The replay-only registry constructor still instantiates AmapWeather before
    # this evaluator replaces it with OfflineWeatherTool. Give that unreachable
    # live adapter a non-secret sentinel so evaluation never depends on an
    # online credential. No request can use it because replacement happens
    # before the TravelEnvironment is constructed.
    live_registry = build_distillation_registry(
        root,
        amap_api_key="replay-only-disabled",
        web_mode="replay_only",
        web_backend="selfhosted",
        snapshot_path=root / "data/intermediate/web_snapshot",
        web_run_id="night_full_eval_v1",
    )
    tools = [
        OfflineWeatherTool() if name == "weather" else live_registry.get(name)
        for name in live_registry.names
    ]
    if any(tool is None for tool in tools):
        raise AssertionError("full replay registry is incomplete")
    return TravelEnvironment(
        ToolRegistry(tools),
        max_attempts=1,
        cache_dir=root / "outputs/night_run/eval_tool_cache",
        cache_namespace="full_replay_snapshot_v4",
    )


def evaluate_one(
    row, model, tokenizer, template, tools, train_tool, mock_environment, args
) -> dict[str, Any]:
    started = perf_counter()
    messages = [{"role": "user", "content": row["query"]}]
    tool_sequence: list[str] = []
    raw_outputs: list[str] = []
    errors: list[str] = []
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
        repeated = repeated or any(
            signature == json.dumps(previous, sort_keys=True)
            for previous in [json.loads(m["content"]) for m in messages if m["role"] == "tool_call"]
        )
        tool_sequence.append(payload["name"])
        messages.append(
            {"role": "tool_call", "content": json.dumps(payload, ensure_ascii=False)}
        )
        observation = _observation(payload, train_tool, mock_environment)
        if (observation.get("error") or {}).get("code") == "snapshot_miss":
            replay_misses += 1
        messages.append(
            {
                "role": "tool_response",
                "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
            }
        )
    result = {
        "query_id": row["query_id"],
        "query": row["query"],
        "expected_tool": row["expected_tool"],
        "tool_sequence": tool_sequence,
        "valid_tool_calls": len(tool_sequence),
        "invalid_tool_calls": invalid,
        "final_answer": final_answer,
        "final_answer_present": bool(final_answer),
        "turns": len(raw_outputs),
        "tool_calls": len(tool_sequence),
        "loop": repeated,
        "raw_outputs": raw_outputs,
        "messages": messages,
        "errors": errors,
        "latency_seconds": perf_counter() - started,
        "unknown_exception": False,
    }
    result["constraint_satisfied"] = bool(
        result["final_answer_present"] and row["expected_tool"] in result["tool_sequence"]
    )
    result["replay_misses"] = replay_misses
    result["task_score"] = completion_score(result)
    result["task_complete"] = result["task_score"] == 1.0
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_attempts = sum(r["valid_tool_calls"] + r["invalid_tool_calls"] for r in records)
    valid_calls = sum(r["valid_tool_calls"] for r in records)
    invalid_calls = sum(r["invalid_tool_calls"] for r in records)
    tool_efficiency = statistics.fmean(
        0.0
        if not r["tool_calls"]
        else (1.0 if r["final_answer_present"] else 0.5) * (0.0 if r["loop"] else 1.0)
        for r in records
    )
    return {
        "queries": len(records),
        "normal_answer_completion_rate": sum(bool(r["final_answer"]) for r in records)
        / len(records),
        "valid_tool_call_rate": valid_calls / total_attempts if total_attempts else 0.0,
        "invalid_tool_call_rate": invalid_calls / total_attempts if total_attempts else 0.0,
        "final_answer_rate": sum(r["final_answer_present"] for r in records) / len(records),
        "task_completion_rate": sum(r["task_complete"] for r in records) / len(records),
        "constraint_satisfaction_rate": sum(r["constraint_satisfied"] for r in records)
        / len(records),
        "mean_task_score": statistics.fmean(r["task_score"] for r in records),
        "average_turns": statistics.fmean(r["turns"] for r in records),
        "average_tool_calls": statistics.fmean(r["tool_calls"] for r in records),
        "loop_rate": sum(r["loop"] for r in records) / len(records),
        "tool_efficiency": tool_efficiency,
        "tool_usage": dict(Counter(tool for r in records for tool in r["tool_sequence"])),
        "unknown_exceptions": sum(r["unknown_exception"] for r in records),
        "replay_misses": sum(r["replay_misses"] for r in records),
        "public_network_requests": 0,
        "average_latency_seconds": statistics.fmean(r["latency_seconds"] for r in records),
    }


def write_report(args, summary: dict[str, Any]) -> None:
    lines = [
        f"# {args.label} Frozen Agent Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Model: `{args.model}`",
        f"- Adapter: `{args.adapter}`" if args.adapter else "- Adapter: none",
        f"- Frozen dev: `{args.dev}`",
        f"- Dev SHA-256: `{sha256_file(args.dev)}`",
        "- Template: `qwen3_nothinking` + native `hermes` agent format",
        f"- Seed: {args.seed}",
        f"- Tool backend: `{args.tool_backend}`",
        "- Public web requests: 0 (the frozen dev contains only local train fixtures)",
        "- Supervised validation loss: N/A (the immutable dev rows contain queries and expected "
        "tools, but no assistant labels)",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"| {key} | `{json.dumps(value, sort_keys=True)}` |")
        elif isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    if args.adapter:
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
    # ms-swift 3.8.x uses the vLLM mode to encode generation prompts without labels.
    # The resulting input_ids are backend-neutral and are consumed by Transformers here.
    template.set_mode("vllm")
    tools = build_agent_tool_schemas()
    train_tool = TrainSearchSimulator(Path("data/fixtures/train_inventory.json"))
    if args.tool_backend == "mock_all":
        mock_environment = TravelEnvironment(build_mock_registry())
    elif args.tool_backend == "full_replay":
        mock_environment = build_full_replay_environment(Path.cwd())
    else:
        mock_environment = None
    rows = _load_rows(args.dev)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    args.output_dir.mkdir(parents=True)
    records = []
    for index, row in enumerate(rows, 1):
        try:
            record = evaluate_one(
                row, model, tokenizer, template, tools, train_tool, mock_environment, args
            )
        except Exception as exc:
            record = {
                "query_id": row["query_id"],
                "query": row["query"],
                "expected_tool": row["expected_tool"],
                "tool_sequence": [],
                "valid_tool_calls": 0,
                "invalid_tool_calls": 0,
                "final_answer": None,
                "final_answer_present": False,
                "turns": 0,
                "tool_calls": 0,
                "loop": False,
                "raw_outputs": [],
                "messages": [],
                "errors": [f"{type(exc).__name__}: {exc}"],
                "latency_seconds": 0.0,
                "unknown_exception": True,
                "constraint_satisfied": False,
                "replay_misses": 0,
                "task_score": 0.0,
                "task_complete": False,
            }
        records.append(record)
        (args.output_dir / f"{index:02d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"evaluated {index}/{len(rows)} {row['query_id']}", flush=True)
    summary = summarize(records)
    summary.update(
        {
            "model": str(args.model),
            "adapter": str(args.adapter) if args.adapter else None,
            "label": args.label,
            "dev_sha256": sha256_file(args.dev),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tool_backend": args.tool_backend,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
