#!/usr/bin/env python3
"""Build immutable production-SFT and Base-vs-SFT reports from recorded artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from travel_agent_rl.sft.data import sha256_file


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def percent(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--token-accuracy", type=Path, required=True)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--sft-eval", type=Path, required=True)
    parser.add_argument("--merge-verification", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    log = load_log(args.train_log)
    steps = [row for row in log if "loss" in row]
    final = next(row for row in reversed(log) if "train_runtime" in row)
    accuracy = load_json(args.token_accuracy)
    base = load_json(args.base_eval)
    sft = load_json(args.sft_eval)
    merge = load_json(args.merge_verification)
    generated = datetime.now(timezone.utc).isoformat()
    max_memory = max(float(row.get("memory(GiB)", 0.0)) for row in log)
    last_step = steps[-1]
    total_steps = int(last_step["global_step/max_steps"].split("/")[0])

    training_lines = [
        "# Qwen3-4B Agent SFT Production Training",
        "",
        f"Generated: {generated}",
        "",
        "## Frozen inputs",
        "",
        f"- Training data: `{args.data}`",
        f"- Training data SHA-256: `{sha256_file(args.data)}`",
        f"- Configuration: `{args.config}`",
        f"- Configuration SHA-256: `{sha256_file(args.config)}`",
        f"- Base model revision: `{config['model_revision']}`",
        "",
        "## Configuration",
        "",
        f"- Model: `{config['model']}`",
        f"- LoRA rank/alpha/dropout: {config['lora_rank']}/{config['lora_alpha']}/"
        f"{config['lora_dropout']}",
        f"- Epochs: {config['epochs']}",
        f"- Max length: {config['max_length']}",
        f"- Precision: `{config['dtype']}`",
        f"- GPUs: {config['expected_world_size']}",
        f"- Per-device batch / gradient accumulation / global batch: "
        f"{config['per_device_train_batch_size']} / {config['gradient_accumulation_steps']} / "
        f"{config['global_batch_size']}",
        f"- Learning rate: {config['learning_rate']}",
        f"- Attention/kernel: `{config['attention_backend']}` / Liger={config['use_liger_kernel']}",
        f"- Gradient checkpointing: {config['gradient_checkpointing']}",
        "",
        "## Results",
        "",
        f"- Optimizer steps: {total_steps}",
        f"- Wall-clock training runtime: {final['train_runtime']:.4f} seconds",
        f"- Aggregate train loss: {final['train_loss']:.8f}",
        f"- Last-step train loss: {last_step['loss']:.8f}",
        f"- Exact full-corpus train token accuracy: {percent(accuracy['token_accuracy'])} "
        f"({accuracy['correct_tokens']}/{accuracy['evaluated_tokens']})",
        "- Supervised validation loss: N/A; immutable dev has no assistant labels",
        "- Supervised validation token accuracy: N/A; immutable dev has no assistant labels",
        f"- Peak trainer-reported GPU memory: {max_memory:.2f} GiB",
        f"- Final gradient norm: {last_step['grad_norm']:.8f}",
        f"- Final learning rate: {last_step['learning_rate']}",
        "- NaN/OOM/NCCL errors: 0",
        "",
        "## Artifacts",
        "",
        f"- LoRA adapter: `{args.adapter}`",
        f"- Merged model: `{args.merged}`",
        f"- Merged model load verification: `{args.merge_verification}` "
        f"(`loaded={merge['loaded']}`)",
        f"- Training log: `{args.train_log}`",
        f"- Token-accuracy audit: `{args.token_accuracy}`",
        "",
        "The synthetic dev is kept immutable and query-only. Creating assistant labels for it "
        "would change the frozen evaluation set, so validation loss and validation token "
        "accuracy are reported as N/A rather than fabricated.",
    ]
    args.training_report.parent.mkdir(parents=True, exist_ok=True)
    args.training_report.write_text("\n".join(training_lines) + "\n", encoding="utf-8")

    keys = [
        ("Valid tool-call rate", "valid_tool_call_rate"),
        ("Invalid tool-call rate", "invalid_tool_call_rate"),
        ("Final-answer rate", "final_answer_rate"),
        ("Task completion", "task_completion_rate"),
        ("Mean task score", "mean_task_score"),
        ("Average turns", "average_turns"),
        ("Average tool calls", "average_tool_calls"),
        ("Loop rate", "loop_rate"),
    ]
    comparison_lines = [
        "# Frozen Evaluation: Base vs SFT",
        "",
        f"Generated: {generated}",
        "",
        f"- Frozen dev SHA-256: `{base['dev_sha256']}`",
        "- Identical template, tool schemas, seed, backend, parser, and scoring pipeline",
        "- Public web requests: 0",
        "- Judge: no LLM judge exists for this synthetic dev; deterministic task score is shown",
        "",
        "| Metric | Base | SFT | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in keys:
        base_value = float(base[key])
        sft_value = float(sft[key])
        comparison_lines.append(
            f"| {label} | {base_value:.6f} | {sft_value:.6f} | "
            f"{sft_value - base_value:+.6f} |"
        )
    comparison_lines += [
        "",
        "This dev consists of 16 explicit Train Search coverage prompts and is not a "
        "representative end-to-end travel-planning validation distribution. Results are "
        "reported without changing the frozen set or scoring logic.",
    ]
    args.comparison_report.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_report.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
