#!/usr/bin/env python3
"""Compute exact teacher-forced token accuracy without materializing full-sequence logits."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from peft import PeftModel
from swift.llm import get_model_tokenizer, get_template

from travel_agent_rl.sft.data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-chunk-tokens", type=int, default=64)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    torch.manual_seed(args.seed)
    device = f"cuda:{local_rank}"
    model, tokenizer = get_model_tokenizer(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_impl="flash_attention_2",
        max_model_len=32768,
    )
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
    template.set_mode("train")
    rows = [json.loads(line) for line in args.data.open() if line.strip()]
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    base = model.get_base_model()
    correct = 0
    count = 0
    local_rows = 0
    started = perf_counter()
    for index in range(rank, len(rows), world_size):
        row = rows[index]
        item = template.encode({"messages": row["messages"], "tools": row["tools"]})
        input_ids = torch.tensor([item["input_ids"]], device=device)
        labels = torch.tensor(item["labels"], device=device)
        with torch.inference_mode():
            hidden = base.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0, :-1]
            targets = labels[1:]
            selected = targets.ne(-100)
            hidden = hidden[selected]
            targets = targets[selected]
            for start in range(0, targets.numel(), args.vocab_chunk_tokens):
                end = start + args.vocab_chunk_tokens
                predictions = base.lm_head(hidden[start:end]).argmax(dim=-1)
                correct += int(predictions.eq(targets[start:end]).sum().item())
                count += int(targets[start:end].numel())
        local_rows += 1
        if local_rows % 25 == 0:
            print(f"rank {rank}: {local_rows} rows", flush=True)
    totals = torch.tensor([correct, count, local_rows], dtype=torch.long, device=device)
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if rank == 0:
        elapsed = perf_counter() - started
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": str(args.model),
            "adapter": str(args.adapter),
            "data": str(args.data),
            "data_sha256": sha256_file(args.data),
            "rows": int(totals[2].item()),
            "correct_tokens": int(totals[0].item()),
            "evaluated_tokens": int(totals[1].item()),
            "token_accuracy": float(totals[0].item() / totals[1].item()),
            "world_size": world_size,
            "method": "exact chunked lm_head argmax on all unmasked causal targets",
            "elapsed_seconds": elapsed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
