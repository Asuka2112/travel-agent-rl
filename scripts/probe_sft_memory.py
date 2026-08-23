#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import deepspeed
from peft import LoraConfig, get_peft_model
from swift.llm import get_model_tokenizer, get_template
from torch.nn.parallel import DistributedDataParallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed Qwen3 LoRA memory probe")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/intermediate/sft/ms_swift_agent_881.jsonl")
    )
    parser.add_argument("--query-id", default="agent_star:easy_ncsq03")
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--strategy", choices=("ddp", "zero2_cpu", "zero3"), default="ddp"
    )
    parser.add_argument("--liger", action="store_true")
    return parser.parse_args()


def load_row(path: Path, query_id: str) -> dict:
    for line in path.open():
        row = json.loads(line)
        if row["query_id"] == query_id:
            return row
    raise KeyError(query_id)


def nvidia_smi() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_qwen3(fused_linear_cross_entropy=True)

    model, tokenizer = get_model_tokenizer(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_impl="flash_attn",
        max_model_len=32768,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = None
    if args.strategy == "ddp":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-4)
        if world_size > 1:
            model = DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=False
            )
    elif args.strategy == "zero3":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-4)
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config={
                "bf16": {"enabled": True},
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "zero_optimization": {
                    "stage": 3,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e7,
                    "stage3_prefetch_bucket_size": 5e7,
                    "stage3_param_persistence_threshold": 1e5,
                },
            },
        )
    else:
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=trainable_parameters,
            config={
                "bf16": {"enabled": True},
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": 1e-4,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "torch_adam": True,
                    },
                },
                "zero_optimization": {
                    "stage": 2,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_scatter": True,
                    "allgather_partitions": True,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                },
                "gradient_clipping": 1.0,
            },
        )

    row = load_row(args.dataset, args.query_id)
    template = get_template(
        tokenizer.model_meta.template,
        tokenizer,
        max_length=None,
        truncation_strategy="raise",
        agent_template="hermes",
        loss_scale="default",
    )
    template.set_mode("train")
    encoded = template.encode(row)
    original_length = len(encoded["input_ids"])
    if original_length > args.target_length:
        raise ValueError(
            f"real sample length {original_length} exceeds target {args.target_length}"
        )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    extra = args.target_length - original_length
    input_ids = torch.tensor(
        [encoded["input_ids"] + [pad_id] * extra], dtype=torch.long, device=local_rank
    )
    labels = torch.tensor(
        [encoded["labels"] + [-100] * extra], dtype=torch.long, device=local_rank
    )
    # Ones intentionally make the masked suffix part of the attention workload. This is a
    # conservative memory stress test; the immutable training row itself is not changed.
    attention_mask = torch.ones_like(input_ids)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(local_rank)
    if world_size > 1:
        dist.barrier()
    started = perf_counter()
    status = "ok"
    error = None
    loss_value = None
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss.item()}")
        if args.strategy in {"zero2_cpu", "zero3"}:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            assert optimizer is not None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        torch.cuda.synchronize(local_rank)
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = f"{type(exc).__name__}: {exc}"
        torch.cuda.empty_cache()
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = perf_counter() - started
    record = {
        "rank": local_rank,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[local_rank],
        "status": status,
        "error": error,
        "target_length": args.target_length,
        "real_sample_length": original_length,
        "masked_stress_suffix_tokens": extra,
        "query_id": args.query_id,
        "loss": loss_value,
        "allocated_mib": round(torch.cuda.memory_allocated(local_rank) / 2**20, 2),
        "reserved_mib": round(torch.cuda.memory_reserved(local_rank) / 2**20, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(local_rank) / 2**20, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(local_rank) / 2**20, 2),
        "step_time_seconds": round(elapsed, 4),
    }
    gathered = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, record)
    else:
        gathered = [record]
    if local_rank == 0:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite probe: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "model": str(args.model),
            "dtype": "bfloat16",
            "lora_rank": 128,
            "lora_alpha": 256,
            "gradient_checkpointing": True,
            "attention": "flash_attention_2",
            "strategy": args.strategy,
            "liger_fused_linear_cross_entropy": args.liger,
            "optimizer_step": True,
            "stress_method": "real trajectory plus loss-masked attended suffix",
            "nvidia_smi": nvidia_smi(),
            "ranks": gathered,
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if world_size > 1:
        try:
            dist.destroy_process_group()
        except dist.DistBackendError:
            pass
    if status != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
