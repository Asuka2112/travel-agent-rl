#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import deepspeed
import flash_attn
import swift
import torch
import transformers
import vllm
from vllm import LLM, SamplingParams


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = sorted(args.model.glob("model-*.safetensors"))
    if not weights:
        raise SystemExit(f"no model shards under {args.model}")

    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    prompt = "<|im_start|>user\nReply with exactly: OK<|im_end|>\n<|im_start|>assistant\n"
    result = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=8))[0].outputs[0]
    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": result.text.strip() == "OK",
        "generated_text": result.text,
        "finish_reason": result.finish_reason,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "max_model_len": args.max_model_len,
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "ms_swift": swift.__version__,
            "vllm": vllm.__version__,
            "deepspeed": deepspeed.__version__,
            "flash_attn": flash_attn.__version__,
        },
        "model_shards": {path.name: sha256(path) for path in weights},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
    if not evidence["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
