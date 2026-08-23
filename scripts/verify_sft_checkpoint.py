#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from swift.llm import get_model_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = get_model_tokenizer(
        str(args.base),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_impl="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" in name
    )
    payload = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "base": str(args.base),
        "adapter": str(args.adapter),
        "adapter_loaded": True,
        "adapter_parameters": adapter_parameters,
        "trainable_parameters_in_inference_mode": trainable,
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model.base_model.model).__name__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
