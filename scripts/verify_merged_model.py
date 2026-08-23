#!/usr/bin/env python3
"""Load a merged SFT checkpoint and emit an immutable verification record."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from swift.llm import get_model_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = get_model_tokenizer(
        str(args.model),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_impl="sdpa",
        max_model_len=32768,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lora_parameter_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    payload = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "loaded": True,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "parameter_count": parameter_count,
        "lora_parameter_names_remaining": len(lora_parameter_names),
        "config_exists": (args.model / "config.json").is_file(),
        "tokenizer_config_exists": (args.model / "tokenizer_config.json").is_file(),
        "safetensor_shards": len(list(args.model.glob("*.safetensors"))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
