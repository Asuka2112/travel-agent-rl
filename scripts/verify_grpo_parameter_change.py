#!/usr/bin/env python3
"""Verify a deterministic policy tensor changed across a GRPO checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

TARGETS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.10.self_attn.o_proj.weight",
)


def tensor_path(model_dir: Path, name: str) -> Path:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        return model_dir / mapping[name]
    single = model_dir / "model.safetensors"
    if single.exists():
        return single
    raise FileNotFoundError(f"no safetensors index or single shard in {model_dir}")


def load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    with safe_open(tensor_path(model_dir, name), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = []
    for name in TARGETS:
        before = load_tensor(args.before, name)
        after = load_tensor(args.after, name)
        if before.shape != after.shape:
            raise AssertionError(f"shape changed for {name}: {before.shape} != {after.shape}")
        delta = after.float() - before.float()
        comparisons.append(
            {
                "name": name,
                "shape": list(before.shape),
                "before_sha256": tensor_sha256(before),
                "after_sha256": tensor_sha256(after),
                "changed_elements": int(torch.count_nonzero(delta).item()),
                "max_absolute_delta": float(delta.abs().max().item()),
                "mean_absolute_delta": float(delta.abs().mean().item()),
            }
        )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "before": str(args.before),
        "after": str(args.after),
        "parameters_changed": any(item["changed_elements"] > 0 for item in comparisons),
        "comparisons": comparisons,
    }
    if not result["parameters_changed"]:
        raise AssertionError("sampled full-parameter tensors did not change")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
