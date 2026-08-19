#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for row in rows
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def rank(seed: int, purpose: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{query_id}".encode()).hexdigest()


def largest_remainder(counts: Counter[str], total: int) -> dict[str, int]:
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/data/processed_split.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    accepted = read_jsonl(root / config["judge_accepted_path"])
    required = int(config["sft_size"]) + int(config["rl_size"])
    if len(accepted) < required:
        raise SystemExit(
            f"only {len(accepted)} high-quality trajectories; require {required}; "
            "thresholds were not changed"
        )

    frozen_rows = read_jsonl(root / config["frozen_queries_path"])
    frozen = {row["query_id"]: row for row in frozen_rows}
    accepted_ids = {row["query_id"] for row in accepted}
    if len(accepted_ids) != len(accepted):
        raise SystemExit("judge-accepted trajectories contain duplicate query IDs")
    if not accepted_ids <= frozen.keys():
        raise SystemExit("judge-accepted data contains queries outside frozen 1,835")

    seed = int(config["seed"])
    by_difficulty: dict[str, list[dict[str, Any]]] = {}
    for row in accepted:
        difficulty = str(row["metadata"].get("difficulty") or "unknown").title()
        by_difficulty.setdefault(difficulty, []).append(row)
    quotas = largest_remainder(
        Counter({key: len(rows) for key, rows in by_difficulty.items()}),
        int(config["rl_size"]),
    )
    rl_ids: set[str] = set()
    for difficulty, rows in by_difficulty.items():
        ordered = sorted(
            rows,
            key=lambda row: (rank(seed, "rl", row["query_id"]), row["query_id"]),
        )
        rl_ids.update(row["query_id"] for row in ordered[: quotas[difficulty]])
    remaining = [row for row in accepted if row["query_id"] not in rl_ids]
    sft = sorted(
        remaining,
        key=lambda row: (rank(seed, "sft", row["query_id"]), row["query_id"]),
    )[: int(config["sft_size"])]
    sft.sort(key=lambda row: row["query_id"])
    sft_ids = {row["query_id"] for row in sft}
    if sft_ids & rl_ids:
        raise AssertionError("SFT and RL IDs overlap")
    rl = [frozen[query_id] for query_id in sorted(rl_ids)]

    dev_source = read_jsonl(root / config["coverage_dev_path"])
    dev = sorted(dev_source, key=lambda row: row["query_id"])[:16]
    dev_ids = {row["query_id"] for row in dev}
    if dev_ids & (sft_ids | rl_ids):
        raise AssertionError("synthetic dev overlaps train splits")

    outputs = {
        "sft": (root / config["sft_path"], jsonl_bytes(sft)),
        "rl": (root / config["rl_path"], jsonl_bytes(rl)),
        "dev": (root / config["dev_path"], jsonl_bytes(dev)),
    }
    for path, payload in outputs.values():
        atomic_write(path, payload)

    frozen_manifest = json.loads(
        (root / config["frozen_manifest_path"]).read_text(encoding="utf-8")
    )
    if frozen_manifest["contamination"]["protected_overlap"] != 0:
        raise SystemExit("frozen parent manifest reports contamination")
    manifest = {
        "algorithm_version": "fixed-hash-stratified-v1",
        "seed": seed,
        "high_quality_input_count": len(accepted),
        "sft_count": len(sft),
        "rl_count": len(rl),
        "dev_count": len(dev),
        "rl_difficulty_quotas": dict(sorted(quotas.items())),
        "sft_rl_overlap": 0,
        "dev_training_overlap": 0,
        "all_training_ids_subset_of_frozen_1835": True,
        "parent_frozen_sha256": frozen_manifest["output_sha256"],
        "parent_contamination": frozen_manifest["contamination"],
        "files": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256(payload)}
            for name, (path, payload) in outputs.items()
        },
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write(root / config["manifest_path"], manifest_payload)

    report = f"""# Contamination report

- SFT/RL split algorithm: `fixed-hash-stratified-v1`, seed `{seed}`
- SFT trajectories: {len(sft)}
- RL queries: {len(rl)}
- Synthetic engineering dev queries: {len(dev)} (coverage-only, excluded from train/eval)
- SFT ↔ RL query-ID overlap: 0
- Dev ↔ SFT/RL query-ID overlap: 0
- SFT and RL are strict subsets of frozen `train_queries_1835`: yes
- Parent frozen data SHA-256: `{frozen_manifest["output_sha256"]}`
- Parent TravelPlanner/frozen-evaluation contamination result: 0 overlaps
- Parent protected comparisons: {frozen_manifest["contamination"]["protected_comparisons"]}

The split does not introduce new queries. Its contamination guarantee follows from strict
query-ID lineage to the immutable, previously exact/constraint/fuzzy-checked 1,835-query
parent manifest. No Judge threshold was changed to construct these counts.
"""
    atomic_write(root / config["contamination_report_path"], report.encode("utf-8"))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
