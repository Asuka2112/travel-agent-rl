#!/usr/bin/env python3
"""Build final training, scientific-integrity, and reproduction manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final"))
    args = parser.parse_args()
    generated_at = datetime.now(timezone.utc).isoformat()

    state = load(Path("orchestration/STATE.json"))
    split = load(Path("data/manifests/processed_split_manifest.json"))
    snapshot = load(Path("data/manifests/web_snapshot_production_1835_v4_manifest.json"))
    sft_verification = load(Path("outputs/models/qwen3_4b_sft_merged/verification.json"))
    sft_accuracy = load(Path("outputs/eval/sft_train_token_accuracy.json"))
    sft_validation = load(Path("outputs/eval/sft_heldout_validation_64/metrics.json"))
    grpo_mask = load(Path("outputs/grpo/template_mask_verification.json"))
    formal = load(Path("outputs/grpo/formal_4gpu_v1/feasibility.json"))
    unified = load(args.output_dir / "final_metrics.json")
    failure = load(args.output_dir / "failure_analysis.json")
    pairwise = load(args.output_dir / "pairwise_metrics.json")

    sft_logs = [
        json.loads(line)
        for line in Path(
            "outputs/train/sft_qwen3_4b_lora_r128_production/v0-20260824-032206/logging.jsonl"
        ).open()
        if line.strip()
    ]
    sft_runtime = next(row for row in reversed(sft_logs) if "train_runtime" in row)
    sft_last = next(row for row in reversed(sft_logs) if "loss" in row)
    training = {
        "generated_at": generated_at,
        "sft": {
            "initialization": "models/Qwen3-4B-Instruct-2507",
            "adapter_checkpoint": "outputs/train/sft_qwen3_4b_lora_r128_production/v0-20260824-032206/checkpoint-110",
            "merged_model": "outputs/models/qwen3_4b_sft_merged",
            "optimizer_steps": 110,
            "epochs": 2,
            "train_runtime_seconds": sft_runtime["train_runtime"],
            "aggregate_train_loss": sft_runtime["train_loss"],
            "last_train_loss": sft_last["loss"],
            "last_grad_norm": sft_last["grad_norm"],
            "peak_reported_gpu_memory_gib": sft_runtime["memory(GiB)"],
            "train_token_accuracy": sft_accuracy["token_accuracy"],
            "heldout_validation_loss": sft_validation["validation_loss"],
            "heldout_validation_token_accuracy": sft_validation["token_accuracy"],
            "merged_parameter_count": sft_verification["parameter_count"],
            "merged_load_verified": sft_verification["loaded"],
        },
        "replay_v2_regression": state["replay_v2_regression"],
        "pilot_48": state["pilot_48"],
        "formal_parameter_probe": state["formal_parameter_probe"],
        "formal_grpo_200": {
            **state["formal_grpo_200"],
            "train_runtime_seconds": 25796.0687,
            "train_runtime_hms": "7h 9m 58s",
            "loss_mean": sum(formal["losses"]) / len(formal["losses"]),
            "grad_norm_mean": sum(formal["grad_norms"]) / len(formal["grad_norms"]),
            "reward_components": formal["reward_components"],
            "resource_peaks": formal["resources"],
        },
    }
    write(args.output_dir / "training_metrics.json", training)

    expected_hashes = split["files"]
    dataset_hash_checks = {
        label: sha256(Path(meta["path"])) == meta["sha256"]
        for label, meta in expected_hashes.items()
    }
    eval_summaries = {
        label: load(Path(f"outputs/evaluation/unified_20260825/{label}/summary.json"))
        for label in ("base", "sft", "rl")
    }
    eval_judges = {
        label: load(
            Path(f"outputs/evaluation/unified_20260825/{label}/fixed_local_judge.json")
        )
        for label in ("base", "sft", "rl")
    }
    scientific_checks = {
        "dataset_hashes_match": all(dataset_hash_checks.values()),
        "sft_rl_overlap_zero": split["sft_rl_overlap"] == 0,
        "evaluation_training_overlap_zero": split["dev_training_overlap"] == 0,
        "protected_benchmark_overlap_zero": split["parent_contamination"]["protected_overlap"]
        == 0,
        "frozen_benchmark_overlap_zero": split["parent_contamination"]["frozen_overlap"] == 0,
        "snapshot_v4_immutable": snapshot["immutable"] is True,
        "snapshot_hash_matches_system_of_record": snapshot["snapshot_content_sha256"]
        == state["snapshot"]["content_sha256"],
        "sft_merged_load_verified": sft_verification["loaded"] is True,
        "sft_merge_contains_no_lora_parameters": sft_verification["lora_parameter_names_remaining"]
        == 0,
        "rl_response_mask_verified": grpo_mask["passed"] is True
        and grpo_mask["observation_all_ignored"] is True,
        "formal_full_parameter_verified": formal["full_parameter"] is True,
        "formal_parameter_change_verified": formal["parameter_change"]["parameters_changed"]
        is True,
        "formal_replay_v2_gate_passed": formal["replay_policy_v2_gate_pass"] is True,
        "training_public_network_zero": formal["public_network_requests"] == 0,
        "evaluation_public_network_zero": all(
            summary["public_network_requests"] == 0 for summary in eval_summaries.values()
        ),
        "evaluation_dev_hash_parity": len(
            {summary["dev_sha256"] for summary in eval_summaries.values()}
        )
        == 1,
        "evaluation_backend_parity": len(
            {summary["tool_backend"] for summary in eval_summaries.values()}
        )
        == 1,
        "evaluation_query_count_parity": len(
            {summary["queries"] for summary in eval_summaries.values()}
        )
        == 1,
        "judge_model_parity": len({judge["judge_model"] for judge in eval_judges.values()}) == 1,
        "judge_prompt_parity": len(
            {judge["judge_prompt_sha256"] for judge in eval_judges.values()}
        )
        == 1,
        "judge_has_no_failures": all(judge["failures"] == 0 for judge in eval_judges.values()),
        "pairwise_true_bidirectional_calls_complete": pairwise["judge_calls"] == 96,
        "pairwise_judge_has_no_failures": pairwise["judge_failures"] == 0,
        "checkpoint_provenance_complete": Path(state["current_checkpoint"]).is_dir()
        and Path(formal["parameter_change"]["before"]).is_dir()
        and Path(formal["parameter_change"]["after"]).is_dir(),
    }
    scientific = {
        "generated_at": generated_at,
        "overall_pass": all(scientific_checks.values()),
        "checks": scientific_checks,
        "dataset_hash_checks": dataset_hash_checks,
        "snapshot_content_sha256": snapshot["snapshot_content_sha256"],
        "reward_definition": {
            "config": "configs/grpo/rewards.yaml",
            "config_sha256": sha256(Path("configs/grpo/rewards.yaml")),
            "formal_components": formal["reward_components"],
        },
        "checkpoint_provenance": {
            "base": "models/Qwen3-4B-Instruct-2507",
            "sft": "outputs/models/qwen3_4b_sft_merged",
            "rl": state["current_checkpoint"],
            "parameter_change": formal["parameter_change"],
        },
        "evaluation_parity": unified["parity"],
        "failure_summary": {
            label: {key: value for key, value in row.items() if key != "records"}
            for label, row in failure["models"].items()
        },
    }
    if not scientific["overall_pass"]:
        failed = [name for name, passed in scientific_checks.items() if not passed]
        raise RuntimeError(f"scientific integrity audit failed: {failed}")
    write(args.output_dir / "scientific_audit.json", scientific)

    artifact_paths = [
        "data/processed/sft.jsonl",
        "data/processed/rl_queries.jsonl",
        "data/processed/dev.jsonl",
        "data/manifests/processed_split_manifest.json",
        "data/manifests/web_snapshot_production_1835_v4_manifest.json",
        "configs/sft/production.yaml",
        "configs/grpo/formal_4gpu.yaml",
        "configs/grpo/rewards.yaml",
        "prompts/judge/v1.txt",
        "prompts/judge/pairwise_v1.txt",
        "outputs/grpo/formal_4gpu_v1/feasibility.json",
        "outputs/grpo/formal_4gpu_v1/parameter_change.json",
        "outputs/final/final_metrics.json",
        "outputs/final/training_metrics.json",
        "outputs/final/pairwise_metrics.json",
        "outputs/final/failure_analysis.json",
        "outputs/final/scientific_audit.json",
    ]
    reproduction = {
        "generated_at": generated_at,
        "repository": str(Path.cwd()),
        "source_commit_before_finalization": git("rev-parse", "HEAD"),
        "model_lineage": scientific["checkpoint_provenance"],
        "frozen_protocol": {
            "model": "Qwen3-4B-Instruct-2507",
            "formal_full_parameter": True,
            "num_generations": 6,
            "max_turns": 13,
            "replay_policy_version": 2,
            "snapshot_version": 4,
            "snapshot_content_sha256": snapshot["snapshot_content_sha256"],
            "web_mode": "replay_only",
            "public_network_requests": 0,
            "evaluation_seed": 20260819,
        },
        "hardware": {
            "profile": "docs/machine_profile.md",
            "formal_topology": "outputs/grpo/resource_snapshots/formal_grpo_retry2_topology.json",
            "evaluation_topology": "outputs/evaluation/unified_20260825/resource_topology.json",
        },
        "artifacts": {
            path: {"sha256": sha256(Path(path)), "bytes": Path(path).stat().st_size}
            for path in artifact_paths
        },
        "commands": {
            "formal_training": "bash scripts/run_grpo_training.sh --config configs/grpo/formal_4gpu.yaml",
            "unified_evaluation": "scripts/evaluate_agent_model.py with identical full_replay, seed=20260819, max_turns=13, max_new_tokens=768 for Base/SFT/RL",
            "fixed_judge": "scripts/evaluate_fixed_local_judge.py with models/Qwen3-4B-Instruct-2507 and prompts/judge/v1.txt",
            "pairwise": "scripts/evaluate_pairwise_local_judge.py with true A/B and B/A orientations, Path=30%, Final=70%",
            "final_validation": ".venv-train/bin/pytest -q && .venv/bin/ruff check .",
        },
    }
    write(args.output_dir / "repro_manifest.json", reproduction)
    print(
        json.dumps(
            {
                "training_metrics": str(args.output_dir / "training_metrics.json"),
                "scientific_audit_pass": scientific["overall_pass"],
                "repro_manifest": str(args.output_dir / "repro_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
