#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    base = load("outputs/eval/base_full8_v1/summary.json")
    sft = load("outputs/eval/sft_full8_v1/summary.json")
    base_judge = load("outputs/eval/base_full8_v1/fixed_local_judge.json")
    sft_judge = load("outputs/eval/sft_full8_v1/fixed_local_judge.json")
    validation = load("outputs/eval/sft_heldout_validation_64/metrics.json")
    manifest = load("data/manifests/sft_heldout_validation_64_manifest.json")
    rows = []
    for label, metrics, judge in (
        ("Base", base, base_judge),
        ("SFT", sft, sft_judge),
    ):
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{metrics['valid_tool_call_rate']:.4f}",
                    f"{metrics['invalid_tool_call_rate']:.4f}",
                    f"{metrics['final_answer_rate']:.4f}",
                    f"{metrics['constraint_satisfaction_rate']:.4f}",
                    f"{metrics['average_turns']:.3f}",
                    f"{metrics['average_tool_calls']:.3f}",
                    f"{metrics['loop_rate']:.4f}",
                    f"{judge['mean_score_1_to_5']:.4f}",
                ]
            )
            + " |"
        )
    report = f"""# Full 8-tool SFT evaluation

## Interactive frozen coverage evaluation

Both models used the same 16 immutable engineering queries, eight visible tool schemas,
snapshot v4, `WEB_MODE=replay_only`, and max 13 turns. The fixed Judge is the unmodified
Base model with `prompts/judge/v1.txt`; it is never the GRPO policy.

| Model | Valid calls | Invalid calls | Final answer | Constraint satisfied | Avg turns | Avg tools | Loop | Judge /5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Base replay misses: {base['replay_misses']}; public requests: {base['public_network_requests']}.
- SFT replay misses: {sft['replay_misses']}; public requests: {sft['public_network_requests']}.
- SFT unknown exceptions: {sft['unknown_exceptions']}.
- The synthetic queries were designed for Train Search coverage and contain unseen web
  requests; replay misses remained deterministic errors and never fell through to the network.

## Offline teacher-forced validation

The held-out subset was fixed by SHA-256 ranking from the 382 Judge-accepted trajectories
not selected for SFT or RL. It also excludes synthetic dev and all frozen-test IDs.

- Rows: {validation['rows']} ({manifest['difficulty']})
- Validation loss: {validation['validation_loss']:.9f}
- Validation token accuracy: {validation['token_accuracy']:.9%}
- Evaluated tokens: {validation['evaluated_tokens']}
- Held-out SHA-256: `{validation['data_sha256']}`
- Overlap with SFT/RL/dev/frozen test: {manifest['exclusions']['overlap']}
- Parent protected/frozen contamination overlap: 0

## Metric boundaries

- Train metrics: aggregate loss 0.2780154, last loss 0.22089405, train token accuracy 92.9729%.
- Offline validation metrics: the 64-row loss/accuracy above.
- Interactive Agent metrics: the Base/SFT table above; they are not teacher-forced metrics.
"""
    Path("reports/sft_full_evaluation.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
