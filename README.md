# Travel Agentic RL

Reproducible post-training pipeline for long-horizon travel tool use with
Qwen3-4B-Instruct-2507, teacher trajectory distillation, SFT, full-parameter multi-turn
GRPO, immutable replay, and parity-locked evaluation.

## Final status

The end-to-end Data -> Distillation -> Filtering -> SFT -> Full-Parameter Agentic GRPO
-> Evaluation pipeline is complete.

| Stage | Result | Primary evidence |
| --- | --- | --- |
| Frozen data | 1,835 queries; protected/frozen contamination 0 | `reports/distillation_1835.md` |
| Distillation/filtering | 1,820 generated; 1,463 Judge-accepted | `reports/distillation_1835.json` |
| SFT | 110 steps; merged Qwen3-4B model load verified | `reports/sft_training.md` |
| Replay Policy v2 | gate passed; public network 0 | `reports/GRPO_REPLAY_V2_REGRESSION.md` |
| GRPO Pilot | 48 queries, G=4, max_turns=8, checkpoint-16 | `reports/GRPO_PILOT_REPORT.md` |
| Parameter Probe | G=6, max_turns=13, checkpoint-3 | `reports/GRPO_FORMAL_PROBE_REPORT.md` |
| Formal GRPO | 200 steps, 1,200 rollouts, checkpoint-200, gate passed | `reports/GRPO_FORMAL_REPORT.md` |
| Unified evaluation | Base/SFT/RL same 16-query dev and fixed Judge | `outputs/final/final_metrics.json` |
| Pairwise evaluation | Base vs SFT, Base vs RL, SFT vs RL; true A/B+B/A | `outputs/final/pairwise_metrics.json` |
| Final audit | tests, ruff, scientific integrity and repro manifest | `FINAL_REPORT.md` |

## Headline metrics

| Metric | Base | SFT | RL |
| --- | ---: | ---: | ---: |
| Task success | 0.8750 | 0.6875 | 1.0000 |
| Constraint satisfaction | 0.8750 | 0.6875 | 1.0000 |
| Valid tool-call rate | 0.8889 | 1.0000 | 1.0000 |
| Final-answer rate | 0.8750 | 0.6875 | 1.0000 |
| Loop rate | 0.0000 | 0.3125 | 0.0000 |
| Tool efficiency | 0.9375 | 0.6563 | 1.0000 |
| Fixed Judge / 5 | 4.7768 | 4.0714 | 4.9911 |

Formal GRPO used Qwen3-4B, full-parameter updates, G=6, max_turns=13, eight
tools, Replay Policy v2, immutable snapshot v4, and zero public-network requests. It
completed exactly 200 unique RL queries x 6 generations. Reward mean was 4.511838,
mean KL was 0.008048, and the advantage non-zero ratio was 0.970149.

The unified dev set contains only 16 synthetic engineering tool-coverage queries. These
results establish the completed reproduction on that frozen protocol; they are not a
broad travel-planning benchmark claim. See `FINAL_REPORT.md` for limitations and full
provenance.

## Reproduction entry points

Create the base environment and run deterministic checks:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt -r requirements/dev.txt
pytest -q
ruff check .
```

The exact frozen inputs, hashes, model lineage, topology logs, commands, and output
artifacts are listed in `outputs/final/repro_manifest.json`. The orchestration system of
record is `orchestration/STATE.json`; detailed stage commands are documented in
`docs/reproduction.md` and the stage reports.
