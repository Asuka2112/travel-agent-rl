# Final project handoff

The Travel Agentic RL reproduction is complete through Data, Distillation, Filtering,
SFT, one Formal Full-Parameter GRPO run, unified Base/SFT/RL evaluation, true
bidirectional pairwise evaluation, failure analysis, final scientific audit, and
reproduction manifest, plus frozen official TravelPlanner external validation.

Formal GRPO completed 200/200 optimizer steps at
`outputs/grpo/formal_4gpu_v1/v1-20260824-191232/checkpoint-200` with exactly 1,200
rollouts (200 unique queries x G=6), max_turns=13, full-parameter updates, immutable
snapshot v4, Replay Policy v2, and zero public-network requests. The gate and sampled
parameter-change proof pass; no OOM/NCCL/NaN/Inf occurred.

Unified evaluation used the identical 16-query dev set and fixed Base Judge for all
models. Task success was Base 0.875, SFT 0.6875, RL 1.0. True A/B+B/A pairwise results
were Base vs SFT 10/4/2, Base vs RL 10/4/2, and SFT vs RL 0/10/6 (left win/tie/loss).
The disagreement between independent Judge scores and direct pairwise judgments is
retained and discussed in `FINAL_REPORT.md`.

Official TravelPlanner validation used the frozen 180-query split, identical generation
protocol for Base/SFT/RL, deterministic final-answer-only adapter, and the unmodified
official evaluator at commit `e52c87f4ac348a3410c46dc3553c519db5ec5e23`.
Commonsense micro was Base 2.7778%, SFT 1.8750%, RL 28.5417%; commonsense macro,
hard micro/macro, and Final Pass were 0 for all three models. Project Task Success,
reported separately from official Final Pass, was Base 6/180, SFT 5/180, RL 78/180.
The contamination re-audit scanned 2,932 training-derived rows with all overlap counts
zero. No retraining or post-result tuning occurred, and all frozen model files—including
checkpoint-200—passed post-evaluation hash verification.

Authoritative deliverables:

- `FINAL_REPORT.md`
- `outputs/final/final_metrics.json`
- `outputs/final/training_metrics.json`
- `outputs/final/pairwise_metrics.json`
- `outputs/final/failure_analysis.json`
- `outputs/final/scientific_audit.json`
- `outputs/final/repro_manifest.json`
- `outputs/final/external_validation_metrics.json`
- `outputs/final/external_validation_scientific_audit.json`
- `outputs/final/external_validation_repro_manifest.json`
- `reports/TRAVELPLANNER_EXTERNAL_VALIDATION.md`
- `orchestration/STATE.json`

All project trainer, rollout, Judge, monitor, evaluation, and pairwise services are
stopped. External GPU processes were never modified. Do not rerun completed stages unless
starting a separately named follow-up experiment. Resolve the final source commit with
`git rev-parse HEAD`; the final commit message is
`feat: add frozen TravelPlanner external validation`.
