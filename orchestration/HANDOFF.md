# Final project handoff

The Travel Agentic RL reproduction is complete through Data, Distillation, Filtering,
SFT, one Formal Full-Parameter GRPO run, unified Base/SFT/RL evaluation, true
bidirectional pairwise evaluation, failure analysis, final scientific audit, and
reproduction manifest, plus frozen official TravelPlanner external validation and the
approved adapter-only corrected frozen-output evaluation.

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

The original official TravelPlanner result is retained unchanged but labeled **invalid /
adapter-contaminated external evaluation**. Its day-prose fallback placed the entire day in
`attraction` and dropped other official fields. The approved deterministic Adapter v2 was
tested only on the official example, non-validation synthetic/project fixtures, and a
fixed 20 Base + 20 SFT + 20 RL faithfulness audit before hash freeze. The final audit found
zero hallucinated fields, wrong field assignments, or entity mutations. No parser LLM was
used.

Adapter v2 was applied exactly once to the 540 unchanged saved final answers, with zero
model-inference runs. The unmodified official evaluator at commit
`e52c87f4ac348a3410c46dc3553c519db5ec5e23` produced corrected commonsense micro of
Base 30/1,440 (2.0833%), SFT 21/1,440 (1.4583%), and RL 429/1,440 (29.7917%).
Commonsense macro, hard micro/macro, and Final Pass remained zero for all models. Parsed
plans were structurally valid 8/8 Base, 6/6 SFT, and 106/106 RL, but official completeness
was 0/180 for every model; hard constraints therefore executed on 0/180 tasks. The old
adapter contaminated constraint inputs, but the residual zero Final Pass is a genuine
frozen-output delivery/completeness/generalization gap rather than a universal remaining
schema bug.

The earlier read-only failure decomposition reproduced every original official aggregate
numerator from 540 per-task verdict sets. It found a systematic adapter/schema defect:
all 166 RL final answers have an adapter finding, all 106 delivered RL plans fail
completeness/city-route/transportation, and the 411 micro passes include 318 vacuous
restaurant/attraction/accommodation passes on missing fields. No confirmed official
database mismatch was isolated. Its evidence and the corrected adapter audit trail are
both retained.

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
- `outputs/final/travelplanner_failure_decomposition.json`
- `reports/TRAVELPLANNER_FAILURE_DECOMPOSITION.md`
- `outputs/final/travelplanner_corrected_metrics.json`
- `outputs/final/travelplanner_adapter_v2/`
- `reports/TRAVELPLANNER_ADAPTER_REPAIR.md`
- `reports/TRAVELPLANNER_CORRECTED_VALIDATION.md`
- `orchestration/STATE.json`

All project trainer, rollout, Judge, monitor, evaluation, and pairwise services are
stopped. External GPU processes were never modified. Do not rerun completed stages unless
starting a separately named follow-up experiment. Resolve the final source commit with
`git rev-parse HEAD`; the adapter pre-formal freeze commit is
`0bb26016054ca8ea9f1e0701a0c4ada8e76a9e5f`, and the final corrected-validation commit is
the current Git HEAD.
