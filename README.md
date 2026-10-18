# Travel Agentic RL

An auditable post-training pipeline for long-horizon travel tool use with
Qwen3-4B-Instruct-2507: trajectory construction and filtering, SFT cold start,
multi-turn GRPO, reward design, frozen tool replay, and paired model evaluation.

## Latest progress — 2026-10-19

The frozen 80-query internal evaluation and the user-requested supplemental
judging are complete. **GRPO-v1 remains the evidence-supported internal delivery
model.** No new training, checkpoint changes, or paid API calls were needed to
complete judging; experiments and automatic continuation are stopped.

- **Four-model comparison:** Base, SFT-v2, GRPO-v1, and GRPO-v2 on the same frozen
  queries and eight-tool replay protocol.
- **Two Judges:** GPT and Gemini 3.1 Pro; three model pairs, with A/B and B/A swaps.
- **Complete score-slot coverage:** 480/480 per Judge, 960/960 combined, including
  36 explicitly labeled manual batch supplements. This is **not** a claim of
  960 independent successful calls or fully independent blind judgments.
- **Fixed weighting:** 30% tool-path quality and 70% final-answer quality; the
  frozen test set, candidate outputs, and seven scoring dimensions are unchanged.

Read the [final metrics report](reports/final_metrics_report_063.md),
[machine-readable metrics](reports/final80_metrics_063.json), and
[publication/provenance notes](docs/FINAL_EVALUATION_STATUS.md).

## Frozen internal comparison

| Model | Structural success / 80 | Final answers / 80 | Valid tool-call rate | Loops / 80 |
| --- | ---: | ---: | ---: | ---: |
| Base | 2 (2.50%) | 5 | 21.05% | 0 |
| SFT-v2 | 55 (68.75%) | 80 | 100% | 5 |
| **GRPO-v1** | **68 (85.00%)** | 76 | 100% | 1 |
| GRPO-v2 | 61 (76.25%) | 80 | 100% | 3 |

GRPO-v1 improves structural success over SFT-v2 by **16.25 percentage points**.
Structural success is an internal protocol/delivery measure, **not validated task
success**, factual correctness, or evidence of external generalization. Producing
an answer or a valid tool call does not establish that travel constraints are met.

## Dual-Judge pairwise results

All rows cover the same 80 queries with both candidate positions. W/T/L is from
the **left-hand model's** perspective. Position disagreements are reported
separately and mapped to aggregate ties by the predeclared rule.

| Judge | Model pair (left / right) | W / T / L | Position disagreements / 80 |
| --- | --- | ---: | ---: |
| GPT | Base / SFT-v2 | 2 / 0 / 78 | 0 |
| GPT | Base / GRPO-v1 | 2 / 1 / 77 | 1 |
| GPT | SFT-v2 / GRPO-v1 | 16 / 9 / 55 | 9 |
| Gemini 3.1 Pro | Base / SFT-v2 | 2 / 5 / 73 | 5 |
| Gemini 3.1 Pro | Base / GRPO-v1 | 1 / 5 / 74 | 5 |
| Gemini 3.1 Pro | SFT-v2 / GRPO-v1 | 8 / 15 / 57 | 14 |

Against SFT-v2, **GRPO-v1 wins/ties/loses 55/9/16 under GPT and 57/15/8 under
Gemini**. Their direct win rates are 68.75% and 71.25%; half-tie preference scores
are 74.375% and 80.625%, respectively. Preference scores are not task-success rates.
GPT/Gemini agree on 71.25% of the SFT-v2/GRPO-v1 aggregate outcomes.

The report includes weighted score differences, query-level bootstrap intervals,
Wilson intervals, Judge agreement, and failure cases. Shared-context batch
effects are not modeled by those intervals; they are not uncertainty guarantees
for an entirely independent experiment.

## Provenance and scope

- Original run: 473 valid GPT and 451 valid Gemini ratings. The 7 GPT and 29
  Gemini missing ratings were supplemented through user-authorized shared-context
  batches; all original failures are retained, not overwritten or converted to ties.
- The actual browser-transmitted bytes and provider identity/settings of the
  user-pasted supplemental responses were not independently observed. Earlier
  shared/manual Gemini protocol exceptions also remain documented.
- Excluding all 36 new supplements and 7 prior Gemini protocol-exception ratings
  leaves the prior-stage preference evidence supporting GRPO-v1. See the
  [strict sensitivity summary](reports/final80_sensitivity_057.json).
- The actual replay provenance of these saved outputs is **snapshot v1**, with
  **2,185 normalized observations replay-consistent** in the earlier audit. This
  supersedes the older final-80 narrative's snapshot-v4 attribution; the historical
  training/replay-v2 reports are separate experiments.
- Formal external-Judge pairwise comparisons cover Base/SFT-v2, Base/GRPO-v1,
  and SFT-v2/GRPO-v1. GRPO-v2 is included in structural comparison only; no
  external dual-Judge preference result is invented for it.

The project overview and [resume-ready project summary](reports/resume_project_latest.md)
focus on completed data construction, SFT, GRPO, reward design, and the measured
internal results. Human review and external benchmark history remain in the full
audit reports, are not presented as completed achievements, and are not additional
resume-completion tasks.

## Evidence and historical results

| Evidence | Location |
| --- | --- |
| Latest full internal report | [Final report 063](reports/final_metrics_report_063.md) |
| Latest pairwise and agreement metrics | [Metrics JSON](reports/final80_metrics_063.json) |
| Strict pre-supplement sensitivity | [Sensitivity JSON](reports/final80_sensitivity_057.json) |
| Compact integrity audit and artifact hashes | [Audit JSON](reports/final80_audit_063.json) |
| Current publication and artifact guide | [Status guide](docs/FINAL_EVALUATION_STATUS.md) |
| Data construction / distillation | [Distillation report](reports/distillation_1835.md) |
| Original SFT / GRPO engineering | [SFT report](reports/sft_training.md), [formal GRPO report](reports/GRPO_FORMAL_REPORT.md) |
| Historical 16-query engineering evaluation | [Original final report](FINAL_REPORT.md), [previous README](docs/history/README_before_final80.md) |
| Historical external evaluation audit | [Corrected validation](reports/TRAVELPLANNER_CORRECTED_VALIDATION.md), [adapter audit](reports/TRAVELPLANNER_ADAPTER_REPAIR.md) |

The older 16-query synthetic engineering results are retained for provenance and
must not be pooled with, or substituted for, the frozen 80-query results above.
No external-generalization improvement is claimed. Checkpoints and raw browser
account/session captures are not published as part of this documentation update.

## Reproduction entry points

For the original implementation and deterministic project checks:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt -r requirements/dev.txt
pytest -q
ruff check .
```

These are existing project checks, not a command to retrain or resubmit Judge
requests. The latest completion-stage offline regression run passed 17 tests;
the full original training-stage test count is historical, not a new run.
Full raw artifacts and the immutable supplemental overlay remain in the server
archive, identified in the [artifact guide](docs/FINAL_EVALUATION_STATUS.md).
