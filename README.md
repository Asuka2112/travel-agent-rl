# Travel Agentic RL

Post-training a travel-planning agent with Qwen3-4B-Instruct-2507, structured tool
use, supervised fine-tuning, and multi-turn Group Relative Policy Optimization.

**Project workflow:** Data construction → SFT → Agentic GRPO → Frozen evaluation.

## Project status

Updated 2026-10-20. Training and the frozen internal evaluation are complete.
**GRPO is the selected internal delivery model.** The project presentation uses
three labels only: **Base**, **SFT** (supervised comparison), and **GRPO** (selected
delivery model). Evaluated checkpoint identities and training lineage remain in
the archived provenance; presentation labels do not change the underlying models.

## What was built

- Travel-task data construction, trajectory filtering, and SFT cold start.
- Multi-turn GRPO training with tool execution, trajectory rewards, a fixed local
  training Judge, and checkpoint/resume support.
- An eight-tool frozen replay protocol and a protected 80-query internal test set.
- GPT and Gemini 3.1 Pro pairwise judging with A/B and B/A position swaps, weighted
  **30% tool-path quality + 70% final-answer quality**.

## Internal results

| Model | Structural success / 80 | Final answers / 80 | Valid tool-call rate | Loops / 80 |
| --- | ---: | ---: | ---: | ---: |
| Base | 2 (2.50%) | 5 | 21.05% | 0 |
| SFT | 55 (68.75%) | 80 | 100% | 5 |
| **GRPO** | **68 (85.00%)** | 76 | 100% | **1** |

The final GRPO model has **16.25 percentage points** higher structural success
than the SFT comparison model, with tool-call loops reduced from **5/80 to 1/80**.
These are comparisons of frozen outputs, not a claim that the displayed SFT
checkpoint is the direct training parent of the selected GRPO checkpoint.
Structural success is an internal protocol/delivery metric, **not verified task
success**, factual correctness, or evidence of external generalization.

## Dual-Judge comparison

Every comparison covers the same 80 queries in both candidate positions. W/T/L
is from the **left-hand model's** perspective.

| Judge | Comparison | Win / Tie / Loss | Position disagreements / 80 |
| --- | --- | ---: | ---: |
| GPT | Base / SFT | 2 / 0 / 78 | 0 |
| GPT | Base / GRPO | 2 / 1 / 77 | 1 |
| GPT | SFT / GRPO | 16 / 9 / 55 | 9 |
| Gemini 3.1 Pro | Base / SFT | 2 / 5 / 73 | 5 |
| Gemini 3.1 Pro | Base / GRPO | 1 / 5 / 74 | 5 |
| Gemini 3.1 Pro | SFT / GRPO | 8 / 15 / 57 | 14 |

Against SFT, **GRPO wins/ties/loses 55/9/16 under GPT and 57/15/8 under Gemini**.
Direct win rates are 68.75% and 71.25%, respectively. Both Judges support GRPO as
the internal delivery choice; this is not a guarantee of superiority on every task.

There are 480 usable score slots per Judge. The combined 960 slots include 36
user-authorized shared-context batch supplements, not 960 independent successful
calls. Position disagreement, batch dependence, and unverified manual-transmission
details remain documented; the conclusion also holds in the strict earlier-stage
sensitivity analysis. Aggregate ties caused by position disagreement do not mean
the candidates are genuinely equal in quality.

## Project documents

- [Final project results and uncertainty](reports/project_results.md)
- [Machine-readable project metrics](reports/project_metrics.json)
- [Resume-ready project description](reports/resume_project_latest.md)
- [Evaluation scope and archived provenance](docs/FINAL_EVALUATION_STATUS.md)

The primary presentation follows one SFT/GRPO project workflow rather than an
experiment-version catalogue. Original reports, negative results, frozen data,
and checkpoint provenance are retained unchanged in the archive; no discarded
experiment is relabeled as the selected model.

## Deterministic project checks

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt -r requirements/dev.txt
pytest -q
ruff check .
```

These commands describe existing project checks, not a request to retrain or
resubmit Judge calls. Evaluation completion used offline checks; experiments and
automatic continuation remain stopped. No checkpoints or raw browser-account
captures are published by this presentation update.
