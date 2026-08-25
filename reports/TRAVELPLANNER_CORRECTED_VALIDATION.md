# TravelPlanner Corrected Frozen-Output Validation

Generated: 2026-08-25T18:20:43.006440+00:00

The original result is retained and labeled **invalid / adapter-contaminated external evaluation**. The new result is labeled **adapter-corrected frozen-output evaluation**.

## Corrected official metrics (n=180 per model)

| Metric | Base | SFT | RL |
|---|---:|---:|---:|
| Commonsense Constraint Micro Pass Rate | 30/1440 (2.0833%) | 21/1440 (1.4583%) | 429/1440 (29.7917%) |
| Commonsense Constraint Macro Pass Rate | 0/180 (0.0000%) | 0/180 (0.0000%) | 0/180 (0.0000%) |
| Hard Constraint Micro Pass Rate | 0/420 (0.0000%) | 0/420 (0.0000%) | 0/420 (0.0000%) |
| Hard Constraint Macro Pass Rate | 0/180 (0.0000%) | 0/180 (0.0000%) | 0/180 (0.0000%) |
| Final Pass Rate | 0/180 (0.0000%) | 0/180 (0.0000%) | 0/180 (0.0000%) |

## Adapter delivery and gates

| Rate | Base | SFT | RL |
|---|---:|---:|---:|
| Parsed / delivered | 8/180 (4.4444%) | 6/180 (3.3333%) | 106/180 (58.8889%) |
| Schema-valid overall | 8/180 (4.4444%) | 6/180 (3.3333%) | 106/180 (58.8889%) |
| Schema-valid given parsed | 8/8 (100.0000%) | 6/6 (100.0000%) | 106/106 (100.0000%) |
| Completeness gate pass | 0/180 (0.0000%) | 0/180 (0.0000%) | 0/180 (0.0000%) |
| Parser failure | 172/180 (95.5556%) | 174/180 (96.6667%) | 74/180 (41.1111%) |
| Sandbox pass | 2/180 (1.1111%) | 0/180 (0.0000%) | 13/180 (7.2222%) |
| Hard constraints executed | 0/180 (0.0000%) | 0/180 (0.0000%) | 0/180 (0.0000%) |

## Before / after

| Model | Old Commonsense Micro | Corrected Commonsense Micro | Old Final Pass | Corrected Final Pass |
|---|---:|---:|---:|---:|
| BASE | 40/1440 (2.7778%) | 30/1440 (2.0833%) | 0/180 (0.0000%) | 0/180 (0.0000%) |
| SFT | 27/1440 (1.8750%) | 21/1440 (1.4583%) | 0/180 (0.0000%) | 0/180 (0.0000%) |
| RL | 411/1440 (28.5417%) | 429/1440 (29.7917%) | 0/180 (0.0000%) | 0/180 (0.0000%) |

## Interpretation

Adapter contamination was real: corrected field separation changed official constraint verdicts (most visibly RL commonsense micro), while removing the v1 prose-as-attraction fallback reduced Base/SFT vacuous passes. However, every delivered corrected plan is structurally schema-valid and completeness remains 0/180 for all three models. Consequently no plan reached the completeness+sandbox gate and hard constraints were executed for 0/180 tasks. Final Pass therefore remains 0/180 for Base, SFT, and RL.

The evidence rejects a universal remaining schema bug: delivered v2 plans have 100% structural validity, the fixed audit found zero hallucinated fields, wrong assignments, or entity mutations, and official format checking completed. The remaining zero Final Pass is a genuine frozen-output delivery/completeness/generalization gap under the official protocol, compounded by conservative parser failures where no faithful daily plan could be extracted. It is not evidence supporting retraining or post-hoc tuning.

All old files and metrics remain unchanged. Exact hashes and per-task gate verdicts are in `outputs/final/travelplanner_corrected_metrics.json` and `outputs/final/travelplanner_adapter_v2/constraint_diagnostics.json`.
