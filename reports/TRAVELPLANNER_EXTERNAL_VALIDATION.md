# TravelPlanner External Validation

Generated: 2026-08-25T17:27:43.043665+00:00

This is a frozen post-hoc evaluation on the official 180-query TravelPlanner validation split. No validation feedback was used for training, prompt changes, tool changes, or hyperparameter tuning.

## Deterministic official metrics

| Model | Commonsense Micro | Commonsense Macro | Hard Micro | Hard Macro | Final Pass |
|---|---:|---:|---:|---:|---:|
| BASE | 2.78% | 0.00% | 0.00% | 0.00% | 0.00% |
| SFT | 1.88% | 0.00% | 0.00% | 0.00% | 0.00% |
| RL | 28.54% | 0.00% | 0.00% | 0.00% | 0.00% |

All official values above were produced by the unmodified official evaluator at revision `e52c87f4ac348a3410c46dc3553c519db5ec5e23`. They are not Judge scores.

## Final Pass and project Task Success uncertainty

| Model | Official Final Pass (95% Wilson CI) | Project Task Success (95% Wilson CI) |
|---|---:|---:|
| BASE | 0/180 (0.00%; 0.00%–2.09%) | 6/180 (3.33%; 1.54%–7.08%) |
| SFT | 0/180 (0.00%; 0.00%–2.09%) | 5/180 (2.78%; 1.19%–6.34%) |
| RL | 0/180 (0.00%; 0.00%–2.09%) | 78/180 (43.33%; 36.31%–50.64%) |

Project Task Success means that the final answer deterministically adapted to exactly the requested number of official-schema days; it must not be read as official Final Pass.

## Project trajectory metrics

| Model | Valid Tool | Final Answer | Avg Turns | Avg Tool Calls | Loop |
|---|---:|---:|---:|---:|---:|
| BASE | 58.84% | 16.67% | 2.178 | 1.183 | 0.00% |
| SFT | 100.00% | 13.89% | 12.444 | 12.306 | 23.33% |
| RL | 100.00% | 92.22% | 7.339 | 6.417 | 3.89% |

## Internal metrics retained separately

Internal Frozen Eval uses n=16 synthetic engineering coverage queries: Base 14/16 (0.875), SFT 11/16 (0.6875), RL 16/16 (1.0). Pointwise Judge scores and the existing A/B+B/A pairwise results apply only to that internal set and do not replace the deterministic TravelPlanner metrics.

Pairwise position disagreements: base vs sft=4/16, base vs rl=4/16, sft vs rl=10/16; total=18/48.

## Scientific controls

Contamination audit: **passed**, scanning 2932 training-derived rows with zero exact/canonical, constraint-fingerprint, or thresholded fuzzy overlaps. Base, SFT, and RL used identical prompt, greedy decoding, eight-tool replay environment, turn/token limits, post-processing, and official evaluation code. Frozen model hashes were rechecked after evaluation; public generation requests were zero.
