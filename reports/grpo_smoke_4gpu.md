# Smoke GRPO Feasibility

Generated: 2026-08-24T06:32:30.082152+00:00

- Gate pass: **False**
- Full-parameter pipeline feasibility: True
- Strict replay gate: False
- Optimizer steps: 5 (minimum 5)
- Full-parameter trainer: True
- Checkpoints: 5
- Checkpoint resume observed: True
- Rollouts: 30
- Mean rollout latency: 4.873983302836617
- Mean turns: 3.0
- Mean tool calls: 2.933333333333333
- Reward mean / mean group std: 2.891676282 / 0.16214967
- Advantage non-zero ratio: 0.7142857142857143
- Mean KL: 0.000132312
- Eight schemas visible in every rollout: True
- Tool/observation counts aligned: True
- Replay misses / public requests: 1 / 0
- No NaN/Inf: True

## Tool usage

| Tool | Calls |
|---|---:|
| `flight_search` | 9 |
| `poi_search` | 54 |
| `route_planning` | 16 |
| `web_search` | 9 |

## Reward components

| Reward | Mean | Mean logged std | Nonconstant |
|---|---:|---:|---:|
| `FinalAnswerBoundaryORM` | 0.066666668 | 0.103279554 | True |
| `FixedLocalJudgeORM` | 0.341676198 | 0.069268778 | True |
| `StageAwarenessORM` | 1.0 | 0.0 | False |
| `ToolEfficiencyORM` | 0.5 | 0.06324555800000001 | True |
| `ValidAgentStepORM` | 0.8833333379999999 | 0.012909944000000001 | True |
| `ValidToolCallORM` | 0.85 | 0.0 | False |

The machine-readable JSON is the source of truth for all values in this report.
