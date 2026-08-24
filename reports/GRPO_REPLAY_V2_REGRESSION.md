# Replay V2 Regression GRPO Feasibility

Generated: 2026-08-24T07:21:11.058696+00:00

- Gate pass: **True**
- Full-parameter pipeline feasibility: True
- Replay policy v2 gate: True
- Optimizer steps: 6 (minimum 6)
- Full-parameter trainer: True
- Checkpoints: 1
- Checkpoint resume observed: False
- Rollouts: 6
- Mean rollout latency: 6.10065392156442
- Mean turns: 3.0
- Mean tool calls: 2.8333333333333335
- Reward mean / mean group std: 3.02750015 / 0.34743637
- Advantage non-zero ratio: 0.6666666666666666
- Mean KL: 0.00024581
- Eight schemas visible in every rollout: True
- Tool/observation counts aligned: True
- Replay misses / public requests: 6 / 0
- Replay miss unique / affected queries: 1 / 3
- Miss recovered / final failure: 6 / 0
- Misses deterministic / logged / non-fabricated: True / True / True
- No NaN/Inf: True

## Tool usage

| Tool | Calls |
|---|---:|
| `flight_search` | 10 |
| `route_planning` | 1 |
| `web_search` | 6 |

## Reward components

| Reward | Mean | Mean logged std | Nonconstant |
|---|---:|---:|---:|
| `FinalAnswerBoundaryORM` | 0.16666667 | 0.40824831 | True |
| `FixedLocalJudgeORM` | 0.17333333 | 0.27034739 | True |
| `StageAwarenessORM` | 1.0 | 0.0 | False |
| `ToolEfficiencyORM` | 0.58333331 | 0.20412415 | True |
| `ValidAgentStepORM` | 0.89583331 | 0.05103104 | True |
| `ValidToolCallORM` | 1.0 | 0.0 | False |

The machine-readable JSON is the source of truth for all values in this report.
