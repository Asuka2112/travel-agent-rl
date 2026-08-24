# Formal Probe GRPO Feasibility

Generated: 2026-08-24T11:02:54.877015+00:00

- Gate pass: **True**
- Full-parameter pipeline feasibility: True
- Replay policy v2 gate: True
- Optimizer steps: 3 (minimum 3)
- Full-parameter trainer: True
- Checkpoints: 3
- Checkpoint resume observed: True
- Rollouts: 18
- Mean rollout latency: 33.05336484851109
- Mean turns: 12.944444444444445
- Mean tool calls: 12.222222222222221
- Reward mean / mean group std: 3.9818255133333333 / 0.5040362899999999
- Advantage non-zero ratio: 1.0
- Mean KL: 7.939333333333334e-05
- Eight schemas visible in every rollout: True
- Tool/observation counts aligned: True
- Replay misses / public requests: 67 / 0
- Replay miss unique / affected queries: 66 / 3
- Miss recovered / final failure: 63 / 4
- Misses deterministic / logged / non-fabricated: True / True / True
- No NaN/Inf: True

## Tool usage

| Tool | Calls |
|---|---:|
| `flight_search` | 32 |
| `nearby_search` | 3 |
| `poi_search` | 83 |
| `route_planning` | 7 |
| `train_search` | 2 |
| `web_scraping` | 5 |
| `web_search` | 88 |

## Reward components

| Reward | Mean | Mean logged std | Nonconstant |
|---|---:|---:|---:|
| `FinalAnswerBoundaryORM` | 0.72222223 | 0.35470678333333333 | True |
| `FixedLocalJudgeORM` | 0.36211639999999995 | 0.040503653333333334 | True |
| `StageAwarenessORM` | 1.0 | 0.0 | False |
| `ToolEfficiencyORM` | 0.8148147866666666 | 0.21842357999999998 | True |
| `ValidAgentStepORM` | 0.9900793833333333 | 0.012668093333333333 | True |
| `ValidToolCallORM` | 1.0 | 0.0 | False |

The machine-readable JSON is the source of truth for all values in this report.
