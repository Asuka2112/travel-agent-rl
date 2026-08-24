# Formal 200 GRPO Feasibility

Generated: 2026-08-24T19:06:38.850412+00:00

- Gate pass: **True**
- Full-parameter pipeline feasibility: True
- Replay policy v2 gate: True
- Optimizer steps: 200 (minimum 200)
- Full-parameter trainer: True
- Checkpoints: 10
- Checkpoint resume observed: True
- Rollouts: 1200
- Mean rollout latency: 25.226796887945383
- Mean turns: 9.170833333333333
- Mean tool calls: 8.1925
- Reward mean / mean group std: 4.5118380472 / 0.0899533458
- Advantage non-zero ratio: 0.9701492537313433
- Mean KL: 0.008047504
- Eight schemas visible in every rollout: True
- Tool/observation counts aligned: True
- Replay misses / public requests: 1398 / 0
- Replay miss unique / affected queries: 1292 / 131
- Miss recovered / final failure: 1350 / 48
- Misses deterministic / logged / non-fabricated: True / True / True
- No NaN/Inf: True

## Tool usage

| Tool | Calls |
|---|---:|
| `flight_search` | 2230 |
| `nearby_search` | 153 |
| `poi_search` | 4950 |
| `route_planning` | 670 |
| `train_search` | 45 |
| `weather` | 23 |
| `web_scraping` | 91 |
| `web_search` | 1669 |

## Reward components

| Reward | Mean | Mean logged std | Nonconstant |
|---|---:|---:|---:|
| `FinalAnswerBoundaryORM` | 0.97833333185 | 0.0429878202 | True |
| `FixedLocalJudgeORM` | 0.54687323715 | 0.028230689500000003 | True |
| `StageAwarenessORM` | 1.0 | 0.0 | False |
| `ToolEfficiencyORM` | 0.9748102795 | 0.0447320052 | True |
| `ValidAgentStepORM` | 0.99922619295 | 0.0015352786 | True |
| `ValidToolCallORM` | 1.0 | 0.0 | False |

The machine-readable JSON is the source of truth for all values in this report.
