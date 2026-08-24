# Pilot GRPO Feasibility

Generated: 2026-08-24T10:41:12.920656+00:00

- Gate pass: **True**
- Full-parameter pipeline feasibility: True
- Replay policy v2 gate: True
- Optimizer steps: 16 (minimum 16)
- Full-parameter trainer: True
- Checkpoints: 14
- Checkpoint resume observed: True
- Rollouts: 192
- Mean rollout latency: 37.338613719155546
- Mean turns: 7.901041666666667
- Mean tool calls: 7.114583333333333
- Reward mean / mean group std: 3.717907904375 / 0.274222819375
- Advantage non-zero ratio: 0.4444444444444444
- Mean KL: 0.000765724375
- Eight schemas visible in every rollout: True
- Tool/observation counts aligned: True
- Replay misses / public requests: 250 / 0
- Replay miss unique / affected queries: 236 / 43
- Miss recovered / final failure: 236 / 14
- Misses deterministic / logged / non-fabricated: True / True / True
- No NaN/Inf: True

## Tool usage

| Tool | Calls |
|---|---:|
| `flight_search` | 266 |
| `nearby_search` | 35 |
| `poi_search` | 632 |
| `route_planning` | 75 |
| `train_search` | 6 |
| `weather` | 9 |
| `web_scraping` | 9 |
| `web_search` | 334 |

## Reward components

| Reward | Mean | Mean logged std | Nonconstant |
|---|---:|---:|---:|
| `FinalAnswerBoundaryORM` | 0.78645833625 | 0.384588165 | True |
| `FixedLocalJudgeORM` | 0.0 | 0.0 | False |
| `StageAwarenessORM` | 1.0 | 0.0 | False |
| `ToolEfficiencyORM` | 0.88662574375 | 0.194085403125 | True |
| `ValidAgentStepORM` | 0.988136581875 | 0.021366019375 | True |
| `ValidToolCallORM` | 1.0 | 0.0 | False |

The machine-readable JSON is the source of truth for all values in this report.
