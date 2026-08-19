# Project specification

## Objective

Build an auditable research pipeline:

```text
licensed travel queries -> normalized queries -> eight-tool ReAct environment
-> teacher trajectories -> rule/format/judge filtering -> SFT and RL datasets
-> Qwen3-4B-Instruct-2507 LoRA SFT -> merged validation -> online multi-turn
GRPO -> Base/SFT/RL frozen evaluation
```

## Non-negotiable constraints

- Base model: `Qwen/Qwen3-4B-Instruct-2507`, pinned to a recorded revision.
- Agent protocol uses `<tool-call>` JSON and `<answer>` boundaries.
- Maximum action turns default to 13 and remain configurable.
- GRPO rollout is online: model action, environment observation, next action.
- Target topology is GPUs 0-1 for vLLM and GPUs 2-7 for full-parameter ZeRO-3,
  but jobs launch only after a fresh availability check.
- API keys remain in environment variables and are never committed.
- Frozen evaluation data is immutable after its manifest is committed.
- Planned, running, completed, and failed states are reported separately.
- All reported results must be derived from saved machine-readable artifacts.

## Initial data targets

| Artifact | Target |
| --- | ---: |
| Raw candidate queries | >= 1,500 |
| Successful distilled trajectories | aspirational >= 1,200 |
| Filtered high-quality trajectories | 800-1,200 |
| SFT trajectories | 800-1,000 |
| RL queries | 200-400 |
| Frozen test queries | >= 80 |

Targets may change only through a documented decision with observed evidence.

## Quality gates

1. Dataset license, revision, checksums, split provenance, and contamination
   report exist before training.
2. Unit, integration, environment, data, reward, and evaluation tests pass.
3. SFT progresses through 100-sample smoke and 100-step sanity gates.
4. GRPO progresses through 3-turn/2-generation online update validation.
5. GPU ownership and free memory are captured immediately before every job.

