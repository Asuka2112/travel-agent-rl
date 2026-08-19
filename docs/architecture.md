# Architecture

## Data and training flow

```text
source manifests
  -> normalize/deduplicate/contamination gate
  -> query catalog + frozen split manifest
  -> multi-turn environment + teacher
  -> raw trajectory ledger (append-only, resumable)
  -> rule filter -> canonical formatter -> versioned LLM judge
  -> SFT trajectories + RL query-only data
  -> LoRA SFT -> merge -> inference gate
  -> online multi-turn GRPO -> evaluation
  -> artifact-derived report
```

## Environment boundary

`ToolRequest` and `ToolResponse` are the only public tool boundary. A registry
maps stable tool names to adapters. Every response carries status, normalized
error information, attempt count, cache state, and latency. Mock adapters are
deterministic and share the same schema as future live adapters.

The agent state machine owns turns and termination. Tools never decide whether
an answer is final. A scheduler will bridge that state machine to ms-swift's
`MultiTurnScheduler`, preserving genuine model/environment alternation.

## Fault tolerance

- External calls are bounded by timeout and retry policies.
- Rate limits and provider exceptions become structured tool errors.
- Raw distillation ledgers are append-only and checkpointed.
- Filtering stages consume immutable inputs and write versioned outputs.
- Experiment manifests bind code commit, data checksums, config, environment,
  GPU snapshot, command, exit code, and produced artifacts.

## GPU topology

The desired final layout is vLLM on two GPUs and ZeRO-3 training on six GPUs.
Numeric GPU IDs are assigned only at job launch after a fresh ownership and
free-memory check; static IDs in a config are targets, not reservations.

