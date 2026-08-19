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

Production adapters use AMap Web Service for POI, nearby, weather, and routes;
Firecrawl v2 for web search/scrape; and committed deterministic inventories for
flight/train results. Provider HTTP failures, invalid JSON, rate limits, and
timeouts are normalized at the boundary. Successful responses can be persisted
in an atomic disk cache; API keys are accepted only from environment variables.

The agent state machine owns turns and termination. Tools never decide whether
an answer is final. A scheduler will bridge that state machine to ms-swift's
`MultiTurnScheduler`, preserving genuine model/environment alternation.

The state machine rejects malformed protocol blocks, unknown tools and invalid
arguments through structured observations, detects repeated identical actions,
and enters a force-answer state at the configured action-turn limit.

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
