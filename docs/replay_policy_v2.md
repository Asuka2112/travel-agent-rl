# Replay Policy v2

Status: active for GRPO from 2026-08-24.

## Decision

The immutable version-4 web snapshot remains the sole web evidence source for
GRPO. Its content and SHA-256 are unchanged. GRPO always runs with
`WEB_MODE=replay_only`; it never falls back to the public network and never uses
Firecrawl Cloud.

A request absent from version 4 is a deterministic tool result, not a process
failure. The observation is a structured error with:

- error code `snapshot_miss`;
- the original query or URL;
- an empty result list;
- `retryable=false`;
- `public_network_used=false`.

The observation contains no invented document or travel fact. It is appended as
a tool-response message, excluded from policy loss, and the agent may use its
next turn to choose Flight Search, Train Search, POI Search, Route Planning, or
another available frozen tool. Rewards do not impose a catastrophic miss
penalty; existing task-completion, stage-awareness, tool-efficiency, and Judge
signals capture any downstream effect.

## Production gate

The superseded gate required zero replay misses. Version 2 instead requires:

1. zero public-network requests;
2. every miss is deterministic, structured, and logged;
3. every miss exposes an empty, non-fabricated observation;
4. no silent success or network fallback;
5. rollout alignment remains valid and execution can continue after a miss.

Telemetry records the tool, arguments, turn, query id, recovery count, final
failure count, mask/logprob alignment, and public-network count. The previous
`GRPO_SMOKE_BLOCKER.md`, smoke metrics, and snapshot-v4 hash remain immutable.

## Provenance

The implementing Git commit is recorded after regression verification. Until
that commit is created, reports identify the working-tree base commit and the
final reproducibility manifest supplies the implementing commit.
