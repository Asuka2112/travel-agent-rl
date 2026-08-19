# Production gates A/B/C

## Gate B: 200-query Web budget diagnosis

- Exhausted queries: 116/200; exhaustion observations: 138
- Search attempted/executed: 291/195 (per all query: 1.455; per Web-using query: 2.273)
- Scrape attempted/executed: 103/61 (per all query: 0.515; per Web-using query: 0.805)
- Exhausted final-pass rate: 84.48%; non-exhausted: 88.10%; risk difference: -3.61 pp
- Exhausted mean Judge/constraint score: 4.326/4.216
- Non-exhausted mean Judge/constraint score: 4.808/4.598
- Exhausted judged trajectories with constraint or completeness below 4: 24/111
- Exact affected Query IDs and per-query outcomes: `reports/web_budget_exhaustion_200.jsonl`

The old cap was statistically associated with lower final pass, but many exhausted
trajectories still passed; exhaustion was a frequent artificial truncation rather than a
one-to-one failure cause.

## Frozen 50 paired production-budget rerun

| Metric | Old total cap 2 | Production total cap 4 |
| --- | ---: | ---: |
| Generation success | 100.00% | 100.00% |
| Final pass | 86.00% | 80.00% |
| Average tool calls | 9.280 | 10.760 |
| Average turns | 9.260 | 10.660 |
| Web exhaustion queries | 22.000 | 19.000 |
| Web exhaustion calls | 28.000 | 21.000 |
| Average Judge score | 4.574 | 4.648 |
| Average latency (s) | 30.888 | 29.172 |

Production uses a **total public-Web action cap of 4**. This is an upper bound, not a
quota. The model/revision, temperature, seed, max turns, Judge, and Query IDs stayed fixed.
Prompt v4 adds the required Train description and stops refining the same fact after an
irrelevant result. Final pass was not optimized upward: the observed 40/50 is retained.
The exhausted/non-exhausted final-pass rates are 78.95% and
80.65%, respectively, so the remaining cap events are not a
primary failure mechanism. Repeated actions remain a legitimate Rule-filter reason.

## Gate C: immutable transient-error repair

- v1 was retained unchanged.
- v2 records: 349; successes/errors: 317/32.
- Single retries: 32 transient records; repaired: 2; retained transient failures: 30.
- Deterministic client errors retained without retry: 2.
- v2 snapshot SHA-256: `585453158e841c81a3dccae8fc8b3ce7b2bcb2a7c84ae36edeb984f256a053b1`
- v2 replay: 238/238 hits, 0 misses, 0 public requests, 0 Cloud requests.
- Gate pilot v3 snapshot SHA-256: `3fb612054cb6b9641efa57069ea079fdc8d974eb3fe2bd4c4be76dc5d9fb28c3`
- Budget-4 replay: 129/129 hits, 0 misses, 0 public requests.

## Gate decision

- Train Search functional coverage: pass (see `reports/train_tool_coverage.md`).
- Illegal tool arguments: 0.
- Unknown exceptions after cache fix: 0.
- Web budget as mass artificial failure: resolved with total cap 4.
- Replay-only public Web/Cloud requests: 0/0.
- Full automated test suite: recorded separately at the gate commit.
