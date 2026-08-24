# Full 8-tool SFT evaluation

## Interactive frozen coverage evaluation

Both models used the same 16 immutable engineering queries, eight visible tool schemas,
snapshot v4, `WEB_MODE=replay_only`, and max 13 turns. The fixed Judge is the unmodified
Base model with `prompts/judge/v1.txt`; it is never the GRPO policy.

| Model | Valid calls | Invalid calls | Final answer | Constraint satisfied | Avg turns | Avg tools | Loop | Judge /5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.9412 | 0.0588 | 0.9375 | 0.9375 | 2.000 | 1.000 | 0.0000 | 4.8661 |
| SFT | 1.0000 | 0.0000 | 0.6250 | 0.6250 | 8.125 | 7.500 | 0.2500 | 3.9286 |

- Base replay misses: 0; public requests: 0.
- SFT replay misses: 42; public requests: 0.
- SFT unknown exceptions: 0.
- The synthetic queries were designed for Train Search coverage and contain unseen web
  requests; replay misses remained deterministic errors and never fell through to the network.

## Offline teacher-forced validation

The held-out subset was fixed by SHA-256 ranking from the 382 Judge-accepted trajectories
not selected for SFT or RL. It also excludes synthetic dev and all frozen-test IDs.

- Rows: 64 ({'Easy': 27, 'Hard': 8, 'Medium': 29})
- Validation loss: 0.223847093
- Validation token accuracy: 92.789570591%
- Evaluated tokens: 47251
- Held-out SHA-256: `d1482b75a123ce6ba94ac9048b8d4b58261386ed49c1ea824591daa218fe86d5`
- Overlap with SFT/RL/dev/frozen test: 0
- Parent protected/frozen contamination overlap: 0

## Metric boundaries

- Train metrics: aggregate loss 0.2780154, last loss 0.22089405, train token accuracy 92.9729%.
- Offline validation metrics: the 64-row loss/accuracy above.
- Interactive Agent metrics: the Base/SFT table above; they are not teacher-forced metrics.
