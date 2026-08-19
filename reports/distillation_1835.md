# Production distillation 1,835 final report

## Counts

- Frozen queries: 1835
- Generated successfully: 1820 (99.18%)
- Generation failures: 15 — `{'model_call_limit_exceeded': 11, 'TimeoutError: timed out': 4}`
- Rule filter: 1699 accepted; `{'repeated_action': 121}`
- Format/schema: 1699 accepted; 0 rejected
- Judge: 1463 accepted; `{'judge_below_threshold': 236}`
- Final split: SFT 881, RL 200, synthetic dev 16

## Pass rates

| Difficulty | Frozen | Generated | Judge accepted |
| --- | ---: | ---: | ---: |
| Easy | 804 | 801 (99.63%) | 689 (85.70%) |
| Hard | 381 | 375 (98.43%) | 261 (68.50%) |
| Medium | 650 | 644 (99.08%) | 513 (78.92%) |

| Scope | Frozen | Generated | Judge accepted |
| --- | ---: | ---: | ---: |
| multi_city | 782 | 771 (98.59%) | 613 (78.39%) |
| single_city | 1053 | 1049 (99.62%) | 850 (80.72%) |

## Tool/runtime audit

- Tool calls: `{'flight_search': 3091, 'nearby_search': 261, 'poi_search': 8467, 'route_planning': 1338, 'train_search': 213, 'weather': 131, 'web_scraping': 374, 'web_search': 5943}`
- Train Search: 213 calls across 211 queries; 213 successful observations
- Average turns/tool calls: 10.824/10.889
- Average generation latency: 33.172 seconds/query
- Web budget exhaustion: 886 observations across 789 queries (43.35%)
- Replay: 4278/4278 hits, 0 misses, 0 live requests, 0 Cloud requests
- Unknown exceptions: 0

Judge attempt 1 produced 1,514 timeouts and one JSON parse exception after an unsafe
concurrency increase. All 1,699 original checkpoint files are retained under the archived
attempt-1 directory. Only exception rows were retried at concurrency 8; the Judge prompt
and thresholds were unchanged, and the final run has zero remaining exceptions.

## Integrity

- SFT/RL overlap: 0
- Dev/training overlap: 0
- Protected benchmark overlap inherited from frozen parent: 0
- Snapshot content SHA-256: `dbd160b6de284208c3906e9da0768cabfe3a77315e64280ea59685eefff15f92`
- SFT SHA-256: `b63a8464d58191cd5825cdd54899f443d1f90f57cece678a89fff23839de8008`
- RL SHA-256: `ca9411657993bb3cc4c754ec4a47583d3207ab6dc9cb4bfa8451e934148e9c80`
- Dev SHA-256: `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`
- Source code commit: `e002008ddfe2fc46ea94c65a6791e65075329564`

SFT training has **not** been started.
