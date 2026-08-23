# SFT Token Length Audit

Generated: 2026-08-23T17:27:14.928037+00:00

Lengths use the frozen Qwen3-4B-Instruct-2507 tokenizer, `qwen3_nothinking` template, native `hermes` agent template, complete eight-tool schema, and no truncation.

## Overall distribution

| Statistic | Tokens |
|---|---:|
| min | 2865 |
| mean | 5477.25 |
| median | 5714 |
| p75 | 6161 |
| p90 | 6479 |
| p95 | 6726 |
| p99 | 7106 |
| max | 7401 |

## Buckets

| Bucket | Count | Percent |
|---|---:|---:|
| <=4K | 103 | 11.69% |
| 4K-8K | 778 | 88.31% |
| 8K-16K | 0 | 0.00% |
| 16K-24K | 0 | 0.00% |
| 24K-32K | 0 | 0.00% |
| >32K | 0 | 0.00% |

## By difficulty

| Difficulty | Count | Min | Mean | Median | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 419 | 3540 | 5619.98 | 5850 | 6246 | 6489 | 6729 | 7080 | 7401 |
| medium | 310 | 3346 | 5397.77 | 5567.5 | 6086 | 6439 | 6682 | 7077 | 7144 |
| hard | 152 | 2865 | 5245.9 | 5504.5 | 6079 | 6515 | 6959 | 7266 | 7363 |

## Samples exceeding 32,768

Count: **0**.

No truncation policy is activated because every trajectory fits intact.

## Longest 20 trajectories

| Query | Difficulty | Turns | Tool calls | Token length |
|---|---:|---:|---:|---:|
| `agent_star:easy_qjhujs` | easy | 28 | 13 | 7401 |
| `agent_star:hard_tobe94` | hard | 30 | 14 | 7363 |
| `agent_star:easy_iz9pzx` | easy | 28 | 13 | 7330 |
| `agent_star:hard_m2ogr8` | hard | 28 | 13 | 7266 |
| `agent_star:hard_4m8yah` | hard | 34 | 16 | 7156 |
| `agent_star:medium_euep90` | medium | 28 | 13 | 7144 |
| `agent_star:medium_yre2m1` | medium | 30 | 14 | 7136 |
| `agent_star:hard_9uvf1u` | hard | 28 | 13 | 7110 |
| `agent_star:easy_ncsq03` | easy | 28 | 13 | 7106 |
| `agent_star:easy_5ydyq1` | easy | 28 | 13 | 7097 |
| `agent_star:medium_gnjxk2` | medium | 28 | 13 | 7085 |
| `agent_star:easy_bwbta2` | easy | 28 | 13 | 7080 |
| `agent_star:medium_kdsu9w` | medium | 28 | 13 | 7077 |
| `agent_star:easy_juob4q` | easy | 28 | 13 | 7054 |
| `agent_star:hard_rx62l6` | hard | 28 | 13 | 7054 |
| `agent_star:hard_8j7qry` | hard | 28 | 13 | 7028 |
| `agent_star:easy_sri6uy` | easy | 28 | 13 | 7020 |
| `agent_star:hard_jo2e07` | hard | 28 | 13 | 6982 |
| `agent_star:medium_5hdue8` | medium | 28 | 13 | 6982 |
| `agent_star:medium_j85p4d` | medium | 28 | 13 | 6970 |