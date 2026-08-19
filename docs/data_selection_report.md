# Frozen 1,835-query training set

## Freeze identity

The training query set was frozen on 2026-08-19 before any pilot or training result was
observed. Re-running the selection command is idempotent; it refuses to overwrite any
existing artifact whose bytes differ.

| Field | Value |
| --- | --- |
| Source | `xxwu/Agent-STAR-TravelDataset`, `TravelTotal_17K` candidate pool |
| Source revision | `d644daaffe2c6a1ec63d86c70fe4c639215f28ee` |
| Candidate-pool SHA-256 | `36a575c22ba0af78e2b2cde9be639fcf325365b116fb9702e0472452aeaca358` |
| Selection algorithm | version 1 |
| Selection seed | `20260819` |
| Near-duplicate threshold | RapidFuzz ratio 98.5 |
| Frozen JSONL SHA-256 | `cf2f694419f7a96e868d07507a0a26d648cfd885ce69cda7daae36b3d3d82e6e` |

The 50- and 200-query pilots use seed `20260850`. Their immutable SHA-256 values are
`e1087f739dbfe234fc530402a6b5c962379aea77e4bb29f64194b55d19aa3377` and
`2a321060d59370b9079474ca5c581c55ec6068aedae3d98b977dac43394eecf3`, respectively.
The 50-query set is a stratified subset of the 200-query set.

## Selection method

Difficulty quotas are proportional to the candidate pool using largest-remainder
allocation. Within each Easy, Medium, and Hard stratum, a deterministic greedy score
rewards under-represented origin, destination, origin/destination route, duration,
city count, and constraint signature. A SHA-256-derived seed fraction provides stable
tie-breaking. Exact canonical duplicates and candidates at or above the configured
near-text threshold are not selected.

## Distribution

| Dimension | Counts |
| --- | --- |
| Difficulty | Easy 804; Medium 650; Hard 381 |
| Trip scope | Single city 1,053; multi city 782 |
| Duration | 3 days 1,053; 5 days 392; 7 days 390 |
| Budget | 1,835 |
| Accommodation constraint | 717 |
| Dietary constraint | 504 |
| Transportation constraint | 381 |
| Multi-constraint | 381 |
| Unique origins / destinations | 238 / 235 |

No selected source query contains an attraction-specific constraint under the frozen
classifier (`attraction`, `sightseeing`, `landmark`, or `museum`). Agent-STAR's source
schema supplies general itinerary requests but no dedicated attraction constraint field,
so this report records attraction-constraint coverage as zero rather than inferring or
manufacturing it. POI/attraction planning remains part of the required answer behavior.

The most frequent origin is Dallas (48, 2.62%); the most frequent destination is Texas
(44, 2.40%). No single origin or destination exceeds 3% of the frozen set. Full entity
counts and all 1,835 record-level selections are stored in the manifest.

## Contamination audit

The selected set was checked against:

- Agent-STAR `TravelPlanner_Val180`;
- TravelPlanner validation and test CSVs;
- `data/manifests/frozen_test_ids.json`;
- exact canonical text hashes, constraint fingerprints, and fuzzy normalized text at
  threshold 92.

The audit performed 2,495,600 selected/protected fuzzy comparisons. Frozen-ID overlap,
protected exact/constraint overlap, and protected fuzzy overlap were all zero. The
selection code fails closed on any future overlap.

## Artifacts

- `data/manifests/train_queries_1835.jsonl`
- `data/manifests/train_queries_1835_manifest.json`
- `data/manifests/distill_pilot_50.jsonl`
- `data/manifests/distill_pilot_200.jsonl`

Verification command:

```bash
.venv/bin/python scripts/freeze_train_queries.py \
  --config configs/data/train_selection_1835.yaml
sha256sum data/manifests/train_queries_1835.jsonl \
  data/manifests/distill_pilot_50.jsonl \
  data/manifests/distill_pilot_200.jsonl
```
