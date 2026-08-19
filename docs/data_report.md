# M1 data report

Generated from the first completed data preparation run on 2026-08-19. Counts
below come from `data/processed/data_stats.json`; artifact hashes and immutable
split IDs are recorded under `data/manifests/`.

## Sources and policy

- Agent-STAR TravelDataset revision
  `d644daaffe2c6a1ec63d86c70fe4c639215f28ee`, declared MIT, supplied the
  17K synthetic candidate pool.
- TravelPlanner revision `8736504ecfc31b7f8b7e40122873c337e83fff7c`,
  declared CC-BY-4.0, supplied protected validation and official test queries.
- Agent-STAR's bundled TravelPlanner validation data was also protected.
- No protected TravelPlanner validation/test query was selected for training.

## Counts

| Stage | Records |
| --- | ---: |
| Raw Agent-STAR candidates | 17,594 |
| Protected reference queries | 1,360 |
| Excluded candidates | 108 |
| Accepted unique catalog | 17,486 |
| Frozen test | 100 |
| Remaining training candidates | 17,386 |

Exclusion flags were `duplicate_constraints` (86), `duplicate_query` (2), and
`protected_constraint_fingerprint` (22). A record can carry more than one flag,
so flag counts need not sum to the number of excluded records. The 92% fuzzy
threshold produced no additional exclusions in this snapshot.

## Distribution

| Split | Easy | Medium | Hard | 3 days | 5 days | 7 days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training candidates | 7,615 | 6,162 | 3,609 | 3,239 | 7,255 | 6,892 |
| Frozen test | 39 | 37 | 24 | 21 | 38 | 41 |

## Contamination and freeze method

Queries were Unicode/whitespace normalized, punctuation-folded for canonical
hashing, and compared by exact/canonical hash, structured travel-constraint
fingerprint, and high-threshold RapidFuzz similarity. Exact query and full
constraint fingerprints were also used for candidate deduplication.

After filtering, a deterministic SHA-256 ordering seeded with `20260819`
selected 100 frozen records. The frozen IDs and artifact SHA-256 are committed.
Validation confirms zero query-ID overlap and zero constraint-fingerprint
overlap between frozen and training splits.

## Limitations

The frozen set is independently held out from the synthetic Agent-STAR pool,
not a substitute for the protected official TravelPlanner test benchmark.
Semantic contamination beyond structured equality and 92% string similarity
will be revisited with embedding-neighbor review before teacher distillation.

