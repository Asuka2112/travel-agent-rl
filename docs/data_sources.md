# Data source and license gate

Checked on 2026-08-19. Downloading does not imply approval for training; every
source must pass revision, checksum, split, and license verification first.

| Source | Candidate use | Observed license | Split policy |
| --- | --- | --- | --- |
| OSU TravelPlanner | schema and protected benchmark reference | dataset card declares CC-BY-4.0; code repository is MIT | official validation/test are protected and excluded from all training/distillation |
| Agent-STAR TravelDataset | primary synthetic query candidates | Hugging Face card declares MIT | exclude bundled TravelPlanner validation from training; use synthetic 17K pool with deduplication |
| Agent-STAR TravelDatabase | local grounding for US travel tools | Hugging Face card declares MIT | fixed 2022 snapshot; build a checksum-verified SQLite database for distillation, never use it as an evaluation label |

Primary references:

- https://github.com/OSU-NLP-Group/TravelPlanner
- https://huggingface.co/datasets/osunlp/TravelPlanner
- https://github.com/WxxShirley/Agent-STAR
- https://huggingface.co/datasets/xxwu/Agent-STAR-TravelDataset
- https://huggingface.co/datasets/xxwu/Agent-STAR-TravelDatabase

The travel database is pinned to revision
`985c8c7ba3e6beb986b3b0a55a2b064d18f2e68e`. Its five CSV checksums and row
counts, plus the generated SQLite checksum, are recorded in
`data/manifests/data_snapshot.json`. `scripts/build_travel_database.py` refuses
to build when any raw checksum differs and publishes the SQLite file atomically.

## Required acquisition manifest fields

`source_id`, upstream URL, commit/revision, retrieved timestamp, declared
license, license file hash, artifact hashes, original split, local split,
record count, exclusions, and transformation version.

## Contamination policy

Normalize Unicode, whitespace, dates, locations, budgets, and punctuation;
then run exact hash, canonical hash, high character/token similarity, and
semantic-nearest-neighbor checks against all official test/validation items.
Flagged records are reviewed or excluded. The frozen test set is created once,
stored with checksums, and never used in prompts, distillation, SFT, or RL.
