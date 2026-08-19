# Data source and license gate

Checked on 2026-08-19. Downloading does not imply approval for training; every
source must pass revision, checksum, split, and license verification first.

| Source | Candidate use | Observed license | Split policy |
| --- | --- | --- | --- |
| OSU TravelPlanner | schema, train/validation candidates, benchmark context | MIT repository; verify dataset artifacts and transitive source terms | official test is evaluation-only and excluded from all training/distillation |
| Agent-STAR TravelDataset | primary synthetic query candidates | Hugging Face card declares MIT | exclude bundled TravelPlanner validation from training; use synthetic 17K pool with deduplication |

Primary references:

- https://github.com/OSU-NLP-Group/TravelPlanner
- https://huggingface.co/datasets/osunlp/TravelPlanner
- https://github.com/WxxShirley/Agent-STAR
- https://huggingface.co/datasets/xxwu/Agent-STAR-TravelDataset

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

