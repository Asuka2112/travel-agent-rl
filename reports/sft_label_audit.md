# SFT Label and Loss-Mask Audit

Generated: 2026-08-23T17:27:14.909398+00:00

- Frozen source rows: 881
- Audited distinct rows: 15
- Model/tokenizer: `models/Qwen3-4B-Instruct-2507`
- ms-swift template: `qwen3_nothinking`
- Agent template: `hermes`
- Loss scale: `default`
- Dataset SHA-256: `f9b8b2be0289f48bb89ce769a351bf652d2b74de61706d8a57d3b6644a346111`

The loss mask is stored as exact token-index run-length spans: `loss=0` is masked and `loss=1` is trained. Each row also preserves the complete decoded input and labels.

| Query | Difficulty | Tokens | Trainable | Calls | Train | Web | All checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `agent_star:easy_01djts` | easy | 6314 | 824 | 13 | False | True | True |
| `agent_star:easy_041h7r` | easy | 5999 | 982 | 13 | True | True | True |
| `agent_star:easy_07g2fi` | easy | 6454 | 836 | 13 | True | True | True |
| `agent_star:medium_050mmw` | medium | 3346 | 487 | 4 | False | False | True |
| `agent_star:medium_0b5l13` | medium | 4025 | 581 | 6 | False | False | True |
| `agent_star:medium_0kuamj` | medium | 4984 | 682 | 11 | False | True | True |
| `agent_star:hard_00tffl` | hard | 3655 | 367 | 6 | False | False | True |
| `agent_star:hard_06hskv` | hard | 5137 | 695 | 12 | False | True | True |
| `agent_star:hard_0awg4b` | hard | 5718 | 692 | 13 | False | True | True |
| `agent_star:easy_0ewavl` | easy | 4064 | 636 | 7 | True | False | True |
| `agent_star:easy_0fgakf` | easy | 4111 | 669 | 7 | True | False | True |
| `agent_star:easy_0s362r` | easy | 5183 | 676 | 11 | True | True | True |
| `agent_star:easy_0aih6l` | easy | 4955 | 718 | 9 | False | True | True |
| `agent_star:easy_0grbzt` | easy | 5796 | 800 | 12 | False | True | True |
| `agent_star:easy_0mak2k` | easy | 6853 | 906 | 13 | False | True | True |

## Required checks

- PASS: User content does not compute loss (15/15)
- PASS: Tool responses/observations do not compute loss (15/15)
- PASS: Every tool call is learned (15/15)
- PASS: The final assistant answer is learned (15/15)
- PASS: No multi-turn assistant action is accidentally masked (15/15)
- PASS: The complete eight-tool schema enters the prompt (15/15)
- PASS: No sample has all labels equal to -100 (15/15)
- PASS: Observation text is absent from trainable labels (15/15)

## Per-sample decoded artifacts

### `agent_star:easy_01djts`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_01djts`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_041h7r`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_041h7r`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_07g2fi`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_07g2fi`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:medium_050mmw`

Directory: `outputs/audit/sft_label_audit/agent_star_medium_050mmw`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:medium_0b5l13`

Directory: `outputs/audit/sft_label_audit/agent_star_medium_0b5l13`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:medium_0kuamj`

Directory: `outputs/audit/sft_label_audit/agent_star_medium_0kuamj`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:hard_00tffl`

Directory: `outputs/audit/sft_label_audit/agent_star_hard_00tffl`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:hard_06hskv`

Directory: `outputs/audit/sft_label_audit/agent_star_hard_06hskv`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:hard_0awg4b`

Directory: `outputs/audit/sft_label_audit/agent_star_hard_0awg4b`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_0ewavl`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_0ewavl`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_0fgakf`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_0fgakf`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_0s362r`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_0s362r`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_0aih6l`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_0aih6l`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_0grbzt`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_0grbzt`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.

### `agent_star:easy_0mak2k`

Directory: `outputs/audit/sft_label_audit/agent_star_easy_0mak2k`

Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.
