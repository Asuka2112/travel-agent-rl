# sft_smoke_checkpoint_10_full10 Frozen Agent Evaluation

Generated: 2026-08-23T18:47:34.477389+00:00

- Model: `models/Qwen3-4B-Instruct-2507`
- Adapter: `outputs/train/sft_smoke_32/v0-20260824-023302/checkpoint-10`
- Frozen dev: `data/processed/dev.jsonl`
- Dev SHA-256: `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`
- Template: `qwen3_nothinking` + native `hermes` agent format
- Seed: 20260819
- Public web requests: 0 (the frozen dev contains only local train fixtures)
- Supervised validation loss: N/A (the immutable dev rows contain queries and expected tools, but no assistant labels)

## Metrics

| Metric | Value |
|---|---:|
| queries | 10 |
| normal_answer_completion_rate | 0.600000 |
| valid_tool_call_rate | 0.961538 |
| invalid_tool_call_rate | 0.038462 |
| final_answer_rate | 0.600000 |
| task_completion_rate | 0.600000 |
| mean_task_score | 0.775000 |
| average_turns | 3.200000 |
| average_tool_calls | 2.500000 |
| loop_rate | 0.100000 |
| tool_usage | `{"poi_search": 15, "train_search": 10}` |
| unknown_exceptions | 0 |
| average_latency_seconds | 19.120003 |
| model | models/Qwen3-4B-Instruct-2507 |
| adapter | outputs/train/sft_smoke_32/v0-20260824-023302/checkpoint-10 |
| label | sft_smoke_checkpoint_10_full10 |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-23T18:47:34.477207+00:00 |
