# sft_smoke_checkpoint_20_probe2 Frozen Agent Evaluation

Generated: 2026-08-23T18:43:58.856816+00:00

- Model: `models/Qwen3-4B-Instruct-2507`
- Adapter: `outputs/train/sft_smoke_32/v0-20260824-023302/checkpoint-20`
- Frozen dev: `data/processed/dev.jsonl`
- Dev SHA-256: `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`
- Template: `qwen3_nothinking` + native `hermes` agent format
- Seed: 20260819
- Public web requests: 0 (the frozen dev contains only local train fixtures)
- Supervised validation loss: N/A (the immutable dev rows contain queries and expected tools, but no assistant labels)

## Metrics

| Metric | Value |
|---|---:|
| queries | 2 |
| normal_answer_completion_rate | 0.000000 |
| valid_tool_call_rate | 1.000000 |
| invalid_tool_call_rate | 0.000000 |
| final_answer_rate | 0.000000 |
| task_completion_rate | 0.000000 |
| mean_task_score | 0.500000 |
| average_turns | 6.000000 |
| average_tool_calls | 6.000000 |
| loop_rate | 0.000000 |
| tool_usage | `{"poi_search": 6, "train_search": 2, "web_search": 4}` |
| unknown_exceptions | 0 |
| average_latency_seconds | 14.309215 |
| model | models/Qwen3-4B-Instruct-2507 |
| adapter | outputs/train/sft_smoke_32/v0-20260824-023302/checkpoint-20 |
| label | sft_smoke_checkpoint_20_probe2 |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-23T18:43:58.856631+00:00 |
