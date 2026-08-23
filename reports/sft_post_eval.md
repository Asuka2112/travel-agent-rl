# SFT-post Frozen Agent Evaluation

Generated: 2026-08-23T19:49:34.957505+00:00

- Model: `outputs/models/qwen3_4b_sft_merged`
- Adapter: none
- Frozen dev: `data/processed/dev.jsonl`
- Dev SHA-256: `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`
- Template: `qwen3_nothinking` + native `hermes` agent format
- Seed: 20260819
- Tool backend: `frozen_train`
- Public web requests: 0 (the frozen dev contains only local train fixtures)
- Supervised validation loss: N/A (the immutable dev rows contain queries and expected tools, but no assistant labels)

## Metrics

| Metric | Value |
|---|---:|
| queries | 16 |
| normal_answer_completion_rate | 0.000000 |
| valid_tool_call_rate | 1.000000 |
| invalid_tool_call_rate | 0.000000 |
| final_answer_rate | 0.000000 |
| task_completion_rate | 0.000000 |
| mean_task_score | 0.500000 |
| average_turns | 6.000000 |
| average_tool_calls | 6.000000 |
| loop_rate | 0.000000 |
| tool_usage | `{"poi_search": 56, "train_search": 16, "web_search": 24}` |
| unknown_exceptions | 0 |
| average_latency_seconds | 7.881395 |
| model | outputs/models/qwen3_4b_sft_merged |
| adapter | None |
| label | SFT-post |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-23T19:49:34.957335+00:00 |
| tool_backend | frozen_train |
