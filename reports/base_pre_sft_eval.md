# Base-pre-SFT Frozen Agent Evaluation

Generated: 2026-08-23T17:37:20.520263+00:00

- Model: `models/Qwen3-4B-Instruct-2507`
- Frozen dev: `data/processed/dev.jsonl`
- Dev SHA-256: `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`
- Template: `qwen3_nothinking` + native `hermes` agent format
- Seed: 20260819
- Public web requests: 0 (the frozen dev contains only local train fixtures)
- Supervised validation loss: N/A (the immutable dev rows contain queries and expected tools, but no assistant labels)

## Metrics

| Metric | Value |
|---|---:|
| queries | 16 |
| normal_answer_completion_rate | 1.000000 |
| valid_tool_call_rate | 1.000000 |
| invalid_tool_call_rate | 0.000000 |
| final_answer_rate | 1.000000 |
| task_completion_rate | 1.000000 |
| mean_task_score | 1.000000 |
| average_turns | 2.000000 |
| average_tool_calls | 1.000000 |
| loop_rate | 0.000000 |
| tool_usage | `{"train_search": 16}` |
| unknown_exceptions | 0 |
| average_latency_seconds | 20.777075 |
| model | models/Qwen3-4B-Instruct-2507 |
| label | Base-pre-SFT |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-23T17:37:20.520091+00:00 |
