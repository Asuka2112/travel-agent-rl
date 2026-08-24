# Base Frozen Agent Evaluation

Generated: 2026-08-24T19:16:33.719715+00:00

- Model: `models/Qwen3-4B-Instruct-2507`
- Adapter: none
- Frozen dev: `data/processed/dev.jsonl`
- Dev SHA-256: `66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712`
- Template: `qwen3_nothinking` + native `hermes` agent format
- Seed: 20260819
- Tool backend: `full_replay`
- Public web requests: 0 (the frozen dev contains only local train fixtures)
- Supervised validation loss: N/A (the immutable dev rows contain queries and expected tools, but no assistant labels)

## Metrics

| Metric | Value |
|---|---:|
| queries | 16 |
| normal_answer_completion_rate | 0.875000 |
| valid_tool_call_rate | 0.888889 |
| invalid_tool_call_rate | 0.111111 |
| final_answer_rate | 0.875000 |
| task_completion_rate | 0.875000 |
| constraint_satisfaction_rate | 0.875000 |
| mean_task_score | 0.906250 |
| average_turns | 2.000000 |
| average_tool_calls | 1.000000 |
| loop_rate | 0.000000 |
| tool_efficiency | 0.937500 |
| tool_usage | `{"train_search": 16}` |
| unknown_exceptions | 0 |
| replay_misses | 0 |
| public_network_requests | 0 |
| average_latency_seconds | 19.509752 |
| model | models/Qwen3-4B-Instruct-2507 |
| adapter | None |
| label | Base |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-24T19:16:33.719529+00:00 |
| tool_backend | full_replay |
