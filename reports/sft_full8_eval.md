# SFT-full8 Frozen Agent Evaluation

Generated: 2026-08-23T20:24:05.691218+00:00

- Model: `outputs/models/qwen3_4b_sft_merged`
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
| normal_answer_completion_rate | 0.625000 |
| valid_tool_call_rate | 1.000000 |
| invalid_tool_call_rate | 0.000000 |
| final_answer_rate | 0.625000 |
| task_completion_rate | 0.625000 |
| constraint_satisfaction_rate | 0.625000 |
| mean_task_score | 0.812500 |
| average_turns | 8.125000 |
| average_tool_calls | 7.500000 |
| loop_rate | 0.250000 |
| tool_usage | `{"flight_search": 1, "nearby_search": 10, "poi_search": 46, "route_planning": 1, "train_search": 16, "web_search": 46}` |
| unknown_exceptions | 0 |
| replay_misses | 42 |
| public_network_requests | 0 |
| average_latency_seconds | 19.570652 |
| model | outputs/models/qwen3_4b_sft_merged |
| adapter | None |
| label | SFT-full8 |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-23T20:24:05.691035+00:00 |
| tool_backend | full_replay |
