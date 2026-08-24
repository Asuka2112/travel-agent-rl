# SFT Frozen Agent Evaluation

Generated: 2026-08-24T19:22:09.533831+00:00

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
| normal_answer_completion_rate | 0.687500 |
| valid_tool_call_rate | 1.000000 |
| invalid_tool_call_rate | 0.000000 |
| final_answer_rate | 0.687500 |
| task_completion_rate | 0.687500 |
| constraint_satisfaction_rate | 0.687500 |
| mean_task_score | 0.843750 |
| average_turns | 7.875000 |
| average_tool_calls | 7.187500 |
| loop_rate | 0.312500 |
| tool_efficiency | 0.656250 |
| tool_usage | `{"flight_search": 1, "nearby_search": 11, "poi_search": 47, "route_planning": 1, "train_search": 16, "web_scraping": 1, "web_search": 38}` |
| unknown_exceptions | 0 |
| replay_misses | 30 |
| public_network_requests | 0 |
| average_latency_seconds | 18.350112 |
| model | outputs/models/qwen3_4b_sft_merged |
| adapter | None |
| label | SFT |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-24T19:22:09.533676+00:00 |
| tool_backend | full_replay |
