# SFT-Smoke-40 Frozen Agent Evaluation

Generated: 2026-08-23T19:16:55.468873+00:00

- Model: `models/Qwen3-4B-Instruct-2507`
- Adapter: `outputs/train/sft_smoke_320_ddp_lr2e5_resume40/v0-20260824-030704/checkpoint-40`
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
| queries | 10 |
| normal_answer_completion_rate | 0.200000 |
| valid_tool_call_rate | 1.000000 |
| invalid_tool_call_rate | 0.000000 |
| final_answer_rate | 0.200000 |
| task_completion_rate | 0.200000 |
| mean_task_score | 0.600000 |
| average_turns | 5.200000 |
| average_tool_calls | 5.000000 |
| loop_rate | 0.000000 |
| tool_usage | `{"poi_search": 22, "train_search": 10, "web_search": 18}` |
| unknown_exceptions | 0 |
| average_latency_seconds | 14.333336 |
| model | models/Qwen3-4B-Instruct-2507 |
| adapter | outputs/train/sft_smoke_320_ddp_lr2e5_resume40/v0-20260824-030704/checkpoint-40 |
| label | SFT-Smoke-40 |
| dev_sha256 | 66d8d8f5ce4ac9e42a265da3f2c28634a54d14a3628150e36b008bb1496c5712 |
| generated_at | 2026-08-23T19:16:55.468694+00:00 |
| tool_backend | frozen_train |
