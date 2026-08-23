# SFT Trajectory-context Smoke Evaluation

Generated: 2026-08-23T19:52:47.247090+00:00

This diagnostic uses immutable teacher observations as context; it does not replace the free-running frozen Agent evaluation. It isolates three behaviors: initial legal tool selection, legal continuation after an observation, and final-answer generation after the complete evidence trajectory.

- Data: `data/intermediate/sft/smoke_320.jsonl`
- Data SHA-256: `1fc3bed9e9bed6e9d2e4399315bc1a4c126130a9b8aa5a34749d9994c1e00ade`
- Adapter: `None`
- Queries: 10
- Total checks: 30
- Protocol pass rate: 96.67%
- Reference action match rate: 76.67%
- Invalid outputs: 0

| Checkpoint | Count | Protocol pass | Reference match | Invalid |
|---|---:|---:|---:|---:|
| initial | 10 | 100.00% | 80.00% | 0 |
| post_first_observation | 10 | 100.00% | 60.00% | 0 |
| pre_final | 10 | 90.00% | 90.00% | 0 |
