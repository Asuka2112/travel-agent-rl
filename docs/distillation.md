# Teacher distillation and filtering

Status: implemented, mock-validated, and local-teacher smoke-validated; full dataset
production remains pending.

The distillation runner is append-only at record granularity. On restart it reads the
existing trajectory JSONL, skips completed `query_id` values, and appends each new result
only after a complete trajectory has been validated. Failures are written separately, so
successful work is never discarded when an endpoint or process stops.

Every trajectory records the teacher model, prompt version, UTC timestamps, turns, tools,
normalized exceptions, token usage, latency, and an optional cost estimate. The three
filtering stages are independent: deterministic rule checks, canonical message/JSON
normalization, and a versioned structured LLM judge.

Offline pipeline smoke (the output is marked mock and must not enter research datasets):

```bash
.venv/bin/python scripts/distill_trajectories.py --mock --limit 5
.venv/bin/python scripts/filter_trajectories.py --skip-judge
```

Credentialed generation reads `TEACHER_*`, `JUDGE_*`, `AMAP_API_KEY`, and
`FIRECRAWL_API_KEY` from the job environment. Secrets are never read from repository
configuration or written to tracked files.

The reproducible local path uses the fixed
`Qwen/Qwen3-30B-A3B-Instruct-2507` revision recorded in
`configs/distill/local_qwen30b.yaml`. Start the localhost-only TP=4 endpoint with
`scripts/serve_local_teacher.sh`, then set `TEACHER_BASE_URL`, `TEACHER_API_KEY`, and
`TEACHER_MODEL` for the distillation process. The API-key value is a non-secret local
placeholder because the endpoint binds only to `127.0.0.1`.

On 2026-08-19 one real 13-tool-turn trajectory completed against live AMap and
Firecrawl plus the deterministic transport simulators. It passed the rule, canonical,
and local structured-judge stages using `configs/filter/local_qwen30b.yaml`. This is a
smoke result, not the claimed full SFT/RL dataset. Model-visible tool observations are
bounded at 2,000 characters while full provider responses remain in the environment's
disk cache, preventing repeated search/scrape results from overflowing the 32K teacher
context.
