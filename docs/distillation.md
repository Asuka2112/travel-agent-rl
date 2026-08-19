# Teacher distillation and filtering

Status: implemented and mock-validated; real teacher/judge run is credential-gated.

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
