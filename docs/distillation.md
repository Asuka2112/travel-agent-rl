# Teacher distillation and filtering

Status: implemented, mock-validated, and local-teacher smoke-validated; full dataset
production remains pending.

The distillation runner checkpoints at record granularity. Each successful query is first
written atomically to an immutable raw JSON file. Aggregate JSONL and index files are
reconstructed from those raw records on startup and every configured checkpoint interval,
so interruption between writes is recoverable. Failures remain separate.

Every trajectory records the teacher model, prompt version, UTC timestamps, turns, tools,
normalized exceptions, token usage, latency, and an optional cost estimate. The three
filtering stages are independent: deterministic rule checks, canonical message/JSON
normalization, and a versioned structured LLM judge. Per-query rule decisions, judge
scores/rationales, and aggregate stage statistics are written as separate audit artifacts.

Offline pipeline smoke (the output is marked mock and must not enter research datasets):

```bash
.venv/bin/python scripts/distill_trajectories.py --mock --limit 5
.venv/bin/python scripts/filter_trajectories.py --skip-judge
```

Credentialed generation reads `TEACHER_*`, `JUDGE_*`, and `AMAP_API_KEY` from the job
environment. Formal web capture is self-hosted and needs no Firecrawl Cloud key. Secrets
are never read from repository configuration or written to tracked files.

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

The US-oriented distillation registry is grounded in the checksum-verified
Agent-STAR TravelDatabase at revision `985c8c7ba3e6beb986b3b0a55a2b064d18f2e68e`.
Its cache namespace includes the data revision. A combined public-web budget limits
executed Firecrawl search/scrape actions to two per trajectory; over-budget model
attempts return a structured local error without calling the provider.

On 2026-08-19 a 10-query pilot using that registry generated 10/10 trajectories.
The rule stage accepted 9/10 (one repeated action); the local structured judge accepted
all 9 rule-valid records, for 9/10 final acceptance. Mean latency was 26.27 seconds,
mean cumulative teacher input was 42,127 tokens, and mean output was 757 tokens. The
executed public-web count was at most two per record and the estimated API cost remained
unset because teacher and judge inference were localhost-only. These are pilot results,
not full-dataset metrics.

The frozen production pilots subsequently passed with teacher-v2 plus audited protocol
repair. The 50-query gate generated 50/50 and retained 43/50 after Rule, Format, and the
unchanged structured Judge (86%). The nested 200-query pilot generated 200/200 and retained
172/200 (86%): seven trajectories were rule-rejected for repeated actions and 21 additional
trajectories were below the Judge threshold. There were no invalid tool names or unknown
exceptions. Full metrics are in `reports/distillation_pilot_50.md` and
`reports/distillation_pilot_200.md`. The project is intentionally stopped before the
1,835-query production distillation.
