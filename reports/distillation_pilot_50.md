# Distillation pilot: 50 frozen queries

## Acceptance summary

- Generation: 50/50 (100.00%)
- Final pass: 43/50 (86.00%)
- Average turns / model calls: 9.26 / 10.34
- Average generation latency: 30.89 seconds/query
- Repeated-action records: 1 (2.00%)
- Invalid tool calls: 0
- Audited protocol corrections: 3 across 3 records
- Exception records: 0
- Firecrawl Cloud calls/credits: 0 / 0

## Tool calls

| Tool | Calls |
| --- | ---: |
| `flight_search` | 90 |
| `nearby_search` | 16 |
| `poi_search` | 246 |
| `route_planning` | 36 |
| `train_search` | 0 |
| `weather` | 2 |
| `web_scraping` | 19 |
| `web_search` | 55 |

## Filtering failures

- Rule: `{"repeated_action": 1}`
- Judge: `{"judge_below_threshold": 6}`
- Tool errors: `{"answer_required": 1, "provider_client_error": 1, "provider_server_error": 8, "repeated_action": 1, "timeout": 1, "tool_budget_exceeded": 28}`

## Web snapshot

- Capture events: 100; hits 44; misses 56; hit rate 44.00%
- Live public-web requests during capture: 56
- Replay verification: `{"firecrawl_cloud_requests": 0, "live_public_web_requests": 0, "mode": "replay_only", "provider_errors_replayed": {"provider_server_error": 4, "timeout": 2}, "returned": 39, "run_id": "pilot_50_v3_replay_verify", "snapshot_hits": 45, "snapshot_misses": 0, "snapshot_not_preexisting": 0, "unique_requests": 45}`

## Projection

At concurrency 8, the measured mean generation latency projects to approximately 1.97 hours for 1,835 queries. This excludes the separate Judge pass and operational buffer.
