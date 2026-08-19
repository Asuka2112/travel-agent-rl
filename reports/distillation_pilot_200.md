# Distillation pilot: 200 frozen queries

## Acceptance summary

- Generation: 200/200 (100.00%)
- Final pass: 172/200 (86.00%)
- Average turns / model calls: 9.78 / 10.88
- Average generation latency: 28.64 seconds/query
- Repeated-action records: 7 (3.50%)
- Invalid tool calls: 0
- Audited protocol corrections: 15 across 12 records
- Exception records: 0
- Firecrawl Cloud calls/credits: 0 / 0

## Tool calls

| Tool | Calls |
| --- | ---: |
| `flight_search` | 338 |
| `nearby_search` | 47 |
| `poi_search` | 1023 |
| `route_planning` | 151 |
| `train_search` | 0 |
| `weather` | 9 |
| `web_scraping` | 103 |
| `web_search` | 291 |

## Filtering failures

- Rule: `{"repeated_action": 7}`
- Judge: `{"judge_below_threshold": 21}`
- Tool errors: `{"answer_required": 6, "provider_client_error": 3, "provider_server_error": 29, "repeated_action": 7, "timeout": 8, "tool_budget_exceeded": 138}`

## Web snapshot

- Capture events: 338; hits 94; misses 244; hit rate 27.81%
- Live public-web requests during capture: 244
- Replay verification: `{"firecrawl_cloud_requests": 0, "live_public_web_requests": 0, "mode": "replay_only", "provider_errors_replayed": {"provider_client_error": 2, "provider_server_error": 23, "timeout": 9}, "returned": 204, "run_id": "pilot_200_v3_replay_verify", "snapshot_hits": 238, "snapshot_misses": 0, "snapshot_not_preexisting": 0, "unique_requests": 238}`

## Projection

At concurrency 8, the measured mean generation latency projects to approximately 1.82 hours for 1,835 queries. This excludes the separate Judge pass and operational buffer.
