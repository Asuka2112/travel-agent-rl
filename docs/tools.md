# Tool providers and credential gate

## Unified tools

| Tool | Production provider | Offline path |
| --- | --- | --- |
| `poi_search` | AMap `/v3/place/text` | deterministic mock |
| `nearby_search` | AMap `/v3/place/around` | deterministic mock |
| `weather` | AMap `/v3/weather/weatherInfo` | deterministic mock |
| `route_planning` | AMap v3 walking/driving/transit | deterministic mock |
| `web_search` | Firecrawl `/v2/search` | deterministic mock |
| `web_scraping` | Firecrawl `/v2/scrape` | deterministic mock |
| `flight_search` | local versioned inventory | same local simulator |
| `train_search` | local versioned inventory | same local simulator |

For Agent-STAR distillation, `build_distillation_registry` replaces US-oriented
POI, nearby, flight, and ground-route calls with the fixed-revision
Agent-STAR TravelDatabase. Weather remains AMap-backed, Firecrawl remains an
optional public-web fallback, and train search remains a deterministic fixture.
Each provider/data revision gets a distinct tool-cache namespace so cached
responses cannot cross provider boundaries.

Official references:

- https://lbs.amap.com/api/webservice/guide/api/search/
- https://lbs.amap.com/api/webservice/guide/api/weatherinfo
- https://docs.firecrawl.dev/api-reference/endpoint/search
- https://docs.firecrawl.dev/api-reference/endpoint/scrape

## Credentials

Set `AMAP_API_KEY` and `FIRECRAWL_API_KEY` in the job environment. Keys are
never accepted in configuration files or serialized tool responses. Run:

```bash
source .venv/bin/activate
python scripts/smoke_live_tools.py
```

The smoke makes paid/quota-bearing provider calls and must not run until both
credentials and intended quota use are confirmed. Missing credentials fail
before a request is sent.

## Reliability

HTTP timeouts, network failures, 429 responses, 5xx responses, invalid JSON,
and provider-declared failures become structured errors. Retryable failures are
retried up to the environment limit. Successful responses can be atomically
cached on disk, and every test can run with provider-free offline fixtures.

## Geographic risk

Most Agent-STAR/TravelPlanner queries are US-oriented. AMap overseas Web
Service access is a premium permission and may not be available on a domestic
key. Until credentialed coverage is measured, non-China travel queries must use
web evidence or a license-cleared local travel database; this is an explicit
M2/M3 risk, not assumed provider coverage.

The local TravelDatabase path now mitigates that risk for 2022 US tasks. It
does not make AMap an overseas provider and does not replace live-provider
coverage tests for China-oriented traffic or weather use cases.
