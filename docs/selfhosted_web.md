# Self-hosted web capture and replay

Formal distillation uses self-hosted Firecrawl v1.7.0 backed by self-hosted SearXNG.
Firecrawl Cloud credentials are neither required nor passed to this stack.

## Runtime topology

| Service | Fixed source | Local endpoint |
| --- | --- | --- |
| Firecrawl API/worker | `firecrawl/firecrawl` tag `v1.7.0`, commit `b57d5f2c4d600b2bf8b34507ee4c874c26c07bb0` | `127.0.0.1:3002` |
| SearXNG | commit `374939b888c8644b408b793fe42d584454631cec` | `127.0.0.1:3080` |
| Playwright service | Firecrawl v1.7.0 plus loopback-bind patch | `127.0.0.1:3081` |
| Valkey | tag `8.1.3`, commit `d063dff523340e5a075f7ddc436e0c2a43d04061` | `127.0.0.1:3079` |

The standard isolated deployment is `deploy/web/docker-compose.yaml`. It does not use or
replace system Redis, Docker networks, or volumes. On the current server the user cannot
access the system Docker socket, so the same components run from `.runtime/` under the
project account. `scripts/manage_selfhost_web.py` starts/stops only verified project PIDs,
uses no `sudo`, and refuses occupied ports. All endpoints are loopback-only.

```bash
python3 scripts/manage_selfhost_web.py status
python3 scripts/manage_selfhost_web.py start
python3 scripts/manage_selfhost_web.py stop
```

Upstream Firecrawl v1.7.0 binds its Playwright helper on all interfaces. The tracked,
idempotent `scripts/patch_firecrawl_playwright.py` makes `HOST=127.0.0.1` effective for
the native runtime. Project-local logs and PID/state files are under `.runtime/` and are
not tracked.

## Agent-facing contract

The Agent still sees the same `web_search` and `web_scraping` schemas. Backend selection
is out of band:

```bash
export WEB_BACKEND=selfhosted
export FIRECRAWL_BASE_URL=http://127.0.0.1:3002/v1
export WEB_MODE=online_capture
export WEB_RUN_ID=pilot_50_capture
```

`WEB_MODE` supports:

- `online_capture`: return an existing snapshot or perform one real self-hosted fetch;
- `replay_only`: return only versioned snapshots and fail closed on a miss;
- `mock`: deterministic test fixtures, excluded from research data.

Formal training and GRPO use `WEB_MODE=replay_only`. `online_capture` is a separate data
acquisition phase.

## Snapshot format

Request keys hash the schema version, tool name, and canonical JSON arguments. Request
records live under `data/intermediate/web_snapshot/v1/requests`; normalized documents are
content-addressed under `.../documents`. Each record includes query/URL, title and
extracted content, normalized text, retrieval timestamp, SHA-256 content hashes, backend,
and HTTP metadata. An append-only event log records hit/miss and whether a live request
occurred. Per-key file locks prevent duplicate concurrent captures.

Snapshot misses in `replay_only` return a structured `snapshot_miss` error without network
fallback. Content is never synthesized: capture records contain the actual response from
the self-hosted Firecrawl request and preserve its source URL.

For the 200-query pilot, all 238 unique trajectory Web requests replayed from an
intentionally unreachable backend address with zero snapshot misses and zero live calls.
The broader accumulated pilot snapshot was frozen in
`data/manifests/web_snapshot_pilot_200_v1_manifest.json`: 349 request records, 335
content-addressed documents, 34 replayable provider errors, and content SHA-256
`0bd3ff0e8bad874cbb2b1e034dd5a8255cc474bea4a3473da0cfafbb87d83d73`.
