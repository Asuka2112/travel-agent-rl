#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from travel_agent_rl.distill import TrajectoryRecord
from travel_agent_rl.tools.base import ToolExecutionError
from travel_agent_rl.tools.firecrawl import FirecrawlScrape, FirecrawlSearch
from travel_agent_rl.tools.web_snapshot import SnapshotWebTool, WebSnapshotStore


def _calls(records: list[TrajectoryRecord]) -> list[tuple[str, dict[str, Any]]]:
    unique: dict[str, tuple[str, dict[str, Any]]] = {}
    for record in records:
        for call, observation in zip(record.tool_calls, record.tool_observations, strict=True):
            if call.name not in {"web_search", "web_scraping"}:
                continue
            error = observation.get("error")
            error_code = error.get("code") if isinstance(error, dict) else None
            if error_code == "tool_budget_exceeded":
                continue
            key = WebSnapshotStore.request_key(call.name, call.arguments)
            unique[key] = (call.name, call.arguments)
    return [unique[key] for key in sorted(unique)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--snapshot-path", type=Path, default=Path("data/intermediate/web_snapshot")
    )
    parser.add_argument("--mode", choices=("online_capture", "replay_only"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        TrajectoryRecord.model_validate_json(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    store = WebSnapshotStore(args.snapshot_path, run_id=args.run_id)
    tools = {
        "web_search": SnapshotWebTool(
            FirecrawlSearch(
                None,
                base_url=args.base_url,
                backend="firecrawl_selfhosted",
            ),
            store,
            mode=args.mode,
        ),
        "web_scraping": SnapshotWebTool(
            FirecrawlScrape(
                None,
                base_url=args.base_url,
                backend="firecrawl_selfhosted",
            ),
            store,
            mode=args.mode,
        ),
    }
    calls = _calls(records)
    provider_errors: dict[str, int] = {}
    snapshot_misses = 0
    returned = 0
    for name, arguments in calls:
        try:
            tools[name].run(arguments)
        except ToolExecutionError as exc:
            provider_errors[exc.code] = provider_errors.get(exc.code, 0) + 1
            if exc.code == "snapshot_miss":
                snapshot_misses += 1
        else:
            returned += 1

    events = [
        json.loads(line)
        for line in store.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("run_id") == args.run_id
    ]
    summary = {
        "mode": args.mode,
        "run_id": args.run_id,
        "unique_requests": len(calls),
        "returned": returned,
        "provider_errors_replayed": provider_errors,
        "snapshot_misses": snapshot_misses,
        "snapshot_hits": sum(bool(event.get("snapshot_hit")) for event in events),
        "snapshot_not_preexisting": sum(not bool(event.get("snapshot_hit")) for event in events),
        "live_public_web_requests": sum(bool(event.get("live_request")) for event in events),
        "firecrawl_cloud_requests": sum(
            event.get("backend") == "firecrawl_cloud" for event in events
        ),
    }
    _write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.mode == "replay_only" and (
        snapshot_misses
        or summary["live_public_web_requests"]
        or summary["firecrawl_cloud_requests"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
