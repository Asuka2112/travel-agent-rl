#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent_rl.tools.base import ToolExecutionError
from travel_agent_rl.tools.firecrawl import FirecrawlScrape, FirecrawlSearch
from travel_agent_rl.tools.web_snapshot import WebSnapshotStore


TRANSIENT_CODES = frozenset({"provider_server_error", "timeout"})


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def import_document(store: WebSnapshotStore, document: dict[str, Any]) -> None:
    path = store.documents_dir / document["content_hash"][:2] / f"{document['content_hash']}.json"
    if not path.exists():
        atomic_write(path, canonical(document))


def migrated_base(
    source: dict[str, Any], store: WebSnapshotStore, created_at: str
) -> dict[str, Any]:
    request_key = store.request_key(source["tool_name"], source["request"], store.schema_version)
    row = {
        **source,
        "schema_version": store.schema_version,
        "request_key": request_key,
        "version_provenance": {
            "source_schema_version": source["schema_version"],
            "source_request_key": source["request_key"],
            "version_created_timestamp": created_at,
        },
    }
    return row


def retry_transient(
    source: dict[str, Any],
    store: WebSnapshotStore,
    tools: dict[str, Any],
    created_at: str,
) -> tuple[dict[str, Any], str]:
    tool = tools[source["tool_name"]]
    retry_timestamp = datetime.now(timezone.utc).isoformat()
    reason = f"single retry for transient {source['response_error']['code']}"
    base = migrated_base(source, store, created_at)
    base.pop("response_error", None)
    try:
        captured = tool.capture(source["request"])
    except ToolExecutionError as exc:
        retry_response: dict[str, Any] = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "code": exc.code,
                "message": str(exc),
                "retryable": exc.retryable,
                "provider": exc.provider,
            },
        }
        base.update(
            {
                "retrieval_timestamp": retry_timestamp,
                "backend": getattr(tool, "backend", "unknown"),
                "http_metadata": {},
                "content_hash": None,
                "documents": [],
                "response_error": retry_response["error"],
            }
        )
        outcome = f"error:{exc.code}"
    else:
        documents = store._documents(source["tool_name"], source["request"], captured.data)
        for document in documents:
            import_document(store, document)
        retry_response = {
            "status": "ok",
            "data": captured.data,
            "http_metadata": captured.http_metadata,
            "backend": captured.backend,
        }
        base.update(
            {
                "retrieval_timestamp": retry_timestamp,
                "backend": captured.backend,
                "http_metadata": captured.http_metadata,
                "content_hash": content_hash(captured.data),
                "documents": documents,
                "response_data": captured.data,
            }
        )
        outcome = "success"
    base["repair_audit"] = {
        "reason": reason,
        "url": source.get("url"),
        "query": source.get("query"),
        "original_timestamp": source["retrieval_timestamp"],
        "retry_timestamp": retry_timestamp,
        "original_response": source,
        "retry_response": retry_response,
    }
    return base, outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", type=Path, default=Path("data/intermediate/web_snapshot")
    )
    parser.add_argument("--source-version", type=int, default=1)
    parser.add_argument("--target-version", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:3002/v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_version == args.target_version:
        raise SystemExit("source and target snapshot versions must differ")

    source_root = args.snapshot_path / f"v{args.source_version}"
    source_paths = sorted((source_root / "requests").rglob("*.json"))
    if not source_paths:
        raise SystemExit("source snapshot has no requests")
    target_root = args.snapshot_path / f"v{args.target_version}"
    if list((target_root / "requests").rglob("*.json")):
        raise SystemExit(f"target snapshot already contains requests: {target_root}")

    store = WebSnapshotStore(
        args.snapshot_path,
        run_id=f"snapshot-repair-v{args.target_version}",
        schema_version=args.target_version,
    )
    tools = {
        "web_search": FirecrawlSearch(None, base_url=args.base_url, backend="firecrawl_selfhosted"),
        "web_scraping": FirecrawlScrape(
            None, base_url=args.base_url, backend="firecrawl_selfhosted"
        ),
    }
    created_at = datetime.now(timezone.utc).isoformat()
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in source_paths]
    completed: list[tuple[dict[str, Any], str]] = []
    transient = [
        row for row in sources if (row.get("response_error") or {}).get("code") in TRANSIENT_CODES
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(retry_transient, row, store, tools, created_at): row
            for row in transient
        }
        for future in as_completed(futures):
            completed.append(future.result())

    retry_by_source = {
        row["version_provenance"]["source_request_key"]: (row, outcome)
        for row, outcome in completed
    }
    audit_rows = []
    outcome_counts: dict[str, int] = {}
    for source in sources:
        source_key = source["request_key"]
        if source_key in retry_by_source:
            row, outcome = retry_by_source[source_key]
        else:
            row = migrated_base(source, store, created_at)
            error = source.get("response_error")
            if error:
                outcome = f"retained:{error['code']}"
                row["repair_audit"] = {
                    "reason": "deterministic error retained without retry",
                    "url": source.get("url"),
                    "query": source.get("query"),
                    "original_timestamp": source["retrieval_timestamp"],
                    "retry_timestamp": None,
                    "original_response": source,
                    "retry_response": None,
                }
            else:
                outcome = "copied_success"
                for document in source.get("documents", []):
                    import_document(store, document)
        target_path = store.request_path(row["request_key"])
        atomic_write(target_path, canonical(row))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if "repair_audit" in row:
            audit_rows.append(
                {
                    "source_request_key": source_key,
                    "target_request_key": row["request_key"],
                    "tool_name": row["tool_name"],
                    "outcome": outcome,
                    **row["repair_audit"],
                }
            )

    audit_payload = b"".join(canonical(row) + b"\n" for row in audit_rows)
    atomic_write(args.audit_output, audit_payload)
    summary = {
        "source_version": args.source_version,
        "target_version": args.target_version,
        "source_records": len(sources),
        "target_records": len(list(store.requests_dir.rglob("*.json"))),
        "retry_records": len(transient),
        "audit_records": len(audit_rows),
        "outcomes": dict(sorted(outcome_counts.items())),
        "audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
        "firecrawl_cloud_requests": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
