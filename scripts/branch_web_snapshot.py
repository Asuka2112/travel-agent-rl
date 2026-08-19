#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent_rl.tools.web_snapshot import WebSnapshotStore


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", type=Path, default=Path("data/intermediate/web_snapshot")
    )
    parser.add_argument("--source-version", type=int, required=True)
    parser.add_argument("--target-version", type=int, required=True)
    args = parser.parse_args()
    if args.source_version == args.target_version:
        raise SystemExit("source and target snapshot versions must differ")

    source_root = args.snapshot_path / f"v{args.source_version}"
    source_paths = sorted((source_root / "requests").rglob("*.json"))
    if not source_paths:
        raise SystemExit("source snapshot has no requests")
    store = WebSnapshotStore(
        args.snapshot_path,
        run_id=f"branch-v{args.source_version}-to-v{args.target_version}",
        schema_version=args.target_version,
    )
    if list(store.requests_dir.rglob("*.json")):
        raise SystemExit(f"target snapshot already contains requests: v{args.target_version}")

    branched_at = datetime.now(timezone.utc).isoformat()
    records = 0
    errors = 0
    for source_path in source_paths:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        request_key = store.request_key(
            source["tool_name"], source["request"], store.schema_version
        )
        row = {
            **source,
            "schema_version": store.schema_version,
            "request_key": request_key,
            "branch_provenance": {
                "source_schema_version": source["schema_version"],
                "source_request_key": source["request_key"],
                "branched_at": branched_at,
            },
        }
        atomic_write(store.request_path(request_key), canonical(row))
        for document in source.get("documents", []):
            document_path = (
                store.documents_dir
                / document["content_hash"][:2]
                / f"{document['content_hash']}.json"
            )
            if not document_path.exists():
                atomic_write(document_path, canonical(document))
        records += 1
        errors += int("response_error" in source)
    summary = {
        "source_version": args.source_version,
        "target_version": args.target_version,
        "records": records,
        "error_records": errors,
        "source_request_tree_sha256": hashlib.sha256(
            canonical(
                [
                    {
                        "path": str(path.relative_to(args.snapshot_path)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in source_paths
                ]
            )
        ).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
