#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", type=Path, default=Path("data/intermediate/web_snapshot")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    version_root = args.snapshot_path / f"v{args.version}"
    request_paths = sorted((version_root / "requests").rglob("*.json"))
    document_paths = sorted((version_root / "documents").rglob("*.json"))
    if not request_paths:
        raise SystemExit("snapshot contains no request records")

    file_entries = [
        {
            "path": str(path.relative_to(args.snapshot_path)),
            "sha256": sha256_bytes(path.read_bytes()),
            "size": path.stat().st_size,
        }
        for path in [*request_paths, *document_paths]
    ]
    requests = [json.loads(path.read_text(encoding="utf-8")) for path in request_paths]
    retrievals = sorted(row["retrieval_timestamp"] for row in requests)
    frozen_at = (
        json.loads(args.output.read_text(encoding="utf-8"))["frozen_at"]
        if args.output.is_file()
        else datetime.now(timezone.utc).isoformat()
    )
    manifest = {
        "immutable": True,
        "snapshot_schema_version": args.version,
        "frozen_at": frozen_at,
        "snapshot_content_sha256": sha256_bytes(canonical(file_entries)),
        "request_records": len(request_paths),
        "document_records": len(document_paths),
        "successful_requests": sum("response_data" in row for row in requests),
        "error_requests": sum("response_error" in row for row in requests),
        "retrieval_started_at": retrievals[0],
        "retrieval_finished_at": retrievals[-1],
        "backend_distribution": dict(sorted(Counter(row["backend"] for row in requests).items())),
        "tool_distribution": dict(sorted(Counter(row["tool_name"] for row in requests).items())),
        "files": file_entries,
    }
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    if args.output.exists():
        if args.output.read_bytes() != payload:
            raise SystemExit(f"refusing to overwrite different frozen manifest: {args.output}")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
