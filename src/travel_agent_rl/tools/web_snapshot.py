from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from .base import BaseTool, ToolExecutionError
from .firecrawl import CapturedToolResult

WebMode = Literal["online_capture", "replay_only", "mock"]
SNAPSHOT_SCHEMA_VERSION = 1


class CapturableTool(Protocol):
    name: str
    description: str
    argument_schema: dict[str, Any]

    def capture(self, arguments: dict[str, Any]) -> CapturedToolResult: ...


def normalize_web_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n")]
    normalized: list[str] = []
    for line in lines:
        if line or (normalized and normalized[-1]):
            normalized.append(line)
    return "\n".join(normalized).strip()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


class WebSnapshotStore:
    def __init__(self, root: Path, *, run_id: str = "unspecified") -> None:
        self.root = root
        self.run_id = run_id
        self.requests_dir = root / f"v{SNAPSHOT_SCHEMA_VERSION}" / "requests"
        self.documents_dir = root / f"v{SNAPSHOT_SCHEMA_VERSION}" / "documents"
        self.locks_dir = root / f"v{SNAPSHOT_SCHEMA_VERSION}" / "locks"
        self.events_path = root / f"v{SNAPSHOT_SCHEMA_VERSION}" / "events.jsonl"
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def request_key(tool_name: str, arguments: dict[str, Any]) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
            )
        )

    def request_path(self, key: str) -> Path:
        return self.requests_dir / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.request_path(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def record_event(self, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.locks_dir / "events.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "run_id": self.run_id,
                            **event,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            fcntl.flock(lock, fcntl.LOCK_UN)

    def capture(
        self,
        tool: CapturableTool,
        arguments: dict[str, Any],
        mode: WebMode,
    ) -> tuple[dict[str, Any], bool]:
        key = self.request_key(tool.name, arguments)
        lock_path = self.locks_dir / f"{key}.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = self.load(key)
            if existing is not None:
                response_error = existing.get("response_error")
                self.record_event(
                    {
                        "tool_name": tool.name,
                        "request_key": key,
                        "mode": mode,
                        "snapshot_hit": True,
                        "live_request": False,
                        "backend": existing["backend"],
                        "error": response_error.get("code") if response_error else None,
                    }
                )
                if response_error:
                    raise ToolExecutionError(
                        response_error["message"],
                        code=response_error["code"],
                        retryable=bool(response_error.get("retryable", False)),
                        provider=response_error.get("provider"),
                    )
                return existing["response_data"], True
            if mode == "replay_only":
                self.record_event(
                    {
                        "tool_name": tool.name,
                        "request_key": key,
                        "mode": mode,
                        "snapshot_hit": False,
                        "live_request": False,
                        "backend": "none",
                        "error": "snapshot_miss",
                    }
                )
                raise ToolExecutionError(
                    f"no versioned web snapshot for {tool.name} request {key}",
                    code="snapshot_miss",
                    provider="web_snapshot",
                )
            try:
                captured = tool.capture(arguments)
            except ToolExecutionError as exc:
                error = {
                    "type": type(exc).__name__,
                    "code": exc.code,
                    "message": str(exc),
                    "retryable": exc.retryable,
                    "provider": exc.provider,
                }
                record = {
                    "schema_version": SNAPSHOT_SCHEMA_VERSION,
                    "request_key": key,
                    "tool_name": tool.name,
                    "request": arguments,
                    "query": arguments.get("query"),
                    "url": arguments.get("url"),
                    "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                    "backend": getattr(tool, "backend", "unknown"),
                    "http_metadata": {},
                    "content_hash": None,
                    "documents": [],
                    "response_error": error,
                }
                _atomic_write(self.request_path(key), _canonical_json(record))
                self.record_event(
                    {
                        "tool_name": tool.name,
                        "request_key": key,
                        "mode": mode,
                        "snapshot_hit": False,
                        "live_request": True,
                        "backend": getattr(tool, "backend", "unknown"),
                        "error": exc.code,
                    }
                )
                raise
            documents = self._documents(tool.name, arguments, captured.data)
            for document in documents:
                document_path = (
                    self.documents_dir
                    / document["content_hash"][:2]
                    / (f"{document['content_hash']}.json")
                )
                if not document_path.exists():
                    _atomic_write(document_path, _canonical_json(document))
            record = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "request_key": key,
                "tool_name": tool.name,
                "request": arguments,
                "query": arguments.get("query"),
                "url": arguments.get("url"),
                "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": captured.backend,
                "http_metadata": captured.http_metadata,
                "content_hash": _sha256(_canonical_json(captured.data)),
                "documents": documents,
                "response_data": captured.data,
            }
            _atomic_write(self.request_path(key), _canonical_json(record))
            self.record_event(
                {
                    "tool_name": tool.name,
                    "request_key": key,
                    "mode": mode,
                    "snapshot_hit": False,
                    "live_request": True,
                    "backend": captured.backend,
                }
            )
            return captured.data, False

    @staticmethod
    def _documents(
        tool_name: str, arguments: dict[str, Any], data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if tool_name == "web_search":
            raw_documents = data.get("results", [])
        else:
            raw_documents = [data.get("document", {})]
        documents: list[dict[str, Any]] = []
        for raw in raw_documents:
            if not isinstance(raw, dict):
                continue
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            url = raw.get("url") or metadata.get("sourceURL") or arguments.get("url")
            title = raw.get("title") or metadata.get("title")
            content = raw.get("markdown") or raw.get("description") or raw.get("content") or ""
            normalized = normalize_web_text(str(content))
            documents.append(
                {
                    "query": arguments.get("query"),
                    "url": url,
                    "title": title,
                    "extracted_content": str(content),
                    "normalized_text": normalized,
                    "content_hash": _sha256(normalized.encode("utf-8")),
                    "http_metadata": {
                        key: metadata[key]
                        for key in ("statusCode", "contentType", "sourceURL")
                        if key in metadata
                    },
                }
            )
        return documents


class SnapshotWebTool(BaseTool):
    cacheable = False

    def __init__(
        self,
        provider: CapturableTool,
        store: WebSnapshotStore,
        *,
        mode: WebMode,
    ) -> None:
        if mode == "mock":
            raise ValueError("mock mode uses deterministic mock web tools")
        self.provider = provider
        self.store = store
        self.mode = mode
        self.name = provider.name
        self.description = provider.description
        self.argument_schema = provider.argument_schema

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        data, hit = self.store.capture(self.provider, arguments, self.mode)
        return {**data, "_snapshot": {"hit": hit, "mode": self.mode}}
