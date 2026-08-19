from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .base import BaseTool, ToolExecutionError
from .http import JsonHttpClient
from .mock import _object_schema

FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"


@dataclass(frozen=True)
class CapturedToolResult:
    data: dict[str, Any]
    http_metadata: dict[str, Any]
    backend: str


class FirecrawlTool(BaseTool):
    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        *,
        base_url: str = FIRECRAWL_BASE_URL,
        backend: str = "firecrawl_cloud",
    ) -> None:
        if backend == "firecrawl_cloud" and not api_key:
            raise ValueError("FIRECRAWL_API_KEY is required for Firecrawl Cloud")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.client = client or httpx.Client(timeout=httpx.Timeout(30.0))
        self.http = JsonHttpClient(self.client, "firecrawl")

    def _post(self, endpoint: str, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        payload, metadata = self.http.request_json_with_metadata(
            "POST",
            f"{self.base_url}{endpoint}",
            json_body=body,
            headers=headers,
        )
        if payload.get("success") is False:
            raise ToolExecutionError(
                str(payload.get("error", "Firecrawl request failed")),
                code="firecrawl_error",
                provider="firecrawl",
            )
        metadata["base_url"] = self.base_url
        return payload, metadata


class FirecrawlSearch(FirecrawlTool):
    name = "web_search"
    description = "Search the web through Firecrawl."
    argument_schema = _object_schema(
        ["query"],
        {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.capture(arguments).data

    def capture(self, arguments: dict[str, Any]) -> CapturedToolResult:
        payload, metadata = self._post(
            "/search",
            {"query": arguments["query"], "limit": arguments.get("limit", 5)},
        )
        return CapturedToolResult(
            data={"results": payload.get("data", [])},
            http_metadata=metadata,
            backend=self.backend,
        )


class FirecrawlScrape(FirecrawlTool):
    name = "web_scraping"
    description = "Scrape a public web URL through Firecrawl."
    argument_schema = _object_schema(
        ["url"],
        {"url": {"type": "string", "pattern": "^https?://"}},
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.capture(arguments).data

    def capture(self, arguments: dict[str, Any]) -> CapturedToolResult:
        payload, metadata = self._post(
            "/scrape",
            {"url": arguments["url"], "formats": ["markdown"]},
        )
        return CapturedToolResult(
            data={"document": payload.get("data", {})},
            http_metadata=metadata,
            backend=self.backend,
        )
