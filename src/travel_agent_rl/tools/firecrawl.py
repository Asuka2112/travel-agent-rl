from __future__ import annotations

from typing import Any

import httpx

from .base import BaseTool, ToolExecutionError
from .http import JsonHttpClient
from .mock import _object_schema

FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"


class FirecrawlTool(BaseTool):
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY is required")
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=httpx.Timeout(30.0))
        self.http = JsonHttpClient(self.client, "firecrawl")

    def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = self.http.request_json(
            "POST",
            f"{FIRECRAWL_BASE_URL}{endpoint}",
            json_body=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if payload.get("success") is False:
            raise ToolExecutionError(
                str(payload.get("error", "Firecrawl request failed")),
                code="firecrawl_error",
                provider="firecrawl",
            )
        return payload


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
        payload = self._post(
            "/search",
            {"query": arguments["query"], "limit": arguments.get("limit", 5)},
        )
        return {"results": payload.get("data", [])}


class FirecrawlScrape(FirecrawlTool):
    name = "web_scraping"
    description = "Scrape a public web URL through Firecrawl."
    argument_schema = _object_schema(
        ["url"],
        {"url": {"type": "string", "pattern": "^https?://"}},
    )

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = self._post(
            "/scrape",
            {"url": arguments["url"], "formats": ["markdown"]},
        )
        return {"document": payload.get("data", {})}

