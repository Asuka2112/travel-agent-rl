from __future__ import annotations

from typing import Any

import httpx

from .base import ToolExecutionError, TransientToolError


class JsonHttpClient:
    def __init__(self, client: httpx.Client, provider: str) -> None:
        self.client = client
        self.provider = provider

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise TransientToolError(
                f"{self.provider} request timed out",
                code="timeout",
                provider=self.provider,
            ) from exc
        except httpx.RequestError as exc:
            raise TransientToolError(
                f"{self.provider} network error: {exc}",
                code="network_error",
                provider=self.provider,
            ) from exc

        if response.status_code == 429:
            raise TransientToolError(
                f"{self.provider} rate limit exceeded",
                code="rate_limited",
                provider=self.provider,
            )
        if response.status_code >= 500:
            raise TransientToolError(
                f"{self.provider} server error: HTTP {response.status_code}",
                code="provider_server_error",
                provider=self.provider,
            )
        if response.status_code >= 400:
            raise ToolExecutionError(
                f"{self.provider} rejected request: HTTP {response.status_code}",
                code="provider_client_error",
                provider=self.provider,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ToolExecutionError(
                f"{self.provider} returned invalid JSON",
                code="invalid_provider_json",
                provider=self.provider,
            ) from exc
        if not isinstance(payload, dict):
            raise ToolExecutionError(
                f"{self.provider} returned a non-object JSON response",
                code="invalid_provider_schema",
                provider=self.provider,
            )
        return payload

