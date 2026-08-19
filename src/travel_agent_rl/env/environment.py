from __future__ import annotations

import hashlib
import json
from time import perf_counter
from jsonschema import ValidationError

from travel_agent_rl.tools.base import (
    ToolError,
    ToolExecutionError,
    ToolRequest,
    ToolResponse,
    ToolStatus,
)
from travel_agent_rl.tools.registry import ToolRegistry


class TravelEnvironment:
    """Schema-safe tool runner with deterministic in-memory caching and retries."""

    def __init__(self, registry: ToolRegistry, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.registry = registry
        self.max_attempts = max_attempts
        self._cache: dict[str, ToolResponse] = {}

    @staticmethod
    def _cache_key(request: ToolRequest) -> str:
        canonical = json.dumps(
            {"tool_name": request.tool_name, "arguments": request.arguments},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def execute(self, request: ToolRequest) -> ToolResponse:
        started = perf_counter()
        tool = self.registry.get(request.tool_name)
        if tool is None:
            return self._error(
                request,
                code="unknown_tool",
                message=f"unknown tool: {request.tool_name}",
                attempts=1,
                started=started,
            )

        key = self._cache_key(request)
        if key in self._cache:
            cached = self._cache[key].model_copy(deep=True)
            cached.request_id = request.request_id
            cached.cached = True
            cached.latency_ms = (perf_counter() - started) * 1000
            return cached

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = tool.invoke(request)
                response.attempts = attempt
                self._cache[key] = response.model_copy(deep=True)
                return response
            except ValidationError as exc:
                return self._error(
                    request,
                    code="invalid_arguments",
                    message=exc.message,
                    attempts=attempt,
                    started=started,
                )
            except ToolExecutionError as exc:
                if exc.retryable and attempt < self.max_attempts:
                    continue
                return self._error(
                    request,
                    code=exc.code,
                    message=str(exc),
                    attempts=attempt,
                    started=started,
                    retryable=exc.retryable,
                )
            except Exception as exc:  # provider exceptions are normalized at the boundary
                return self._error(
                    request,
                    code="unhandled_tool_error",
                    message=f"{type(exc).__name__}: {exc}",
                    attempts=attempt,
                    started=started,
                )
        raise AssertionError("retry loop must return")

    @staticmethod
    def _error(
        request: ToolRequest,
        *,
        code: str,
        message: str,
        attempts: int,
        started: float,
        retryable: bool = False,
    ) -> ToolResponse:
        return ToolResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            status=ToolStatus.ERROR,
            error=ToolError(code=code, message=message, retryable=retryable),
            attempts=attempts,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
