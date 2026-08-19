from __future__ import annotations

import hashlib
import json
import os
from time import perf_counter
from pathlib import Path
from jsonschema import ValidationError
from pydantic import ValidationError as PydanticValidationError
from uuid import uuid4

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

    def __init__(
        self,
        registry: ToolRegistry,
        max_attempts: int = 2,
        cache_dir: Path | None = None,
        cache_namespace: str = "default",
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not cache_namespace.strip():
            raise ValueError("cache_namespace must not be empty")
        self.registry = registry
        self.max_attempts = max_attempts
        self.cache_dir = cache_dir
        self.cache_namespace = cache_namespace
        self._cache: dict[str, ToolResponse] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, request: ToolRequest) -> str:
        canonical = json.dumps(
            {
                "cache_namespace": self.cache_namespace,
                "tool_name": request.tool_name,
                "arguments": request.arguments,
            },
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
        if tool.cacheable:
            cached_response = self._load_cached(key)
            if cached_response is not None:
                cached = cached_response.model_copy(deep=True)
                cached.request_id = request.request_id
                cached.cached = True
                cached.latency_ms = (perf_counter() - started) * 1000
                return cached

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = tool.invoke(request)
                response.attempts = attempt
                if tool.cacheable:
                    self._cache[key] = response.model_copy(deep=True)
                    self._store_cached(key, response)
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
                    provider=exc.provider,
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
        provider: str | None = None,
    ) -> ToolResponse:
        return ToolResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            status=ToolStatus.ERROR,
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
                provider=provider,
            ),
            attempts=attempts,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def _load_cached(self, key: str) -> ToolResponse | None:
        if key in self._cache:
            return self._cache[key]
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            response = ToolResponse.model_validate_json(path.read_text())
        except (PydanticValidationError, UnicodeDecodeError):
            quarantine = path.with_name(f"{path.name}.corrupt.{uuid4().hex}")
            path.replace(quarantine)
            return None
        self._cache[key] = response
        return response

    def _store_cached(self, key: str, response: ToolResponse) -> None:
        if self.cache_dir is None:
            return
        target = self.cache_dir / f"{key}.json"
        temporary = self.cache_dir / f".{key}.{os.getpid()}.{uuid4().hex}.tmp"
        temporary.write_text(response.model_dump_json())
        temporary.replace(target)
