from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from time import perf_counter
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, model_validator


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    provider: str | None = None


class ToolRequest(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: uuid4().hex)


class ToolResponse(BaseModel):
    request_id: str
    tool_name: str
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None
    attempts: int = Field(default=1, ge=1)
    cached: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_error_contract(self) -> ToolResponse:
        if self.status is ToolStatus.OK and self.error is not None:
            raise ValueError("successful tool responses cannot carry errors")
        if self.status is ToolStatus.ERROR and self.error is None:
            raise ValueError("error tool responses must carry a normalized error")
        return self


class ToolExecutionError(RuntimeError):
    code = "tool_execution_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.provider = provider


class TransientToolError(ToolExecutionError):
    code = "transient_tool_error"
    retryable = True


class BaseTool(ABC):
    name: str
    description: str
    argument_schema: dict[str, Any]

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        Draft202012Validator(self.argument_schema).validate(arguments)

    def invoke(self, request: ToolRequest) -> ToolResponse:
        started = perf_counter()
        self.validate_arguments(request.arguments)
        data = self.run(request.arguments)
        return ToolResponse(
            request_id=request.request_id,
            tool_name=self.name,
            status=ToolStatus.OK,
            data=data,
            latency_ms=(perf_counter() - started) * 1000,
        )

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an already validated call."""
