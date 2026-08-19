from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from travel_agent_rl.agent.protocol import AgentMessage


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ChatResult(BaseModel):
    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ChatBackend(Protocol):
    model: str

    def complete(self, messages: Sequence[AgentMessage], system_prompt: str) -> ChatResult: ...


@dataclass(frozen=True)
class OpenAICompatibleChatBackend:
    """Small dependency-free client for OpenAI-compatible teacher endpoints."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 90.0
    max_tokens: int = 4096
    temperature: float = 0.1

    def complete(self, messages: Sequence[AgentMessage], system_prompt: str) -> ChatResult:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *[message.model_dump() for message in messages],
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"teacher HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"teacher connection failed: {exc.reason}") from exc

        try:
            choice = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("teacher response did not contain message content") from exc
        raw_usage = body.get("usage") or {}
        usage = TokenUsage(
            input_tokens=raw_usage.get("prompt_tokens", 0),
            output_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )
        return ChatResult(content=choice, model=body.get("model", self.model), usage=usage)
