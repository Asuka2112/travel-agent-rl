from __future__ import annotations

from travel_agent_rl.agent.protocol import AgentMessage
from travel_agent_rl.distill import ChatResult, RetryingChatBackend, TokenUsage


class FlakyBackend:
    model = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, system_prompt):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary")
        return ChatResult(content="<answer>ok</answer>", model=self.model, usage=TokenUsage())


def test_retrying_backend_recovers_transient_failure() -> None:
    backend = FlakyBackend()
    retrying = RetryingChatBackend(backend, max_attempts=3, initial_backoff_seconds=0)
    result = retrying.complete([AgentMessage(role="user", content="hi")], "prompt")
    assert result.content == "<answer>ok</answer>"
    assert backend.calls == 3
