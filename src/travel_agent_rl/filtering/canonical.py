from __future__ import annotations

import json

from travel_agent_rl.agent.protocol import AgentMessage, FinalAnswer, ToolCall, parse_agent_output
from travel_agent_rl.distill.models import TrajectoryRecord


def canonicalize_trajectory(trajectory: TrajectoryRecord) -> TrajectoryRecord:
    messages: list[AgentMessage] = []
    for message in trajectory.messages:
        content = message.content.strip()
        if message.role == "assistant":
            parsed = parse_agent_output(content)
            if isinstance(parsed, ToolCall):
                payload = json.dumps(
                    {"name": parsed.name, "arguments": parsed.arguments},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                content = f"<tool-call>\n{payload}\n</tool-call>"
            elif isinstance(parsed, FinalAnswer):
                content = f"<answer>\n{parsed.content.strip()}\n</answer>"
        elif message.role == "tool":
            first_line, *suffix = content.splitlines()
            payload = json.loads(first_line)
            content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if suffix:
                content = "\n".join((content, *[line.strip() for line in suffix if line.strip()]))
        messages.append(AgentMessage(role=message.role, content=content))
    return trajectory.model_copy(update={"messages": messages}, deep=True)
