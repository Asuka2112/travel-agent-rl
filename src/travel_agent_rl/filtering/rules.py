from __future__ import annotations

import json
from collections import Counter

from pydantic import BaseModel, Field

from travel_agent_rl.agent.protocol import FinalAnswer, ToolCall, parse_agent_output
from travel_agent_rl.distill.models import TrajectoryRecord


class RuleFilterResult(BaseModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


def apply_rule_filter(
    trajectory: TrajectoryRecord,
    allowed_tools: set[str],
    *,
    max_turns: int = 13,
) -> RuleFilterResult:
    reasons: list[str] = []
    if not trajectory.metadata.success:
        reasons.append("generation_failed")
    if not trajectory.final_answer or not trajectory.final_answer.strip():
        reasons.append("missing_final_answer")
    if trajectory.metadata.number_of_turns > max_turns:
        reasons.append("max_turns_exceeded")

    action_counts: Counter[str] = Counter()
    saw_final = False
    expected_role = "assistant"
    for message in trajectory.messages[1:]:
        if message.role != expected_role:
            reasons.append("invalid_message_alternation")
            break
        if message.role == "assistant":
            try:
                parsed = parse_agent_output(message.content)
            except (ValueError, TypeError, json.JSONDecodeError):
                reasons.append("invalid_assistant_format")
                expected_role = "tool"
                continue
            if isinstance(parsed, FinalAnswer):
                saw_final = True
                expected_role = "done"
            elif isinstance(parsed, ToolCall):
                if parsed.name not in allowed_tools:
                    reasons.append("illegal_tool")
                key = json.dumps(parsed.model_dump(mode="json"), sort_keys=True)
                action_counts[key] += 1
                if action_counts[key] > 1:
                    reasons.append("repeated_action")
                expected_role = "tool"
        elif message.role == "tool":
            if not message.content.strip():
                reasons.append("empty_observation")
            else:
                try:
                    json.loads(message.content.splitlines()[0])
                except json.JSONDecodeError:
                    reasons.append("invalid_observation_json")
            expected_role = "assistant"
        else:
            reasons.append("invalid_message_role")
            break

    if not saw_final:
        reasons.append("missing_final_message")
    if expected_role != "done":
        reasons.append("incomplete_trajectory")
    return RuleFilterResult(accepted=not reasons, reasons=list(dict.fromkeys(reasons)))
