from __future__ import annotations

from datetime import datetime, timezone

from travel_agent_rl.agent.protocol import AgentMessage, ToolCall
from travel_agent_rl.distill import ChatResult, TokenUsage, TrajectoryMetadata, TrajectoryRecord
from travel_agent_rl.filtering import (
    StructuredJudge,
    apply_rule_filter,
    canonicalize_trajectory,
)


def trajectory() -> TrajectoryRecord:
    now = datetime.now(timezone.utc)
    return TrajectoryRecord(
        query_id="q1",
        query="plan",
        messages=[
            AgentMessage(role="user", content=" plan "),
            AgentMessage(
                role="assistant",
                content=' <tool-call> {"arguments":{"city":"Beijing","query":"museum"},"name":"poi_search"} </tool-call> ',
            ),
            AgentMessage(role="tool", content=' {"status":"ok","data":{"name":"Museum"}} '),
            AgentMessage(role="assistant", content=" <answer> Go. </answer> "),
        ],
        tool_calls=[ToolCall(name="poi_search", arguments={"query": "museum", "city": "Beijing"})],
        tool_observations=[{"status": "ok", "data": {"name": "Museum"}}],
        final_answer="Go.",
        metadata=TrajectoryMetadata(
            query_id="q1",
            teacher_model="teacher",
            prompt_version="v1",
            started_at=now,
            finished_at=now,
            number_of_turns=1,
            tools_used=["poi_search"],
            success=True,
            latency_seconds=0,
        ),
    )


def test_rule_filter_accepts_complete_trajectory() -> None:
    result = apply_rule_filter(trajectory(), {"poi_search"})
    assert result.accepted
    assert result.reasons == []


def test_rule_filter_rejects_illegal_tool() -> None:
    result = apply_rule_filter(trajectory(), {"weather"})
    assert not result.accepted
    assert "illegal_tool" in result.reasons


def test_canonicalizer_normalizes_json_and_tags() -> None:
    canonical = canonicalize_trajectory(trajectory())
    assert canonical.messages[1].content == (
        '<tool-call>\n{"arguments":{"city":"Beijing","query":"museum"},"name":"poi_search"}\n</tool-call>'
    )
    assert canonical.messages[2].content == '{"data":{"name":"Museum"},"status":"ok"}'
    assert canonical.messages[3].content == "<answer>\nGo.\n</answer>"


class JudgeBackend:
    model = "judge"

    def complete(self, messages, system_prompt):
        return ChatResult(
            model=self.model,
            content="""```json
{"scores":{"task_relevance":5,"completeness":4,"constraint_satisfaction":4,
"factual_grounding":5,"tool_use_correctness":5,"tool_efficiency":4,
"final_answer_quality":4},"rationale":"Grounded and efficient."}
```""",
            usage=TokenUsage(),
        )


def test_structured_judge_applies_configured_threshold() -> None:
    decision = StructuredJudge(JudgeBackend(), "prompt", mean_threshold=4).evaluate(trajectory())
    assert decision.accepted
    assert decision.scores.mean > 4
    assert decision.prompt_version == "judge-v1"
