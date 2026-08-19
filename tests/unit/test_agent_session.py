import json

from travel_agent_rl.agent import AgentSession, SessionStatus
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import BaseTool, build_mock_registry
from travel_agent_rl.tools.mock import _object_schema
from travel_agent_rl.tools.registry import ToolRegistry


def _call(name: str, arguments: dict) -> str:
    return f"<tool-call>{json.dumps({'name': name, 'arguments': arguments})}</tool-call>"


class _OversizedTool(BaseTool):
    name = "oversized"
    description = "Return a deliberately oversized provider response."
    argument_schema = _object_schema([], {})

    def run(self, arguments: dict) -> dict:
        return {
            "documents": [
                {"title": f"document-{index}", "content": "evidence " * 2_000}
                for index in range(10)
            ]
        }


def test_session_executes_tool_then_accepts_answer() -> None:
    session = AgentSession(TravelEnvironment(build_mock_registry()), max_turns=3)
    transition = session.step(_call("weather", {"city": "Hangzhou", "date": "2026-08-20"}))
    assert transition.status is SessionStatus.RUNNING
    assert transition.error_code is None
    final = session.step("<answer>Pack light.</answer>")
    assert final.status is SessionStatus.FINISHED
    assert final.final_answer == "Pack light."


def test_session_detects_repeat_and_forces_answer_at_limit() -> None:
    session = AgentSession(TravelEnvironment(build_mock_registry()), max_turns=2)
    action = _call("web_search", {"query": "Hangzhou"})
    session.step(action)
    repeated = session.step(action)
    assert repeated.error_code == "repeated_action"
    assert repeated.status is SessionStatus.FORCE_ANSWER
    rejected = session.step(action)
    assert rejected.error_code == "answer_required"
    final = session.step("<answer>Done.</answer>")
    assert final.status is SessionStatus.FINISHED


def test_invalid_output_consumes_turn_and_is_detectable() -> None:
    session = AgentSession(TravelEnvironment(build_mock_registry()), max_turns=1)
    transition = session.step("not protocol")
    assert transition.error_code == "invalid_agent_output"
    assert transition.status is SessionStatus.FORCE_ANSWER


def test_session_bounds_model_visible_tool_observation() -> None:
    session = AgentSession(
        TravelEnvironment(ToolRegistry([_OversizedTool()])),
        max_turns=2,
    )
    transition = session.step(_call("oversized", {}))
    assert transition.observation is not None
    assert len(transition.observation) <= 2_000
    payload = json.loads(transition.observation)
    assert payload["status"] == "ok"
    assert payload["tool_name"] == "oversized"
    assert "truncated" in transition.observation


def test_session_enforces_combined_public_web_budget() -> None:
    session = AgentSession(
        TravelEnvironment(build_mock_registry()),
        max_turns=5,
        max_public_web_actions=2,
    )
    first = session.step(_call("web_search", {"query": "one"}))
    second = session.step(_call("web_scraping", {"url": "https://example.test/two"}))
    blocked = session.step(_call("web_search", {"query": "three"}))

    assert first.error_code is None
    assert second.error_code is None
    assert blocked.error_code == "tool_budget_exceeded"
    assert blocked.observation is not None
    assert json.loads(blocked.observation)["agent_state"]["remaining_public_web_actions"] == 0
