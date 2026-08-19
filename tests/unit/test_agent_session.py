from travel_agent_rl.agent import AgentSession, SessionStatus
from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import build_mock_registry


def _call(name: str, arguments: dict) -> str:
    import json

    return f"<tool-call>{json.dumps({'name': name, 'arguments': arguments})}</tool-call>"


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

