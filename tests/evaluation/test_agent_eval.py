import pytest

from travel_agent_rl.evaluation.agent_eval import (
    completion_score,
    parse_native_agent_output,
    validate_tool_call,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "train_search",
            "parameters": {
                "type": "object",
                "required": ["origin"],
                "properties": {"origin": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }
]


def test_parse_native_tool_call_and_answer() -> None:
    kind, payload = parse_native_agent_output(
        '<tool_call>{"name":"train_search","arguments":{"origin":"Boston"}}</tool_call>'
    )
    assert kind == "tool_call"
    assert payload["name"] == "train_search"
    assert parse_native_agent_output("Final itinerary")[0] == "answer"


def test_parse_native_rejects_mixed_call() -> None:
    with pytest.raises(ValueError, match="malformed"):
        parse_native_agent_output("prefix <tool_call>{}</tool_call>")


def test_validate_tool_call() -> None:
    assert validate_tool_call(
        {"name": "train_search", "arguments": {"origin": "Boston"}}, TOOLS
    ) == (True, None)
    valid, reason = validate_tool_call(
        {"name": "train_search", "arguments": {}}, TOOLS
    )
    assert valid is False
    assert "required" in reason


def test_completion_score() -> None:
    assert completion_score(
        {
            "expected_tool": "train_search",
            "tool_sequence": ["train_search"],
            "final_answer": "The simulated schedule is suitable.",
            "invalid_tool_calls": 0,
        }
    ) == 1.0
