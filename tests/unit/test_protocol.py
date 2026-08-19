import pytest

from travel_agent_rl.agent import FinalAnswer, ToolCall, parse_agent_output


def test_parse_tool_call() -> None:
    parsed = parse_agent_output(
        '<tool-call>{"name":"weather","arguments":{"city":"Hangzhou"}}</tool-call>'
    )
    assert isinstance(parsed, ToolCall)
    assert parsed.name == "weather"


def test_parse_answer() -> None:
    parsed = parse_agent_output("<answer>A concise itinerary.</answer>")
    assert isinstance(parsed, FinalAnswer)


def test_parse_repairs_single_character_closing_tag_typo() -> None:
    parsed = parse_agent_output(
        '<tool-call>{"name":"weather","arguments":{"city":"Hangzhou"}}</tool-call}'
    )
    assert isinstance(parsed, ToolCall)
    assert parsed.name == "weather"


@pytest.mark.parametrize(
    "text", ["plain text", "<tool-call>{bad}</tool-call>", "<answer></answer>"]
)
def test_reject_invalid_protocol(text: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_agent_output(text)
