from __future__ import annotations

import json

import pytest

from travel_agent_rl.sft.data import build_agent_tool_schemas, convert_frozen_trajectory


def test_build_agent_tool_schemas_has_exact_eight_tools() -> None:
    tools = build_agent_tool_schemas()
    assert [tool["function"]["name"] for tool in tools] == [
        "poi_search",
        "nearby_search",
        "weather",
        "route_planning",
        "web_search",
        "web_scraping",
        "flight_search",
        "train_search",
    ]


def test_convert_frozen_trajectory_to_native_agent_roles() -> None:
    source = {
        "query_id": "q1",
        "metadata": {"difficulty": "hard"},
        "messages": [
            {"role": "user", "content": "plan"},
            {
                "role": "assistant",
                "content": '<tool-call>{"name":"train_search","arguments":{"origin":"A"}}</tool-call>',
            },
            {"role": "tool", "content": '{"options":[]}'},
            {"role": "assistant", "content": "<answer>done</answer>"},
        ],
    }
    result = convert_frozen_trajectory(source)
    assert result["query_id"] == "q1"
    assert result["difficulty"] == "hard"
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "tool_call",
        "tool_response",
        "assistant",
    ]
    assert json.loads(result["messages"][1]["content"])["name"] == "train_search"
    assert result["messages"][-1]["content"] == "done"
    assert len(json.loads(result["tools"])) == 8


def test_convert_rejects_non_final_trajectory() -> None:
    source = {
        "query_id": "q1",
        "messages": [
            {"role": "user", "content": "plan"},
            {
                "role": "assistant",
                "content": '<tool-call>{"name":"train_search","arguments":{}}</tool-call>',
            },
        ],
    }
    with pytest.raises(ValueError, match="end with"):
        convert_frozen_trajectory(source)
