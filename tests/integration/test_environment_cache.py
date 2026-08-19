import pytest

from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import ToolRequest, build_mock_registry


@pytest.mark.integration
def test_equivalent_calls_use_cache_but_keep_request_identity() -> None:
    environment = TravelEnvironment(build_mock_registry())
    first = environment.execute(
        ToolRequest(tool_name="web_search", arguments={"query": "Hangzhou"})
    )
    second = environment.execute(
        ToolRequest(tool_name="web_search", arguments={"query": "Hangzhou"})
    )
    assert first.cached is False
    assert second.cached is True
    assert first.request_id != second.request_id
    assert first.data == second.data


@pytest.mark.integration
def test_disk_cache_survives_environment_instance(tmp_path) -> None:
    request = ToolRequest(tool_name="web_search", arguments={"query": "Hangzhou"})
    first_environment = TravelEnvironment(build_mock_registry(), cache_dir=tmp_path)
    first = first_environment.execute(request)
    second_environment = TravelEnvironment(build_mock_registry(), cache_dir=tmp_path)
    second = second_environment.execute(
        ToolRequest(tool_name="web_search", arguments={"query": "Hangzhou"})
    )
    assert first.cached is False
    assert second.cached is True
    assert second.data == first.data

