from travel_agent_rl.env import TravelEnvironment
from travel_agent_rl.tools import ToolRequest, ToolStatus, build_mock_registry
from travel_agent_rl.tools.mock import SMOKE_ARGUMENTS


def test_all_eight_mock_tools_share_response_contract() -> None:
    registry = build_mock_registry()
    assert set(registry.names) == set(SMOKE_ARGUMENTS)
    environment = TravelEnvironment(registry)
    for name, arguments in SMOKE_ARGUMENTS.items():
        response = environment.execute(ToolRequest(tool_name=name, arguments=arguments))
        assert response.status is ToolStatus.OK
        assert response.data["mock"] is True
        assert response.error is None


def test_invalid_arguments_are_normalized() -> None:
    environment = TravelEnvironment(build_mock_registry())
    response = environment.execute(ToolRequest(tool_name="weather", arguments={}))
    assert response.status is ToolStatus.ERROR
    assert response.error is not None
    assert response.error.code == "invalid_arguments"


def test_unknown_tool_is_normalized() -> None:
    environment = TravelEnvironment(build_mock_registry())
    response = environment.execute(ToolRequest(tool_name="book_hotel", arguments={}))
    assert response.error is not None
    assert response.error.code == "unknown_tool"
