import pytest
from pydantic import ValidationError

from travel_agent_rl.tools.base import ToolError, ToolResponse, ToolStatus


def test_error_response_requires_normalized_error() -> None:
    with pytest.raises(ValidationError):
        ToolResponse(request_id="1", tool_name="weather", status=ToolStatus.ERROR)


def test_success_response_rejects_error() -> None:
    with pytest.raises(ValidationError):
        ToolResponse(
            request_id="1",
            tool_name="weather",
            status=ToolStatus.OK,
            error=ToolError(code="bad", message="bad"),
        )
