from __future__ import annotations

from collections.abc import Iterable

from .base import BaseTool
from .mock import MOCK_TOOL_DEFINITIONS, DeterministicMockTool


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] = ()) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))


def build_mock_registry() -> ToolRegistry:
    return ToolRegistry(
        DeterministicMockTool(
            name=item["name"],
            description=item["description"],
            argument_schema=item["schema"],
            fixture=item["fixture"],
        )
        for item in MOCK_TOOL_DEFINITIONS
    )

