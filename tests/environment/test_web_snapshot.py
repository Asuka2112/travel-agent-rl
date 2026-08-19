from __future__ import annotations

import json

import pytest

from travel_agent_rl.tools.base import ToolExecutionError
from travel_agent_rl.tools.firecrawl import CapturedToolResult, FirecrawlSearch
from travel_agent_rl.tools.web_snapshot import SnapshotWebTool, WebSnapshotStore


class FakeSearch(FirecrawlSearch):
    def __init__(self) -> None:
        self.calls = 0
        self.backend = "test_selfhosted"

    def capture(self, arguments):
        self.calls += 1
        return CapturedToolResult(
            data={
                "results": [
                    {
                        "url": "https://example.com/",
                        "title": " Example ",
                        "description": "Line one.\r\nLine two.",
                    }
                ]
            },
            http_metadata={"status_code": 200, "headers": {"content-type": "application/json"}},
            backend=self.backend,
        )


class FailingSearch(FakeSearch):
    def capture(self, arguments):
        self.calls += 1
        raise ToolExecutionError(
            "upstream rejected this URL",
            code="upstream_rejected",
            provider="test_selfhosted",
        )


def test_online_capture_then_replay_without_provider_call(tmp_path) -> None:
    provider = FakeSearch()
    store = WebSnapshotStore(tmp_path, run_id="pilot")
    capture = SnapshotWebTool(provider, store, mode="online_capture")
    first = capture.run({"query": "test", "limit": 5})
    replay = SnapshotWebTool(provider, store, mode="replay_only")
    second = replay.run({"query": "test", "limit": 5})

    assert provider.calls == 1
    assert first["_snapshot"]["hit"] is False
    assert second["_snapshot"]["hit"] is True
    request_files = list((tmp_path / "v1" / "requests").rglob("*.json"))
    document_files = list((tmp_path / "v1" / "documents").rglob("*.json"))
    assert len(request_files) == 1
    assert len(document_files) == 1
    assert json.loads(document_files[0].read_text())["normalized_text"] == ("Line one.\nLine two.")


def test_capture_failure_is_versioned_and_replayed_without_provider_call(tmp_path) -> None:
    provider = FailingSearch()
    store = WebSnapshotStore(tmp_path, run_id="failure")
    capture = SnapshotWebTool(provider, store, mode="online_capture")
    with pytest.raises(ToolExecutionError, match="upstream rejected"):
        capture.run({"query": "bad", "limit": 5})

    replay = SnapshotWebTool(provider, store, mode="replay_only")
    with pytest.raises(ToolExecutionError, match="upstream rejected"):
        replay.run({"query": "bad", "limit": 5})

    assert provider.calls == 1
    request_file = next((tmp_path / "v1" / "requests").rglob("*.json"))
    assert json.loads(request_file.read_text())["response_error"]["code"] == ("upstream_rejected")
