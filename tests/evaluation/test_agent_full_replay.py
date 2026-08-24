from __future__ import annotations

import importlib.util
from pathlib import Path

from travel_agent_rl.tools.base import ToolRequest
from travel_agent_rl.tools.registry import build_mock_registry


def load_evaluator_module():
    script = Path(__file__).parents[2] / "scripts" / "evaluate_agent_model.py"
    spec = importlib.util.spec_from_file_location("evaluate_agent_model", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_replay_environment_needs_no_live_amap_credential(tmp_path, monkeypatch):
    module = load_evaluator_module()
    monkeypatch.delenv("AMAP_API_KEY", raising=False)

    def fake_registry(root, **kwargs):
        assert root == tmp_path
        assert kwargs["amap_api_key"] == "replay-only-disabled"
        assert kwargs["web_mode"] == "replay_only"
        return build_mock_registry()

    monkeypatch.setattr(module, "build_distillation_registry", fake_registry)
    environment = module.build_full_replay_environment(tmp_path)
    result = environment.execute(
        ToolRequest(tool_name="weather", arguments={"city": "110000", "date": "2026-08-25"})
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "offline_weather_unavailable"
