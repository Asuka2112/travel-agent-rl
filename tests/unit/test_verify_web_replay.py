import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "verify_web_replay.py"
SPEC = importlib.util.spec_from_file_location("verify_web_replay", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
reached_web_provider = MODULE.reached_web_provider


def test_session_rejected_web_actions_are_not_replayed() -> None:
    for code in ("answer_required", "repeated_action", "tool_budget_exceeded"):
        assert not reached_web_provider({"error": {"code": code}})


def test_provider_errors_and_successes_are_replayed() -> None:
    assert reached_web_provider({"status": "ok", "data": {}})
    assert reached_web_provider({"error": {"code": "provider_server_error"}})
