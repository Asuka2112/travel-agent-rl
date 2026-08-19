import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "filter_trajectories.py"
SPEC = importlib.util.spec_from_file_location("filter_trajectories", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_judge_exceptions_are_retried_but_quality_rejections_are_resumed() -> None:
    assert MODULE.should_retry_judge_row({"drop_reason": ["judge_exception:TimeoutError"]})
    assert not MODULE.should_retry_judge_row({"drop_reason": ["judge_below_threshold"]})
    assert not MODULE.should_retry_judge_row({"drop_reason": []})
