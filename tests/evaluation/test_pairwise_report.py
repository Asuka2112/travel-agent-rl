import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "report_pairwise_evaluation.py"
SPEC = importlib.util.spec_from_file_location("pairwise_report", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def scores(value: float) -> dict[str, float]:
    return {
        "task_relevance": value,
        "completeness": value,
        "constraint_satisfaction": value,
        "factual_grounding": value,
        "tool_use_correctness": value,
        "tool_efficiency": value,
        "final_answer_quality": value,
    }


def test_weighted_score_uses_frozen_30_70_split() -> None:
    result = MODULE.weighted_score(
        {**scores(5), "tool_efficiency": 1, "final_answer_quality": 1}
    )
    assert result["weighted"] == pytest.approx(0.78)


def test_decide_is_symmetric() -> None:
    assert MODULE.decide(0.8, 0.2) == "left"
    assert MODULE.decide(0.2, 0.8) == "right"
    assert MODULE.decide(0.5, 0.5) == "tie"


def test_pairwise_audit_swaps_position_without_changing_logical_winner() -> None:
    candidates = {
        "base": {"q": {"scores": scores(4)}},
        "rl": {"q": {"scores": scores(3)}},
    }
    result = MODULE.evaluate_pair("base", "rl", candidates)
    assert result["win"] == 1
    assert result["position_disagreement"] == 0
    assert result["records"][0]["ab"]["winner"] == "base"
    assert result["records"][0]["ba"]["winner"] == "base"
