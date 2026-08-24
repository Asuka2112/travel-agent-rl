from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "evaluate_pairwise_local_judge.py"
SPEC = importlib.util.spec_from_file_location("true_pairwise", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_weighted_score_uses_thirty_seventy_protocol() -> None:
    scores = {name: 5.0 for name in MODULE.PATH_DIMENSIONS}
    scores.update({name: 1.0 for name in MODULE.FINAL_DIMENSIONS})
    result = MODULE.weighted_score(scores)
    assert result["path"] == 1.0
    assert result["final_answer"] == 0.2
    assert result["weighted"] == pytest.approx(0.44)


def test_bidirectional_disagreement_becomes_audited_tie() -> None:
    rows = [
        {"query_id": "q1", "winner": "base"},
        {"query_id": "q1", "winner": "rl"},
    ]
    result = MODULE.aggregate_pair("base", "rl", rows)
    assert result["position_disagreement"] == 1
    assert result["tie"] == 1
    assert result["pairwise_score"] == 0.5
