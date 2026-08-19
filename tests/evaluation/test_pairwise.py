import pytest

from travel_agent_rl.evaluation import bidirectional_pairs


def test_pairwise_evaluation_swaps_position() -> None:
    assert bidirectional_pairs("sft", "grpo") == (("sft", "grpo"), ("grpo", "sft"))


def test_pairwise_rejects_identical_candidates() -> None:
    with pytest.raises(ValueError):
        bidirectional_pairs("base", "base")

