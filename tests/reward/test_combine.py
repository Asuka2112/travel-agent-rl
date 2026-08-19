import pytest

from travel_agent_rl.rewards import REWARD_COMPONENTS, combine_rewards


def test_six_required_reward_components_are_explicit() -> None:
    assert len(REWARD_COMPONENTS) == 6
    scores = {name: 1.0 for name in REWARD_COMPONENTS}
    weights = {name: 0.5 for name in REWARD_COMPONENTS}
    assert combine_rewards(scores, weights) == pytest.approx(3.0)


def test_unknown_reward_is_rejected() -> None:
    with pytest.raises(ValueError):
        combine_rewards({"opaque_reward": 1.0}, {})

