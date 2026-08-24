from __future__ import annotations

import math

from travel_agent_rl.rewards import REWARD_COMPONENTS, RewardEngine

SCHEMAS = {
    "poi_search": {
        "type": "object",
        "properties": {"city": {"type": "string", "minLength": 1}},
        "required": ["city"],
        "additionalProperties": False,
    }
}
USER = {"role": "user", "content": "Plan a trip"}
CALL = {
    "role": "assistant",
    "content": '<tool-call>{"name":"poi_search","arguments":{"city":"Paris"}}</tool-call>',
}
OBS = {"role": "tool", "content": '{"status":"ok","data":{"items":["Louvre"]}}'}
FINAL = {"role": "assistant", "content": "<answer>Visit the Louvre.</answer>"}


def test_six_case_reward_sanity() -> None:
    cases = {
        "good": ([USER, CALL, OBS, FINAL], 0.9),
        "bad_tool": (
            [
                USER,
                {
                    "role": "assistant",
                    "content": '<tool-call>{"name":"unknown","arguments":{}}</tool-call>',
                },
                OBS,
                FINAL,
            ],
            0.2,
        ),
        "loop": ([USER, CALL, OBS, CALL, OBS, FINAL], 0.4),
        "malformed": ([USER, {"role": "assistant", "content": "<tool_call>broken"}], None),
        "premature": ([USER, FINAL], 0.3),
        "efficient_valid": ([USER, CALL, OBS, FINAL], 0.8),
    }
    engine = RewardEngine(
        weights={name: 1.0 for name in REWARD_COMPONENTS},
        tool_schemas=SCHEMAS,
    )
    scored = {
        name: engine.score(messages, judge_result=judge, progress=0.25)
        for name, (messages, judge) in cases.items()
    }
    assert scored["good"].total > scored["bad_tool"].total
    assert scored["good"].total > scored["loop"].total
    assert scored["good"].total > scored["malformed"].total
    assert scored["good"].total > scored["premature"].total
    assert all(math.isfinite(result.total) for result in scored.values())
    for component in REWARD_COMPONENTS:
        values = {result.components[component] for result in scored.values()}
        assert len(values) > 1, f"constant component: {component}"
