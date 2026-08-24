#!/usr/bin/env python3
"""Run the six-trajectory GRPO reward preflight and save machine-readable evidence."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from travel_agent_rl.rewards import REWARD_COMPONENTS, RewardEngine
from travel_agent_rl.sft.data import build_agent_tool_schemas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo/reward_sanity.json"))
    args = parser.parse_args()
    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in build_agent_tool_schemas()
    }
    user = {"role": "user", "content": "Plan a trip to Paris."}
    call = {
        "role": "assistant",
        "content": '<tool-call>{"name":"poi_search","arguments":{"city":"Paris","query":"museum"}}</tool-call>',
    }
    observation = {"role": "tool", "content": '{"status":"ok","data":{"items":[]}}'}
    answer = {"role": "assistant", "content": "<answer>Evidence-grounded plan.</answer>"}
    cases = {
        "good": ([user, call, observation, answer], 0.9),
        "bad_tool": (
            [user, {"role": "assistant", "content": '<tool-call>{"name":"bad","arguments":{}}</tool-call>'}, observation, answer],
            0.2,
        ),
        "loop": ([user, call, observation, call, observation, answer], 0.4),
        "malformed": ([user, {"role": "assistant", "content": "<tool_call>broken"}], None),
        "premature_answer": ([user, answer], 0.3),
        "efficient_valid": ([user, call, observation, answer], 0.8),
    }
    engine = RewardEngine.from_yaml(Path("configs/grpo/rewards.yaml"), tool_schemas=schemas)
    results = {
        name: engine.score(messages, judge_result=judge, progress=0.0).to_dict()
        for name, (messages, judge) in cases.items()
    }
    if not results["good"]["total"] > results["bad_tool"]["total"]:
        raise AssertionError("good trajectory did not outrank bad tool trajectory")
    if any(not math.isfinite(item["total"]) for item in results.values()):
        raise FloatingPointError("non-finite reward")
    for component in REWARD_COMPONENTS:
        if len({item["components"][component] for item in results.values()}) == 1:
            raise AssertionError(f"constant reward component: {component}")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": results,
        "statistics": engine.statistics.summary(),
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
