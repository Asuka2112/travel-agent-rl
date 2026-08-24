#!/usr/bin/env python3
"""Verify real qwen3/hermes encoding masks tool observations out of GRPO loss."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from swift.llm import get_model_tokenizer, get_template
from swift.trainers.rlhf_trainer.utils import replace_assistant_response_with_ids

from travel_agent_rl.sft.data import build_agent_tool_schemas


def find_subsequence(values: list[int], needle: list[int]) -> int:
    for index in range(len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    raise ValueError("sentinel token sequence not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/grpo/template_mask_verification.json")
    )
    args = parser.parse_args()
    _, tokenizer = get_model_tokenizer(str(args.model), load_model=False)
    template = get_template(
        tokenizer.model_meta.template,
        tokenizer,
        max_length=32768,
        truncation_strategy="raise",
        agent_template="hermes",
        loss_scale="default",
    )
    template.set_mode("train")
    observation_sentinel = "OBSERVATION_NEVER_IN_POLICY_LOSS_8C19E3"
    call = '<tool_call>{"name":"train_search","arguments":{"origin":"Boston","destination":"New York City","date":"2022-01-03"}}</tool_call>'
    answer = "SIMULATED_TRAIN_EVIDENCE_FINAL_72FD"
    messages = [
        {"role": "user", "content": "Plan a rail trip."},
        {"role": "assistant", "content": call},
        {"role": "tool_response", "content": observation_sentinel},
        {"role": "assistant", "content": answer},
    ]
    call_ids = tokenizer.encode(call, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    completion_ids = [call_ids, answer_ids]
    loss_masks = [[1] * len(call_ids), [1] * len(answer_ids)]
    replaced = replace_assistant_response_with_ids(
        deepcopy(messages), completion_ids, loss_masks
    )
    item = template.encode(
        {
            "messages": replaced,
            "tools": json.dumps(build_agent_tool_schemas(), ensure_ascii=False),
        }
    )
    input_ids = item["input_ids"]
    labels = item["labels"]
    observation_ids = tokenizer.encode(observation_sentinel, add_special_tokens=False)
    observation_start = find_subsequence(input_ids, observation_ids)
    observation_labels = labels[observation_start : observation_start + len(observation_ids)]
    call_start = find_subsequence(input_ids, call_ids)
    answer_start = find_subsequence(input_ids, answer_ids)
    call_labels = labels[call_start : call_start + len(call_ids)]
    answer_labels = labels[answer_start : answer_start + len(answer_ids)]
    payload = {
        "model": str(args.model),
        "template": "qwen3_nothinking/hermes",
        "response_turns": len(completion_ids),
        "token_mask_lengths_aligned": all(
            len(ids) == len(mask) for ids, mask in zip(completion_ids, loss_masks)
        ),
        "observation_tokens": len(observation_ids),
        "observation_all_ignored": all(label == -100 for label in observation_labels),
        "first_action_all_trained": all(label != -100 for label in call_labels),
        "final_answer_all_trained": all(label != -100 for label in answer_labels),
        "passed": False,
    }
    payload["passed"] = all(
        (
            payload["token_mask_lengths_aligned"],
            payload["observation_all_ignored"],
            payload["first_action_all_trained"],
            payload["final_answer_all_trained"],
        )
    )
    if not payload["passed"]:
        raise AssertionError(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
