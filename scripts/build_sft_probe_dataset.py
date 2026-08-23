#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from swift.llm import get_model_tokenizer, get_template

from travel_agent_rl.sft.data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/intermediate/sft/ms_swift_agent_881.jsonl")
    )
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--query-id", default="agent_star:easy_ncsq03")
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main_row(args) -> dict:
    for line in args.dataset.open():
        row = json.loads(line)
        if row["query_id"] == args.query_id:
            return row
    raise KeyError(args.query_id)


def main() -> None:
    args = parse_args()
    source = main_row(args)
    _, tokenizer = get_model_tokenizer(str(args.model), load_model=False)
    template = get_template(
        tokenizer.model_meta.template,
        tokenizer,
        max_length=None,
        truncation_strategy="raise",
        agent_template="hermes",
        loss_scale="default",
        sequence_parallel_size=4,
    )
    template.set_mode("train")
    observation_index = next(
        index
        for index, message in enumerate(source["messages"])
        if message["role"] == "tool_response"
    )
    marker = "\n[LOSS_MASKED_MEMORY_STRESS_SUFFIX]\n"

    def candidate(repetitions: int) -> tuple[dict, int]:
        row = copy.deepcopy(source)
        row["query_id"] = f"probe:{args.target_length}:{source['query_id']}"
        row["messages"][observation_index]["content"] += marker + " x" * repetitions
        return row, len(template.encode(row)["input_ids"])

    low, high = 0, args.target_length
    while candidate(high)[1] < args.target_length:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if candidate(middle)[1] <= args.target_length:
            low = middle
        else:
            high = middle
    row, encoded_length = candidate(low)
    if encoded_length > args.target_length or args.target_length - encoded_length > 4:
        raise RuntimeError(
            f"failed to construct near-exact stress length: {encoded_length}/{args.target_length}"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe dataset: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_query_id": source["query_id"],
        "source_dataset_sha256": sha256_file(args.dataset),
        "probe_dataset_sha256": sha256_file(args.output),
        "target_length": args.target_length,
        "encoded_length": encoded_length,
        "stress_suffix_tokens_are_loss_masked": True,
        "final_answer_preserved": row["messages"][-1] == source["messages"][-1],
        "assistant_actions_preserved": [
            message for message in row["messages"] if message["role"] == "tool_call"
        ]
        == [message for message in source["messages"] if message["role"] == "tool_call"],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
