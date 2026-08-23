#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from swift.llm import get_model_tokenizer, get_template

from travel_agent_rl.sft.audit import length_bucket, nearest_rank, select_label_audit_rows
from travel_agent_rl.sft.data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Qwen3/ms-swift SFT labels and lengths")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/intermediate/sft/ms_swift_agent_881.jsonl")
    )
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--label-report", type=Path, default=Path("reports/sft_label_audit.md"))
    parser.add_argument(
        "--length-report", type=Path, default=Path("reports/sft_token_length_report.md")
    )
    parser.add_argument(
        "--artifacts", type=Path, default=Path("outputs/audit/sft_label_audit")
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def encode_all(rows: list[dict], model: Path):
    _, tokenizer = get_model_tokenizer(str(model), load_model=False)
    template_type = tokenizer.model_meta.template
    template = get_template(
        template_type,
        tokenizer,
        max_length=None,
        truncation_strategy="raise",
        agent_template="hermes",
        loss_scale="default",
    )
    template.set_mode("train")
    encoded = []
    for index, row in enumerate(rows, 1):
        item = template.encode(row)
        labels = item["labels"]
        if not labels or all(label == -100 for label in labels):
            raise RuntimeError(f"all labels are masked for {row['query_id']}")
        encoded.append(item)
        if index % 100 == 0:
            print(f"encoded {index}/{len(rows)}", flush=True)
    return tokenizer, template, encoded


def _rle_mask(labels: list[int], tokenizer) -> list[dict]:
    result = []
    start = 0
    trainable = labels[0] != -100
    for index in range(1, len(labels) + 1):
        next_trainable = labels[index] != -100 if index < len(labels) else not trainable
        if next_trainable == trainable:
            continue
        token_ids = [token for token in labels[start:index] if token != -100]
        result.append(
            {
                "start": start,
                "end_exclusive": index,
                "length": index - start,
                "loss": 1 if trainable else 0,
                "trainable_text": (
                    tokenizer.decode(token_ids, skip_special_tokens=False) if token_ids else ""
                ),
            }
        )
        start = index
        trainable = next_trainable
    return result


def _audit_one(row: dict, item: dict, tokenizer, template, directory: Path) -> dict:
    ids = item["input_ids"]
    labels = item["labels"]
    prompt = tokenizer.decode(ids, skip_special_tokens=False)
    labels_decoded = template.safe_decode(labels)
    trainable_ids = [label for label in labels if label != -100]
    trainable = tokenizer.decode(trainable_ids, skip_special_tokens=False)
    tool_calls = [m for m in row["messages"] if m["role"] == "tool_call"]
    observations = [m for m in row["messages"] if m["role"] == "tool_response"]
    final_answer = row["messages"][-1]["content"]
    user_content = row["messages"][0]["content"]

    checks = {
        "user_masked": user_content not in trainable,
        "tool_responses_masked": all(m["content"] not in trainable for m in observations),
        "all_tool_calls_trainable": trainable.count("<tool_call>") == len(tool_calls),
        "final_answer_trainable": final_answer in trainable,
        "all_assistant_actions_present": all(
            json.loads(message["content"])["name"] in trainable for message in tool_calls
        ),
        "tool_schema_in_prompt": all(
            f'"name": "{tool["function"]["name"]}"' in prompt
            for tool in json.loads(row["tools"])
        ),
        "has_trainable_labels": any(label != -100 for label in labels),
        "observation_not_forced": all(m["content"] not in labels_decoded for m in observations),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"label audit failed for {row['query_id']}: {failed}")

    directory.mkdir(parents=True, exist_ok=False)
    (directory / "input_ids.decoded.txt").write_text(prompt, encoding="utf-8")
    (directory / "labels.decoded.txt").write_text(labels_decoded, encoding="utf-8")
    (directory / "loss_mask.rle.json").write_text(
        json.dumps(_rle_mask(labels, tokenizer), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "query_id": row["query_id"],
        "difficulty": row["difficulty"],
        "tokens": len(ids),
        "trainable_tokens": len(trainable_ids),
        "tool_calls": len(tool_calls),
        "uses_train": any("train_search" in m["content"] for m in tool_calls),
        "uses_web": any(
            name in m["content"] for m in tool_calls for name in ("web_search", "web_scraping")
        ),
        "checks": checks,
        "directory": str(directory),
    }


def write_label_report(args, rows, encoded, tokenizer, template) -> None:
    selected = select_label_audit_rows(rows)
    by_id = {row["query_id"]: (row, item) for row, item in zip(rows, encoded, strict=True)}
    args.artifacts.mkdir(parents=True, exist_ok=False)
    audits = []
    for row in selected:
        source, item = by_id[row["query_id"]]
        safe_id = row["query_id"].replace(":", "_").replace("/", "_")
        audits.append(_audit_one(source, item, tokenizer, template, args.artifacts / safe_id))

    lines = [
        "# SFT Label and Loss-Mask Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Frozen source rows: {len(rows)}",
        f"- Audited distinct rows: {len(audits)}",
        f"- Model/tokenizer: `{args.model}`",
        f"- ms-swift template: `{tokenizer.model_meta.template}`",
        "- Agent template: `hermes`",
        "- Loss scale: `default`",
        f"- Dataset SHA-256: `{sha256_file(args.dataset)}`",
        "",
        "The loss mask is stored as exact token-index run-length spans: `loss=0` is masked and "
        "`loss=1` is trained. Each row also preserves the complete decoded input and labels.",
        "",
        "| Query | Difficulty | Tokens | Trainable | Calls | Train | Web | All checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for audit in audits:
        lines.append(
            f"| `{audit['query_id']}` | {audit['difficulty']} | {audit['tokens']} | "
            f"{audit['trainable_tokens']} | {audit['tool_calls']} | {audit['uses_train']} | "
            f"{audit['uses_web']} | {all(audit['checks'].values())} |"
        )
    lines += ["", "## Required checks", ""]
    labels = {
        "user_masked": "User content does not compute loss",
        "tool_responses_masked": "Tool responses/observations do not compute loss",
        "all_tool_calls_trainable": "Every tool call is learned",
        "final_answer_trainable": "The final assistant answer is learned",
        "all_assistant_actions_present": "No multi-turn assistant action is accidentally masked",
        "tool_schema_in_prompt": "The complete eight-tool schema enters the prompt",
        "has_trainable_labels": "No sample has all labels equal to -100",
        "observation_not_forced": "Observation text is absent from trainable labels",
    }
    for key, label in labels.items():
        lines.append(f"- PASS: {label} ({sum(a['checks'][key] for a in audits)}/{len(audits)})")
    lines += ["", "## Per-sample decoded artifacts", ""]
    for audit in audits:
        lines += [
            f"### `{audit['query_id']}`",
            "",
            f"Directory: `{audit['directory']}`",
            "",
            "Files: `input_ids.decoded.txt`, `labels.decoded.txt`, `loss_mask.rle.json`.",
            "",
        ]
    args.label_report.parent.mkdir(parents=True, exist_ok=True)
    args.label_report.write_text("\n".join(lines), encoding="utf-8")


def _stats(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "mean": round(statistics.fmean(values), 2),
        "median": statistics.median(values),
        "p75": nearest_rank(values, 75),
        "p90": nearest_rank(values, 90),
        "p95": nearest_rank(values, 95),
        "p99": nearest_rank(values, 99),
        "max": max(values),
    }


def write_length_report(args, rows, encoded) -> None:
    records = []
    for row, item in zip(rows, encoded, strict=True):
        tool_calls = sum(m["role"] == "tool_call" for m in row["messages"])
        records.append(
            {
                "query_id": row["query_id"],
                "difficulty": row["difficulty"],
                "turns": len(row["messages"]),
                "tool_calls": tool_calls,
                "token_length": len(item["input_ids"]),
            }
        )
    lengths = [record["token_length"] for record in records]
    overall = _stats(lengths)
    buckets = Counter(length_bucket(length) for length in lengths)
    by_difficulty = {
        difficulty: _stats(
            [record["token_length"] for record in records if record["difficulty"] == difficulty]
        )
        for difficulty in ("easy", "medium", "hard")
    }
    overlong = [record for record in records if record["token_length"] > 32768]
    longest = sorted(records, key=lambda record: (-record["token_length"], record["query_id"]))[:20]
    lines = [
        "# SFT Token Length Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Lengths use the frozen Qwen3-4B-Instruct-2507 tokenizer, `qwen3_nothinking` template, "
        "native `hermes` agent template, complete eight-tool schema, and no truncation.",
        "",
        "## Overall distribution",
        "",
        "| Statistic | Tokens |",
        "|---|---:|",
    ]
    lines += [f"| {key} | {value} |" for key, value in overall.items()]
    lines += [
        "",
        "## Buckets",
        "",
        "| Bucket | Count | Percent |",
        "|---|---:|---:|",
    ]
    for bucket in ("<=4K", "4K-8K", "8K-16K", "16K-24K", "24K-32K", ">32K"):
        count = buckets[bucket]
        lines.append(f"| {bucket} | {count} | {count / len(records) * 100:.2f}% |")
    lines += [
        "",
        "## By difficulty",
        "",
        "| Difficulty | Count | Min | Mean | Median | P75 | P90 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for difficulty, stats in by_difficulty.items():
        count = sum(record["difficulty"] == difficulty for record in records)
        lines.append(
            f"| {difficulty} | {count} | {stats['min']} | {stats['mean']} | "
            f"{stats['median']} | {stats['p75']} | {stats['p90']} | {stats['p95']} | "
            f"{stats['p99']} | {stats['max']} |"
        )
    lines += [
        "",
        "## Samples exceeding 32,768",
        "",
        f"Count: **{len(overlong)}**.",
        "",
    ]
    if overlong:
        lines += [
            "No sample has been silently truncated. The deterministic policy for training is: retain "
            "the system/tool schema, user request, every assistant tool call, the final tool round, and "
            "the complete final assistant answer; compact earlier tool responses only by removing "
            "low-priority response metadata and tail list items in original message order until the "
            "encoded sequence fits. Emit a per-row before/after hash and refuse the row if the final "
            "answer or any assistant action would be removed.",
            "",
            "| Query | Difficulty | Tokens | Excess | Turns | Tool calls |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for record in sorted(overlong, key=lambda item: -item["token_length"]):
            lines.append(
                f"| `{record['query_id']}` | {record['difficulty']} | {record['token_length']} | "
                f"{record['token_length'] - 32768} | {record['turns']} | {record['tool_calls']} |"
            )
    else:
        lines.append("No truncation policy is activated because every trajectory fits intact.")
    lines += [
        "",
        "## Longest 20 trajectories",
        "",
        "| Query | Difficulty | Turns | Tool calls | Token length |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in longest:
        lines.append(
            f"| `{record['query_id']}` | {record['difficulty']} | {record['turns']} | "
            f"{record['tool_calls']} | {record['token_length']} |"
        )
    args.length_report.parent.mkdir(parents=True, exist_ok=True)
    args.length_report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset)
    if len(rows) != 881:
        raise ValueError(f"expected 881 rows, got {len(rows)}")
    tokenizer, template, encoded = encode_all(rows, args.model)
    write_label_report(args, rows, encoded, tokenizer, template)
    write_length_report(args, rows, encoded)
    print(json.dumps({"rows": len(rows), "label_report": str(args.label_report), "length_report": str(args.length_report)}))


if __name__ == "__main__":
    main()
