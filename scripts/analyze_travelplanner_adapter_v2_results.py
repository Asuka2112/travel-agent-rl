#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("base", "sft", "rl")
COMMONSENSE_KEYS = (
    "is_reasonable_visiting_city",
    "is_valid_restaurants",
    "is_valid_attractions",
    "is_valid_accommodation",
    "is_valid_transportation",
    "is_valid_information_in_current_city",
    "is_valid_information_in_sandbox",
    "is_not_absent",
)


def load_validation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("days", "visiting_city_number", "people_number", "budget"):
            row[key] = int(row[key])
        for key in ("date", "local_constraint", "reference_information"):
            if isinstance(row.get(key), str):
                row[key] = ast.literal_eval(row[key])
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_official_modules(root: Path):
    root = root.resolve()
    evaluation = root / "evaluation"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(evaluation))
    os.chdir(evaluation)
    commonsense = importlib.import_module("commonsense_constraint")
    hard = importlib.import_module("hard_constraint")
    return commonsense, hard


def status(value: Any) -> str:
    if value is None:
        return "not_evaluated"
    if value[0] is None:
        return "not_applicable"
    return "pass" if bool(value[0]) else "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose frozen Adapter v2 official inputs")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = load_validation(args.validation)
    if len(validation) != 180:
        raise ValueError(f"expected 180 validation rows, got {len(validation)}")
    commonsense_module, hard_module = load_official_modules(args.official_root)
    labels: dict[str, Any] = {}
    for label in LABELS:
        label_root = args.results_root / label
        submissions = load_jsonl(label_root / "official_submission.jsonl")
        metrics = json.loads((label_root / "official_metrics.json").read_text(encoding="utf-8"))
        if len(submissions) != 180:
            raise ValueError(f"{label}: expected 180 submissions")
        tasks = []
        hard_executed = 0
        for idx, (query, submission) in enumerate(zip(validation, submissions)):
            plan = submission.get("plan")
            info = commonsense_module.evaluation(query, plan) if plan else None
            verdicts = {
                key: status(info[key] if info else None) for key in COMMONSENSE_KEYS
            }
            gate = bool(
                info
                and info["is_not_absent"][0]
                and info["is_valid_information_in_sandbox"][0]
            )
            hard_status: dict[str, str] = {}
            if gate:
                hard_executed += 1
                hard_info = hard_module.evaluation(query, plan)
                hard_status = {key: status(value) for key, value in hard_info.items()}
            tasks.append(
                {
                    "idx": idx,
                    "parser_valid": plan is not None,
                    "schema_valid": bool(
                        plan
                        and all(
                            set(row)
                            == {
                                "day",
                                "current_city",
                                "transportation",
                                "breakfast",
                                "attraction",
                                "lunch",
                                "dinner",
                                "accommodation",
                            }
                            for row in plan
                        )
                    ),
                    "completeness_pass": bool(info and info["is_not_absent"][0]),
                    "sandbox_pass": bool(
                        info and info["is_valid_information_in_sandbox"][0]
                    ),
                    "hard_executed": gate,
                    "commonsense": verdicts,
                    "hard": hard_status,
                }
            )
        labels[label] = {
            "n": 180,
            "parser_valid": sum(task["parser_valid"] for task in tasks),
            "schema_valid": sum(task["schema_valid"] for task in tasks),
            "parser_failure": sum(not task["parser_valid"] for task in tasks),
            "completeness_pass": sum(task["completeness_pass"] for task in tasks),
            "sandbox_pass": sum(task["sandbox_pass"] for task in tasks),
            "hard_constraints_executed": hard_executed,
            "commonsense_status_counts": {
                key: dict(Counter(task["commonsense"][key] for task in tasks))
                for key in COMMONSENSE_KEYS
            },
            "official_metrics": metrics,
            "tasks": tasks,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"labels": labels}, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                label: {
                    key: labels[label][key]
                    for key in (
                        "parser_valid",
                        "schema_valid",
                        "parser_failure",
                        "completeness_pass",
                        "sandbox_pass",
                        "hard_constraints_executed",
                    )
                }
                for label in LABELS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
