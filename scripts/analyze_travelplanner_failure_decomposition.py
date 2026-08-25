#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
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
HARD_KEYS = (
    "valid_cost",
    "valid_room_rule",
    "valid_cuisine",
    "valid_room_type",
    "valid_transportation",
)
PLAN_FIELDS = (
    "day",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose frozen TravelPlanner failures")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-seed", type=int, default=20260826)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


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


def load_official_modules(root: Path):
    root = root.resolve()
    evaluation = root / "evaluation"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(evaluation))
    os.chdir(evaluation)
    commonsense = importlib.import_module("commonsense_constraint")
    hard = importlib.import_module("hard_constraint")
    return commonsense, hard


def verdict(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "not_evaluated", "pass": None, "message": None}
    passed, message = value
    if passed is None:
        status = "not_applicable"
    else:
        status = "pass" if bool(passed) else "fail"
    return {"status": status, "pass": None if passed is None else bool(passed), "message": message}


def applicable_hard(query: dict[str, Any]) -> set[str]:
    local = query["local_constraint"]
    keys = {"valid_cost"}
    mapping = {
        "house rule": "valid_room_rule",
        "cuisine": "valid_cuisine",
        "room type": "valid_room_type",
        "transportation": "valid_transportation",
    }
    keys.update(target for source, target in mapping.items() if local.get(source) is not None)
    return keys


def schema_findings(record: dict[str, Any], query: dict[str, Any]) -> list[dict[str, str]]:
    answer = record.get("final_answer")
    plan = record.get("official_plan")
    findings = []
    if not answer:
        return findings
    if plan is None:
        findings.append(
            {
                "type": "parsing_failure",
                "evidence": "final answer present but deterministic adapter produced null plan",
            }
        )
        if re.search(r"```(?:json)?|[\[{]", answer, re.I):
            findings.append(
                {
                    "type": "wrong_json_schema",
                    "evidence": "answer contains JSON-like syntax but no official plan was produced",
                }
            )
        return findings
    if len(plan) != query["days"]:
        findings.append(
            {
                "type": "malformed_itinerary_structure",
                "evidence": f"plan rows={len(plan)}, required days={query['days']}",
            }
        )
    expected_days = [str(index) for index in range(1, len(plan) + 1)]
    actual_days = [str(row.get("day", "")) for row in plan]
    if actual_days != expected_days:
        findings.append(
            {
                "type": "malformed_itinerary_structure",
                "evidence": f"day sequence={actual_days}, expected={expected_days}",
            }
        )
    missing = Counter()
    wrong_rows = 0
    for row in plan:
        if set(row) != set(PLAN_FIELDS):
            wrong_rows += 1
        for field in PLAN_FIELDS[1:]:
            if row.get(field) in (None, "", "-"):
                missing[field] += 1
    if wrong_rows:
        findings.append(
            {
                "type": "wrong_json_schema",
                "evidence": f"{wrong_rows} plan rows do not have exactly the official schema keys",
            }
        )
    if missing:
        findings.append(
            {
                "type": "missing_required_field",
                "evidence": ", ".join(f"{key}={count}" for key, count in sorted(missing.items())),
            }
        )
    if any(row.get("current_city") in (None, "", "-") for row in plan):
        findings.append(
            {
                "type": "malformed_itinerary_structure",
                "evidence": "one or more day rows lack current_city",
            }
        )
    return findings


def sandbox_entity(record: dict[str, Any], message: str | None) -> tuple[str | None, str | None]:
    if not message:
        return None, None
    day_match = re.search(r"day (\d+)", message, re.I)
    if not day_match or not record.get("official_plan"):
        return None, None
    row = record["official_plan"][int(day_match.group(1)) - 1]
    lowered = message.casefold()
    for field in ("breakfast", "lunch", "dinner", "accommodation", "attraction", "transportation"):
        if field in lowered:
            return field, str(row.get(field, ""))
    if "flight number" in lowered or "self-driving" in lowered or "taxi" in lowered:
        return "transportation", str(row.get("transportation", ""))
    return None, None


def environment_findings(
    record: dict[str, Any],
    sandbox_message: str | None,
    schema_findings: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    adapter = []
    environment = []
    field, entity = sandbox_entity(record, sandbox_message)
    if not field or not entity or entity in ("-", ""):
        return adapter, environment
    if schema_findings:
        adapter.append(
            {
                "type": "entity_validation_confounded_by_adapter",
                "evidence": (
                    f"official sandbox rejected {field}={entity!r}, but the same plan already "
                    "has parser/schema findings; no database mismatch is attributed"
                ),
            }
        )
        return adapter, environment
    if field == "transportation":
        parseable = bool(re.search(r"\bfrom\s+.+?\s+to\s+.+", entity, re.I))
    else:
        parseable = bool(re.match(r"^.+?,\s*[^,]+(?:\([^)]*\))?$", entity))
    if not parseable:
        adapter.append(
            {
                "type": "entity_normalization_failure",
                "evidence": f"{field}={entity!r} does not match the official evaluator format",
            }
        )
        return adapter, environment
    kind = (
        "tool_result_unavailable_in_official_sandbox"
        if field == "transportation"
        else "entity_absent_from_official_database"
    )
    environment.append({"type": kind, "evidence": f"{field}={entity!r}; {sandbox_message}"})
    tool_text = "\n".join(
        str(message.get("content", ""))
        for message in record.get("messages", [])
        if message.get("role") == "tool_response"
    ).casefold()
    name = entity.split(",", 1)[0].strip().casefold()
    if len(name) >= 3 and name in tool_text:
        environment.append(
            {
                "type": "local_environment_entity_differs_from_official_record",
                "evidence": f"entity {name!r} appears in local tool response but failed official sandbox",
            }
        )
    return adapter, environment


def model_failure_categories(
    commonsense: dict[str, dict[str, Any]],
    diagnostic_hard: dict[str, dict[str, Any]],
    schema: list[dict[str, str]],
) -> list[str]:
    missing_structure = any(
        finding["type"] in {"missing_required_field", "malformed_itinerary_structure"}
        for finding in schema
    )
    categories = set()
    mapping = {
        "is_reasonable_visiting_city": "city_coverage",
        "is_valid_restaurants": "other_official_constraints",
        "is_valid_attractions": "attraction",
        "is_valid_accommodation": "accommodation",
        "is_valid_transportation": "transportation",
        "is_valid_information_in_current_city": "temporal_consistency",
    }
    for key, category in mapping.items():
        if commonsense[key]["status"] == "fail" and not missing_structure:
            categories.add(category)
    hard_mapping = {
        "valid_cost": "budget",
        "valid_room_rule": "accommodation",
        "valid_cuisine": "cuisine",
        "valid_room_type": "accommodation",
        "valid_transportation": "transportation",
    }
    for key, category in hard_mapping.items():
        if diagnostic_hard[key]["status"] == "fail" and not missing_structure:
            categories.add(category)
    return sorted(categories)


def count_status(tasks: list[dict[str, Any]], group: str, keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: dict(Counter(task[group][key]["status"] for task in tasks))
        for key in keys
    }


def taxonomy_counts(tasks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    planning = Counter(
        category for task in tasks for category in task["model_planning_failure_categories"]
    )
    adapter = Counter(
        finding["type"] for task in tasks for finding in task["adapter_findings"]
    )
    environment = Counter(
        finding["type"] for task in tasks for finding in task["environment_findings"]
    )
    return {
        "model_planning": {
            key: planning[key]
            for key in (
                "budget",
                "transportation",
                "accommodation",
                "cuisine",
                "attraction",
                "temporal_consistency",
                "city_coverage",
                "other_official_constraints",
            )
        },
        "output_adapter": {
            key: adapter[key]
            for key in (
                "parsing_failure",
                "missing_required_field",
                "wrong_json_schema",
                "entity_normalization_failure",
                "malformed_itinerary_structure",
                "entity_validation_confounded_by_adapter",
            )
        },
        "environment_database": {
            key: environment[key]
            for key in (
                "entity_absent_from_official_database",
                "tool_result_unavailable_in_official_sandbox",
                "local_environment_entity_differs_from_official_record",
            )
        },
    }


def sample_extremes(tasks: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    delivered = [task for task in tasks if task["delivered"]]
    minimum = min(task["distance_to_final_pass"] for task in delivered)
    maximum = max(task["distance_to_final_pass"] for task in tasks)
    closest_pool = [task for task in delivered if task["distance_to_final_pass"] == minimum]
    worst_pool = [task for task in tasks if task["distance_to_final_pass"] == maximum]

    def choose(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chosen = rng.sample(pool, min(10, len(pool)))
        return [
            {
                "idx": task["idx"],
                "query": task["query"],
                "distance_to_final_pass": task["distance_to_final_pass"],
                "failed_constraints": task["failed_constraints"],
                "adapter_findings": task["adapter_findings"],
                "environment_findings": task["environment_findings"],
                "final_answer_excerpt": task["final_answer_excerpt"],
                "plan_nonempty_fields": task["plan_nonempty_fields"],
            }
            for task in sorted(chosen, key=lambda item: item["idx"])
        ]

    return {
        "seed": seed,
        "closest_distance": minimum,
        "closest_pool_size": len(closest_pool),
        "closest_10": choose(closest_pool),
        "worst_distance": maximum,
        "worst_pool_size": len(worst_pool),
        "worst_10": choose(worst_pool),
    }


def main() -> None:
    args = parse_args()
    args.official_root = args.official_root.resolve()
    args.validation = args.validation.resolve()
    args.raw_root = args.raw_root.resolve()
    args.metrics_root = args.metrics_root.resolve()
    args.output = args.output.resolve()
    validation = load_validation(args.validation)
    commonsense_module, hard_module = load_official_modules(args.official_root)
    models = {}
    for label in LABELS:
        official_metrics = json.loads(
            (args.metrics_root / f"{label}_official_metrics.json").read_text(encoding="utf-8")
        )
        project_metrics = json.loads(
            (args.metrics_root / f"{label}_project_metrics.json").read_text(encoding="utf-8")
        )
        tasks = []
        for idx, query in enumerate(validation):
            record = json.loads(
                (args.raw_root / label / f"{idx + 1:03d}.json").read_text(encoding="utf-8")
            )
            plan = record.get("official_plan")
            commonsense_raw = None
            official_hard_raw = None
            diagnostic_hard_raw = None
            evaluator_exception = None
            if plan:
                try:
                    commonsense_raw = commonsense_module.evaluation(query, plan)
                    if (
                        commonsense_raw["is_not_absent"][0]
                        and commonsense_raw["is_valid_information_in_sandbox"][0]
                    ):
                        official_hard_raw = hard_module.evaluation(query, plan)
                    diagnostic_hard_raw = hard_module.evaluation(query, plan)
                except Exception as exc:  # preserve official-parser failures as evidence
                    evaluator_exception = f"{type(exc).__name__}: {exc}"
            commonsense = {
                key: verdict(commonsense_raw.get(key) if commonsense_raw else None)
                for key in COMMONSENSE_KEYS
            }
            official_hard = {
                key: verdict(official_hard_raw.get(key) if official_hard_raw else None)
                for key in HARD_KEYS
            }
            diagnostic_hard = {
                key: verdict(diagnostic_hard_raw.get(key) if diagnostic_hard_raw else None)
                for key in HARD_KEYS
            }
            schema = schema_findings(record, query)
            sandbox_message = commonsense["is_valid_information_in_sandbox"]["message"]
            normalization, environment = environment_findings(record, sandbox_message, schema)
            schema.extend(normalization)
            applicable = applicable_hard(query)
            failed = [
                f"commonsense.{key}"
                for key in COMMONSENSE_KEYS
                if commonsense[key]["status"] in {"fail", "not_evaluated"}
            ]
            failed.extend(
                f"hard.{key}"
                for key in applicable
                if diagnostic_hard[key]["status"] in {"fail", "not_evaluated"}
            )
            plan_nonempty = Counter()
            for row in plan or []:
                for field in PLAN_FIELDS[1:]:
                    if row.get(field) not in (None, "", "-"):
                        plan_nonempty[field] += 1
            final_answer = record.get("final_answer") or ""
            planning_categories = model_failure_categories(commonsense, diagnostic_hard, schema)
            if not final_answer:
                planning_categories = sorted({*planning_categories, "other_official_constraints"})
            task = {
                "idx": idx,
                "query": query["query"],
                "internal_task_success": bool(record.get("task_complete")),
                "final_answer_present": bool(final_answer),
                "delivered": bool(plan),
                "official_final_pass": bool(
                    commonsense_raw
                    and official_hard_raw
                    and all(value[0] is not False for value in commonsense_raw.values())
                    and all(value[0] is not False for value in official_hard_raw.values())
                ),
                "commonsense": commonsense,
                "official_hard": official_hard,
                "diagnostic_ungated_hard": diagnostic_hard,
                "applicable_hard_constraints": sorted(applicable),
                "failed_constraints": sorted(failed),
                "distance_to_final_pass": len(failed),
                "adapter_findings": schema,
                "environment_findings": environment,
                "model_planning_failure_categories": planning_categories,
                "evaluator_exception": evaluator_exception,
                "plan_nonempty_fields": dict(plan_nonempty),
                "final_answer_excerpt": final_answer[:600],
            }
            tasks.append(task)

        computed = {
            "commonsense_micro_numerator": sum(
                task["commonsense"][key]["status"] == "pass"
                for task in tasks
                for key in COMMONSENSE_KEYS
            ),
            "commonsense_macro_numerator": sum(
                task["delivered"]
                and all(task["commonsense"][key]["status"] != "fail" for key in COMMONSENSE_KEYS)
                for task in tasks
            ),
            "hard_micro_numerator": sum(
                task["official_hard"][key]["status"] == "pass"
                for task in tasks
                for key in HARD_KEYS
            ),
            "hard_macro_numerator": sum(
                any(task["official_hard"][key]["status"] != "not_evaluated" for key in HARD_KEYS)
                and all(task["official_hard"][key]["status"] != "fail" for key in HARD_KEYS)
                for task in tasks
            ),
            "final_pass_numerator": sum(task["official_final_pass"] for task in tasks),
        }
        expected = official_metrics["metrics"]
        parity = {
            "commonsense_micro": computed["commonsense_micro_numerator"]
            == expected["Commonsense Constraint Micro Pass Rate"]["numerator"],
            "commonsense_macro": computed["commonsense_macro_numerator"]
            == expected["Commonsense Constraint Macro Pass Rate"]["numerator"],
            "hard_micro": computed["hard_micro_numerator"]
            == expected["Hard Constraint Micro Pass Rate"]["numerator"],
            "hard_macro": computed["hard_macro_numerator"]
            == expected["Hard Constraint Macro Pass Rate"]["numerator"],
            "final_pass": computed["final_pass_numerator"]
            == expected["Final Pass Rate"]["numerator"],
        }
        if not all(parity.values()):
            raise RuntimeError(f"{label} per-task evaluator parity failed: {parity} {computed}")
        internal_success = [task for task in tasks if task["internal_task_success"]]
        models[label] = {
            "official_metrics": expected,
            "project_metrics": project_metrics,
            "delivery": {
                "numerator": sum(task["delivered"] for task in tasks),
                "denominator": 180,
            },
            "format_valid": {
                "numerator": sum(task["internal_task_success"] for task in tasks),
                "denominator": 180,
                "definition": project_metrics["task_success"]["definition"],
            },
            "constraint_counts": {
                "commonsense": count_status(tasks, "commonsense", COMMONSENSE_KEYS),
                "official_hard": count_status(tasks, "official_hard", HARD_KEYS),
                "diagnostic_ungated_hard": count_status(
                    tasks, "diagnostic_ungated_hard", HARD_KEYS
                ),
            },
            "distance_to_final_pass_histogram": dict(
                sorted(Counter(task["distance_to_final_pass"] for task in tasks).items())
            ),
            "one_constraint_away": sum(task["distance_to_final_pass"] == 1 for task in tasks),
            "two_constraints_away": sum(task["distance_to_final_pass"] == 2 for task in tasks),
            "parser_failure_tasks": sum(
                any(finding["type"] == "parsing_failure" for finding in task["adapter_findings"])
                for task in tasks
            ),
            "schema_or_adapter_failure_tasks": sum(bool(task["adapter_findings"]) for task in tasks),
            "official_database_mismatch_tasks": sum(
                bool(task["environment_findings"]) for task in tasks
            ),
            "taxonomy_task_counts": taxonomy_counts(tasks),
            "internal_task_success_subset": {
                "n": len(internal_success),
                "final_pass": sum(task["official_final_pass"] for task in internal_success),
                "failed_constraint_counts": dict(
                    Counter(
                        constraint for task in internal_success for constraint in task["failed_constraints"]
                    )
                ),
                "adapter_finding_task_counts": dict(
                    Counter(
                        finding["type"]
                        for task in internal_success
                        for finding in task["adapter_findings"]
                    )
                ),
                "distance_histogram": dict(
                    sorted(Counter(task["distance_to_final_pass"] for task in internal_success).items())
                ),
            },
            "evaluator_parity": parity,
            "tasks": tasks,
        }
    models["rl"]["random_manual_audit_sample"] = sample_extremes(
        models["rl"]["tasks"], args.audit_seed
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "no_outputs_modified_or_regenerated": True,
        "official_evaluator_revision": "e52c87f4ac348a3410c46dc3553c519db5ec5e23",
        "distance_definition": (
            "Number of failed/not-evaluated official commonsense checks plus failed/not-evaluated "
            "applicable hard checks from diagnostic ungated calls. Official hard verdicts remain "
            "marked not_evaluated when the official commonsense gate blocked them."
        ),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                label: {
                    "delivery": models[label]["delivery"],
                    "one_away": models[label]["one_constraint_away"],
                    "two_away": models[label]["two_constraints_away"],
                    "parser_failures": models[label]["parser_failure_tasks"],
                    "adapter_failures": models[label]["schema_or_adapter_failure_tasks"],
                    "database_mismatches": models[label]["official_database_mismatch_tasks"],
                }
                for label in LABELS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
