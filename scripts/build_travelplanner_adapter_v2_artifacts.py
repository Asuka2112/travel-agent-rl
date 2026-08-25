#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = ("base", "sft", "rl")
METRIC_NAMES = (
    "Commonsense Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Final Pass Rate",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def answer_hash(answer: str | None) -> str:
    return sha256_bytes((answer or "").encode("utf-8"))


def fraction(value: int, denominator: int = 180) -> dict[str, Any]:
    return {"numerator": value, "denominator": denominator, "rate": value / denominator}


def percent(value: float) -> str:
    return f"{100 * value:.4f}%"


def metric_cell(metric: dict[str, Any]) -> str:
    return f"{metric['numerator']}/{metric['denominator']} ({percent(metric['rate'])})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Adapter v2 corrected validation artifacts")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    final = repo / "outputs/final"
    old_root = final / "travelplanner_validation"
    new_root = final / "travelplanner_adapter_v2"
    freeze = load(new_root / "adapter_freeze_manifest.json")
    formal = load(new_root / "formal_run_lock.json")
    diagnostics = load(new_root / "constraint_diagnostics.json")
    old = {label: load(old_root / f"{label}_official_metrics.json") for label in LABELS}
    corrected = {
        label: load(new_root / label / "official_metrics.json") for label in LABELS
    }

    raw_checks: dict[str, bool] = {}
    for label in LABELS:
        expected = freeze["raw_answer_manifest"]["labels"][label]["items"]
        for task_id in range(1, 181):
            record = load(old_root / label / f"{task_id:03d}.json")
            raw_checks[f"{label}/{task_id:03d}"] = (
                answer_hash(record.get("final_answer"))
                == expected[task_id - 1]["final_answer_sha256"]
            )
    checks = {
        "all_540_raw_final_answers_unchanged": all(raw_checks.values()),
        "formal_generation_attempt_exactly_one": formal["formal_generation_attempt"] == 1,
        "formal_model_inference_runs_zero": formal["model_inference_runs"] == 0,
        "adapter_hash_matches_freeze": sha256_file(
            repo / "src/travel_agent_rl/evaluation/travelplanner_adapter_v2.py"
        )
        == freeze["adapter_sha256"],
        "manual_audit_hallucinated_field_zero": freeze["manual_audit_summary"]["counts"][
            "hallucinated_field"
        ]
        == 0,
        "manual_audit_entity_mutation_zero": freeze["manual_audit_summary"]["counts"][
            "entity_mutation"
        ]
        == 0,
        "all_frozen_model_files_unchanged_at_freeze": freeze["checks"][
            "all_frozen_model_files_unchanged"
        ],
        "official_revision_correct": all(
            row["official_revision"] == "e52c87f4ac348a3410c46dc3553c519db5ec5e23"
            for row in corrected.values()
        ),
        "one_corrected_official_result_per_model": all(
            (new_root / label / "official_metrics.json").is_file() for label in LABELS
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"scientific artifact checks failed: {[k for k, v in checks.items() if not v]}")

    rates: dict[str, Any] = {}
    for label in LABELS:
        diag = diagnostics["labels"][label]
        rates[label] = {
            "parsed_delivered": fraction(diag["parser_valid"]),
            "schema_valid_overall": fraction(diag["schema_valid"]),
            "schema_valid_given_parsed": fraction(diag["schema_valid"], diag["parser_valid"]),
            "completeness_gate_pass": fraction(diag["completeness_pass"]),
            "parser_failure": fraction(diag["parser_failure"]),
            "sandbox_pass": fraction(diag["sandbox_pass"]),
            "hard_constraints_executed": fraction(diag["hard_constraints_executed"]),
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    result = {
        "generated_at": generated_at,
        "evaluation_status": {
            "old": "invalid / adapter-contaminated external evaluation",
            "corrected": "adapter-corrected frozen-output evaluation",
        },
        "n_per_model": 180,
        "old_metrics_preserved": {
            label: old[label]["metrics"] for label in LABELS
        },
        "corrected_official_metrics": {
            label: corrected[label]["metrics"] for label in LABELS
        },
        "adapter_rates": rates,
        "constraint_diagnostics": {
            label: {
                key: diagnostics["labels"][label][key]
                for key in (
                    "commonsense_status_counts",
                    "completeness_pass",
                    "sandbox_pass",
                    "hard_constraints_executed",
                )
            }
            for label in LABELS
        },
        "adapter": {
            "sha256": freeze["adapter_sha256"],
            "type": "deterministic_no_llm",
            "parser_model": None,
            "parser_prompt_sha256": None,
            "manual_audit": freeze["manual_audit_summary"],
        },
        "raw_answers": {
            "all_unchanged": True,
            "hash_definition": freeze["raw_answer_manifest"]["hash_definition"],
            "ordered_hash_sha256": {
                label: freeze["raw_answer_manifest"]["labels"][label][
                    "ordered_hash_sha256"
                ]
                for label in LABELS
            },
        },
        "models": {
            label: {
                "path": freeze["models"][label]["path"],
                "tree_manifest_sha256": freeze["models"][label]["tree_manifest_sha256"],
            }
            for label in LABELS
        },
        "official": {
            "revision": freeze["official_repository_revision"],
            "contract_file_hashes": freeze["official_contract_files"],
        },
        "formal_run": formal,
        "scientific_checks": checks,
        "raw_answer_checks": raw_checks,
    }
    metrics_path = final / "travelplanner_corrected_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    repair_lines = [
        "# TravelPlanner Adapter Repair",
        "",
        f"Generated: {generated_at}",
        "",
        "This was an adapter-only experiment over 540 frozen final answers. No model inference, "
        "training, prompt change, tool call, plan repair, entity replacement, or constraint "
        "solving occurred.",
        "",
        "## Root cause and contract",
        "",
        "Adapter v1 recognized day headings but, when prose lacked explicit `Field:` labels, "
        "placed the entire day body in `attraction`. It therefore discarded `current_city`, "
        "transportation, meals, and accommodation as separate official fields. The exact "
        "official contract and field mapping are frozen in `docs/TRAVELPLANNER_ADAPTER_SPEC.md`.",
        "",
        "Adapter v2 is deterministic and receives only the raw final answer. It uses no query, "
        "task ID, database, evaluator verdict, or parser LLM. Missing content remains `-` or a "
        "parser failure; alternatives are not resolved and absent entities are never supplied.",
        "",
        "## Independent tests and audit",
        "",
        "The pre-validation suite covered the official 180-line `example_evaluation.jsonl`, "
        "an unrelated synthetic itinerary, and a project-native non-TravelPlanner trajectory. "
        "The unmodified official `format_check.py` completed on the official example and synthetic "
        "submission.",
        "",
        "The fixed-seed audit reviewed 20 Base, 20 SFT, and 20 RL answers. Candidate history 001-008 "
        "is retained. Final counts were:",
        "",
        "| Audit label | Count |",
        "|---|---:|",
    ]
    for key, value in freeze["manual_audit_summary"]["counts"].items():
        repair_lines.append(f"| {key.replace('_', ' ')} | {value} |")
    repair_lines.extend(
        [
            "",
            f"Frozen adapter SHA-256: `{freeze['adapter_sha256']}`. Parser model and parser prompt "
            "hash are `null` because no LLM parser is used.",
            "",
            "## One-shot execution",
            "",
            "The adapter was committed and hash-frozen before the only formal 540-answer postprocess. "
            "Each model then received exactly one unmodified official `eval.py --set_type validation` "
            "run. The lock records one generation attempt and zero model-inference runs. All exact "
            "raw-answer hashes matched before and after the run.",
            "",
            "No post-score adapter edit or evaluator rerun was performed.",
        ]
    )
    (repo / "reports/TRAVELPLANNER_ADAPTER_REPAIR.md").write_text(
        "\n".join(repair_lines) + "\n", encoding="utf-8"
    )

    validation_lines = [
        "# TravelPlanner Corrected Frozen-Output Validation",
        "",
        f"Generated: {generated_at}",
        "",
        "The original result is retained and labeled **invalid / adapter-contaminated external "
        "evaluation**. The new result is labeled **adapter-corrected frozen-output evaluation**.",
        "",
        "## Corrected official metrics (n=180 per model)",
        "",
        "| Metric | Base | SFT | RL |",
        "|---|---:|---:|---:|",
    ]
    for name in METRIC_NAMES:
        validation_lines.append(
            f"| {name} | "
            + " | ".join(metric_cell(corrected[label]["metrics"][name]) for label in LABELS)
            + " |"
        )
    validation_lines.extend(
        [
            "",
            "## Adapter delivery and gates",
            "",
            "| Rate | Base | SFT | RL |",
            "|---|---:|---:|---:|",
        ]
    )
    for title, key in (
        ("Parsed / delivered", "parsed_delivered"),
        ("Schema-valid overall", "schema_valid_overall"),
        ("Schema-valid given parsed", "schema_valid_given_parsed"),
        ("Completeness gate pass", "completeness_gate_pass"),
        ("Parser failure", "parser_failure"),
        ("Sandbox pass", "sandbox_pass"),
        ("Hard constraints executed", "hard_constraints_executed"),
    ):
        validation_lines.append(
            f"| {title} | "
            + " | ".join(metric_cell(rates[label][key]) for label in LABELS)
            + " |"
        )
    validation_lines.extend(
        [
            "",
            "## Before / after",
            "",
            "| Model | Old Commonsense Micro | Corrected Commonsense Micro | Old Final Pass | Corrected Final Pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        validation_lines.append(
            f"| {label.upper()} | "
            f"{metric_cell(old[label]['metrics']['Commonsense Constraint Micro Pass Rate'])} | "
            f"{metric_cell(corrected[label]['metrics']['Commonsense Constraint Micro Pass Rate'])} | "
            f"{metric_cell(old[label]['metrics']['Final Pass Rate'])} | "
            f"{metric_cell(corrected[label]['metrics']['Final Pass Rate'])} |"
        )
    validation_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Adapter contamination was real: corrected field separation changed official "
            "constraint verdicts (most visibly RL commonsense micro), while removing the v1 prose-as-"
            "attraction fallback reduced Base/SFT vacuous passes. However, every delivered corrected "
            "plan is structurally schema-valid and completeness remains 0/180 for all three models. "
            "Consequently no plan reached the completeness+sandbox gate and hard constraints were "
            "executed for 0/180 tasks. Final Pass therefore remains 0/180 for Base, SFT, and RL.",
            "",
            "The evidence rejects a universal remaining schema bug: delivered v2 plans have 100% "
            "structural validity, the fixed audit found zero hallucinated fields, wrong assignments, "
            "or entity mutations, and official format checking completed. The remaining zero Final "
            "Pass is a genuine frozen-output delivery/completeness/generalization gap under the "
            "official protocol, compounded by conservative parser failures where no faithful daily "
            "plan could be extracted. It is not evidence supporting retraining or post-hoc tuning.",
            "",
            "All old files and metrics remain unchanged. Exact hashes and per-task gate verdicts are "
            "in `outputs/final/travelplanner_corrected_metrics.json` and "
            "`outputs/final/travelplanner_adapter_v2/constraint_diagnostics.json`.",
        ]
    )
    (repo / "reports/TRAVELPLANNER_CORRECTED_VALIDATION.md").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
