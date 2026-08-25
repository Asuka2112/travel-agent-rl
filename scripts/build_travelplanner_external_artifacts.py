#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = ("base", "sft", "rl")
OFFICIAL_NAMES = (
    "Commonsense Constraint Micro Pass Rate",
    "Commonsense Constraint Macro Pass Rate",
    "Hard Constraint Micro Pass Rate",
    "Hard Constraint Macro Pass Rate",
    "Final Pass Rate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen external-validation artifacts")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "outputs/final"
    validation_dir = output / "travelplanner_validation"
    generated_at = datetime.now(timezone.utc).isoformat()
    internal = load(output / "final_metrics.json")
    pairwise = load(output / "pairwise_metrics.json")
    contamination = load(validation_dir / "contamination_audit.json")
    protocol = load(validation_dir / "protocol_manifest.json")
    project = {
        label: load(validation_dir / f"{label}_project_metrics.json") for label in LABELS
    }
    official = {
        label: load(validation_dir / f"{label}_official_metrics.json") for label in LABELS
    }

    checks = {
        "contamination_gate_passed": contamination["gate"] == "passed",
        "contamination_all_overlap_counts_zero": all(
            value == 0
            for key, value in contamination["totals"].items()
            if key != "candidate_rows_scanned"
        ),
        "all_models_have_180_project_records": all(row["n"] == 180 for row in project.values()),
        "all_models_have_180_official_records": all(row["n"] == 180 for row in official.values()),
        "validation_hash_parity": len(
            {row["validation_sha256"] for row in project.values()}
            | {protocol["validation"]["sha256"]}
        )
        == 1,
        "generation_protocol_parity": len(
            {json.dumps(row["protocol"], sort_keys=True) for row in project.values()}
        )
        == 1,
        "official_revision_parity": {
            row["official_revision"] for row in official.values()
        }
        == {"e52c87f4ac348a3410c46dc3553c519db5ec5e23"},
        "official_evaluator_hash_parity": len(
            {row["evaluator_sha256"] for row in official.values()}
        )
        == 1,
        "public_network_requests_zero": all(
            row["public_network_requests"] == 0 for row in project.values()
        ),
        "no_unknown_generation_exceptions": all(
            row["unknown_exceptions"] == 0 for row in project.values()
        ),
    }
    model_file_checks: dict[str, bool] = {}
    for label, model in protocol["models"].items():
        model_root = root / model["path"]
        for entry in model["files"]:
            path = model_root / entry["path"]
            key = f"{label}:{entry['path']}"
            model_file_checks[key] = (
                path.is_file()
                and path.stat().st_size == entry["bytes"]
                and sha256_file(path) == entry["sha256"]
            )
    checks["all_frozen_model_files_unchanged"] = all(model_file_checks.values())
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"external scientific audit failed: {failed}")

    compact_pairwise = [
        {
            key: row[key]
            for key in (
                "left",
                "right",
                "queries",
                "win",
                "tie",
                "loss",
                "position_disagreement",
                "pairwise_score",
            )
        }
        for row in pairwise["pairs"]
    ]
    result = {
        "generated_at": generated_at,
        "evaluation_type": "frozen post-hoc official TravelPlanner validation",
        "n": 180,
        "contamination_audit": {
            "gate": contamination["gate"],
            "candidate_rows_scanned": contamination["totals"]["candidate_rows_scanned"],
            "overlaps": {
                key: value
                for key, value in contamination["totals"].items()
                if key != "candidate_rows_scanned"
            },
        },
        "deterministic_rule_based": {
            label: official[label]["metrics"] for label in LABELS
        },
        "project_metrics": project,
        "internal_frozen_eval": {
            "n": 16,
            "task_success": {
                label: {
                    "numerator": round(internal["metrics"][label]["task_success"] * 16),
                    "denominator": 16,
                    "rate": internal["metrics"][label]["task_success"],
                }
                for label in LABELS
            },
        },
        "pointwise_judge": {
            "scope": "internal frozen 16-query evaluation only",
            "score_1_to_5": {
                label: internal["metrics"][label]["judge_score_1_to_5"] for label in LABELS
            },
        },
        "pairwise_judge": {
            "scope": "preserved internal frozen 16-query A/B plus B/A evaluation only",
            "judge_calls": pairwise["judge_calls"],
            "judge_failures": pairwise["judge_failures"],
            "pairs": compact_pairwise,
            "position_disagreement_total": sum(
                row["position_disagreement"] for row in pairwise["pairs"]
            ),
        },
    }
    metrics_path = output / "external_validation_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    audit = {
        "generated_at": generated_at,
        "overall_pass": True,
        "checks": checks,
        "model_file_checks": model_file_checks,
        "protocol_manifest_sha256": sha256_file(validation_dir / "protocol_manifest.json"),
        "external_metrics_sha256": sha256_file(metrics_path),
    }
    audit_path = output / "external_validation_scientific_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")

    artifacts = [
        metrics_path,
        audit_path,
        validation_dir / "contamination_audit.json",
        validation_dir / "protocol_manifest.json",
        *(validation_dir / f"{label}_project_metrics.json" for label in LABELS),
        *(validation_dir / f"{label}_official_metrics.json" for label in LABELS),
    ]
    repro = {
        "generated_at": generated_at,
        "source_commit_before_final_validation_report": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "protocol_manifest": protocol,
        "official_repository": official["base"]["official_repository"],
        "official_revision": official["base"]["official_revision"],
        "artifacts": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
        "commands": {
            "generation": (
                ".venv-train/bin/python scripts/evaluate_travelplanner_validation.py "
                "--model <frozen-model> --validation data/raw/travelplanner/validation.csv "
                "--output-dir <immutable-output> --label <base|sft|rl> --device cuda:<safe-gpu>"
            ),
            "official_evaluation": (
                "python scripts/run_travelplanner_official_eval.py --official-root <official-repo> "
                "--submission <official_submission.jsonl> --output <official_metrics.json>"
            ),
            "audit": "python scripts/build_travelplanner_external_artifacts.py",
        },
    }
    (output / "external_validation_repro_manifest.json").write_text(
        json.dumps(repro, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# TravelPlanner External Validation",
        "",
        f"Generated: {generated_at}",
        "",
        "This is a frozen post-hoc evaluation on the official 180-query TravelPlanner validation "
        "split. No validation feedback was used for training, prompt changes, tool changes, or "
        "hyperparameter tuning.",
        "",
        "## Deterministic official metrics",
        "",
        "| Model | Commonsense Micro | Commonsense Macro | Hard Micro | Hard Macro | Final Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        metrics = official[label]["metrics"]
        lines.append(
            f"| {label.upper()} | "
            + " | ".join(percent(metrics[name]["rate"]) for name in OFFICIAL_NAMES)
            + " |"
        )
    lines.extend(
        [
            "",
            "All official values above were produced by the unmodified official evaluator at "
            "revision `e52c87f4ac348a3410c46dc3553c519db5ec5e23`. They are not Judge scores.",
            "",
            "## Final Pass and project Task Success uncertainty",
            "",
            "| Model | Official Final Pass (95% Wilson CI) | Project Task Success (95% Wilson CI) |",
            "|---|---:|---:|",
        ]
    )
    for label in LABELS:
        final_pass = official[label]["metrics"]["Final Pass Rate"]
        task = project[label]["task_success"]
        lines.append(
            f"| {label.upper()} | {final_pass['numerator']}/{final_pass['denominator']} "
            f"({percent(final_pass['rate'])}; "
            f"{percent(final_pass['ci95_wilson'][0])}–{percent(final_pass['ci95_wilson'][1])}) | "
            f"{task['numerator']}/{task['denominator']} ({percent(task['rate'])}; "
            f"{percent(task['ci95_wilson'][0])}–{percent(task['ci95_wilson'][1])}) |"
        )
    lines.extend(
        [
            "",
            "Project Task Success means that the final answer deterministically adapted to exactly "
            "the requested number of official-schema days; it must not be read as official Final Pass.",
            "",
            "## Project trajectory metrics",
            "",
            "| Model | Valid Tool | Final Answer | Avg Turns | Avg Tool Calls | Loop |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        row = project[label]
        lines.append(
            f"| {label.upper()} | {percent(row['valid_tool_rate'])} | "
            f"{percent(row['final_answer_rate'])} | {row['average_turns']:.3f} | "
            f"{row['average_tool_calls']:.3f} | {percent(row['loop_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Internal metrics retained separately",
            "",
            "Internal Frozen Eval uses n=16 synthetic engineering coverage queries: Base 14/16 "
            "(0.875), SFT 11/16 (0.6875), RL 16/16 (1.0). Pointwise Judge scores and the "
            "existing A/B+B/A pairwise results apply only to that internal set and do not replace "
            "the deterministic TravelPlanner metrics.",
            "",
            "Pairwise position disagreements: "
            + ", ".join(
                f"{row['left']} vs {row['right']}={row['position_disagreement']}/{row['queries']}"
                for row in compact_pairwise
            )
            + f"; total={result['pairwise_judge']['position_disagreement_total']}/48.",
            "",
            "## Scientific controls",
            "",
            f"Contamination audit: **{contamination['gate']}**, scanning "
            f"{contamination['totals']['candidate_rows_scanned']} training-derived rows with zero "
            "exact/canonical, constraint-fingerprint, or thresholded fuzzy overlaps. Base, SFT, and "
            "RL used identical prompt, greedy decoding, eight-tool replay environment, turn/token "
            "limits, post-processing, and official evaluation code. Frozen model hashes were "
            "rechecked after evaluation; public generation requests were zero.",
        ]
    )
    report = root / "reports/TRAVELPLANNER_EXTERNAL_VALIDATION.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": str(metrics_path), "report": str(report), "checks": checks}))


if __name__ == "__main__":
    main()
