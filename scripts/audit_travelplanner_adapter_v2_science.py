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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Final scientific audit for Adapter v2")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    new = repo / "outputs/final/travelplanner_adapter_v2"
    old = repo / "outputs/final/travelplanner_validation"
    freeze = load(new / "adapter_freeze_manifest.json")
    formal = load(new / "formal_run_lock.json")
    corrected = load(repo / "outputs/final/travelplanner_corrected_metrics.json")
    contamination = load(new / "contamination_audit.post_adapter.json")

    model_file_checks: dict[str, bool] = {}
    for label, model in freeze["models"].items():
        for entry in model["files"]:
            path = repo / model["path"] / entry["path"]
            model_file_checks[f"{label}:{entry['path']}"] = (
                path.is_file()
                and path.stat().st_size == entry["bytes"]
                and sha256_file(path) == entry["sha256"]
            )
    submission_checks = {}
    for label in LABELS:
        submission = new / label / "official_submission.jsonl"
        metrics = load(new / label / "official_metrics.json")
        submission_checks[label] = sha256_file(submission) == metrics["submission_sha256"]
    old_git_diff = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "0bb26016054ca8ea9f1e0701a0c4ada8e76a9e5f^",
            "--",
            str(old.relative_to(repo)),
        ],
        cwd=repo,
        check=False,
    ).returncode
    checks = {
        "all_model_files_unchanged_after_formal": all(model_file_checks.values()),
        "all_540_raw_answers_unchanged_after_formal": corrected["raw_answers"][
            "all_unchanged"
        ],
        "all_submission_hashes_match_official_results": all(submission_checks.values()),
        "formal_generation_attempt_exactly_one": formal["formal_generation_attempt"] == 1,
        "model_inference_runs_zero": formal["model_inference_runs"] == 0,
        "contamination_gate_passed": contamination["gate"] == "passed",
        "contamination_overlaps_zero": all(
            value == 0
            for key, value in contamination["totals"].items()
            if key != "candidate_rows_scanned"
        ),
        "old_validation_artifacts_unchanged_in_worktree": old_git_diff == 0,
        "adapter_still_matches_freeze": sha256_file(
            repo / "src/travel_agent_rl/evaluation/travelplanner_adapter_v2.py"
        )
        == freeze["adapter_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"scientific audit failed: {[key for key, value in checks.items() if not value]}")

    artifacts = [
        repo / "docs/TRAVELPLANNER_ADAPTER_SPEC.md",
        repo / "reports/TRAVELPLANNER_ADAPTER_REPAIR.md",
        repo / "reports/TRAVELPLANNER_CORRECTED_VALIDATION.md",
        repo / "outputs/final/travelplanner_corrected_metrics.json",
        new / "adapter_freeze_manifest.json",
        new / "formal_run_lock.json",
        new / "constraint_diagnostics.json",
        new / "contamination_audit.post_adapter.json",
        *[new / label / "official_submission.jsonl" for label in LABELS],
        *[new / label / "official_metrics.json" for label in LABELS],
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "generated_at": generated_at,
        "overall_pass": True,
        "checks": checks,
        "model_file_checks": model_file_checks,
        "submission_checks": submission_checks,
    }
    audit_path = new / "scientific_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    artifacts.append(audit_path)
    reproduction = {
        "generated_at": generated_at,
        "evaluation": "adapter-corrected frozen-output evaluation",
        "pre_formal_git_commit": "0bb26016054ca8ea9f1e0701a0c4ada8e76a9e5f",
        "adapter_sha256": freeze["adapter_sha256"],
        "parser_model": None,
        "parser_prompt_sha256": None,
        "official_revision": freeze["official_repository_revision"],
        "raw_answer_ordered_hashes": corrected["raw_answers"]["ordered_hash_sha256"],
        "model_tree_manifest_hashes": {
            label: freeze["models"][label]["tree_manifest_sha256"] for label in LABELS
        },
        "commands": {
            "formal_postprocess": "python scripts/run_travelplanner_adapter_v2_formal.py",
            "official_format_check": (
                "python <official>/postprocess/format_check.py --evaluation_file_path "
                "<label>/official_submission.jsonl"
            ),
            "official_evaluation": (
                "python scripts/run_travelplanner_official_eval.py --official-root <official> "
                "--submission <label>/official_submission.jsonl --output <label>/official_metrics.json"
            ),
            "constraint_diagnostics": (
                "python scripts/analyze_travelplanner_adapter_v2_results.py "
                "--official-root <official> --validation <validation.csv> "
                "--results-root <adapter-results> --output <diagnostics.json>"
            ),
        },
        "artifacts": [
            {
                "path": str(path.relative_to(repo)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }
    (new / "reproduction_manifest.json").write_text(
        json.dumps(reproduction, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
