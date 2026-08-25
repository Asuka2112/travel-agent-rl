#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_FILES = (
    "postprocess/parsing.py",
    "postprocess/element_extraction.py",
    "postprocess/combination.py",
    "postprocess/example_evaluation.jsonl",
    "postprocess/format_check.py",
    "evaluation/eval.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze audited TravelPlanner Adapter v2")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--official-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    official = args.official_root.resolve()
    output = repo / "outputs/final/travelplanner_adapter_v2"
    destination = output / "adapter_freeze_manifest.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite adapter freeze: {destination}")
    adapter = repo / "src/travel_agent_rl/evaluation/travelplanner_adapter_v2.py"
    audit_path = output / "audit/manual_audit_final.json"
    audit = load(audit_path)
    adapter_hash = sha256_file(adapter)
    if audit["adapter_sha256"] != adapter_hash:
        raise RuntimeError("manual audit does not match current adapter source")
    counts = audit["manual_review"]["counts"]
    required = {
        "reviewed_records": audit["manual_review"]["reviewed_records"] == 60,
        "hallucinated_field_zero": counts["hallucinated_field"] == 0,
        "wrong_field_assignment_zero": counts["wrong_field_assignment"] == 0,
        "entity_mutation_zero": counts["entity_mutation"] == 0,
    }
    protocol_path = repo / "outputs/final/travelplanner_validation/protocol_manifest.json"
    protocol = load(protocol_path)
    model_checks: dict[str, bool] = {}
    for label, model in protocol["models"].items():
        model_root = repo / model["path"]
        for entry in model["files"]:
            path = model_root / entry["path"]
            model_checks[f"{label}:{entry['path']}"] = (
                path.is_file()
                and path.stat().st_size == entry["bytes"]
                and sha256_file(path) == entry["sha256"]
            )
    required["all_frozen_model_files_unchanged"] = all(model_checks.values())
    preflight = output / "preflight"
    for name in ("synthetic_format_check.log", "official_example_format_check.log"):
        text = (preflight / name).read_text(encoding="utf-8")
        required[f"{name}_completed_without_traceback"] = "Traceback" not in text
    official_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=official,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required["official_revision_frozen"] = (
        official_revision == "e52c87f4ac348a3410c46dc3553c519db5ec5e23"
    )
    if not all(required.values()):
        raise RuntimeError(f"adapter freeze checks failed: {[k for k, v in required.items() if not v]}")
    raw_manifest_path = output / "raw_answer_manifest.pre_formal.json"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "adapter_frozen_before_formal_validation",
        "adapter_sha256": adapter_hash,
        "adapter_path": str(adapter.relative_to(repo)),
        "adapter_spec_sha256": sha256_file(repo / "docs/TRAVELPLANNER_ADAPTER_SPEC.md"),
        "formal_runner_sha256": sha256_file(
            repo / "scripts/run_travelplanner_adapter_v2_formal.py"
        ),
        "parser": {
            "type": "deterministic_no_llm",
            "model": None,
            "prompt": None,
            "prompt_sha256": None,
        },
        "official_repository_revision": official_revision,
        "official_contract_files": {
            name: sha256_file(official / name) for name in OFFICIAL_FILES
        },
        "raw_answer_manifest_sha256": sha256_file(raw_manifest_path),
        "raw_answer_manifest": load(raw_manifest_path),
        "frozen_protocol_manifest_sha256": sha256_file(protocol_path),
        "validation": protocol["validation"],
        "models": protocol["models"],
        "manual_audit_sha256": sha256_file(audit_path),
        "manual_audit_summary": audit["manual_review"],
        "preflight": {
            name: {
                "sha256": sha256_file(preflight / name),
                "bytes": (preflight / name).stat().st_size,
            }
            for name in (
                "synthetic_submission.jsonl",
                "synthetic_format_check.log",
                "official_example_format_check.log",
            )
        },
        "checks": required,
        "model_file_checks": model_checks,
        "formal_evaluation_attempt_limit": 1,
        "model_inference_permitted": False,
    }
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
