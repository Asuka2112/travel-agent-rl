#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from travel_agent_rl.evaluation.travelplanner_adapter_v2 import (
    adapt_final_answer_v2,
    validate_official_schema,
)


LABELS = ("base", "sft", "rl")
AUDIT_SEED = 20260826
AUDIT_SIZE = 20


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_answer(answer: str | None) -> str:
    return sha256_bytes((answer or "").encode("utf-8"))


def load_record(root: Path, label: str, task_id: int) -> dict[str, Any]:
    return json.loads((root / label / f"{task_id:03d}.json").read_text(encoding="utf-8"))


def raw_answer_manifest(root: Path) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for label in LABELS:
        items = []
        for task_id in range(1, 181):
            record = load_record(root, label, task_id)
            items.append(
                {
                    "task_id": task_id,
                    "query_id": record.get("query_id"),
                    "final_answer_sha256": sha256_answer(record.get("final_answer")),
                }
            )
        joined = "\n".join(item["final_answer_sha256"] for item in items).encode("ascii")
        labels[label] = {
            "count": len(items),
            "ordered_hash_sha256": sha256_bytes(joined),
            "items": items,
        }
    return {"hash_definition": "sha256(exact UTF-8 final_answer bytes)", "labels": labels}


def create_selection(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    selections: dict[str, list[int]] = {}
    for offset, label in enumerate(LABELS):
        rng = random.Random(AUDIT_SEED + offset)
        selections[label] = sorted(rng.sample(range(1, 181), AUDIT_SIZE))
    payload = {
        "seed": AUDIT_SEED,
        "method": "random.Random(seed + label_offset).sample(range(1, 181), 20)",
        "selection_precedes_adapter-score evaluation": True,
        "labels": selections,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def next_candidate(audit_dir: Path) -> Path:
    existing = sorted(audit_dir.glob("candidate_*.json"))
    number = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 1
    return audit_dir / f"candidate_{number:03d}.json"


def build_audit(root: Path, out_dir: Path, adapter_path: Path) -> Path:
    audit_dir = out_dir / "audit"
    selection = create_selection(audit_dir / "selection.json")
    records = []
    for label in LABELS:
        for task_id in selection["labels"][label]:
            record = load_record(root, label, task_id)
            answer = record.get("final_answer")
            parsed = adapt_final_answer_v2(answer)
            valid, errors = validate_official_schema(parsed)
            records.append(
                {
                    "label": label,
                    "task_id": task_id,
                    "query_id": record.get("query_id"),
                    "raw_final_answer_sha256": sha256_answer(answer),
                    "raw_final_answer": answer,
                    "parsed_plan": parsed,
                    "parser_valid": parsed is not None,
                    "schema_valid": valid,
                    "schema_errors": errors,
                    "manual_audit": {
                        "faithful_extraction": None,
                        "field_loss": None,
                        "hallucinated_field": None,
                        "wrong_field_assignment": None,
                        "entity_mutation": None,
                        "notes": None,
                    },
                }
            )
    path = next_candidate(audit_dir)
    payload = {
        "candidate": path.stem,
        "adapter_sha256": sha256_file(adapter_path),
        "selection_file_sha256": sha256_file(audit_dir / "selection.json"),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("audit",), required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    raw_root = repo / "outputs/final/travelplanner_validation"
    out_dir = repo / "outputs/final/travelplanner_adapter_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "raw_answer_manifest.pre_formal.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(raw_answer_manifest(raw_root), indent=2) + "\n", encoding="utf-8"
        )
    path = build_audit(
        raw_root,
        out_dir,
        repo / "src/travel_agent_rl/evaluation/travelplanner_adapter_v2.py",
    )
    print(path)


if __name__ == "__main__":
    main()
