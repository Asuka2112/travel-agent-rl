from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .prepare import sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_snapshot(root: Path) -> dict[str, Any]:
    manifest_path = root / "data/manifests/data_snapshot.json"
    manifest = json.loads(manifest_path.read_text())
    verified_artifacts: dict[str, int] = {}
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, expected in manifest["artifacts"].items():
        path = root / expected["path"]
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"artifact hash mismatch for {path}")
        rows = _read_jsonl(path)
        if len(rows) != expected["records"]:
            raise ValueError(f"artifact count mismatch for {path}")
        verified_artifacts[name] = len(rows)
        rows_by_name[name] = rows

    train_ids = {row["query_id"] for row in rows_by_name["candidate_train.jsonl"]}
    frozen_ids = {row["query_id"] for row in rows_by_name["frozen_test.jsonl"]}
    if train_ids & frozen_ids:
        raise ValueError("train and frozen query IDs overlap")
    train_constraints = {row["constraint_sha256"] for row in rows_by_name["candidate_train.jsonl"]}
    frozen_constraints = {row["constraint_sha256"] for row in rows_by_name["frozen_test.jsonl"]}
    if train_constraints & frozen_constraints:
        raise ValueError("train and frozen constraint fingerprints overlap")

    frozen_manifest = json.loads((root / "data/manifests/frozen_test_ids.json").read_text())
    if frozen_ids != set(frozen_manifest["ids"]):
        raise ValueError("frozen ID manifest does not match frozen artifact")
    if sha256_file(root / "data/processed/frozen_test.jsonl") != frozen_manifest["artifact_sha256"]:
        raise ValueError("frozen artifact hash does not match frozen ID manifest")

    return {
        "success": True,
        "verified_artifacts": verified_artifacts,
        "train_frozen_id_overlap": 0,
        "train_frozen_constraint_overlap": 0,
    }
