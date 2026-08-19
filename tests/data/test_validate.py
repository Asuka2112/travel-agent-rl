import json
from pathlib import Path

import pytest

from travel_agent_rl.data.prepare import sha256_file
from travel_agent_rl.data.validate import validate_snapshot


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_snapshot_checks_hashes_and_split_overlap(tmp_path: Path) -> None:
    train = [{"query_id": "train:1", "constraint_sha256": "a"}]
    frozen = [{"query_id": "test:1", "constraint_sha256": "b"}]
    excluded: list[dict] = []
    catalog = train + frozen
    artifacts = {
        "query_catalog.jsonl": catalog,
        "candidate_train.jsonl": train,
        "frozen_test.jsonl": frozen,
        "contamination_excluded.jsonl": excluded,
    }
    artifact_manifest = {}
    for name, rows in artifacts.items():
        path = tmp_path / "data/processed" / name
        _write_jsonl(path, rows)
        artifact_manifest[name] = {
            "path": str(path.relative_to(tmp_path)),
            "sha256": sha256_file(path),
            "records": len(rows),
        }
    manifest_dir = tmp_path / "data/manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "data_snapshot.json").write_text(
        json.dumps({"artifacts": artifact_manifest})
    )
    frozen_path = tmp_path / "data/processed/frozen_test.jsonl"
    (manifest_dir / "frozen_test_ids.json").write_text(
        json.dumps({"ids": ["test:1"], "artifact_sha256": sha256_file(frozen_path)})
    )
    assert validate_snapshot(tmp_path)["success"] is True

    frozen_path.write_text(frozen_path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_snapshot(tmp_path)

