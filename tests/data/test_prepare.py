import csv
import json
from pathlib import Path

import yaml

from travel_agent_rl.data.prepare import canonicalize_query, normalize_query, prepare_data


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _record(record_id: int, origin: str, destination: str, query: str) -> dict:
    return {
        "id": record_id,
        "org": origin,
        "dest": destination,
        "days": 3,
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "visiting_city_number": 1,
        "people_number": 1,
        "local_constraint": {"cuisine": None},
        "budget": 1000 + record_id,
        "query": query,
        "level": "easy",
    }


def test_query_normalization_is_stable() -> None:
    assert normalize_query("  Full\u3000width\n text  ") == "Full width text"
    assert canonicalize_query("Budget: $1,000!") == "budget 1 000"


def test_prepare_data_excludes_protected_and_freezes_disjoint_split(tmp_path: Path) -> None:
    candidates = [
        _record(1, "A", "B", "Protected trip"),
        _record(2, "C", "D", "Candidate two"),
        _record(3, "E", "F", "Candidate three"),
        _record(4, "G", "H", "Candidate four"),
    ]
    protected = [_record(99, "A", "B", "Protected trip")]
    candidate_path = tmp_path / "data/raw/agent/candidates.jsonl"
    agent_protected = tmp_path / "data/raw/agent/protected.jsonl"
    travel_validation = tmp_path / "data/raw/travel/validation.csv"
    travel_test = tmp_path / "data/raw/travel/test.csv"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(agent_protected, protected)
    _write_csv(travel_validation, [_record(100, "I", "J", "Validation trip")])
    _write_csv(travel_test, [_record(101, "K", "L", "Test trip")])

    config = {
        "sources": {
            "agent_star": {
                "repository": "agent",
                "revision": "a" * 40,
                "declared_license": "mit",
                "url": "https://example.test/agent",
                "candidate_file": str(candidate_path.relative_to(tmp_path)),
                "protected_files": [str(agent_protected.relative_to(tmp_path))],
            },
            "travelplanner": {
                "repository": "travel",
                "revision": "b" * 40,
                "declared_license": "cc-by-4.0",
                "url": "https://example.test/travel",
                "protected_files": [
                    str(travel_validation.relative_to(tmp_path)),
                    str(travel_test.relative_to(tmp_path)),
                ],
            },
        },
        "policy": {
            "frozen_test_size": 1,
            "minimum_candidates_after_cleaning": 2,
            "fuzzy_contamination_threshold": 99.0,
            "seed": 7,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    stats = prepare_data(config_path, tmp_path)
    assert stats["raw_candidates"] == 4
    assert stats["accepted_unique"] == 3
    assert stats["frozen_test"] == 1
    frozen = json.loads((tmp_path / "data/manifests/frozen_test_ids.json").read_text())
    train_ids = {
        json.loads(line)["query_id"]
        for line in (tmp_path / "data/processed/candidate_train.jsonl").read_text().splitlines()
    }
    assert not set(frozen["ids"]) & train_ids

