from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

CONSTRAINT_FIELDS = (
    "org",
    "dest",
    "days",
    "date",
    "visiting_city_number",
    "people_number",
    "local_constraint",
    "budget",
)


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def canonicalize_query(text: str) -> str:
    normalized = normalize_query(text).casefold()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = normalize_query(value)
        if value[:1] in {"[", "{"}:
            try:
                return _canonical_value(ast.literal_eval(value))
            except (SyntaxError, ValueError):
                pass
        return value.casefold()
    if isinstance(value, dict):
        return {str(key).casefold(): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return str(value).casefold()


def constraint_fingerprint(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    payload = {field: _canonical_value(record.get(field)) for field in fields}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(encoded.encode("utf-8"))


def _record_fields(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        field
        for field in CONSTRAINT_FIELDS
        if field in record and record[field] not in (None, "", [], {})
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _protected_index(paths: list[Path]) -> tuple[list[str], dict[tuple[str, ...], set[str]]]:
    canonical_queries: list[str] = []
    fingerprints: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for path in paths:
        rows = _read_csv(path) if path.suffix == ".csv" else _read_jsonl(path)
        for row in rows:
            query = normalize_query(str(row["query"]))
            canonical_queries.append(canonicalize_query(query))
            fields = _record_fields(row)
            if fields:
                fingerprints[fields].add(constraint_fingerprint(row, fields))
    return canonical_queries, fingerprints


def _source_file_manifest(path: Path) -> dict[str, Any]:
    if path.suffix == ".csv":
        count = len(_read_csv(path))
    elif path.suffix == ".jsonl":
        count = len(_read_jsonl(path))
    else:
        count = None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "records": count,
    }


def prepare_data(config_path: Path, root: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    policy = config["policy"]
    agent_config = config["sources"]["agent_star"]
    travel_config = config["sources"]["travelplanner"]
    candidate_path = root / agent_config["candidate_file"]
    protected_paths = [
        *(root / path for path in agent_config["protected_files"]),
        *(root / path for path in travel_config["protected_files"]),
    ]

    protected_queries, protected_fingerprints = _protected_index(protected_paths)
    protected_hashes = {sha256_bytes(query.encode("utf-8")) for query in protected_queries}
    threshold = float(policy["fuzzy_contamination_threshold"])

    raw_candidates = _read_jsonl(candidate_path)
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_query_hashes: set[str] = set()
    seen_constraints: set[str] = set()

    for position, source in enumerate(raw_candidates):
        query = normalize_query(str(source["query"]))
        canonical = canonicalize_query(query)
        query_hash = sha256_bytes(canonical.encode("utf-8"))
        full_fields = _record_fields(source)
        full_fingerprint = constraint_fingerprint(source, full_fields)
        reasons: list[str] = []

        if query_hash in protected_hashes:
            reasons.append("protected_exact_or_canonical")
        for fields, fingerprints in protected_fingerprints.items():
            if all(field in source for field in fields):
                if constraint_fingerprint(source, fields) in fingerprints:
                    reasons.append("protected_constraint_fingerprint")
                    break
        if not reasons:
            fuzzy_match = process.extractOne(
                canonical,
                protected_queries,
                scorer=fuzz.ratio,
                score_cutoff=threshold,
            )
            if fuzzy_match is not None:
                reasons.append(f"protected_fuzzy_{fuzzy_match[1]:.2f}")
        if query_hash in seen_query_hashes:
            reasons.append("duplicate_query")
        if full_fingerprint in seen_constraints:
            reasons.append("duplicate_constraints")

        query_id = f"agent_star:{source.get('id', position)}"
        record = {
            "query_id": query_id,
            "query": query,
            "canonical_sha256": query_hash,
            "constraint_sha256": full_fingerprint,
            "source": {
                "dataset": agent_config["repository"],
                "revision": agent_config["revision"],
                "row": position,
                "source_id": source.get("id"),
            },
            "attributes": {key: value for key, value in source.items() if key != "query"},
        }
        if reasons:
            record["exclusion_reasons"] = sorted(set(reasons))
            excluded.append(record)
            continue
        seen_query_hashes.add(query_hash)
        seen_constraints.add(full_fingerprint)
        accepted.append(record)

    minimum = int(policy["minimum_candidates_after_cleaning"])
    if len(accepted) < minimum:
        raise RuntimeError(f"only {len(accepted)} clean candidates; required at least {minimum}")

    seed = int(policy["seed"])
    ordered = sorted(
        accepted,
        key=lambda row: sha256_bytes(f"{seed}:{row['query_id']}".encode("utf-8")),
    )
    frozen_size = int(policy["frozen_test_size"])
    frozen = ordered[:frozen_size]
    train = ordered[frozen_size:]
    frozen_ids = {row["query_id"] for row in frozen}
    if frozen_ids & {row["query_id"] for row in train}:
        raise AssertionError("frozen and train splits overlap")

    processed_dir = root / "data/processed"
    artifact_rows = {
        "query_catalog.jsonl": accepted,
        "candidate_train.jsonl": train,
        "frozen_test.jsonl": frozen,
        "contamination_excluded.jsonl": excluded,
    }
    for name, rows in artifact_rows.items():
        _write_jsonl(processed_dir / name, rows)

    exclusion_counts: dict[str, int] = defaultdict(int)
    for row in excluded:
        for reason in row["exclusion_reasons"]:
            exclusion_counts[reason] += 1
    stats = {
        "raw_candidates": len(raw_candidates),
        "protected_reference_queries": len(protected_queries),
        "excluded_total": len(excluded),
        "accepted_unique": len(accepted),
        "train_candidates": len(train),
        "frozen_test": len(frozen),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
    }
    _write_json(processed_dir / "data_stats.json", stats)

    manifest = {
        "pipeline_version": 1,
        "seed": seed,
        "source_revisions": {
            name: {
                "repository": item["repository"],
                "revision": item["revision"],
                "declared_license": item["declared_license"],
                "url": item["url"],
            }
            for name, item in config["sources"].items()
        },
        "source_files": [
            _source_file_manifest(candidate_path),
            *(_source_file_manifest(path) for path in protected_paths),
        ],
        "artifacts": {
            name: {
                "path": str(processed_dir / name),
                "sha256": sha256_file(processed_dir / name),
                "records": len(rows),
            }
            for name, rows in artifact_rows.items()
        },
        "stats": stats,
    }
    _write_json(root / "data/manifests/data_snapshot.json", manifest)
    _write_json(
        root / "data/manifests/frozen_test_ids.json",
        {
            "seed": seed,
            "size": len(frozen),
            "ids": sorted(frozen_ids),
            "artifact_sha256": manifest["artifacts"]["frozen_test.jsonl"]["sha256"],
        },
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/sources.yaml"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.config, args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

