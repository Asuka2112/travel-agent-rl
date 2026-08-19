from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from rapidfuzz import fuzz, process

from .prepare import (
    _protected_index,
    canonicalize_query,
    constraint_fingerprint,
    sha256_bytes,
    sha256_file,
)

SELECTION_ALGORITHM_VERSION = 1
DIFFICULTIES = ("easy", "medium", "hard")


def constraint_types(record: dict[str, Any]) -> tuple[str, ...]:
    attributes = record["attributes"]
    local = attributes.get("local_constraint") or {}
    types: list[str] = []
    if attributes.get("budget") not in (None, ""):
        types.append("budget")
    if local.get("house rule") or local.get("room type"):
        types.append("accommodation")
    if local.get("cuisine"):
        types.append("dietary")
    if local.get("transportation"):
        types.append("transportation")
    if re.search(r"\b(attraction|sightseeing|landmark|museum)\b", record["query"], re.I):
        types.append("attraction")
    if len(types) > 2:
        types.append("multi_constraint")
    return tuple(types)


def _difficulty(record: dict[str, Any]) -> str:
    value = str(record["attributes"].get("level", "")).casefold()
    if value not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty for {record['query_id']}: {value}")
    return value


def _largest_remainder_quotas(counts: Counter[str], size: int) -> dict[str, int]:
    total = sum(counts.values())
    exact = {key: size * counts[key] / total for key in DIFFICULTIES}
    quotas = {key: math.floor(exact[key]) for key in DIFFICULTIES}
    remainder = size - sum(quotas.values())
    order = sorted(DIFFICULTIES, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _stable_fraction(seed: int, query_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _coverage_value(record: dict[str, Any], key: str) -> Any:
    attributes = record["attributes"]
    if key == "origin":
        return attributes.get("org")
    if key == "destination":
        return attributes.get("dest")
    if key == "route":
        return (attributes.get("org"), attributes.get("dest"))
    if key == "days":
        return attributes.get("days")
    if key == "city_count":
        return attributes.get("visiting_city_number")
    if key == "constraints":
        return constraint_types(record)
    raise KeyError(key)


def _select_stratum(
    candidates: list[dict[str, Any]],
    size: int,
    seed: int,
    near_duplicate_threshold: float,
) -> list[dict[str, Any]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    selected_canonical: list[str] = []
    dimensions = ("origin", "destination", "route", "days", "city_count", "constraints")
    coverage = {name: Counter() for name in dimensions}

    while len(selected) < size:
        if not remaining:
            raise RuntimeError(f"only selected {len(selected)} of {size} after near deduplication")

        def score(record: dict[str, Any]) -> tuple[float, float]:
            weighted = 0.0
            for name, weight in (
                ("route", 8.0),
                ("origin", 4.0),
                ("destination", 4.0),
                ("constraints", 3.0),
                ("days", 2.0),
                ("city_count", 2.0),
            ):
                weighted += weight / (1 + coverage[name][_coverage_value(record, name)])
            return weighted, _stable_fraction(seed, record["query_id"])

        ranked = sorted(remaining, key=score, reverse=True)
        chosen: dict[str, Any] | None = None
        for candidate in ranked:
            canonical = canonicalize_query(candidate["query"])
            match = process.extractOne(
                canonical,
                selected_canonical,
                scorer=fuzz.ratio,
                score_cutoff=near_duplicate_threshold,
            )
            if match is None:
                chosen = candidate
                selected_canonical.append(canonical)
                break
        if chosen is None:
            raise RuntimeError("near-duplicate threshold prevents completing selection")
        remaining.remove(chosen)
        selected.append(chosen)
        for name in dimensions:
            coverage[name][_coverage_value(chosen, name)] += 1
    return selected


def select_queries(
    candidates: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    near_duplicate_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter(_difficulty(record) for record in candidates)
    quotas = _largest_remainder_quotas(counts, size)
    selected: list[dict[str, Any]] = []
    for offset, difficulty in enumerate(DIFFICULTIES):
        stratum = [record for record in candidates if _difficulty(record) == difficulty]
        selected.extend(
            _select_stratum(
                stratum,
                quotas[difficulty],
                seed + offset,
                near_duplicate_threshold,
            )
        )
    selected.sort(key=lambda row: row["query_id"])
    return selected, quotas


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _freeze(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"frozen artifact differs and will not be overwritten: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(values: list[Any]) -> dict[str, int]:
        return dict(sorted(Counter(str(value) for value in values).items()))

    origins = [row["attributes"].get("org") for row in rows]
    destinations = [row["attributes"].get("dest") for row in rows]
    constraints = Counter(kind for row in rows for kind in constraint_types(row))
    return {
        "difficulty": counts([_difficulty(row) for row in rows]),
        "trip_scope": counts(
            [
                "single_city"
                if int(row["attributes"].get("visiting_city_number", 1)) == 1
                else "multi_city"
                for row in rows
            ]
        ),
        "days": counts([row["attributes"].get("days") for row in rows]),
        "constraint_types": dict(sorted(constraints.items())),
        "unique_origins": len(set(origins)),
        "unique_destinations": len(set(destinations)),
        "top_origins": Counter(origins).most_common(15),
        "top_destinations": Counter(destinations).most_common(15),
        "maximum_origin_share": max(Counter(origins).values()) / len(rows),
        "maximum_destination_share": max(Counter(destinations).values()) / len(rows),
    }


def _assert_uncontaminated(
    rows: list[dict[str, Any]],
    protected_paths: list[Path],
    frozen_ids: set[str],
    fuzzy_threshold: float,
) -> dict[str, int]:
    protected_queries, protected_fingerprints = _protected_index(protected_paths)
    protected_hashes = {sha256_bytes(query.encode()) for query in protected_queries}
    checks = Counter()
    for row in rows:
        if row["query_id"] in frozen_ids:
            raise RuntimeError(f"frozen-evaluation overlap: {row['query_id']}")
        canonical = canonicalize_query(row["query"])
        if sha256_bytes(canonical.encode()) in protected_hashes:
            raise RuntimeError(f"protected exact query overlap: {row['query_id']}")
        source = row["attributes"]
        for fields, fingerprints in protected_fingerprints.items():
            if all(field in source for field in fields):
                if constraint_fingerprint(source, fields) in fingerprints:
                    raise RuntimeError(f"protected constraint overlap: {row['query_id']}")
        if process.extractOne(
            canonical,
            protected_queries,
            scorer=fuzz.ratio,
            score_cutoff=fuzzy_threshold,
        ):
            raise RuntimeError(f"protected fuzzy query overlap: {row['query_id']}")
        checks["protected_comparisons"] += len(protected_queries)
    checks["selected"] = len(rows)
    checks["frozen_overlap"] = 0
    checks["protected_overlap"] = 0
    return dict(checks)


def _pilot_rows(
    selected: list[dict[str, Any]], size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter(_difficulty(record) for record in selected)
    quotas = _largest_remainder_quotas(counts, size)
    rows: list[dict[str, Any]] = []
    for difficulty in DIFFICULTIES:
        stratum = sorted(
            (record for record in selected if _difficulty(record) == difficulty),
            key=lambda record: (_stable_fraction(seed, record["query_id"]), record["query_id"]),
        )
        rows.extend(stratum[: quotas[difficulty]])
    rows.sort(key=lambda row: row["query_id"])
    return rows, quotas


def freeze_training_selection(root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    input_path = root / config["input_path"]
    snapshot = json.loads((root / config["source_manifest_path"]).read_text())
    expected_hash = snapshot["artifacts"][input_path.name]["sha256"]
    if sha256_file(input_path) != expected_hash:
        raise RuntimeError("candidate pool differs from its source manifest")
    candidates = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    selected, quotas = select_queries(
        candidates,
        size=int(config["selection_size"]),
        seed=int(config["seed"]),
        near_duplicate_threshold=float(config["near_duplicate_threshold"]),
    )
    protected_paths = [root / path for path in config["protected_paths"]]
    frozen_ids = set(json.loads((root / config["frozen_ids_path"]).read_text())["ids"])
    contamination = _assert_uncontaminated(
        selected,
        protected_paths,
        frozen_ids,
        float(config["contamination_fuzzy_threshold"]),
    )

    output_path = root / config["output_path"]
    train_bytes = _jsonl_bytes(selected)
    _freeze(output_path, train_bytes)
    pilot_200, pilot_200_quotas = _pilot_rows(selected, 200, int(config["pilot_seed"]))
    pilot_50_quotas = _largest_remainder_quotas(Counter(_difficulty(r) for r in selected), 50)
    pilot_50_ids: set[str] = set()
    for difficulty in DIFFICULTIES:
        ordered = [row for row in pilot_200 if _difficulty(row) == difficulty]
        pilot_50_ids.update(row["query_id"] for row in ordered[: pilot_50_quotas[difficulty]])
    pilot_50 = [row for row in pilot_200 if row["query_id"] in pilot_50_ids]
    pilot_50.sort(key=lambda row: row["query_id"])
    _freeze(root / config["pilot_50_path"], _jsonl_bytes(pilot_50))
    _freeze(root / config["pilot_200_path"], _jsonl_bytes(pilot_200))

    source = snapshot["source_revisions"]["agent_star"]
    records = [
        {
            "query_id": row["query_id"],
            "difficulty": _difficulty(row),
            "constraint_types": list(constraint_types(row)),
            "origin": row["attributes"].get("org"),
            "destination": row["attributes"].get("dest"),
            "days": row["attributes"].get("days"),
            "visiting_city_number": row["attributes"].get("visiting_city_number"),
            "source_row": row["source"].get("row"),
        }
        for row in selected
    ]
    manifest = {
        "immutable": True,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "selection_seed": int(config["seed"]),
        "selection_size": len(selected),
        "difficulty_quotas": quotas,
        "near_duplicate_threshold": float(config["near_duplicate_threshold"]),
        "source_dataset": source["repository"],
        "source_revision": source["revision"],
        "source_pool_path": config["input_path"],
        "source_pool_sha256": expected_hash,
        "output_path": config["output_path"],
        "output_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "pilot_seed": int(config["pilot_seed"]),
        "pilot_50": {
            "path": config["pilot_50_path"],
            "records": len(pilot_50),
            "quotas": pilot_50_quotas,
            "sha256": hashlib.sha256(_jsonl_bytes(pilot_50)).hexdigest(),
        },
        "pilot_200": {
            "path": config["pilot_200_path"],
            "records": len(pilot_200),
            "quotas": pilot_200_quotas,
            "sha256": hashlib.sha256(_jsonl_bytes(pilot_200)).hexdigest(),
        },
        "distribution": _distribution(selected),
        "contamination": contamination,
        "protected_sources": [str(path.relative_to(root)) for path in protected_paths],
        "records": records,
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    _freeze(root / config["manifest_path"], manifest_bytes)
    return manifest
