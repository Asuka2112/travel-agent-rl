#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(root.resolve())),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def model_manifest(path: Path) -> dict[str, Any]:
    files = sorted(file for file in path.rglob("*") if file.is_file())
    entries = [file_entry(file, path) for file in files]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path),
        "files": entries,
        "tree_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze external validation protocol and models")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/travelplanner_external_validation.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/final/travelplanner_validation/protocol_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tracked_code = [
        Path("scripts/audit_travelplanner_contamination.py"),
        Path("scripts/evaluate_travelplanner_validation.py"),
        Path("scripts/run_travelplanner_official_eval.py"),
        Path("src/travel_agent_rl/evaluation/travelplanner.py"),
        args.config,
    ]
    validation = Path(config["split"])
    result = {
        "manifest_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_commit_before_protocol_freeze": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "validation": {
            "path": str(validation),
            "bytes": validation.stat().st_size,
            "sha256": sha256_file(validation),
            "n": config["n"],
        },
        "protocol_files": [file_entry(path, Path.cwd()) for path in tracked_code],
        "models": {
            label: model_manifest(Path(path)) for label, path in config["models"].items()
        },
        "immutable_web_snapshot": {
            "version": 4,
            "content_sha256": "dbd160b6de284208c3906e9da0768cabfe3a77315e64280ea59685eefff15f92",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v["tree_manifest_sha256"] for k, v in result["models"].items()}))


if __name__ == "__main__":
    main()
