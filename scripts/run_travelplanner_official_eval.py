#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from travel_agent_rl.evaluation.travelplanner import wilson_interval


DENOMINATORS = {
    "Delivery Rate": 180,
    "Commonsense Constraint Micro Pass Rate": 1440,
    "Commonsense Constraint Macro Pass Rate": 180,
    "Hard Constraint Micro Pass Rate": 420,
    "Hard Constraint Macro Pass Rate": 180,
    "Final Pass Rate": 180,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unmodified official TravelPlanner evaluator")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite official metrics: {args.output}")
    lines = [line for line in args.submission.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 180:
        raise ValueError(f"official validation requires exactly 180 submissions, got {len(lines)}")
    indices = [json.loads(line)["idx"] for line in lines]
    if indices != list(range(180)):
        raise ValueError("submission idx values must be exactly 0..179 in official order")

    official_root = args.official_root.resolve()
    evaluation = official_root / "evaluation"
    evaluator = evaluation / "eval.py"
    command = [
        sys.executable,
        "eval.py",
        "--set_type",
        "validation",
        "--evaluation_file_path",
        str(args.submission.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=evaluation,
        check=False,
        capture_output=True,
        text=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_stdout = args.output.with_suffix(".official_stdout.txt")
    raw_stderr = args.output.with_suffix(".official_stderr.txt")
    raw_stdout.write_text(completed.stdout, encoding="utf-8")
    raw_stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"official evaluator failed with {completed.returncode}; see {raw_stdout} and {raw_stderr}"
        )

    parsed: dict[str, dict[str, object]] = {}
    for name, denominator in DENOMINATORS.items():
        match = re.search(rf"^{re.escape(name)}:\s*([0-9.eE+-]+)%\s*$", completed.stdout, re.M)
        if not match:
            raise RuntimeError(f"official evaluator did not report {name!r}")
        rate = float(match.group(1)) / 100.0
        numerator = round(rate * denominator)
        if abs(rate - numerator / denominator) > 1e-9:
            raise RuntimeError(f"non-integral official numerator for {name}: {rate} * {denominator}")
        parsed[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": rate,
        }
        if name == "Final Pass Rate":
            parsed[name]["ci95_wilson"] = wilson_interval(numerator, denominator)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=official_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "unmodified official TravelPlanner evaluation/eval.py --set_type validation",
        "official_repository": "https://github.com/OSU-NLP-Group/TravelPlanner",
        "official_revision": revision,
        "evaluator_sha256": sha256_file(evaluator),
        "commonsense_evaluator_sha256": sha256_file(evaluation / "commonsense_constraint.py"),
        "hard_evaluator_sha256": sha256_file(evaluation / "hard_constraint.py"),
        "submission": str(args.submission),
        "submission_sha256": sha256_file(args.submission),
        "n": 180,
        "metrics": parsed,
        "command": command,
        "raw_stdout": str(raw_stdout),
        "raw_stderr": str(raw_stderr),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
