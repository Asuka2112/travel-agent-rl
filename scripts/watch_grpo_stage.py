#!/usr/bin/env python3
"""Persist GRPO stage health without invoking a model or mutating services."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


FATAL_PATTERNS = {
    "oom": re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE),
    "nccl_error": re.compile(r"NCCL[^\n]*(?:error|failed|timeout)", re.IGNORECASE),
    "nan_or_inf": re.compile(r"(?:loss|reward|grad_norm)[^\n]{0,80}(?:nan|inf)", re.IGNORECASE),
    "checkpoint_failure": re.compile(
        r"(?:save|checkpoint)[^\n]{0,80}(?:failed|error|exception)", re.IGNORECASE
    ),
}
COMPLETION_PATTERNS = (
    re.compile(r"train_runtime"),
    re.compile(r"Training completed", re.IGNORECASE),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_text(path: Path, limit: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode(errors="replace")


def tmux_alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def checkpoint_paths(output_dir: Path) -> list[str]:
    paths = [path for path in output_dir.glob("*/checkpoint-*") if path.is_dir()]
    paths.sort(key=lambda path: (int(path.name.rsplit("-", 1)[-1]), str(path)))
    return [str(path) for path in paths]


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--trainer-session", required=True)
    parser.add_argument("--trainer-log", type=Path, required=True)
    parser.add_argument("--rollout-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("outputs/run_state.json"))
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--startup-grace", type=float, default=900.0)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    args = parser.parse_args()

    started = time.monotonic()
    while True:
        trainer_text = tail_text(args.trainer_log)
        rollout_text = tail_text(args.rollout_log)
        combined = trainer_text + "\n" + rollout_text
        failures = [name for name, pattern in FATAL_PATTERNS.items() if pattern.search(combined)]
        disk = shutil.disk_usage(args.output_dir.parent)
        disk_free_gib = disk.free / 2**30
        if disk_free_gib < args.minimum_free_gib:
            failures.append("disk_low")

        alive = tmux_alive(args.trainer_session)
        elapsed = time.monotonic() - started
        completed = any(pattern.search(trainer_text) for pattern in COMPLETION_PATTERNS)
        if failures:
            status = "error"
        elif completed and not alive:
            status = "completed"
        elif not alive and elapsed > args.startup_grace:
            status = "process_dead"
        else:
            status = "running"

        checkpoints = checkpoint_paths(args.output_dir)
        payload: dict[str, object] = {
            "updated_at": utc_now(),
            "stage": args.stage,
            "status": status,
            "trainer_session": args.trainer_session,
            "trainer_alive": alive,
            "failure_signals": sorted(set(failures)),
            "disk_free_gib": disk_free_gib,
            "checkpoint_count": len(checkpoints),
            "latest_checkpoint": checkpoints[-1] if checkpoints else None,
            "trainer_log": str(args.trainer_log),
            "rollout_log": str(args.rollout_log),
        }
        write_state(args.state, payload)
        if status != "running":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
