#!/usr/bin/env python3
"""Atomically update the persistent unattended-run state and append its audit log."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--failure-reason")
    parser.add_argument("--next-manual-action")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--pid", type=int)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    output_dir = args.root / "outputs/night_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    log_path = output_dir / "night_run.log"
    if state_path.exists():
        state: dict[str, Any] = json.loads(state_path.read_text())
    else:
        state = {
            "start_time": utc_now(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.root, text=True
            ).strip(),
            "config": [],
            "history": [],
        }
    event = {
        "time": utc_now(),
        "stage": args.stage,
        "status": args.status,
        "pid": args.pid if args.pid is not None else os.getpid(),
        "checkpoint": args.checkpoint,
        "failure_reason": args.failure_reason,
    }
    state["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.root, text=True
    ).strip()
    state["current_stage"] = args.stage
    state["status"] = args.status
    state["checkpoint"] = args.checkpoint
    state["failure_reason"] = args.failure_reason
    state["next_manual_action"] = args.next_manual_action
    state["pid"] = event["pid"]
    state["config"] = sorted(set([*state.get("config", []), *args.config]))
    state["history"].append(event)
    if args.status in {"completed", "hard_stopped", "failed"}:
        state["end_time"] = utc_now()
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(state_path)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
