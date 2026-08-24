#!/usr/bin/env python3
"""Append auditable GPU/CPU samples for the project GRPO processes."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def project_rss_bytes(marker: str) -> int:
    total = 0
    uid = os.getuid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if marker not in command:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return total


def host_memory() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def gpu_samples(gpu_ids: set[int]) -> list[tuple[int, int, int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, used, total, utilization = (int(value.strip()) for value in line.split(","))
        if index in gpu_ids:
            rows.append((index, used, total, utilization))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--marker", default="travel-agent-rl")
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated physical GPU ids to sample.",
    )
    args = parser.parse_args()
    gpu_ids = {int(value) for value in args.gpu_ids.split(",") if value.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    exists = args.output.exists() and args.output.stat().st_size > 0
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "timestamp_utc",
                    "gpu_index",
                    "gpu_used_mib",
                    "gpu_total_mib",
                    "gpu_utilization_percent",
                    "host_ram_total_bytes",
                    "host_ram_available_bytes",
                    "project_rss_bytes",
                ]
            )
            handle.flush()
        while running:
            timestamp = datetime.now(timezone.utc).isoformat()
            total, available = host_memory()
            rss = project_rss_bytes(args.marker)
            for gpu, used, gpu_total, utilization in gpu_samples(gpu_ids):
                writer.writerow(
                    [timestamp, gpu, used, gpu_total, utilization, total, available, rss]
                )
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
