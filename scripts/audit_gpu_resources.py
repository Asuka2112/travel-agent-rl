#!/usr/bin/env python3
"""Capture a read-only GPU/process audit for the dynamic shared-server policy."""

from __future__ import annotations

import argparse
import json
import pwd
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def query(command: list[str]) -> list[list[str]]:
    output = subprocess.check_output(command, text=True)
    return [
        [value.strip() for value in line.split(",")] for line in output.splitlines() if line.strip()
    ]


def process_info(pid: int) -> dict[str, object]:
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        uid_line = next(line for line in status if line.startswith("Uid:"))
        username = pwd.getpwuid(int(uid_line.split()[1])).pw_name
        command = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
            .strip()
        )
        elapsed = int(
            subprocess.check_output(["ps", "-o", "etimes=", "-p", str(pid)], text=True).strip()
        )
        return {"pid": pid, "user": username, "elapsed_seconds": elapsed, "command": command}
    except (FileNotFoundError, ProcessLookupError, StopIteration, KeyError, ValueError):
        return {"pid": pid, "user": None, "elapsed_seconds": None, "command": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-marker", default="travel-agent-rl")
    parser.add_argument(
        "--selected-gpu-ids",
        default="",
        help="Comma-separated GPU ids selected after the read-only safety audit.",
    )
    args = parser.parse_args()
    selected = {int(value) for value in args.selected_gpu_ids.split(",") if value.strip()}
    gpu_rows = query(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_rows = query(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes_by_uuid: dict[str, list[dict[str, object]]] = {}
    for uuid, pid_value, memory_value in process_rows:
        info = process_info(int(pid_value))
        info["used_memory_mib"] = int(memory_value)
        info["project_process"] = args.project_marker in str(info.get("command") or "")
        processes_by_uuid.setdefault(uuid, []).append(info)
    gpus = []
    for (
        index_value,
        uuid,
        name,
        used_value,
        free_value,
        total_value,
        utilization_value,
        temperature_value,
    ) in gpu_rows:
        index = int(index_value)
        processes = processes_by_uuid.get(uuid, [])
        external_processes = [process for process in processes if not process["project_process"]]
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "used_memory_mib": int(used_value),
                "free_memory_mib": int(free_value),
                "total_memory_mib": int(total_value),
                "utilization_percent": int(utilization_value),
                "temperature_c": int(temperature_value),
                "selected_for_project": index in selected,
                "external_process_count": len(external_processes),
                "resource_state": (
                    "idle_safe"
                    if not external_processes and int(used_value) <= 64
                    else "shared_selected"
                    if index in selected
                    else "external_in_use"
                ),
                "processes": processes,
            }
        )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resource_policy": "dynamic_shared_server_v2",
        "selected_gpu_ids": sorted(selected),
        "selected_gpu_count": len(selected),
        "selection_note": (
            "Selection is an explicit stage decision; external processes are never killed, "
            "reset, or preempted."
        ),
        "gpus": gpus,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
