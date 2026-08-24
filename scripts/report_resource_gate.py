#!/usr/bin/env python3
"""Generate the mandatory post-pilot GPU resource gate."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--gpu-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/RESOURCE_GATE.md"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/grpo/resource_gate.json"),
    )
    args = parser.parse_args()
    smoke = load(args.smoke)
    pilot = load(args.pilot)
    audit = load(args.gpu_audit)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    safe_count = int(audit["safe_authorized_gpu_count"])
    if not smoke["gate_pass"] or not pilot["gate_pass"]:
        status = "hardware_feasibility_failure"
        reason = "the real full-parameter smoke or pilot feasibility gate did not pass"
    elif safe_count >= 7:
        status = "ready_for_formal_grpo"
        reason = "at least seven GPUs are authorized and safe"
    else:
        status = "waiting_for_gpu_resources"
        reason = (
            "full-parameter 4-GPU feasibility passed, but only GPU0-3 are authorized; "
            "formal G=6/max_turns=13 requires at least seven safe GPUs"
        )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "git_commit": commit,
        "safe_authorized_gpu_count": safe_count,
        "authorized_gpu_ids": audit["authorized_gpu_ids"],
        "prohibited_gpu_ids": audit["prohibited_gpu_ids"],
        "smoke_gate_pass": smoke["gate_pass"],
        "pilot_gate_pass": pilot["gate_pass"],
        "formal_num_generations": 6,
        "formal_max_turns": 13,
        "formal_started": False,
        "lora_grpo_used": False,
        "external_processes_modified": False,
        "smoke_metrics": str(args.smoke),
        "pilot_metrics": str(args.pilot),
        "gpu_audit": str(args.gpu_audit),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    process_lines = []
    for gpu in audit["gpus"]:
        processes = "; ".join(
            f"{item['pid']} ({item['user']}, {item['used_memory_mib']} MiB, "
            f"project={item['project_process']})"
            for item in gpu["processes"]
        ) or "none"
        process_lines.append(
            f"| {gpu['index']} | {gpu['policy']} | {gpu['used_memory_mib']} | {processes} |"
        )
    lines = [
        "# GRPO Resource Gate",
        "",
        f"Generated: {result['generated_at']}",
        "",
        f"**status = `{status}`**",
        "",
        reason + ".",
        "",
        "## Feasibility outcome",
        "",
        f"- Full-parameter smoke gate: {smoke['gate_pass']}",
        f"- Full-parameter pilot gate: {pilot['gate_pass']}",
        "- Smoke/pilot topology: GPU0 one TP=1 vLLM server; GPU1-3 ZeRO-3 trainer",
        "- Training strategy: 100% full parameters; no LoRA GRPO",
        f"- Smoke optimizer steps: {smoke['optimizer_steps']}",
        f"- Pilot optimizer steps: {pilot['optimizer_steps']}",
        f"- Smoke replay misses/public requests: {smoke['replay_misses']}/{smoke['public_network_requests']}",
        f"- Pilot replay misses/public requests: {pilot['replay_misses']}/{pilot['public_network_requests']}",
        "",
        "This is a resource-wait state, not an experiment failure or HARD STOP. Formal GRPO was "
        "not changed to LoRA, and formal `G=6`/`max_turns=13` were not reduced.",
        "",
        "## Read-only GPU audit",
        "",
        "| GPU | Policy | Used MiB | Compute processes |",
        "|---:|---|---:|---|",
        *process_lines,
        "",
        f"Current Git commit before the resource-gate commit: `{commit}`.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
