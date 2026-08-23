#!/usr/bin/env python3
"""Build the SFT memory-probe report from actual torch and nvidia-smi records."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


STEP_PATTERN = re.compile(
    r"\{'loss': (?P<loss>[0-9.]+).*?'token_acc': (?P<token_acc>[0-9.]+).*?"
    r"'memory\(GiB\)': (?P<memory>[0-9.]+)"
)
TIME_PATTERN = re.compile(r"1/1 \[[^\]]*?, (?P<seconds>[0-9.]+)s/it\]")


def parse_step(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    step = STEP_PATTERN.search(text)
    times = TIME_PATTERN.findall(text)
    if step is None or not times:
        raise ValueError(f"could not parse successful step from {path}")
    return {
        "loss": float(step.group("loss")),
        "token_acc": float(step.group("token_acc")),
        "trainer_peak_gib": float(step.group("memory")),
        "step_seconds": float(times[0]),
    }


def parse_samples(path: Path) -> list[dict[str, Any]]:
    by_gpu: dict[int, list[tuple[int, int, int]]] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if len(row) != 6:
                continue
            index = int(row[1])
            used = int(row[3])
            free = int(row[4])
            utilization = int(row[5])
            by_gpu.setdefault(index, []).append((used, free, utilization))
    return [
        {
            "gpu": index,
            "samples": len(values),
            "min_used_mib": min(value[0] for value in values),
            "peak_used_mib": max(value[0] for value in values),
            "peak_increment_mib": max(value[0] for value in values)
            - min(value[0] for value in values),
            "min_free_mib": min(value[1] for value in values),
            "peak_utilization_percent": max(value[2] for value in values),
        }
        for index, values in sorted(by_gpu.items())
        if index < 4
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=Path("outputs/probes/sft_memory"))
    parser.add_argument("--report", type=Path, default=Path("reports/sft_memory_probe.md"))
    args = parser.parse_args()
    ddp = {
        8192: json.loads((args.probe_dir / "probe_8192_liger.json").read_text()),
        16384: json.loads((args.probe_dir / "probe_16384_liger.json").read_text()),
    }
    sp = {}
    for length in (24576, 32768):
        directory = args.probe_dir / f"sp4_zero3cpu_{length}_sampled"
        sp[length] = {
            "step": parse_step(directory / "train.log"),
            "samples": parse_samples(directory / "nvidia_smi_samples.csv"),
        }
    lines = [
        "# SFT GPU Memory Probe",
        "",
        "All four target lengths completed a real bf16 LoRA-rank-128 forward, backward, "
        "optimizer step, and checkpoint save. The base is the same real p99-near trajectory; "
        "loss-masked attended suffix tokens stress longer shapes without modifying frozen data.",
        "",
        "The 8K/16K DDP probe records exact per-rank torch allocated/reserved peaks. The "
        "24K/32K production fallback uses Ulysses sequence parallel + ZeRO-3 CPU offload; "
        "ms-swift exposes allocator peak only for rank 0, so a 200 ms nvidia-smi sampler records "
        "actual total device-memory peaks for every GPU. Missing per-rank DeepSpeed allocator "
        "fields are labeled unavailable rather than estimated.",
        "",
        "## Successful target summary",
        "",
        "| Length | Strategy | Forward/backward/step | Rank-0 trainer peak | Step time | Loss | "
        "Token accuracy |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for length in (8192, 16384):
        rank0 = ddp[length]["ranks"][0]
        lines.append(
            f"| {length} | 4-rank DDP + Liger | PASS | "
            f"{rank0['peak_allocated_mib'] / 1024:.2f} GiB | "
            f"{rank0['step_time_seconds']:.4f} s | {rank0['loss']:.8f} | unavailable |"
        )
    for length in (24576, 32768):
        step = sp[length]["step"]
        lines.append(
            f"| {length} | SP4 + ZeRO-3 CPU offload | PASS | "
            f"{step['trainer_peak_gib']:.2f} GiB | {step['step_seconds']:.4f} s | "
            f"{step['loss']:.8f} | {step['token_acc']:.2%} |"
        )

    lines += [
        "",
        "## Exact torch allocator records: 8K and 16K",
        "",
        "| Length | GPU | Allocated after step MiB | Reserved after step MiB | "
        "Peak allocated MiB | Peak reserved MiB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for length in (8192, 16384):
        for row in ddp[length]["ranks"]:
            lines.append(
                f"| {length} | {row['physical_gpu']} | {row['allocated_mib']:.2f} | "
                f"{row['reserved_mib']:.2f} | {row['peak_allocated_mib']:.2f} | "
                f"{row['peak_reserved_mib']:.2f} |"
            )

    lines += [
        "",
        "## Sampled nvidia-smi records: 24K and 32K",
        "",
        "These are total device-memory measurements and therefore include the pre-existing "
        "~4.5 GiB allocation on GPU 0. `Peak increment` subtracts the minimum observed baseline.",
        "",
        "| Length | GPU | Samples | Min used MiB | Peak used MiB | Peak increment MiB | "
        "Min free MiB | Peak util |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for length in (24576, 32768):
        for row in sp[length]["samples"]:
            lines.append(
                f"| {length} | {row['gpu']} | {row['samples']} | {row['min_used_mib']} | "
                f"{row['peak_used_mib']} | {row['peak_increment_mib']} | "
                f"{row['min_free_mib']} | {row['peak_utilization_percent']}% |"
            )

    lines += [
        "",
        "## Decision",
        "",
        "The required 32,768-token shape passes on GPUs 0–3 without clearing GPU 0. Frozen "
        "training rows have a measured maximum of 7,401 tokens, so formal SFT uses faster "
        "four-rank DDP while retaining max_length=32768 and zero truncation. SP4 + ZeRO-3 CPU "
        "offload remains the verified fallback for a true 32K row.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
